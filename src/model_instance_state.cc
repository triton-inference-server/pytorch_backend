// Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#define ENABLE_DEBUG_TRACE_FUNCTION_CALL 1

#include "libtorch.hh"
#include "model_instance_state.hh"

#include "string_utils.hh"

#ifdef TRITON_PYTORCH_ENABLE_TORCHVISION
// Suppress warnings in torch headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma warning(push, 0)
#include <torchvision/ops/ops.h>
#include <torchvision/vision.h>  // Torchvision header
#pragma warning(pop)
#pragma GCC diagnostic pop
#endif  // TRITON_PYTORCH_ENABLE_TORCHVISION

#ifdef TRITON_ENABLE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU


namespace triton::backend::pytorch {

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), device_(torch::kCPU), is_dict_input_(false),
      device_cnt_(0)
{
  DEBUG_TRACE_FUNCTION_CALL();
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    device_ = torch::Device(torch::kCUDA, DeviceId());
    CreateCudaEvents(DeviceId());
#endif
  }

#ifdef TRITON_ENABLE_GPU
  device_cnt_ = torch::cuda::device_count();
#endif

  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), device_, &model_path_, Kind(), &torch_model_));

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
#ifdef TRITON_ENABLE_GPU
    // Since we cannot determine the exact devices used by the model, we create
    // a CUDA stream for every available device to ensure proper synchronization
    // of CUDA streams. This approach may have implications when a timestamp is
    // captured on a device that is not used by the model. Currently, this issue
    // is addressed by synchronizing the CUDA streams before recording
    // timestamps to prevent timestamp skewing. However, in the future, any
    // modifications to the CUDA stream synchronization logic should be handled
    // with caution.
    for (int i = 0; i < device_cnt_; i++) {
      cudaStream_t stream;
      THROW_IF_BACKEND_INSTANCE_ERROR(
          CreateCudaStream(i, 0 /* cuda_stream_priority */, &stream));
      stream_vec_.push_back(stream);
    }
    if (!stream_vec_.empty()) {
      // Create CUDA events on the first device that will be used for collecting
      // inputs/outputs.
      CreateCudaEvents(0);
    }
#endif
  }

  size_t expected_input_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (model_state->ModelConfig().Find("input", &inputs)) {
      expected_input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (model_state->ModelConfig().Find("batch_input", &config_batch_inputs)) {
      batch_input_count_ = config_batch_inputs.ArraySize();
      expected_input_cnt += batch_input_count_;
    }
  }

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  triton::common::TritonJson::Value sequence_batching;
  if (model_state->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    bool have_start, have_end, have_ready, have_corrid;
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_START", false /* required */,
        &have_start));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_END", false /* required */,
        &have_end));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_READY", false /* required */,
        &have_ready));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateTypedSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_CORRID", false /* required */,
        &have_corrid));
    if (have_start) {
      expected_input_cnt += 1;
    }
    if (have_end) {
      expected_input_cnt += 1;
    }
    if (have_ready) {
      expected_input_cnt += 1;
    }
    if (have_corrid) {
      expected_input_cnt += 1;
    }
    // Add the state inputs to the expected count
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      expected_input_cnt += states.ArraySize();
    }
  }
  supports_batching_ = model_state_->MaxBatchSize() > 0;

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs(expected_input_cnt));
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());
}

ModelInstanceState::~ModelInstanceState()
{
  DEBUG_TRACE_FUNCTION_CALL();
  torch_model_.reset();
  ClearCache();

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
#ifdef TRITON_ENABLE_GPU
    for (size_t i = 0; i < stream_vec_.size(); i++) {
      LOG_IF_ERROR(
          ConvertCUDAStatusToTritonError(
              cudaSetDevice(i), TRITONSERVER_ERROR_INTERNAL,
              "Failed to set the device"),
          "Failed to set the device");

      LOG_IF_ERROR(
          ConvertCUDAStatusToTritonError(
              cudaStreamDestroy(stream_vec_[i]), TRITONSERVER_ERROR_INTERNAL,
              "Failed to destroy cuda stream"),
          "~ModelInstanceState error: ");
      stream_vec_[i] = nullptr;
    }
#endif
  }
}

void
ModelInstanceState::AddInputToMap(
    NamingConvention naming_convention,
    const std::vector<std::string> allowed_inputs, const std::string& io_name,
    const uint32_t index)
{
  DEBUG_TRACE_FUNCTION_CALL();
  std::string deliminator = "__";

  if (is_dict_input_) {
    // If dictionary, index is irrelevant but we use the map to store the
    // input names since they are the keys for the dictionary
    input_index_map_[io_name] = index;
  } else {
    switch (naming_convention) {
      case NamingConvention::FORWARD_ARGUMENT: {
        auto itr =
            std::find(allowed_inputs.begin(), allowed_inputs.end(), io_name);
        if (itr != allowed_inputs.end()) {
          input_index_map_[io_name] =
              std::distance(allowed_inputs.begin(), itr);
        }
        return;
      }
      case NamingConvention::NAMED_INDEX: {
        int start_pos = io_name.find(deliminator);
        int ip_index = std::atoi(io_name.substr(start_pos + 2).c_str());
        input_index_map_[io_name] = ip_index;
        return;
      }
      case NamingConvention::STRICT_CONFIG_ORDERING: {
        input_index_map_[io_name] = index;
        return;
      }
    }
  }
}

void
ModelInstanceState::ClearCache()
{
  DEBUG_TRACE_FUNCTION_CALL();
#ifdef TRITON_ENABLE_GPU
  if (device_.is_cuda() ||
      ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0))) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
#endif  // TRITON_ENABLE_GPU
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  DEBUG_TRACE_FUNCTION_CALL();
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

void
ModelInstanceState::CreateCudaEvents(const int32_t& device_id)
{
  DEBUG_TRACE_FUNCTION_CALL();
#ifdef TRITON_ENABLE_GPU
  // Need to set the CUDA context so that the context that events are
  // created on match with contexts that events are recorded with.
  THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
      cudaSetDevice(device_id), TRITONSERVER_ERROR_INTERNAL,
      "Failed to set the device"));
  THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
      cudaEventCreate(&compute_input_start_event_), TRITONSERVER_ERROR_INTERNAL,
      "Failed to create cuda event"));
  THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
      cudaEventCreate(&compute_infer_start_event_), TRITONSERVER_ERROR_INTERNAL,
      "Failed to create cuda event"));
  THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
      cudaEventCreate(&compute_output_start_event_),
      TRITONSERVER_ERROR_INTERNAL, "Failed to create cuda event"));
#endif
}

void
ModelInstanceState::Execute(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count,
    std::vector<torch::jit::IValue>* input_tensors,
    std::vector<torch::jit::IValue>* output_tensors)
{
  DEBUG_TRACE_FUNCTION_CALL();
  NVTX_RANGE(nvtx_, "Execute " + Name());

  torch::jit::IValue model_outputs_;

  try {
    // enable/disable optimized execution
    torch::jit::setGraphExecutorOptimize(
        model_state_->EnabledOptimizedExecution());

    // enable/disable inference mode - supersedes NoGradGuard
    torch::InferenceMode infer_guard(model_state_->EnabledInferenceMode());

    // enable/disable cudnn
    at::globalContext().setUserEnabledCuDNN(model_state_->EnabledCudnn());

    // JIT. No change is made unless parameter is explicitly set.
    if (std::get<0>(model_state_->EnabledJitProfiling())) {
      torch::jit::getProfilingMode() =
          std::get<1>(model_state_->EnabledJitProfiling());
    }

    if (std::get<0>(model_state_->EnabledJitExecutor())) {
      torch::jit::getExecutorMode() =
          std::get<1>(model_state_->EnabledJitExecutor());
    }

    // Fuser. No change is made unless fuser is explicitly set in
    // parameters.
    if (std::get<0>(model_state_->EnabledTensorExprFuser())) {
      torch::jit::setTensorExprFuserEnabled(
          std::get<1>(model_state_->EnabledTensorExprFuser()));
    }

    torch::NoGradGuard no_grad;

    // If input is a dictionary, prepare dictionary from 'input_tensors'.
    if (is_dict_input_) {
      torch::Dict<std::string, torch::Tensor> input_dict;
      for (auto& input_index : input_index_map_) {
        torch::jit::IValue ival = (*input_tensors)[input_index.second];
        input_dict.insert(input_index.first, ival.toTensor());
      }
      std::vector<torch::jit::IValue> input_dict_ivalue = {input_dict};
      model_outputs_ = torch_model_->forward(input_dict_ivalue);
    } else {
      model_outputs_ = torch_model_->forward(*input_tensors);
    }

    if (model_outputs_.isTuple()) {
      auto model_outputs_tuple = model_outputs_.toTuple();
      size_t op_index = 0;
      for (auto& m_op : model_outputs_tuple->elements()) {
        if (m_op.isList()) {
          auto list_output = m_op.toList();
          if (list_output.elementType()->kind() != c10::TypeKind::StringType) {
            throw std::invalid_argument(
                "output at index " + std::to_string(op_index) +
                " must be of type Tensor or List[str], received List[" +
                list_output.elementType()->str() + "]");
          }
          output_tensors->push_back(m_op);
        } else {
          auto tensor_output = m_op.toTensor();
          output_tensors->push_back(m_op);
        }
        op_index++;
      }
    } else if (model_outputs_.isTensor()) {
      output_tensors->push_back(model_outputs_);
    } else if (model_outputs_.isList()) {
      auto list_output = model_outputs_.toList();
      if (list_output.elementType()->kind() != c10::TypeKind::StringType) {
        throw std::invalid_argument(
            "output must be of type Tensor or List[str], received List[" +
            list_output.elementType()->str() + "]");
      }
      output_tensors->push_back(model_outputs_);
    } else {
      throw std::invalid_argument(
          "output must be of type Tensor, List[str] or Tuple containing one of "
          "these two types. It should not be a List / Dictionary of Tensors or "
          "a Scalar");
    }
  }
  catch (std::exception& ex) {
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("PyTorch execute failure: " + std::string(ex.what())).c_str()));
  }
}

float
ModelInstanceState::GetCudaEventElapsedTime(
    const cudaEvent_t& start_event, const cudaEvent_t& end_event)
{
  DEBUG_TRACE_FUNCTION_CALL();
  float duration = 0;
#ifdef TRITON_ENABLE_GPU
  // [FIXME] in the case of cudaEventElapsedTime failure, should handle
  // stats reporting more gracefully as the durations are inaccurate
  LOG_IF_ERROR(
      ConvertCUDAStatusToTritonError(
          cudaEventElapsedTime(&duration, start_event, end_event),
          TRITONSERVER_ERROR_INTERNAL, "Failed to capture elapsed time"),
      "Failed to capture elapsed time");
#endif
  return duration;
}


cudaStream_t
ModelInstanceState::GetCudaStreamByInstanceKind()
{
  DEBUG_TRACE_FUNCTION_CALL();
#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    return stream_;
  } else if (
      (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) &&
      !stream_vec_.empty()) {
    return stream_vec_[0];
  }
#endif
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::GetNamingConvention(
    triton::backend::pytorch::NamingConvention* naming_convention,
    const std::vector<std::string>& allowed_ios)
{
  DEBUG_TRACE_FUNCTION_CALL();
  // Rules for (non-Dictionary) input tensor names:
  // 1. Must be in 'allowed_inputs' (arguments in the forward function)
  // 2. Must follow the naming convention i.e. <name>__<index>
  // 3. If neither of the above conditions are satisfied, enforce strict
  // ordering of model inputs.
  //
  // Rules for output tensor names:
  // 1. Must follow the naming convention i.e. <name>__<index>
  // 2. If not, we enforce strict ordering of model outputs.
  std::string deliminator = "__";
  std::string io_kind = "input";
  *naming_convention = NamingConvention::FORWARD_ARGUMENT;

  // symbolizes output
  if (allowed_ios.size() == 0) {
    io_kind = "output";
    *naming_convention = NamingConvention::NAMED_INDEX;
  }

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(
      model_state_->ModelConfig().MemberAsArray(io_kind.c_str(), &ios));

  if (io_kind == "input") {
    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

      // Validate name
      std::string io_name;
      RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
      auto itr = std::find(allowed_ios.begin(), allowed_ios.end(), io_name);
      if (itr == allowed_ios.end()) {
        *naming_convention = NamingConvention::NAMED_INDEX;
        break;
      }
    }
  }

  // If not, check if inputs follow INDEX
  if (*naming_convention == NamingConvention::NAMED_INDEX) {
    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

      // Validate name
      std::string io_name;
      RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
      int start_pos = io_name.find(deliminator);
      if (start_pos == -1) {
        *naming_convention = NamingConvention::STRICT_CONFIG_ORDERING;
        break;
      } else {
        // check if the index part of the name is not an integer
        std::string index_str = io_name.substr(start_pos + 2);
        bool is_int = true;
        for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
          if (std::isdigit(*itr) == 0) {
            is_int = false;
          }
        }

        if (!is_int) {
          if (io_kind == "input") {
            LOG_MESSAGE(
                TRITONSERVER_LOG_WARN,
                ("input '" + io_name +
                 "' or previous input(s) are neither an input argument to the "
                 "model '" +
                 model_state_->Name() +
                 "' nor do they follow the <name>__<index> naming convention. "
                 "Falling back to enforcing strict ordering from model "
                 "configuration.")
                    .c_str());
          } else {
            LOG_MESSAGE(
                TRITONSERVER_LOG_WARN,
                ("output '" + io_name +
                 "' or previous output(s) of the model '" +
                 model_state_->Name() +
                 "' do not follow the <name>__<index> naming convention. "
                 "Falling back to enforcing strict ordering from model "
                 "configuration.")
                    .c_str());
          }
          *naming_convention = NamingConvention::STRICT_CONFIG_ORDERING;
          break;
        }
      }
    }
  }

  triton::common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    // If we need to manage state for the model, then we need to check
    // the naming of the state adheres to both the input and output conventions
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      if (*naming_convention != NamingConvention::NAMED_INDEX) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("PyTorch model '" + model_state_->Name() +
             "' is using sequence batching with state but not all inputs and "
             "outputs follow the <name>__<index> naming convention. ")
                .c_str());
      }
    }

    for (size_t i = 0; i < states.ArraySize(); i++) {
      triton::common::TritonJson::Value state;
      RETURN_IF_ERROR(states.IndexAsObject(i, &state));
      std::string name_entry =
          io_kind == "input" ? "input_name" : "output_name";
      std::string state_name;
      RETURN_IF_ERROR(state.MemberAsString(name_entry.c_str(), &state_name));
      int start_pos = state_name.find(deliminator);
      if (start_pos == -1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            ("PyTorch model '" + model_state_->Name() +
             "' is using sequence batching with state but state '" +
             state_name +
             "' does not follow the <name>__<index> naming convention. ")
                .c_str());
      } else {
        // check if the index part of the name is not an integer
        std::string index_str = state_name.substr(start_pos + 2);
        bool is_int = true;
        for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
          if (std::isdigit(*itr) == 0) {
            is_int = false;
          }
        }
        if (!is_int) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              ("PyTorch model '" + model_state_->Name() +
               "' is using sequence batching with state but state '" +
               state_name +
               "' does not follow the <name>__<index> naming convention. ")
                  .c_str());
        }
      }
    }
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  DEBUG_TRACE_FUNCTION_CALL();
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    SetCurrentCudaStream(stream_, DeviceId());
  } else if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
    // Replace the default stream of each device with the one we created.
    for (size_t i = 0; i < stream_vec_.size(); i++) {
      SetCurrentCudaStream(stream_vec_[i], i);
    }
  }
#endif

  NVTX_RANGE(nvtx_, "ProcessRequests " + Name());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to PyTorch backend for '" + Name() + "'")
                  .c_str()));
      return;
    }
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  for (size_t i = 0; i < request_count; i++) {
    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size.
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed, err);
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if (!all_response_failed) {
    if ((total_batch_size != 1) &&
        (total_batch_size > (size_t)max_batch_size)) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "batch size " + std::to_string(total_batch_size) + " for '" +
                  Name() + "', max allowed is " +
                  std::to_string(max_batch_size))
                  .c_str()));
    }
  }

  std::vector<const char*> input_names;
  std::vector<torch::jit::IValue> input_tensors;
  bool cuda_copy = false;
  std::unique_ptr<BackendInputCollector> collector;

  // For 'KIND_MODEL', it's fine to use CUDA events to calculate the compute
  // input duration since only one stream will be used for input collection.
  if ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) ||
      ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0))) {
#ifdef TRITON_ENABLE_GPU
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventRecord(
                compute_input_start_event_, GetCudaStreamByInstanceKind()),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
#endif
  }

  if (!all_response_failed) {
    collector.reset(new BackendInputCollector(
        requests, request_count, &responses,
        model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
        GetCudaStreamByInstanceKind(), nullptr, nullptr, 0,
        HostPolicyName().c_str()));
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        SetInputTensors(
            total_batch_size, requests, request_count, &responses,
            collector.get(), &input_names, &input_tensors, &cuda_copy));
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(GetCudaStreamByInstanceKind());
    cuda_copy = false;
  }
#endif

  std::vector<torch::jit::IValue> output_tensors;
  uint64_t compute_start_ns = 0;
  uint64_t compute_infer_start = 0;

  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed,
      RecordBackendTimestamp(
          &compute_start_ns,
          reinterpret_cast<void*>(&compute_infer_start_event_)));

  // For 'KIND_MODEL', capture the timestamp for the compute infer duration.
  if ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0)) {
    SET_TIMESTAMP(compute_infer_start);
  }

  // Run...
  if (!all_response_failed) {
    Execute(&responses, request_count, &input_tensors, &output_tensors);
  }

  // Verify output indices are valid with number of outputs after execution
  bool invalid_index = false;
  int max_index = output_tensors.size() - 1;

  if (!all_response_failed) {
    for (const auto& name : model_state_->ModelOutputs()) {
      int op_index = output_index_map_[name.first];
      if ((op_index < 0) || (op_index > max_index)) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "The output " + std::string(name.first) +
                    " in the model configuration refers to an output index "
                    "which doesn't exist. This model has " +
                    std::to_string(max_index + 1) + " outputs")
                    .c_str()));
        invalid_index = true;
        break;
      }
    }
  }

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
    // For 'KIND_MODEL', multiple streams will be involved, so we need to call
    // 'cudaStreamSynchronize' before reading the output tensors.
    for (auto& stream : stream_vec_) {
      cudaStreamSynchronize(stream);
    }
  }
#endif

  uint64_t compute_end_ns = 0;
  uint64_t compute_output_start = 0;

  if ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0)) {
#ifdef TRITON_ENABLE_GPU
    SET_TIMESTAMP(compute_output_start);
#endif
  } else {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        RecordBackendTimestamp(
            &compute_end_ns,
            reinterpret_cast<void*>(&compute_output_start_event_)));
  }

  if (!all_response_failed) {
    if (!invalid_index) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          ReadOutputTensors(
              total_batch_size, output_tensors, requests, request_count,
              &responses));
    }
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send PyTorch backend response");
    }
  }

  // We don't need an explicit CUDA syncrhonization here since we have already
  // synchronized the stream in the ReadOutputTensors function.
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    float compute_input_duration = GetCudaEventElapsedTime(
        compute_input_start_event_, compute_infer_start_event_);
    float compute_infer_duration = GetCudaEventElapsedTime(
        compute_infer_start_event_, compute_output_start_event_);

    compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
    compute_end_ns = compute_start_ns + (compute_infer_duration * 1e6);
#endif
  } else if (
      (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0)) {
#ifdef TRITON_ENABLE_GPU
    float compute_input_duration = GetCudaEventElapsedTime(
        compute_input_start_event_, compute_infer_start_event_);
    uint64_t compute_infer_duration =
        compute_output_start - compute_infer_start;

    compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
    compute_end_ns = compute_start_ns + compute_infer_duration;
#endif
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  if (!all_response_failed) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            TritonModelInstance(), total_batch_size, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting batch request statistics");
  }
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size,
    const std::vector<torch::jit::IValue>& output_tensors,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  DEBUG_TRACE_FUNCTION_CALL();
  NVTX_RANGE(nvtx_, "ReadOutputTensors " + Name());

  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->MaxBatchSize() > 0, model_state_->EnablePinnedInput(),
      GetCudaStreamByInstanceKind());

  bool cuda_copy = false;
  // The serialized string buffer must be valid until output copies are done
  std::vector<std::unique_ptr<std::string>> string_buffer;
  for (auto& output : model_state_->ModelOutputs()) {
    int op_index = output_index_map_[output.first];
    auto name = output.first;
    auto output_tensor_pair = output.second;

    if (output_tensors[op_index].isTensor()) {
      torch::Tensor output_flat;
      try {
        output_flat =
            output_tensors[op_index].toTensor().contiguous().flatten();
      }
      catch (std::exception& ex) {
        RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("output tensor '") + name + "' is not found")
                .c_str()));
      }

      // Verify output datatype matches datatype from model config
      TRITONSERVER_DataType output_dtype =
          ConvertTorchTypeToDataType(output_flat.scalar_type());
      TRITONSERVER_DataType config_datatype = output_dtype_map_[name];
      if (config_datatype != output_dtype) {
        RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("configuration expects datatype TYPE_") +
             TRITONSERVER_DataTypeString(config_datatype) + " for output '" +
             name + "', model provides TYPE_" +
             TRITONSERVER_DataTypeString(output_dtype))
                .c_str()));
      }

      const char* output_buffer =
          static_cast<const char*>(output_flat.data_ptr());

      // Output tensors may not reside on the same device as model
      torch::Device tensor_device = output_flat.device();
      const auto memory_type = (tensor_device.type() == torch::kCPU)
                                   ? TRITONSERVER_MEMORY_CPU
                                   : TRITONSERVER_MEMORY_GPU;
      const auto memory_id =
          (tensor_device.type() == torch::kCPU) ? 0 : tensor_device.index();

      // Batch output doesn't support string data type yet, as it is not trivial
      // to parse string output
      const BatchOutput* batch_output = StateForModel()->FindBatchOutput(name);
      if (batch_output == nullptr) {
        // Get output shape
        std::vector<int64_t> batchn_shape;
        auto shape = output_tensors[op_index].toTensor().sizes();
        for (auto itr = shape.begin(); itr != shape.end(); itr++) {
          batchn_shape.push_back(*itr);
        }

        if (batchn_shape.size() == 0) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("output '") + name +
               "' is a scalar which is not supported.")
                  .c_str());
        }
        if (output_tensor_pair.first != -1) {
          responder.ProcessTensor(
              name, output_dtype, batchn_shape, output_buffer, memory_type,
              memory_id);
        }
        if (output_tensor_pair.second != -1) {
          std::vector<TRITONBACKEND_State*> states;
          states = responder.ProcessStateTensor(
              name, output_dtype, batchn_shape, output_buffer, memory_type,
              memory_id);
          // Update the states
          for (auto& state : states) {
            RETURN_IF_ERROR(TRITONBACKEND_StateUpdate(state));
          }
        }

      } else {
        responder.ProcessBatchOutput(
            name, *batch_output, output_buffer, memory_type, memory_id);
      }
    } else if (output_tensors[op_index].isList()) {
      // Custom handling for string/bytes tensor...
      torch::List<torch::jit::IValue> output_list =
          output_tensors[op_index].toList();

      // Get output shape
      std::vector<int64_t> batchn_shape{(int64_t)output_list.size()};

      for (size_t idx = 0; idx < responses->size(); idx++) {
        auto& request = requests[idx];
        auto& response = (*responses)[idx];

        if (supports_batching_ != 0) {
          TRITONBACKEND_Input* input;
          TRITONBACKEND_RequestInputByIndex(request, 0 /* index*/, &input);
          const int64_t* shape;
          TRITONBACKEND_InputProperties(
              input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
          batchn_shape[0] = shape[0];
        }

        int64_t tensor_element_cnt = 0;
        RETURN_IF_ERROR(GetElementCount(batchn_shape, &tensor_element_cnt));

        // Only need an response tensor for requested outputs.
        if (response != nullptr) {
          if (output_tensor_pair.first != -1) {
            TRITONBACKEND_Output* response_output;
            RESPOND_AND_SET_NULL_IF_ERROR(
                &response, TRITONBACKEND_ResponseOutput(
                               response, &response_output, name.c_str(),
                               TRITONSERVER_TYPE_BYTES, batchn_shape.data(),
                               batchn_shape.size()));
            string_buffer.emplace_back(new std::string());
            cuda_copy |= SetStringOutputBuffer(
                &output_list, &response, response_output, tensor_element_cnt,
                GetCudaStreamByInstanceKind(), string_buffer.back().get());
          }
        }
        if (output_tensor_pair.second != -1) {
          TRITONBACKEND_State* response_state;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_StateNew(
                             &response_state, request, name.c_str(),
                             TRITONSERVER_TYPE_BYTES, batchn_shape.data(),
                             batchn_shape.size()));

          string_buffer.emplace_back(new std::string());
          cuda_copy |= SetStringStateBuffer(
              &output_list, &response, response_state, tensor_element_cnt,
              GetCudaStreamByInstanceKind(), string_buffer.back().get());
        }
      }
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("output '") + name +
           "' must be of type Tensor or List[str].")
              .c_str());
    }
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  // We have to always synchronize the stream. This is to make sure that
  // the events on the cuda stream are synchronized. Otherwise, the events
  // are only guaranteed to be synchronized if the model provides the output
  // on GPU.
  cudaStreamSynchronize(GetCudaStreamByInstanceKind());
#endif

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::RecordBackendTimestamp(
    uint64_t* timestamp, void* cuda_event)
{
  DEBUG_TRACE_FUNCTION_CALL();
  if ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) ||
      ((Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) && (device_cnt_ > 0))) {
#ifdef TRITON_ENABLE_GPU
    cudaEvent_t* lcuda_event = reinterpret_cast<cudaEvent_t*>(cuda_event);
    RETURN_IF_ERROR(ConvertCUDAStatusToTritonError(
        cudaEventRecord(*lcuda_event, GetCudaStreamByInstanceKind()),
        TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
#endif
  } else {
    SET_TIMESTAMP(*timestamp);
  }
  return nullptr;
}

void
ModelInstanceState::SetCurrentCudaStream(
    const cudaStream_t& stream, const int& device_id)
{
#ifdef TRITON_ENABLE_GPU
  at::cuda::CUDAStream torch_stream =
      at::cuda::getStreamFromExternal(stream, device_id);
  // This function replaces the default stream with the stream we created. It
  // is not necessary to change the current device to the desired device when
  // replacing the default stream for that device. See the documentation here:
  // https://pytorch.org/cppdocs/api/function_namespacec10_1_1cuda_1a6ed50cc0fc16cc7014d9c2f4c3bd098d.html
  at::cuda::setCurrentCUDAStream(torch_stream);
#endif
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<torch::jit::IValue>* input_tensors, bool* cuda_copy)
{
  DEBUG_TRACE_FUNCTION_CALL();
  // InferenceMode should be used to guard all tensors operations
  torch::InferenceMode infer_guard(model_state_->EnabledInferenceMode());

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  input_tensors->resize(input_count + batch_input_count_);

  // The inputs must be in contiguous CPU/GPU memory.
  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
  if (device_.is_cpu()) {
    alloc_perference = {
        {TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};
  } else {
    alloc_perference = {{TRITONSERVER_MEMORY_GPU, device_.index()}};
  }

  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    input_names->emplace_back(input_name);

    // The shape for the entire input patch,
    // [total_batch_size, ...] for non-ragged input and
    // [total_element_count] for ragged input (non-nested tensor)
    std::vector<int64_t> batchn_shape;
    if (StateForModel()->IsInputRagged(input_name)) {
      batchn_shape = std::vector<int64_t>{0};
      for (size_t idx = 0; idx < request_count; idx++) {
        TRITONBACKEND_Input* input;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
        const int64_t* input_shape;
        uint32_t input_dims_count;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]), TRITONBACKEND_InputProperties(
                                      input, nullptr, nullptr, &input_shape,
                                      &input_dims_count, nullptr, nullptr));

        int64_t element_cnt = 0;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            GetElementCount(input_shape, input_dims_count, &element_cnt));
        batchn_shape[0] += element_cnt;
      }
    } else {
      batchn_shape =
          std::vector<int64_t>(input_shape, input_shape + input_dims_count);
      if (supports_batching_) {
        batchn_shape[0] = total_batch_size;
      }
    }

    // The input must be in contiguous CPU/GPU memory.
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    // For 'KIND_MODEL', input will always be in CPU as we don't have a way to
    // query the input types.
    if (device_.is_cpu() || (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL)) {
      alloc_perference = {
          {TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};
    } else {
      alloc_perference = {{TRITONSERVER_MEMORY_GPU, device_.index()}};
    }

    const char* input_buffer;
    size_t batchn_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    RETURN_IF_ERROR(collector->ProcessTensor(
        input_name, nullptr, 0, alloc_perference, &input_buffer,
        &batchn_byte_size, &memory_type, &memory_type_id));

    // Create Torch tensor
    const auto torch_dtype = ConvertDataTypeToTorchType(input_datatype);
    torch::TensorOptions options{torch_dtype.second};
    auto updated_options = (memory_type == TRITONSERVER_MEMORY_GPU)
                               ? options.device(torch::kCUDA, device_.index())
                               : options.device(torch::kCPU);

    if (input_datatype == TRITONSERVER_TYPE_BYTES) {
      // Create the PyTorch list to hold the strings.
      torch::List<std::string> input_list;
      input_list.reserve(batchn_shape[0]);

      for (size_t idx = 0; idx < request_count; idx++) {
        TRITONBACKEND_Input* input;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
        const int64_t* shape;
        uint32_t dims_count;
        uint32_t buffer_count;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_InputPropertiesForHostPolicy(
                input, HostPolicyName().c_str(), nullptr, nullptr, &shape,
                &dims_count, nullptr, &buffer_count));

        int64_t batch_element_cnt = 0;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            GetElementCount(shape, dims_count, &batch_element_cnt));

        *cuda_copy |= SetStringInputTensor(
            &input_list, input, input_name, buffer_count, batch_element_cnt,
            &((*responses)[idx]), GetCudaStreamByInstanceKind(),
            HostPolicyName().c_str());
      }

      (*input_tensors)[input_index_map_[input_name]] = input_list;
    } else {
      if (batchn_byte_size) {
        // Remove constness to align with the signature of torch::from_blob()
        torch::Tensor input_tensor = torch::from_blob(
            const_cast<char*>(input_buffer), batchn_shape, updated_options);
        (*input_tensors)[input_index_map_[input_name]] = input_tensor;
      } else {
        // torch:from_blob seems not working when the input size is 0
        // create zero-length inputs directly
        torch::Tensor input_tensor =
            torch::zeros(batchn_shape, updated_options);
        (*input_tensors)[input_index_map_[input_name]] = input_tensor;
      }
    }
  }

  for (const auto& batch_input : StateForModel()->BatchInputs()) {
    std::vector<int64_t> shape;
    collector->BatchInputShape(batch_input, &shape);

    for (const auto& input_name : batch_input.TargetNames()) {
      input_names->emplace_back(input_name.c_str());

      const char* dst_buffer;
      size_t dst_buffer_byte_size;
      TRITONSERVER_MemoryType dst_memory_type;
      int64_t dst_memory_type_id;

      RESPOND_ALL_AND_SET_NULL_IF_ERROR(
          (*responses), responses->size(),
          collector->ProcessBatchInput(
              batch_input, nullptr, 0, alloc_perference, &dst_buffer,
              &dst_buffer_byte_size, &dst_memory_type, &dst_memory_type_id));

      const auto torch_dtype =
          ConvertDataTypeToTorchType(batch_input.DataType());
      torch::TensorOptions options{torch_dtype.second};
      auto updated_options = (dst_memory_type == TRITONSERVER_MEMORY_GPU)
                                 ? options.device(torch::kCUDA, device_.index())
                                 : options.device(torch::kCPU);

      if (dst_buffer_byte_size) {
        torch::Tensor input_tensor = torch::from_blob(
            const_cast<char*>(dst_buffer), shape, updated_options);
        (*input_tensors)[input_index_map_[input_name]] = input_tensor;
      } else {
        // special handle when input has zero size
        torch::Tensor input_tensor = torch::zeros(shape, updated_options);
        (*input_tensors)[input_index_map_[input_name]] = input_tensor;
      }
    }
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();

  return nullptr;
}

ModelState*
ModelInstanceState::StateForModel() const
{
  return model_state_;
}

TRITONSERVER_Error*
ModelInstanceState::ValidateBooleanSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  DEBUG_TRACE_FUNCTION_CALL();
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    std::string deliminator = "__";
    int ip_index = 0;
    int start_pos = tensor_name.find(deliminator);
    if (start_pos == -1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("input '" + tensor_name +
           "' does not follow <name>__<index> naming convention.")
              .c_str());
    }

    // check if the index part of the name is not an integer
    std::string index_str = tensor_name.substr(start_pos + 2);
    for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
      if (std::isdigit(*itr) == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("input '" + tensor_name +
             "' does not follow <name>__<index> naming convention.")
                .c_str());
      }
    }

    ip_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
    input_index_map_[tensor_name] = ip_index;
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs(const size_t expected_input_cnt)
{
  DEBUG_TRACE_FUNCTION_CALL();
  // Collect all the expected input tensor names and validate that the model
  // configuration specifies only those.
  std::vector<std::string> allowed_inputs;

  const torch::jit::Method& method = torch_model_->get_method("forward");
  const auto& schema = method.function().getSchema();
  const std::vector<c10::Argument>& arguments = schema.arguments();

  // Currently, only models with a single input of type Dict(str, Tensor) are
  // supported. If the model expects more than one input then they must be all
  // be of type Tensor.
  //
  // Ignore the argument at idx 0 if it is of Class type (self param in forward
  // function)
  size_t start_idx = 0;
  if ((arguments.size() > 0) &&
      (arguments.at(0).type()->kind() == c10::TypeKind::ClassType)) {
    start_idx = 1;
  }
  if ((arguments.size() == (1 + start_idx)) &&
      (arguments.at(start_idx).type()->kind() == c10::TypeKind::DictType)) {
    is_dict_input_ = true;
  } else if (arguments.size() > start_idx) {
    // Return error if multiple inputs are of kind DictType
    for (size_t i = start_idx + 1; i < arguments.size(); i++) {
      if (arguments.at(i).type()->kind() == c10::TypeKind::DictType) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "Multiple inputs of kind DictType were detected. Only a single "
            "input of type Dict(str, Tensor) is supported.");
      }
    }

    // Return error if all inputs are not of type Tensor
    for (size_t i = start_idx; i < arguments.size(); i++) {
      if ((arguments.at(i).type()->kind() != c10::TypeKind::TensorType) &&
          (arguments.at(i).type()->kind() != c10::TypeKind::ListType)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("An input of type '") + arguments.at(i).type()->str() +
             "' was detected in the model. Only a single input of type "
             "Dict(str, Tensor) or input(s) of type Tensor are supported.")
                .c_str());
      }
      allowed_inputs.emplace_back(arguments.at(i).name());
    }

    // If all inputs are tensors, match number of expected inputs between model
    // and configuration
    if ((arguments.size() - start_idx) != expected_input_cnt) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects " + std::to_string(expected_input_cnt) +
           " inputs, model provides " +
           std::to_string(arguments.size() - start_idx))
              .c_str());
    }
  }

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));

  if (ios.ArraySize() == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "model configuration must contain at least one input, none were "
        "specified.");
  }

  triton::backend::pytorch::NamingConvention naming_convention;
  RETURN_IF_ERROR(GetNamingConvention(&naming_convention, allowed_inputs));

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Validate name
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    AddInputToMap(naming_convention, allowed_inputs, io_name, i);
    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTorchType(io_dtype);
    if (!pr.first && (io_dtype != "TYPE_STRING")) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for input '" + io_name +
           "' for model '" + model_state_->Name() + "'")
              .c_str());
    }

    // Validate shape for String inputs. Only allow 1 dimension.
    if (io_dtype == "TYPE_STRING") {
      // If a reshape is provided for the input then use that when
      // validating the model shapes.
      std::vector<int64_t> dims;
      triton::common::TritonJson::Value reshape;
      if (io.Find("reshape", &reshape)) {
        RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
      } else {
        RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
      }

      if ((dims.size() + (supports_batching_ ? 1 : 0)) > 1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Triton only supports 1 dimensional List of String as input for "
             "'" +
             std::string(io_name) + "' for model '" + model_state_->Name() +
             "'")
                .c_str());
      }
    }
  }
  triton::common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value state;
        RETURN_IF_ERROR(states.IndexAsObject(i, &state));
        std::string state_name;
        RETURN_IF_ERROR(state.MemberAsString("input_name", &state_name));
        AddInputToMap(naming_convention, allowed_inputs, state_name, i);

        // Validate data type
        std::string state_dtype;
        RETURN_IF_ERROR(state.MemberAsString("data_type", &state_dtype));
        const auto pr = ModelConfigDataTypeToTorchType(state_dtype);
        if (!pr.first && (state_dtype != "TYPE_STRING")) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              ("unsupported datatype " + state_dtype + " for input state '" +
               state_name + "' for model '" + model_state_->Name() + "'")
                  .c_str());
        }

        // Validate shape for String inputs. Only allow 1 dimension.
        if (state_dtype == "TYPE_STRING") {
          std::vector<int64_t> dims;
          if ((dims.size() + (supports_batching_ ? 1 : 0)) > 1) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                ("Triton only supports 1 dimensional List of String as input "
                 "for "
                 "'" +
                 std::string(state_name) + "' for model '" +
                 model_state_->Name() + "'")
                    .c_str());
          }
        }
      }
    }
  }

  triton::common::TritonJson::Value batch_inputs;
  RETURN_IF_ERROR(
      model_state_->ModelConfig().MemberAsArray("batch_input", &batch_inputs));
  size_t i = 0;
  for (const auto& batch_input : StateForModel()->BatchInputs()) {
    for (const auto& input_name : batch_input.TargetNames()) {
      AddInputToMap(
          naming_convention, allowed_inputs, input_name, i + ios.ArraySize());
      i++;
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
  DEBUG_TRACE_FUNCTION_CALL();
  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));
  std::string deliminator = "__";
  int op_index = 0;

  if (ios.ArraySize() == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "model configuration must contain at least one output, none were "
        "specified.");
  }

  NamingConvention naming_convention;
  RETURN_IF_ERROR(GetNamingConvention(&naming_convention, {}));

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Validate name
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    switch (naming_convention) {
      case NamingConvention::NAMED_INDEX: {
        int start_pos = io_name.find(deliminator);
        op_index = std::atoi(io_name.substr(start_pos + 2).c_str());
        break;
      }
      case NamingConvention::STRICT_CONFIG_ORDERING: {
        op_index = i;
        break;
      }
      default:
        break;
    }

    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTorchType(io_dtype);
    if (!pr.first && (io_dtype != "TYPE_STRING")) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for output '" + io_name +
           "' for model '" + model_state_->Name() + "'")
              .c_str());
    }

    // Validate shape for String outputs. Only allow 1 dimension.
    if (io_dtype == "TYPE_STRING") {
      // If a reshape is provided for the output then use that when
      // validating the model shapes.
      std::vector<int64_t> dims;
      triton::common::TritonJson::Value reshape;
      if (io.Find("reshape", &reshape)) {
        RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
      } else {
        RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
      }

      if ((dims.size() + (supports_batching_ ? 1 : 0)) > 1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Triton only supports 1 dimensional List of String as output for "
             "'" +
             std::string(io_name) + "' for model '" + model_state_->Name() +
             "'")
                .c_str());
      }
    }

    output_index_map_[io_name] = op_index;
    output_dtype_map_[io_name] = ConvertTorchTypeToDataType(pr.second);
  }

  triton::common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value state;
        RETURN_IF_ERROR(states.IndexAsObject(i, &state));
        std::string state_name;
        RETURN_IF_ERROR(state.MemberAsString("output_name", &state_name));
        std::string state_dtype;
        RETURN_IF_ERROR(state.MemberAsString("data_type", &state_dtype));
        std::vector<int64_t> dims;
        RETURN_IF_ERROR(ParseShape(state, "dims", &dims));

        // For state, naming convention is enforced to be NAMED_INDEX
        int start_pos = state_name.find(deliminator);
        op_index = std::atoi(state_name.substr(start_pos + 2).c_str());

        const auto pr = ModelConfigDataTypeToTorchType(state_dtype);
        if (!pr.first && (state_dtype != "TYPE_STRING")) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              ("unsupported datatype " + state_dtype + " for state '" +
               state_name + "' for model '" + model_state_->Name() + "'")
                  .c_str());
        }

        // Validate shape for String outputs. Only allow 1 dimension.
        if (state_dtype == "TYPE_STRING") {
          if ((dims.size() + (supports_batching_ ? 1 : 0)) > 1) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                ("Triton only supports 1 dimensional List of String as output "
                 "for "
                 "'" +
                 std::string(state_name) + "' for model '" +
                 model_state_->Name() + "'")
                    .c_str());
          }
        }

        output_index_map_[state_name] = op_index;
        output_dtype_map_[state_name] = ConvertTorchTypeToDataType(pr.second);
      }
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateTypedSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  DEBUG_TRACE_FUNCTION_CALL();
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetTypedSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    std::string deliminator = "__";
    int ip_index = 0;
    int start_pos = tensor_name.find(deliminator);
    if (start_pos == -1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("input '" + tensor_name +
           "' does not follow <name>__<index> naming convention.")
              .c_str());
    }

    // check if the index part of the name is not an integer
    std::string index_str = tensor_name.substr(start_pos + 2);
    for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
      if (std::isdigit(*itr) == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("input '" + tensor_name +
             "' does not follow <name>__<index> naming convention.")
                .c_str());
      }
    }

    // check if the data type is supported by PyTorch
    if (!ModelConfigDataTypeToTorchType(tensor_datatype).first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("input '" + tensor_name + "' type '" + tensor_datatype +
           "' is not supported by PyTorch.")
              .c_str());
    }

    ip_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
    input_index_map_[tensor_name] = ip_index;
  }

  return nullptr;  // success
}


}  // namespace triton::backend::pytorch
