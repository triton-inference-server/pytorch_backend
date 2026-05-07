// Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "model_instance_state.hh"

#include <cstddef>

#include "../libtorch.hh"
#include "../libtorch_utils.h"
#include "../string_utils.hh"
#include "triton/common/nvtx.h"
#include "triton_exception.hh"
#include "triton_utils.hh"

#ifdef TRITON_ENABLE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#endif

namespace triton::backend::pytorch::pt2 {
using TritonModelState = triton::backend::pytorch::pt2::ModelState;
using TritonJsonValue = triton::common::TritonJson::Value;
using TritonNamingConvention = triton::backend::pytorch::NamingConvention;

static const std::string DELIMINATOR{"__"};

ModelInstanceState::ModelInstanceState(
    TritonModelState* model, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance{model, triton_model_instance}, model_{model}
{
  if (!model)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL, "Argument `model` cannot be `null`.")
  if (!triton_model_instance)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Argument `triton_model_instance` cannot be `null`.")

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    DEBUG_TRACE_INFO(
        "device = torch::Device(torch::kCUDA, " << DeviceId() << ")");
    device_ = torch::Device{torch::kCUDA, static_cast<int8_t>(DeviceId())};
    CreateCudaEvents(DeviceId());
  }

  device_count_ = torch::cuda::device_count();
#endif

  model_->LoadModel(
      /* model_file_name= */ ArtifactFilename(),
      /* device= */ device_,
      /* device_count= */ device_count_,
      /* kind= */ Kind());

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    DEBUG_TRACE_INFO(
        "{ Kind(): TRITONSERVER_INSTANCEGROUPKIND_GPU, len(device_count_): "
        << device_count_ << " }");
    // Since we cannot determine the exact devices used by the model, we create
    // a CUDA stream for every available device to ensure proper synchronization
    // of CUDA streams. This approach may have implications when a timestamp is
    // captured on a device that is not used by the model. Currently, this issue
    // is addressed by synchronizing the CUDA streams before recording
    // timestamps to prevent timestamp skewing. However, in the future, any
    // modifications to the CUDA stream synchronization logic should be handled
    // with caution.
    for (int i = 0; i < device_count_; i += 1) {
      cudaStream_t stream;
      if (auto err = CreateCudaStream(i, 0 /* cuda_stream_priority */, &stream))
        throw triton::backend::BackendModelInstanceException(err);

      stream_vector_.push_back(stream);
    }

    if (!stream_vector_.empty()) {
      // Create CUDA events on the first device that will be used for collecting
      // inputs/outputs.
      CreateCudaEvents(0);
    }
  }
#endif

  auto model_inputs = model_->InputMap();
  auto model_outputs = model_->OutputMap();

  for (auto [input_name, input_index] : model_inputs) {
    std::string ordinal_name = "INPUT__" + std::to_string(input_index);
    map_inputs_.emplace(input_name, ordinal_name);
    map_inputs_[input_name].model_index() = input_index;
  }

  for (auto [output_name, output_index] : model_outputs) {
    std::string ordinal_name = "OUTPUT__" + std::to_string(output_index);
    map_outputs_.emplace(output_name, ordinal_name);
    map_outputs_[output_name].model_index() = output_index;
  }

  size_t expected_input_count{model_inputs.size()};

  TritonJsonValue inputs;
  if (model_->ModelConfig().Find("input", &inputs)) {
    expected_input_count = inputs.ArraySize();
    if (expected_input_count != inputs.ArraySize())
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Model instance \""
              << Name()
              << "\" has mismatching input count between model config and "
                 "model definition: model config input count = "
              << inputs.ArraySize()
              << ", model definition input count = " << expected_input_count);
  }

  TritonJsonValue config_batch_inputs;
  if (model_->ModelConfig().Find("batch_input", &config_batch_inputs)) {
    batch_input_count_ = config_batch_inputs.ArraySize();
    expected_input_count += batch_input_count_;
  }

  // If this is a sequence model then make sure that the required inputs are
  // present in the model and have the correct shape and datatype.
  TritonJsonValue sequence_batching;
  if (model_->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    if (ValidateBooleanSequenceControl(
            sequence_batching, "CONTROL_SEQUENCE_START",
            false /* required */)) {
      DEBUG_TRACE_INFO(
          "ValidateBooleanSequenceControl(sequence_batching,"
          " \"CONTROL_SEQUENCE_START\", false /* required */) -> true");
      expected_input_count += 1;
    }

    if (ValidateBooleanSequenceControl(
            sequence_batching, "CONTROL_SEQUENCE_END", false /* required */)) {
      DEBUG_TRACE_INFO(
          "ValidateBooleanSequenceControl(sequence_batching,"
          " \"CONTROL_SEQUENCE_END\", false /* required */) -> true");
      expected_input_count += 1;
    }
    if (ValidateBooleanSequenceControl(
            sequence_batching, "CONTROL_SEQUENCE_READY",
            false /* required */)) {
      DEBUG_TRACE_INFO(
          "ValidateBooleanSequenceControl(sequence_batching,"
          " \"CONTROL_SEQUENCE_READY\", false /* required */) -> true");
      expected_input_count += 1;
    }
    if (ValidateBooleanSequenceControl(
            sequence_batching, "CONTROL_SEQUENCE_CORRID",
            false /* required */)) {
      DEBUG_TRACE_INFO(
          "ValidateBooleanSequenceControl(sequence_batching,"
          " \"CONTROL_SEQUENCE_CORRID\", false /* required */) -> true");
      expected_input_count += 1;
    }

    // Add the state inputs to the expected count.
    TritonJsonValue states;
    if (sequence_batching.Find("state", &states)) {
      expected_input_count += states.ArraySize();
    }
  }

  is_batching_supported_ = model_->MaxBatchSize() > 0;

  DEBUG_TRACE_INFO(
      "{ device_count: " << device_count_
                         << ", expected_input_count: " << expected_input_count
                         << ", batch_input_count_: " << batch_input_count_
                         << ", is_batching_supported: "
                         << (is_batching_supported_ ? "true" : "false")
                         << " }");

  ValidateInputs(expected_input_count);
  ValidateOutputs();
}

ModelInstanceState::~ModelInstanceState()
{
  ClearCache();

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
    DEBUG_TRACE_INFO(
        "{ Kind(): TRITONSERVER_INSTANCEGROUPKIND_MODEL, len(stream_vector_): "
        << stream_vector_.size() << " }");
    for (size_t i = 0; i < stream_vector_.size(); i += 1) {
      if (auto err = ConvertCUDAStatusToTritonError(
              /* cuda_error= */ cudaSetDevice(i),
              /* code= */ TRITONSERVER_ERROR_INTERNAL,
              /* msg= */ "Failed to set device for stream destruction")) {
        TRITON_LOG_ERROR(
            "Failed to set the device while destroying streams for instance \""
            << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
      }

      if (auto err = ConvertCUDAStatusToTritonError(
              /* cuda_error= */ cudaStreamDestroy(stream_vector_[i]),
              /* code= */ TRITONSERVER_ERROR_INTERNAL,
              /* msg= */ "Failed to destroy cuda stream")) {
        TRITON_LOG_ERROR(
            "Failed to destroy cuda stream for instance \""
            << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
      }

      stream_vector_[i] = nullptr;
    }
  }
#endif
}

const std::string&
ModelInstanceState::ArtifactFilename() const
{
  return BackendModelInstance::ArtifactFilename();
}

void
ModelInstanceState::ClearCache()
{
#ifdef TRITON_ENABLE_GPU
  if (device_.is_cuda() ||
      (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0)) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
#endif
}

ModelInstanceState*
ModelInstanceState::Create(
    TritonModelState* model, TRITONBACKEND_ModelInstance* triton_model_instance)
{
  if (!model)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL, "Argument `model` cannot be `null`.");
  if (!triton_model_instance)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Argument `triton_model_instance` cannot be `null`.");

  try {
    return new ModelInstanceState(model, triton_model_instance);
  }
  catch (const triton::backend::pytorch::BackendException& exception) {
    DEBUG_TRACE_ERROR(
        "{ model_name: \"" << model->Name() << "\", error: \""
                           << exception.what() << "\" }");
    THROW_TRITON_EXCEPTION(
        exception.error_code(),
        "Failed to create ModelInstanceState for model \""
            << model->Name() << "\": " << exception.what());
  }
  catch (const std::exception& exception) {
    DEBUG_TRACE_ERROR(
        "{ model_name: \"" << model->Name()
                           << "\""
                              ", error: \""
                           << exception.what() << "\" }");
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to create ModelInstanceState for model \""
            << model->Name() << "\": " << exception.what());
  }
}

void
ModelInstanceState::CreateCudaEvents(int32_t device_id)
{
#ifdef TRITON_ENABLE_GPU
  // Need to set the CUDA context so that the context that events are created on
  // match with contexts that events are recorded with.
  DEBUG_TRACE_INFO("cudaSetDevice(" << device_id << ")");
  if (auto err = cudaSetDevice(device_id)) {
    DEBUG_TRACE_ERROR(
        "cudaSetDevice(" << device_id
                         << ") failed with error: " << cudaGetErrorString(err));
    THROW_TRITON_EXCEPTION(
        err,
        "When creating CUDA events, failed to set the device for "
        "model instance "
        "\"" << Name()
             << "\" : " << cudaGetErrorString(err));
  }

  DEBUG_TRACE_INFO("cudaEventCreate(&compute_input_start_event_)");
  if (auto err = cudaEventCreate(&compute_input_start_event_)) {
    DEBUG_TRACE_ERROR(
        "cudaEventCreate(&compute_input_start_event_) failed with error: "
        << cudaGetErrorString(err));
    THROW_TRITON_EXCEPTION(
        err,
        "When creating CUDA events, failed to create compute input start event "
        "for model instance "
        "\"" << Name()
             << "\" : " << cudaGetErrorString(err));
  }

  DEBUG_TRACE_INFO("cudaEventCreate(&compute_infer_start_event_)");
  if (auto err = cudaEventCreate(&compute_infer_start_event_)) {
    DEBUG_TRACE_ERROR(
        "cudaEventCreate(&compute_infer_start_event_) failed with error: "
        << cudaGetErrorString(err));
    THROW_TRITON_EXCEPTION(
        err,
        "When creating CUDA events, failed to create compute infer start event "
        "for model instance "
        "\"" << Name()
             << "\" : " << cudaGetErrorString(err));
  }

  DEBUG_TRACE_INFO("cudaEventCreate(&compute_output_start_event_)");
  if (auto err = cudaEventCreate(&compute_output_start_event_)) {
    DEBUG_TRACE_ERROR(
        "cudaEventCreate(&compute_output_start_event_) failed with error: "
        << cudaGetErrorString(err));
    THROW_TRITON_EXCEPTION(
        err,
        "When creating CUDA events, failed to create compute output start "
        "event for model instance "
        "\"" << Name()
             << "\" : " << cudaGetErrorString(err));
  }
#endif
}

cudaStream_t
ModelInstanceState::CudaStream()
{
  return BackendModelInstance::CudaStream();
}

int32_t
ModelInstanceState::DeviceId() const
{
  return BackendModelInstance::DeviceId();
}

void
ModelInstanceState::Execute(
    std::vector<TRITONBACKEND_Response*>* responses, uint32_t response_count,
    std::vector<torch::Tensor>& input_tensors,
    std::vector<torch::Tensor>& output_tensors)
{
  NVTX_RANGE(nvtx_, "Execute " + Name());

  std::vector<torch::Tensor> model_outputs;

  try {
    // Enable/disable inference mode based on the model setting.
    // Supersedes NoGradGuard.
    torch::InferenceMode guard{model_->InferenceModeEnabled()};

    // Enable/disable cuDNN.
    at::globalContext().setUserEnabledCuDNN(model_->CudnnEnabled());

    torch::NoGradGuard no_grad_guard;

    DEBUG_TRACE_INFO(
        "calling Forward() for model instance \""
        << Name() << "\" with " << input_tensors.size() << " input tensors.");
    model_outputs = model_->Forward(input_tensors);
    DEBUG_TRACE_INFO(
        "Forward() for model instance \"" << Name() << "\" returned "
                                          << model_outputs.size()
                                          << " output tensors.");

    for (auto& output_tensor : model_outputs) {
      output_tensors.push_back(output_tensor);
    }
  }
  catch (const std::exception& exception) {
    DEBUG_TRACE_ERROR(
        "{ model_name: \"" << model_->Name()
                           << "\""
                              ", error: \""
                           << exception.what() << "\" }");
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            TOSTRING(
                "Inductor model instance \""
                << Name() << "\" execution failure: " << exception.what())
                .c_str()));
  }
}

float
ModelInstanceState::GetCudaEventElapsedTime(
    const cudaEvent_t& start_event, const cudaEvent_t& end_event)
{
  float duration{0};
#ifdef TRITON_ENABLE_GPU
  if (auto err = ConvertCUDAStatusToTritonError(
          cudaEventElapsedTime(&duration, start_event, end_event),
          TRITONSERVER_ERROR_INTERNAL, "Failed to capture elapsed time")) {
    TRITON_LOG_ERROR(
        "Failed to capture elapsed time"
        << " for instance \"" << Name()
        << "\": " << TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
  }
#endif
  return duration;
}

cudaStream_t
ModelInstanceState::GetCudaStreamByInstanceKind()
{
#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    DEBUG_TRACE_INFO(
        "Returning CUDA stream for GPU instance kind for model instance \""
        << Name() << "\".");
    return stream_;
  }

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL &&
      !stream_vector_.empty()) {
    DEBUG_TRACE_INFO(
        "Returning CUDA stream for MODEL instance kind for model instance \""
        << Name() << "\".");
    return stream_vector_[0];
  }
#endif
  DEBUG_TRACE_INFO(
      "No CUDA stream available for model instance \"" << Name() << "\".");
  return nullptr;
}

TritonNamingConvention
ModelInstanceState::GetNamingConvention(
    const std::vector<std::string>& allowed_ios)
{
  // Rules for (non-Dictionary) input tensor names:
  // 1. Must be in 'allowed_inputs' (arguments in the forward function)
  // 2. Must follow the naming convention i.e. <name>__<index>
  // 3. If neither of the above conditions are satisfied, enforce strict
  // ordering of model inputs.
  //
  // Rules for output tensor names:
  // 1. Must follow the naming convention i.e. <name>__<index>
  // 2. If not, we enforce strict ordering of model outputs.
  std::string io_kind{"input"};
  TritonNamingConvention naming_convention{
      TritonNamingConvention::FORWARD_ARGUMENT};

  if (allowed_ios.size() == 0) {
    io_kind = "output";
    naming_convention = TritonNamingConvention::NAMED_INDEX;
  }

  TritonJsonValue ios;
  if (auto err = model_->ModelConfig().MemberAsArray(io_kind.c_str(), &ios)) {
    THROW_TRITON_EXCEPTION(
        err, "Failed to get \"" << io_kind
                                << "\" array from model config for model \""
                                << model_->Name() << "\".");
  }

  if (io_kind == "input") {
    for (size_t i = 0; i < ios.ArraySize(); i += 1) {
      TritonJsonValue io;
      if (auto err = ios.IndexAsObject(i, &io)) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", io_kind: \"" << io_kind << "\""
                               << ", index: " << i << ", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\""
                               << " }");
        THROW_TRITON_EXCEPTION(
            err, "Failed to get \"" << io_kind << "\" object at index " << i
                                    << " from model config for model \""
                                    << model_->Name() << "\".");
      }

      // Validate name.
      std::string io_name;
      if (auto err = io.MemberAsString("name", &io_name)) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", io_kind: \"" << io_kind << "\""
                               << ", index: " << i << ", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\""
                               << " }");
        THROW_TRITON_EXCEPTION(
            err, "Failed to get \"" << io_kind << "\" name at index " << i
                                    << " from model config for model \""
                                    << model_->Name() << "\".");
      }
      DEBUG_TRACE_INFO(
          "{ model_name: \"" << model_->Name() << "\", io_name: \"" << io_name
                             << "\" }");

      auto it = std::find(allowed_ios.begin(), allowed_ios.end(), io_name);
      if (it == allowed_ios.end()) {
        naming_convention = TritonNamingConvention::NAMED_INDEX;
        break;
      }
    }
  }

  if (naming_convention == TritonNamingConvention::NAMED_INDEX) {
    for (size_t i = 0; i < ios.ArraySize(); i += 1) {
      TritonJsonValue io;
      if (auto err = ios.IndexAsObject(i, &io)) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", naming_convention: \"" << naming_convention
                               << "\", io_kind: \"" << io_kind << "\""
                               << ", index: " << i << ", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\""
                               << " }");
        THROW_TRITON_EXCEPTION(
            err, "Failed to get " << io_kind << " object at index " << i
                                  << " from model config for model \""
                                  << model_->Name() << "\".");
      }

      // Validate name.
      std::string io_name;
      if (auto err = io.MemberAsString("name", &io_name)) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", naming_convention: \"" << naming_convention
                               << "\", io_kind: \"" << io_kind << "\""
                               << ", index: " << i << ", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\""
                               << " }");
        THROW_TRITON_EXCEPTION(
            err, "Failed to get \"" << io_kind << "\" name at index " << i
                                    << " from model config for model \""
                                    << model_->Name() << "\".");
      }
      DEBUG_TRACE_INFO(
          "{ model_name: \"" << model_->Name() << "\""
                             << ", index: " << i << ", naming_convention: \""
                             << naming_convention << "\""
                             << ", io_kind: \"" << io_kind << "\""
                             << ", io_name: \"" << io_name << "\" }");

      int start_pos = io_name.find(DELIMINATOR);
      if (start_pos == -1) {
        naming_convention = TritonNamingConvention::STRICT_CONFIG_ORDERING;
        DEBUG_TRACE_INFO(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", index: " << i << ", naming_convention: \""
                               << naming_convention << "\" }");
        break;
      } else {
        // check if the index part of the name is not an integer
        std::string index_str = io_name.substr(start_pos + 2);
        bool is_int{true};
        for (auto it = index_str.begin(); it != index_str.end(); it++) {
          if (std::isdigit(*it) == 0) {
            is_int = false;
          }
        }

        if (!is_int) {
          if (io_kind == "input") {
            TRITON_LOG_WARN(
                "Input \""
                << io_name
                << "\" or previous input(s) are neither an input argument to "
                   "the model \""
                << model_->Name()
                << "\" nor follow the <name>__<index> naming convention."
                   " Defaulting to strict ordering of model inputs.");
          } else {
            TRITON_LOG_WARN(
                "Output \""
                << io_name << "\" or previous output(s) of the model \""
                << model_->Name()
                << "\" do not follow the <name>__<index> naming convention."
                   " Defaulting to strict ordering of model outputs.");
          }

          naming_convention = TritonNamingConvention::STRICT_CONFIG_ORDERING;
          DEBUG_TRACE_INFO(
              "{ model_name: \"" << model_->Name() << "\""
                                 << ", index: " << i
                                 << ", naming_convention: \""
                                 << naming_convention << "\" }");
          break;
        }
      }
    }
  }

  return naming_convention;
}

const std::string&
ModelInstanceState::HostPolicyName() const
{
  return BackendModelInstance::HostPolicyName();
}

TritonModelState*
ModelInstanceState::ModelState() const
{
  return model_;
}

TRITONSERVER_InstanceGroupKind
ModelInstanceState::Kind() const
{
  return BackendModelInstance::Kind();
}

triton::backend::BackendModel*
ModelInstanceState::Model() const
{
  return BackendModelInstance::Model();
}

const std::string&
ModelInstanceState::Name() const
{
  return BackendModelInstance::Name();
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  TRITON_LOG_VERBOSE(
      "TRITONBACKEND_ModelExecute: Running model \""
      << Name() << "\" with " << request_count
      << (request_count == 1 ? " request." : " requests."));

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    SetCurrentCudaStream(stream_, DeviceId());
  } else if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
    // Replace the default stream of each device with the one we created.
    for (size_t i = 0; i < stream_vector_.size(); i += 1) {
      SetCurrentCudaStream(stream_vector_[i], DeviceId());
    }
  }
#endif

  NVTX_RANGE(nvtx_, "ProcessRequests " + Name());

  uint64_t exec_start_ns{0};
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_->MaxBatchSize();
  DEBUG_TRACE_INFO(
      "{ model: \"" << Name() << "\", max_batch_size: " << max_batch_size
                    << " }");

  // For each request collect the total batch size for this inference execution.
  // The batch-size, number of inputs, and size of each input has already been
  // checked so don't need to do that here.
  size_t total_batch_size{0};
  for (size_t i = 0; i < request_count; i += 1) {
    // If we get a nullptr request then something is badly wrong. Fail and
    // release all requests.
    if (!requests[i]) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              TOSTRING(
                  "NULL request given to PyTorch backend for model \"" << Name()
                                                                       << "\".")
                  .c_str()));
      return;
    }
  }

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed{false};

  for (size_t i = 0; i < request_count; i += 1) {
    TRITONBACKEND_Response* response;
    if (auto err = TRITONBACKEND_ResponseNew(&response, requests[i])) {
      responses.emplace_back(nullptr);
      TRITON_LOG_ERROR(
          "Failed to create response for request "
          << i << " of model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
    } else {
      responses.emplace_back(response);
    }
  }

  for (size_t i = 0; i < request_count; i += 1) {
    if (max_batch_size > 0) {
      TRITONBACKEND_Input* input{nullptr};
      auto err = TRITONBACKEND_RequestInputByIndex(requests[i], 0, &input);
      if (!err) {
        const int64_t* shape{nullptr};
        err = TRITONBACKEND_InputProperties(
            /* input= */ input,
            /* name= */ nullptr,
            /* datatype= */ nullptr,
            /* shape= */ &shape,
            /* dims_count= */ nullptr,
            /* byte_size= */ nullptr,
            /* buffer_count= */ nullptr);
        total_batch_size += shape[0];
      }

      if (err) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed, err);
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      total_batch_size += 1;
    }
  }

  if (total_batch_size == 0) {
    DEBUG_TRACE_INFO("{ model: \"" << Name() << "\", batch_size: 0 }");
    return;
  }

  if (!all_response_failed) {
    if (total_batch_size != 1 && total_batch_size > (size_t)max_batch_size) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              TOSTRING(
                  "batch size " << total_batch_size << " for \"" << Name()
                                << "\", max allowed is " << max_batch_size)
                  .c_str()));
    }
  }

  std::vector<const char*> input_names;
  std::vector<torch::Tensor> input_tensors;
  bool cuda_copy{false};
  std::unique_ptr<BackendInputCollector> collector;

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU ||
      (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0)) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventRecord(
                compute_input_start_event_, GetCudaStreamByInstanceKind()),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
  }
#endif

  if (!all_response_failed) {
    collector.reset(new BackendInputCollector{
        /* requests= */ requests,
        /* request_count= */ request_count,
        /* responses= */ &responses,
        /* memory_manager= */ model_->TritonMemoryManager(),
        /* pinned_enabled= */ model_->EnablePinnedInput(),
        /* stream= */ GetCudaStreamByInstanceKind(),
        /* event= */ nullptr,
        /* buffer_ready_event= */ nullptr,
        /* kernel_buffer_threshold= */ 0,
        /* host_policy_name= */ HostPolicyName().c_str()});

    triton_exception triton_ex;
    TRITONSERVER_Error* err{nullptr};
    try {
      SetInputTensors(
          /* total_batch_size= */ total_batch_size,
          /* requests= */ requests,
          /* request_count= */ request_count,
          /* responses= */ &responses,
          /* collector= */ collector.get(),
          /* input_names= */ &input_names,
          /* input_tensors= */ &input_tensors,
          /* cuda_copy= */ &cuda_copy);
    }
    catch (const triton_exception& e) {
      triton_ex = e;
    }
    catch (const std::exception& e) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          TOSTRING(
              "Failed to set input tensors for model instance \""
              << Name() << "\": " << e.what())
              .c_str());
    }
    err = err ? err : triton_ex.get_error();
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed, err);
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(GetCudaStreamByInstanceKind());
    cuda_copy = false;
  }
#endif

  std::vector<torch::Tensor> output_tensors;
  uint64_t compute_start_ns{0};
  uint64_t compute_infer_start{0};

  triton_exception triton_ex;
  TRITONSERVER_Error* err{nullptr};
  try {
    RecordBackendTimestamp(
        &compute_start_ns,
        reinterpret_cast<void*>(&compute_infer_start_event_));
  }
  catch (const triton_exception& e) {
    triton_ex = e;
  }
  catch (const std::exception& e) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        TOSTRING(
            "Failed record compute start timestamp for \""
            << Name() << "\": " << e.what())
            .c_str());
  }
  err = err ? err : triton_ex.get_error();

  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed, err);

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0) {
    SET_TIMESTAMP(compute_infer_start);
  }

  if (!all_response_failed) {
    DEBUG_TRACE_INFO(
        "Execute() { model: \""
        << Name() << "\", request_count: " << request_count
        << ", len(input_tensors): " << input_tensors.size()
        << ", len(output_tensors): " << output_tensors.size() << " }");

    Execute(
        /* responses= */ &responses,
        /* response_count= */ request_count,
        /* input_tensors= */ input_tensors,
        /* output_tensors= */ output_tensors);

    DEBUG_TRACE_INFO(
        "Execute() -> { model: \""
        << Name() << "\", request_count: " << request_count
        << ", len(input_tensors): " << input_tensors.size()
        << ", len(output_tensors): " << output_tensors.size() << " }");
  }

  bool invalid_index{false};
  int max_index{static_cast<int>(output_tensors.size() - 1)};

  if (!all_response_failed) {
    for (const auto& name : model_->ModelOutputs()) {
      int output_index = model_->OutputMap()[name.first];
      if (output_index < 0 || output_index > max_index) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                TOSTRING(
                    "The output \""
                    << name.first
                    << "\" in the model configuration refers to an output "
                       "index which doesn't exist. This model has "
                    << max_index << " outputs.")
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
    for (auto& stream : stream_vector_) {
      cudaStreamSynchronize(stream);
    }
  }
#endif

  uint64_t compute_end_ns{0};
  uint64_t compute_output_start{0};

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0) {
    SET_TIMESTAMP(compute_output_start);
  } else
#endif
  {
    triton_exception triton_ex;
    TRITONSERVER_Error* err{nullptr};
    try {
      RecordBackendTimestamp(
          &compute_end_ns,
          reinterpret_cast<void*>(&compute_output_start_event_));
    }
    catch (const triton_exception& e) {
      triton_ex = e;
    }
    catch (const std::exception& e) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          TOSTRING(
              "Failed to record compute end timestamp for \""
              << Name() << "\": " << e.what())
              .c_str());
    }
    err = err ? err : triton_ex.get_error();
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed, err);
  }

  if (!all_response_failed) {
    if (!invalid_index) {
      TRITONSERVER_Error* err{nullptr};
      try {
        ReadOutputTensors(
            /* total_batch_size= */ total_batch_size,
            /* output_tensors= */ output_tensors,
            /* requests= */ requests,
            /* request_count= */ request_count,
            /* responses= */ responses);
      }
      catch (const triton::backend::pytorch::BackendException& exception) {
        TRITON_LOG_ERROR(
            "Failed to read output tensors for model instance \""
            << Name() << "\": " << exception.what());

        err = TRITONSERVER_ErrorNew(
            exception.error_code(),
            TOSTRING(
                "Failed to read output tensors for model instance \""
                << Name() << "\": " << exception.what())
                .c_str());
      }
      catch (const std::exception& exception) {
        TRITON_LOG_ERROR(
            "Failed to read output tensors for model instance \""
            << Name() << "\": " << exception.what());

        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            TOSTRING(
                "Failed to read output tensors for model instance \""
                << Name() << "\": " << exception.what())
                .c_str());
      }

      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed, err);
    }
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of an earlier
  // error. Note that the responses are not set to nullptr here as we need that
  // indication below to determine if the request we successful or not.
  for (auto& response : responses) {
    if (response) {
      if (auto err = TRITONBACKEND_ResponseSend(
              /* response= */ response,
              /* flags= */ TRITONSERVER_RESPONSE_COMPLETE_FINAL,
              /* userp= */ nullptr)) {
        TRITON_LOG_ERROR(
            "Failed to send PyTorch backend response for model instance \""
            << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
      }
    }
  }

#ifdef TRITON_ENABLE_GPU
  // We don't need an explicit CUDA synchronization here since we have already
  // synchronized the stream in the ReadOutputTensors function.
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    float compute_input_duration = GetCudaEventElapsedTime(
        compute_input_start_event_, compute_infer_start_event_);
    float compute_infer_duration = GetCudaEventElapsedTime(
        compute_infer_start_event_, compute_output_start_event_);

    compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
    compute_end_ns = compute_start_ns + (compute_infer_duration * 1e6);
  } else if (
      Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0) {
    float compute_input_duration = GetCudaEventElapsedTime(
        compute_input_start_event_, compute_infer_start_event_);
    uint64_t compute_infer_duration =
        compute_output_start - compute_infer_start;

    compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
    compute_end_ns = compute_start_ns + compute_infer_duration;
  }
#endif

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; r += 1) {
    auto& request = requests[r];
    if (auto err = TRITONBACKEND_ModelInstanceReportStatistics(
            /* instance= */ TritonModelInstance(),
            /* request= */ request,
            /* success= */ (responses[r] != nullptr),
            /* exec_start_ns= */ exec_start_ns,
            /* exec_end_ns= */ exec_end_ns,
            /* compute_start_ns= */ compute_start_ns,
            /* compute_end_ns= */ compute_end_ns)) {
      TRITON_LOG_ERROR(
          "Failed to report statistics for request "
          << r << " of model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
    }

    if (auto err = TRITONBACKEND_RequestRelease(
            request, TRITONSERVER_REQUEST_RELEASE_ALL)) {
      TRITON_LOG_ERROR(
          "Failed to release request "
          << r << " of model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
    }
  }

  if (!all_response_failed) {
    if (auto err = TRITONBACKEND_ModelInstanceReportBatchStatistics(
            /* instance= */ TritonModelInstance(),
            /* batch_size= */ total_batch_size,
            /* exec_start_ns= */ exec_start_ns,
            /* compute_start_ns= */ compute_start_ns,
            /* compute_end_ns= */ compute_end_ns,
            /* exec_end_ns= */ exec_end_ns)) {
      TRITON_LOG_ERROR(
          "Failed to report batch statistics for model instance \""
          << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<torch::Tensor>& output_tensors,
    TRITONBACKEND_Request** requests, uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>& responses)
{
  if (!requests)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Invalid nullptr requests pointer for reading output tensors"
        " for model instance \""
            << Name() << "\".");

  NVTX_RANGE(nvtx_, "ReadOutputTensors " + Name());

  triton::backend::BackendOutputResponder responder{
      /* requests= */ requests,
      /* request_count= */ request_count,
      /* responses= */ &responses,
      /* memory_manager= */ model_->TritonMemoryManager(),
      /* first_dim_batching= */ false,
      /* pinned_enabled= */ model_->EnablePinnedInput(),
      /* stream= */ GetCudaStreamByInstanceKind()};

  bool cuda_copy{false};
  std::vector<std::shared_ptr<std::string>> string_buffers;

  for (auto& output : model_->ModelOutputs()) {
    auto output_name = output.first;

    if (!map_outputs_.contains(output_name)) {
      if (auto err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              TOSTRING(
                  "Inductor model instance \""
                  << Name() << "\" output tensor \"" << output_name
                  << "\" not found in output map.")
                  .c_str())) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\", output_name: \""
                               << output_name << "\", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\" }");
        THROW_TRITON_EXCEPTION(err, TRITONSERVER_ErrorMessage(err));
      }
    }

    auto output_index = map_outputs_[output_name].model_index();
    auto output_tensor_pair = output.second;

    torch::Tensor output_flat;

    try {
      output_flat = output_tensors[output_index].contiguous().flatten();
    }
    catch (const std::exception& exception) {
      DEBUG_TRACE_ERROR(
          "{ model_name: \"" << model_->Name() << "\", output_name: \""
                             << output_name << "\", error: \""
                             << exception.what() << "\" }");

      if (auto err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              TOSTRING(
                  "Inductor model instance \""
                  << Name() << "\""
                  << " output tensor \"" << output_name << "\""
                  << " not found: " << exception.what())
                  .c_str())) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \""
            << model_->Name() << "\", output_name: \"" << output_name
            << "\", output_index: " << output_index << ", error: \""
            << TRITONSERVER_ErrorMessage(err) << "\" }");
        THROW_TRITON_EXCEPTION(err, TRITONSERVER_ErrorMessage(err));
      }
    }

    auto scalar_type = output_flat.scalar_type();
    auto output_dtype = ConvertTorchTypeToDataType(scalar_type);
    auto config_dtype = map_outputs_[output_name].triton_dtype();

    if (output_dtype != config_dtype) {
      DEBUG_TRACE_ERROR(
          "{ model_name: \""
          << model_->Name() << "\", output_name: \"" << output_name
          << "\", output_index: " << output_index << ", scalar_type: \""
          << torch::toString(scalar_type) << "\", output_dtype: TYPE_"
          << TRITONSERVER_DataTypeString(output_dtype)
          << ", config_dtype: TYPE_"
          << TRITONSERVER_DataTypeString(config_dtype)
          << ", error: \"datatype mismatch\" }");
      if (auto err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              TOSTRING(
                  "Inductor model instance \""
                  << Name() << "\" output tensor \"" << output_name
                  << "\" has datatype TYPE_"
                  << TRITONSERVER_DataTypeString(output_dtype)
                  << " but model configuration expects TYPE_"
                  << TRITONSERVER_DataTypeString(config_dtype) << ".")
                  .c_str())) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \""
            << model_->Name() << "\", output_name: \"" << output_name
            << "\", output_index: " << output_index << ", scalar_type: \""
            << torch::toString(scalar_type) << "\", output_dtype: TYPE_"
            << TRITONSERVER_DataTypeString(output_dtype)
            << ", config_dtype: TYPE_"
            << TRITONSERVER_DataTypeString(config_dtype) << ", error: \" }");
        THROW_TRITON_EXCEPTION(err, TRITONSERVER_ErrorMessage(err));
      }
    }

    auto output_buffer = static_cast<const char*>(output_flat.data_ptr());

    // Output tensors might not reside on the same device as the model instance.
    torch::Device output_device = output_flat.device();
    const auto memory_type = (output_device.type() == torch::kCPU)
                                 ? TRITONSERVER_MEMORY_CPU
                                 : TRITONSERVER_MEMORY_GPU;
    const auto memory_type_id =
        (output_device.type() == torch::kCPU) ? 0 : output_device.index();

    // Batch output doesn't support string data type yet, as it is not trivial
    // to parse string output.
    const triton::backend::BatchOutput* batch_output =
        model_->FindBatchOutput(output_name);

    if (!batch_output) {
      std::vector<int64_t> batch_n_shape;
      auto shape = output_tensors[output_index].sizes();
      for (auto it = shape.begin(); it != shape.end(); it++) {
        batch_n_shape.push_back(*it);
      }

      if (batch_n_shape.size() == 0) {
        auto error_message = TOSTRING(
            "Inductor model instance \""
            << Name() << "\" output tensor \"" << output_name
            << "\" is a scalar which is not supported.");
        auto err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, error_message.c_str());
        THROW_TRITON_EXCEPTION(err, error_message);
      }

      if (output_tensor_pair.first != -1) {
        responder.ProcessTensor(
            /* output_name= */ output_name,
            /* datatype= */ output_dtype,
            /* batchn_shape= */ batch_n_shape,
            /* buffer= */ output_buffer,
            /* memory_type= */ memory_type,
            /* memory_type_id= */ memory_type_id);
      }

      if (output_tensor_pair.second != -1) {
        auto states = responder.ProcessStateTensor(
            /* output_name= */ output_name,
            /* datatype= */ output_dtype,
            /* batchn_shape= */ batch_n_shape,
            /* buffer= */ output_buffer,
            /* memory_type= */ memory_type,
            /* memory_type_id= */ memory_type_id);

        for (auto& state : states) {
          if (auto err = TRITONBACKEND_StateUpdate(state)) {
            DEBUG_TRACE_ERROR(
                "{ model_name: \"" << model_->Name() << "\", output_name: \""
                                   << output_name
                                   << "\", output_index: " << output_index
                                   << ", error: \"failed to update state\" }");
            THROW_TRITON_EXCEPTION(
                err, "Failed to update state for model instance \""
                         << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
          }
        }
      }
    } else {
      responder.ProcessBatchOutput(
          /* name= */ output_name,
          /* batch_output= */ *batch_output,
          /* buffer= */ output_buffer,
          /* memory_type= */ memory_type,
          /* memory_type_id= */ memory_type_id);
    }
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  cudaStreamSynchronize(GetCudaStreamByInstanceKind());
#endif
}

void
ModelInstanceState::RecordBackendTimestamp(
    uint64_t* timestamp, void* cuda_event_ptr)
{
#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU ||
      (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0)) {
    cudaEvent_t& cuda_event = *(reinterpret_cast<cudaEvent_t*>(cuda_event_ptr));
    if (auto err = ConvertCUDAStatusToTritonError(
            cudaEventRecord(cuda_event, GetCudaStreamByInstanceKind()),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record CUDA event")) {
      DEBUG_TRACE_ERROR(
          "{ model_name: \"" << model_->Name() << "\", error: \""
                             << TRITONSERVER_ErrorMessage(err) << "\" }");
      throw triton_exception{err};
    }
  } else
#endif
  {
    SET_TIMESTAMP(*timestamp);
  }
}

void
ModelInstanceState::SetCurrentCudaStream(
    const cudaStream_t& stream, int device_id)
{
#ifdef TRITON_ENABLE_GPU
  DEBUG_TRACE_INFO(
      "Setting current CUDA stream on device "
      << device_id << " for model instance \"" << Name() << "\".");
  at::cuda::CUDAStream torch_stream{
      at::cuda::getStreamFromExternal(stream, device_id)};
  // This function replaces the default stream with the stream we created.
  // It is not necessary to change the current device to the desired device when
  // replacing the default stream for that device. See the documentation here:
  // https://pytorch.org/cppdocs/api/function_namespacec10_1_1cuda_1a6ed50cc0fc16cc7014d9c2f4c3bd098d.html
  at::cuda::setCurrentCUDAStream(torch_stream);
#endif
}

void
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<torch::Tensor>* input_tensors, bool* cuda_copy)
{
  if (!requests)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Argument 'requests' cannot be nullptr.");

  torch::InferenceMode guard{model_->InferenceModeEnabled()};

  uint32_t input_count{0};
  if (auto err = TRITONBACKEND_RequestInputCount(requests[0], &input_count)) {
    TRITON_LOG_ERROR(
        "Failed to get input count"
        << " for request 0 of model instance \"" << Name()
        << "\": " << TRITONSERVER_ErrorMessage(err));
    throw triton_exception{err};
  }

  input_tensors->resize(input_count + batch_input_count_);

  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_preference;
  if (device_.is_cpu()) {
    alloc_preference = {
        {TRITONSERVER_MEMORY_CPU_PINNED, 0},
        {TRITONSERVER_MEMORY_CPU, 0},
    };
  } else {
    alloc_preference = {
        {TRITONSERVER_MEMORY_GPU, device_.index()},
    };
  }

  for (size_t input_idx = 0; input_idx < input_count; input_idx += 1) {
    TRITONBACKEND_Input* input{nullptr};
    if (auto err =
            TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input)) {
      TRITON_LOG_ERROR(
          "Failed to get input "
          << input_idx << " for request 0 of model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      throw triton_exception{err};
    }

    const char* input_name{nullptr};
    TRITONSERVER_DataType input_datatype{TRITONSERVER_TYPE_INVALID};
    const int64_t* input_shape{nullptr};
    uint32_t input_dims_count{0};

    if (auto err = TRITONBACKEND_InputProperties(
            /* input= */ input,
            /* name= */ &input_name,
            /* datatype= */ &input_datatype,
            /* shape= */ &input_shape,
            /* dims_count= */ &input_dims_count,
            /* byte_size= */ nullptr,
            /* buffer_count= */ nullptr)) {
      TRITON_LOG_ERROR(
          "Failed to get properties for input "
          << input_idx << " for request 0 of model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      throw triton_exception{err};
    }

    input_names->emplace_back(input_name);

    std::vector<int64_t> batch_n_shape{};

    if (model_->IsInputRagged(input_name)) {
      DEBUG_TRACE_INFO(
          "{ model: \"" << Name() << "\", input: \"" << input_name
                        << "\", is_ragged: true }");

      batch_n_shape = std::vector<int64_t>{0};
      for (size_t request_index = 0; request_index < request_count;
           request_index += 1) {
        TRITONBACKEND_Input* input{nullptr};
        if (auto err = TRITONBACKEND_RequestInput(
                requests[request_index], input_name, &input)) {
          RESPOND_AND_SET_NULL_IF_ERROR(
              &((*responses)[request_index]),
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  TOSTRING(
                      "Failed to get input \""
                      << input_name << "\" for request " << request_index
                      << " of model instance \"" << Name()
                      << "\": " << TRITONSERVER_ErrorMessage(err))
                      .c_str()));
          TRITONSERVER_ErrorDelete(err);
        }

        const int64_t* input_shape{nullptr};
        uint32_t input_dims_count{0};
        if (auto err = TRITONBACKEND_InputProperties(
                /* input= */ input,
                /* name= */ nullptr,
                /* datatype= */ nullptr,
                /* shape= */ &input_shape,
                /* dims_count= */ &input_dims_count,
                /* byte_size= */ nullptr,
                /* buffer_count= */ nullptr)) {
          RESPOND_AND_SET_NULL_IF_ERROR(
              &((*responses)[request_index]),
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  TOSTRING(
                      "Failed to get properties for input \""
                      << input_name << "\" for request " << request_index
                      << " of model instance \"" << Name()
                      << "\": " << TRITONSERVER_ErrorMessage(err))
                      .c_str()));
          TRITONSERVER_ErrorDelete(err);
        }

        int64_t element_cnt{0};
        if (auto err =
                GetElementCount(input_shape, input_dims_count, &element_cnt)) {
          RESPOND_AND_SET_NULL_IF_ERROR(
              &((*responses)[request_index]),
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  TOSTRING(
                      "Failed to get element count for input \""
                      << input_name << "\" for request " << request_index
                      << " of model instance \"" << Name()
                      << "\": " << TRITONSERVER_ErrorMessage(err))
                      .c_str()));
          TRITONSERVER_ErrorDelete(err);
        }

        batch_n_shape[0] += element_cnt;
      }
    } else {
      batch_n_shape =
          std::vector<int64_t>{input_shape, input_shape + input_dims_count};
      if (is_batching_supported_) {
        DEBUG_TRACE_INFO(
            "{ model: \"" << Name() << "\", input: \"" << input_name
                          << "\", original_shape: "
                          << triton::backend::ShapeToString(batch_n_shape)
                          << ", batch_n_shape[0]=total_batch_size: "
                          << total_batch_size << " }");
        batch_n_shape[0] = total_batch_size;
      }
    }

    // The input must be in contiguous CPU/GPU memory.
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_preference;
    // For 'KIND_MODEL', input will always be in CPU as we don't have a way to
    // query the input types.
    if (device_.is_cpu() || (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL)) {
      alloc_preference = {
          {TRITONSERVER_MEMORY_CPU_PINNED, 0},
          {TRITONSERVER_MEMORY_CPU, 0},
      };

      DEBUG_TRACE_INFO(
          "{ model: \""
          << Name() << "\", input: \"" << input_name
          << "\", allocation_preference: [ { type: \""
          << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_CPU_PINNED)
          << "\", id: 0 }, { type: \""
          << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_CPU)
          << "\", id: 0 } ] }");
    } else {
      alloc_preference = {
          {TRITONSERVER_MEMORY_GPU, device_.index()},
      };

      DEBUG_TRACE_INFO(
          "{ model: \"" << Name() << "\", input: \"" << input_name
                        << "\", allocation_preference: [ { type: \""
                        << TRITONSERVER_MemoryTypeString(
                               TRITONSERVER_MEMORY_GPU)
                        << "\", id: " << device_.index() << " } ] }");
    }

    const char* input_buffer{nullptr};
    size_t batch_n_byte_size{0};
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id{0};

    if (auto err = collector->ProcessTensor(
            /* input_name= */ input_name,
            /* buffer= */ nullptr,
            /* buffer_byte_size= */ 0,
            /* allowed_input_types= */ alloc_preference,
            /* dst_buffer= */ &input_buffer,
            /* dst_buffer_byte_size= */ &batch_n_byte_size,
            /* dst_memory_type= */ &memory_type,
            /* dst_memory_type_id= */ &memory_type_id)) {
      TRITON_LOG_ERROR(
          "Failed to process input \""
          << input_name << "\" for model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      throw triton_exception{err};
    }

    const std::pair<bool, torch::ScalarType> torch_dtype =
        ConvertDataTypeToTorchType(input_datatype);
    torch::TensorOptions options{torch_dtype.second};
    auto updated_options = (memory_type == TRITONSERVER_MEMORY_GPU)
                               ? options.device(torch::kCUDA, device_.index())
                               : options.device(torch::kCPU);

    if (input_datatype == TRITONSERVER_TYPE_BYTES) {
      DEBUG_TRACE_INFO(
          "{ model: \"" << Name() << "\", input: \"" << input_name
                        << "\", input_datatype: TYPE_BYTES, batch_n_byte_size: "
                        << batch_n_byte_size << " }");

      // Create the PyTorch list to hold the strings.
      torch::List<std::string> input_list{};
      input_list.reserve(batch_n_shape[0]);

      for (size_t idx = 0; idx < request_count; idx += 1) {
        TRITONBACKEND_Input* input{nullptr};
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]), TRITONBACKEND_RequestInput(
                                      /* request= */ requests[idx],
                                      /* name= */ input_name,
                                      /* input= */ &input));

        const int64_t* shape{nullptr};
        uint32_t dims_count{0};
        uint32_t buffer_count{0};

        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_InputPropertiesForHostPolicy(
                /* input= */ input,
                /* host_policy_name= */ HostPolicyName().c_str(),
                /* name= */ nullptr,
                /* datatype= */ nullptr,
                /* shape= */ &shape,
                /* dims_count= */ &dims_count,
                /* byte_size= */ nullptr,
                /* buffer_count= */ &buffer_count));

        int64_t batch_element_count = 0;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]), GetElementCount(
                                      /* dims= */ shape,
                                      /* dims_cnt= */ dims_count,
                                      /* cnt= */ &batch_element_count));

        *cuda_copy |= SetStringInputTensor(
            /* input_list= */ &input_list,
            /* input= */ input,
            /* name= */ input_name,
            /* buffer_count= */ buffer_count,
            /* request_element_cnt= */ batch_element_count,
            /* response= */ &((*responses)[idx]),
            /* cuda_stream= */ GetCudaStreamByInstanceKind(),
            /* host_policy_name= */ HostPolicyName().c_str());
      }

      // TODO: make this work
      // (*input_tensors)[map_inputs_[input_name].model_index()] = input_list;
      DEBUG_TRACE_ERROR(
          "(*input_tensors)[map_inputs_[input_name].model_index()] = "
          "input_list is not supported yet for input \""
          << input_name << "\" of model instance \"" << Name() << "\".");
    } else {
      auto input_index = map_inputs_[input_name].model_index();

      if (batch_n_byte_size > 0) {
        (*input_tensors)[input_index] = torch::from_blob(
            const_cast<char*>(input_buffer), torch::IntArrayRef(batch_n_shape),
            updated_options);
      } else {
        // Create an empty tensor for zero-sized input.
        (*input_tensors)[input_index] =
            torch::zeros(batch_n_shape, updated_options);
      }
    }
  }

  for (const auto& batch_input : model_->BatchInputs()) {
    std::vector<int64_t> shape{};
    collector->BatchInputShape(batch_input, &shape);

    for (const auto& input_name : batch_input.TargetNames()) {
      input_names->emplace_back(input_name.c_str());

      const char* dst_buffer{nullptr};
      size_t dst_buffer_byte_size{0};
      TRITONSERVER_MemoryType dst_memory_type{TRITONSERVER_MEMORY_CPU};
      int64_t dst_memory_type_id{0};

      RESPOND_ALL_AND_SET_NULL_IF_ERROR(
          (*responses), responses->size(),
          collector->ProcessBatchInput(
              /* batch_input= */ batch_input,
              /* buffer= */ nullptr,
              /* buffer_byte_size= */ 0,
              /* allowed_input_types= */ alloc_preference,
              /* dst_buffer= */ &dst_buffer,
              /* dst_buffer_byte_size= */ &dst_buffer_byte_size,
              /* dst_memory_type= */ &dst_memory_type,
              /* dst_memory_type_id= */ &dst_memory_type_id));

      const auto torch_dtype =
          ConvertDataTypeToTorchType(batch_input.DataType());
      torch::TensorOptions options{torch_dtype.second};
      auto updated_options = (dst_memory_type == TRITONSERVER_MEMORY_GPU)
                                 ? options.device(torch::kCUDA, device_.index())
                                 : options.device(torch::kCPU);

      auto input_index = map_inputs_[input_name].model_index();

      if (dst_buffer_byte_size) {
        (*input_tensors)[input_index] = torch::from_blob(
            const_cast<char*>(dst_buffer), shape, updated_options);
      } else {
        // special handle when input has zero size
        (*input_tensors)[input_index] = torch::zeros(shape, updated_options);
      }
    }
  }

  *cuda_copy |= collector->Finalize();
}

TRITONBACKEND_ModelInstance*
ModelInstanceState::TritonModelInstance()
{
  return BackendModelInstance::TritonModelInstance();
}

bool
ModelInstanceState::ValidateBooleanSequenceControl(
    TritonJsonValue& sequence_batching, const std::string& control_kind,
    bool required)
{
  THROW_TRITON_EXCEPTION(
      TRITONSERVER_ERROR_INTERNAL,
      "Boolean sequence control validation is not supported for model instance "
      "\"" << Name()
           << "\".");
  std::string tensor_name;
  std::string tensor_dtype;
  if (auto err = GetBooleanSequenceControlProperties(
          /* batcher= */ sequence_batching,
          /* model_name= */ model_->Name(),
          /* control_kind= */ control_kind,
          /* required= */ required,
          /* tensor_name= */ &tensor_name,
          /* tensor_datatype= */ &tensor_dtype,
          /* fp32_false_value= */ nullptr,
          /* fp32_true_value= */ nullptr,
          /* int32_false_value= */ nullptr,
          /* int32_true_value= */ nullptr,
          /* bool_false_value= */ nullptr,
          /* bool_true_value= */ nullptr)) {
    DEBUG_TRACE_ERROR(
        "Failed to validate boolean sequence control for model instance \""
        << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
    THROW_TRITON_EXCEPTION(
        err, "Failed to validate boolean sequence control for model instance \""
                 << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
  }

  bool have_control{!tensor_name.empty()};

  if (have_control) {
    int input_index{0};
    int start_pos{static_cast<int>(tensor_name.find(DELIMINATOR))};

    if (start_pos == -1) {
      DEBUG_TRACE_ERROR(
          "Input \""
          << tensor_name
          << "\" does not follow <name>__<index> naming convention.");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Input \""
              << tensor_name
              << "\" does not follow <name>__<index> naming convention.");
    }

    // Check if the index part of the name is not an integer.
    std::string index_str = tensor_name.substr(start_pos + 2);
    for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
      if (std::isdigit(*itr) == 0) {
        DEBUG_TRACE_ERROR(
            "Input \""
            << tensor_name
            << "\" does not follow <name>__<index> naming convention.");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ERROR_INTERNAL,
            "Input \""
                << tensor_name
                << "\" does not follow <name>__<index> naming convention.");
      }
    }

    input_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
  }

  return have_control;
}

void
ModelInstanceState::ValidateInputs(const size_t expected_input_count)
{
  if (map_inputs_.size() != expected_input_count) {
    DEBUG_TRACE_ERROR(
        "Failed to load model \""
        << Name() << "\" configuration expects " << expected_input_count
        << " inputs, but model expects " << map_inputs_.size() << " inputs.");
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to load model \""
            << Name() << "\" configuration expects " << expected_input_count
            << " inputs, but model expects " << map_inputs_.size()
            << " inputs.");
  }

  /* CANNOT VALIDATE INPUTS BY DTYPE DUE TO LACK OF INFORMATION FROM MODEL */

  TritonJsonValue inputs;
  if (auto err = model_->ModelConfig().MemberAsArray("input", &inputs)) {
    DEBUG_TRACE_ERROR(
        "Failed to get model \"" << Name() << "\" input configuration: "
                                 << TRITONSERVER_ErrorMessage(err));
    THROW_TRITON_EXCEPTION(
        err, "Failed to get model \"" << Name() << "\" input configuration: "
                                      << TRITONSERVER_ErrorMessage(err));
  }

  if (inputs.ArraySize() != expected_input_count) {
    DEBUG_TRACE_ERROR(
        "Failed to load model \""
        << Name() << "\" configuration expects " << expected_input_count
        << " inputs, but model configuration has " << inputs.ArraySize()
        << " inputs.");
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to load model \""
            << Name() << "\" configuration expects " << expected_input_count
            << " inputs, but model configuration has " << inputs.ArraySize()
            << " inputs.");
  }

  for (size_t i = 0; i < inputs.ArraySize(); i += 1) {
    size_t server_index{i};

    TritonJsonValue input;
    if (auto err = inputs.IndexAsObject(i, &input)) {
      DEBUG_TRACE_ERROR(
          "Failed to get input " << i << " for model instance \"" << Name()
                                 << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get input "
                   << i << " for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    std::string input_name;
    if (auto err = input.MemberAsString("name", &input_name)) {
      DEBUG_TRACE_ERROR(
          "Failed to get name for input "
          << i << " for model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get name for input "
                   << i << " for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    if (!map_inputs_.contains(input_name)) {
      DEBUG_TRACE_ERROR(
          "Input \"" << input_name << "\" for model instance \"" << Name()
                     << "\" is not found in model input map.");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Input \"" << input_name << "\" for model instance \"" << Name()
                     << "\" is not found in model input map.");
    }

    map_inputs_[input_name].server_index() = server_index;

    // Validate dtype
    std::string input_dtype_string;
    if (auto err = input.MemberAsString("data_type", &input_dtype_string)) {
      DEBUG_TRACE_ERROR(
          "Failed to get data type for input \""
          << input_name << "\" for model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get data type for input \""
                   << input_name << "\" for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    const auto pr = ModelConfigDataTypeToTorchType(input_dtype_string);
    if (!pr.first && (input_dtype_string != "TYPE_STRING")) {
      DEBUG_TRACE_ERROR(
          "Unsupported datatype " << input_dtype_string << " for input \""
                                  << input_name << "\" for model instance \""
                                  << Name() << "\".");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Unsupported datatype " << input_dtype_string << " for input \""
                                  << input_name << "\" for model instance \""
                                  << Name() << "\".");
    }
    auto torch_dtype = pr.second;

    map_inputs_[input_name].torch_dtype() = torch_dtype;
    map_inputs_[input_name].triton_dtype() =
        ConvertTorchTypeToDataType(torch_dtype);

    std::vector<int64_t> input_shape{};
    // Validate shape for String inputs. Only allow 1 dimension.
    if (input_dtype_string == "TYPE_STRING") {
      // If a reshape is provided for the input then use that when validating
      // the model shapes.
      TritonJsonValue reshape;
      if (input.Find("reshape", &reshape)) {
        if (auto err = ParseShape(reshape, "shape", &input_shape)) {
          DEBUG_TRACE_ERROR(
              "Failed to parse reshape dims for input \""
              << input_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to parse reshape shape for input \""
                       << input_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }
      } else {
        if (auto err = ParseShape(input, "dims", &input_shape)) {
          DEBUG_TRACE_ERROR(
              "Failed to parse dims for input \""
              << input_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to parse dims for input \""
                       << input_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }
      }
    } else {
      if (auto err = ParseShape(input, "dims", &input_shape)) {
        DEBUG_TRACE_ERROR(
            "Failed to parse dims for input \""
            << input_name << "\" for model instance \"" << Name()
            << "\": " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        input_shape = {-1};
      }
    }

    map_inputs_[input_name].shape() = input_shape;

    DEBUG_TRACE_INFO(
        "{ model: \"" << Name() << "\", input_name: \"" << input_name
                      << "\", input_index: "
                      << map_inputs_[input_name].model_index()
                      << ", input_dtype: " << torch_dtype << ", input_shape: "
                      << triton::backend::ShapeToString(input_shape)
                      << ", server_index: "
                      << map_inputs_[input_name].server_index() << " }");
  }

  TritonJsonValue sequence_batching;
  if (model_->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    throw std::runtime_error("Sequence batching is not supported yet.");
    /*
    TritonJsonValue states;
    if (sequence_batching.Find("state", &states))
    {
      for (size_t i = 0; i < states.ArraySize(); i += 1)
      {
        TritonJsonValue state;
        if (auto err = states.IndexAsObject(i, &state))
        {
          DEBUG_TRACE_ERROR("Failed to get sequence state " << i << " for
          model instance \"" << Name() << "\": " <<
          TRITONSERVER_ErrorMessage(err)); THROW_TRITON_EXCEPTION(err,
                                 "Failed to get sequence state " << i << "
                                 for model instance \"" << Name() << "\": "
                                 << TRITONSERVER_ErrorMessage(err));
        }

        std::string state_name;
        if (auto err = state.MemberAsString("input_name", &state_name))
        {
          DEBUG_TRACE_ERROR("Failed to get input name for sequence state " <<
          i << " for model instance \"" << Name() << "\": " <<
          TRITONSERVER_ErrorMessage(err)); THROW_TRITON_EXCEPTION(err,
                                 "Failed to get input name for sequence state
                                 " << i << " for model instance \"" << Name()
                                 << "\": " <<
                                 TRITONSERVER_ErrorMessage(err));
        }

        // TODO: FIXME
        // AddInputToMap(naming_convention, allowed_inputs, state_name, i);

        // Validate dtype
        std::string state_dtype;
        if (auto err = state.MemberAsString("data_type", &state_dtype))
        {
          DEBUG_TRACE_ERROR("Failed to get data type for sequence state input
          \"" << state_name << "\" for model instance \"" << Name() << "\": "
          << TRITONSERVER_ErrorMessage(err)); THROW_TRITON_EXCEPTION(err,
                                 "Failed to get data type for sequence state
                                 input \"" << state_name << "\" for model
                                 instance \"" << Name() << "\": " <<
                                 TRITONSERVER_ErrorMessage(err));
        }

        DEBUG_TRACE_INFO("{ name: \"" << Name() << "\""
                         << ", state_name: \"" << state_name << "\""
                         << ", state_dtype: \"" << state_dtype << "\""
                         << " }");

        const auto pr = ModelConfigDataTypeToTorchType(state_dtype);
        if (!pr.first)
        {
          DEBUG_TRACE_ERROR("Unsupported datatype " << state_dtype << " for
          sequence state input \"" << state_name << "\" for model instance
          \"" << Name() << "\".");
          THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                                 "Unsupported datatype " << state_dtype << "
                                 for sequence state input \"" << state_name
                                 << "\" for model instance \"" << Name() <<
                                 "\".");
        }

        // Validate shape for String inputs. Only allow 1 dimension.
        if (state_dtype == "TYPE_STRING")
        {
          if (is_batching_supported_)
          {
            DEBUG_TRACE_ERROR("Triton only supports 1-dimensional string
            inputs for sequence state input \"" << state_name << "\" for
            model \"" << Name() << "\" when batching is enabled.");
            THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INVALID_ARG,
                                   "Triton only supports 1-dimensional string
                                   inputs for sequence " "state input \"" <<
                                   state_name << "\" for model \"" << Name()
                                   << "\".");
          }
        }
      }
    }
    */
  }

  /*
    The code below likely needs to be fixed up as this is where batch input were
    assigned to the input map and the naming convention and allowed inputs were
    used to validate the batch input target names and assign them to the input
    map. This is also where the batch_input_count_ was determined based on the
    number of batch inputs and the number of target names for each batch input.
  */
#if false
  for (const auto& batch_input : model_->BatchInputs()) {
    for (const auto& input_name : batch_input.TargetNames()) {
      // TODO: FIXME
      // AddInputToMap(/* naming_convention= */naming_convention,
      //               /* allowed_inputs= */allowed_inputs,
      //               /* input_name= */input_name,
      //               /* index= */i + inputs.ArraySize());
    }
  }
#endif

  DEBUG_TRACE_INFO(
      "{ name: \"" << Name()
                   << "\", total_batch_input_target_names: " << uint32_t i =
          model_->BatchInputs().size() * batch_input.TargetNames().size();
      << ", batch_input_count: " << batch_input_count_ << " }");
}

void
ModelInstanceState::ValidateOutputs()
{
  TritonJsonValue outputs;
  if (auto err = model_->ModelConfig().MemberAsArray("output", &outputs)) {
    DEBUG_TRACE_ERROR(
        "Failed to get model \"" << Name() << "\" output configuration: "
                                 << TRITONSERVER_ErrorMessage(err));
    THROW_TRITON_EXCEPTION(
        err, "Failed to get model \"" << Name() << "\" output configuration: "
                                      << TRITONSERVER_ErrorMessage(err));
  }

  int output_index{0};

  if (outputs.ArraySize() == 0) {
    DEBUG_TRACE_ERROR(
        "Configuration for model \""
        << Name()
        << "\" must define at least one output, none were specified.");
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Configuration for model \""
            << Name()
            << "\" must define at least one output, none were specified.");
  }

  for (size_t i = 0; i < outputs.ArraySize(); i += 1) {
    size_t server_index{i};

    TritonJsonValue output;
    if (auto err = outputs.IndexAsObject(i, &output)) {
      DEBUG_TRACE_ERROR(
          "Failed to get output " << i << " for model instance \"" << Name()
                                  << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get output "
                   << i << " for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    // Validate name
    std::string output_name;
    if (auto err = output.MemberAsString("name", &output_name)) {
      DEBUG_TRACE_ERROR(
          "Failed to get name for output "
          << i << " for model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get name for output "
                   << i << " for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    if (!map_outputs_.contains(output_name)) {
      DEBUG_TRACE_ERROR(
          "Output \"" << output_name << "\" for model instance \"" << Name()
                      << "\" is not found in model output map.");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Output \"" << output_name << "\" for model instance \"" << Name()
                      << "\" is not found in model output map.");
    }

    output_index = i;
    map_outputs_[output_name].server_index() = server_index;

    // Validate data type.
    std::string output_dtype;
    if (auto err = output.MemberAsString("data_type", &output_dtype)) {
      DEBUG_TRACE_ERROR(
          "Failed to get data type for output \""
          << output_name << "\" for model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get data type for output \""
                   << output_name << "\" for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    const auto torch_type = ModelConfigDataTypeToTorchType(output_dtype);
    if (!torch_type.first && (output_dtype != "TYPE_STRING")) {
      DEBUG_TRACE_ERROR(
          "Unsupported datatype " << output_dtype << " for output \""
                                  << output_name << "\" for model instance \""
                                  << Name() << "\".");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Unsupported datatype " << output_dtype << " for output \""
                                  << output_name << "\" for model instance \""
                                  << Name() << "\".");
    }

    map_outputs_[output_name].torch_dtype() = torch_type.second;
    map_outputs_[output_name].triton_dtype() =
        ConvertTorchTypeToDataType(torch_type.second);

    std::vector<int64_t> output_shape{};
    // Validate shape for String outputs. Only allow 1 dimension.
    if (output_dtype == "TYPE_STRING") {
      TritonJsonValue reshape;

      if (output.Find("reshape", &reshape)) {
        if (auto err = ParseShape(reshape, "shape", &output_shape)) {
          DEBUG_TRACE_ERROR(
              "Failed to parse reshape shape for output \""
              << output_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to parse reshape shape for output \""
                       << output_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        DEBUG_TRACE_INFO(
            "{ name: \"" << Name() << "\""
                         << ", output_name: \"" << output_name << "\""
                         << ", output_index: " << output_index
                         << ", output_dtype: \"" << output_dtype << "\""
                         << ", reshape: true"
                         << ", len(output_shape): " << output_shape.size()
                         << ", server_index: "
                         << map_outputs_[output_name].server_index() << " }");
      } else {
        if (auto err = ParseShape(output, "dims", &output_shape)) {
          DEBUG_TRACE_ERROR(
              "Failed to parse dims for output \""
              << output_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to parse dims for output \""
                       << output_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        DEBUG_TRACE_INFO(
            "{ name: \"" << Name() << "\""
                         << ", output_name: \"" << output_name << "\""
                         << ", output_index: " << output_index << ""
                         << ", output_dtype: \"" << output_dtype << "\""
                         << ", reshape: false"
                         << ", len(output_shape): " << output_shape.size()
                         << ", server_index: "
                         << map_outputs_[output_name].server_index() << " }");
      }

      map_outputs_[output_name].shape() = output_shape;

      if ((output_shape.size() + (is_batching_supported_ ? 1 : 0)) > 1) {
        DEBUG_TRACE_ERROR(
            "Triton only supports 1 dimensional List of String as output for \""
            << output_name << "\" for model instance \"" << Name() << "\".");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ERROR_INTERNAL,
            "Triton only supports 1 dimensional List of String as output for \""
                << output_name << "\" for model instance \"" << Name()
                << "\".");
      }
    } else {
      if (auto err = ParseShape(output, "dims", &output_shape)) {
        DEBUG_TRACE_ERROR(
            "Failed to parse dims for output \""
            << output_name << "\" for model instance \"" << Name()
            << "\": " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        output_shape = {-1};
      }

      map_outputs_[output_name].shape() = output_shape;
    }
  }

  TritonJsonValue sequence_batching;
  if (model_->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    throw std::runtime_error("Sequence batching is not supported yet.");
    /*
    TritonJsonValue states;
    if (sequence_batching.Find("state", &states))
    {
      for (size_t i = 0; i < states.ArraySize(); i += 1)
      {
        TritonJsonValue state;
        if (auto err = states.IndexAsObject(i, &state))
        {
          DEBUG_TRACE_ERROR("Failed to get sequence state " << i
                            << " for model instance \"" << Name() << "\": "
                            << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(err,
                                 "Failed to get sequence state " << i
                                 << " for model instance \"" << Name() <<
                                 "\": " << TRITONSERVER_ErrorMessage(err));
        }

        std::string state_name;
        if (auto err = state.MemberAsString("output_name", &state_name))
        {
          DEBUG_TRACE_ERROR("Failed to get output name for sequence state "
          << i
                            << " for model instance \"" << Name() << "\": "
                            << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(err,
                                 "Failed to get output name for sequence
                                 state " << i
                                 << " for model instance \"" << Name() <<
                                 "\": " << TRITONSERVER_ErrorMessage(err));
        }

        std::string state_dtype;
        if (auto err = state.MemberAsString("data_type", &state_dtype))
        {
          DEBUG_TRACE_ERROR("Failed to get data type for sequence state
          output \"" << state_name << "\""
                            " for model instance \"" << Name() << "\": " <<
                            TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(err,
                                 "Failed to get data type for sequence state
                                 output \"" << state_name << "\"" " for model
                                 instance \"" << Name() << "\": " <<
                                 TRITONSERVER_ErrorMessage(err));
        }

        std::vector<int64_t> dims;
        if (auto err = ParseShape(state, "dims", &dims))
        {
          DEBUG_TRACE_ERROR("Failed to parse dims for sequence state output
          \"" << state_name << "\""
                            " for model instance \"" << Name() << "\": " <<
                            TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(err,
                                 "Failed to parse dims for sequence state
                                 output \"" << state_name << "\"" " for model
                                 instance \"" << Name() << "\": " <<
                                 TRITONSERVER_ErrorMessage(err));
        }

        DEBUG_TRACE_INFO("{ name: \"" << Name() << "\""
                         << ", output_index: " << output_index
                         << ", states[" << i << "]:"
                           << " { state_name: \"" << state_name << "\""
                           << ", state_dtype: \"" << state_dtype << "\""
                           << ", len(dims): " << dims.size() << " }"
                         << " }");

        int start_pos = state_name.find(DELIMINATOR);
        output_index = std::atoi(state_name.substr(start_pos + 2).c_str());

        const auto pr = ModelConfigDataTypeToTorchType(state_dtype);
        if (!pr.first && state_dtype != "TYPE_STRING")
        {
          DEBUG_TRACE_ERROR("Unsupported datatype " << state_dtype << " for
          sequence state output \"" << state_name << "\" for model instance
          \"" << Name() << "\".");
          THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                                 "Unsupported datatype " << state_dtype << "
                                 for sequence state output" " \"" <<
                                 state_name << "\" for model instance \"" <<
                                 Name() << "\".");
        }

        // Validate shape for String outputs. Only allow 1 dimension.
        if (state_dtype == "TYPE_STRING")
        {
          if ((dims.size() + (is_batching_supported_ ? 1 : 0)) > 1)
          {
            DEBUG_TRACE_ERROR("Triton only supports 1-dimensional string
            outputs for sequence state output"
                              " \"" << state_name << "\" for model instance
                              \"" << Name() << "\".");
            THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                                   "Triton only supports 1-dimensional string
                                   outputs for sequence state output" " \""
                                   << state_name << "\" for model instance
                                   \"" << Name() << "\".");
          }
        }

        // model_->OutputMap()[state_name] = op_index;
        map_outputs_[state_name].triton_dtype() =
        ConvertTorchTypeToDataType(pr.second); DEBUG_TRACE_INFO("{ name: \""
        << Name() << "\""
                         << ", state_name: \"" << state_name << "\""
                         << ", output_index: " <<
                         map_outputs_[state_name].server_index()
                         << ", torch_type: " <<
                         map_outputs_[state_name].torch_dtype()
                         << ", output_dtype: \"" <<
                         TRITONSERVER_DataTypeString(map_outputs_[state_name].triton_dtype())
                         << "\""
                         << " }");
      }
    }
    */
  }
}

bool
ModelInstanceState::ValidateTypedSequenceControl(
    TritonJsonValue& sequence_batching, const std::string& control_kind,
    bool required)
{
  THROW_TRITON_EXCEPTION(
      TRITONSERVER_ERROR_INTERNAL,
      "Typed sequence control validation is not supported for model instance \""
          << Name() << "\".");

  std::string tensor_name;
  std::string tensor_dtype;

  if (auto err = GetTypedSequenceControlProperties(
          /* batcher= */ sequence_batching,
          /* model_name= */ model_->Name(),
          /* control_kind= */ control_kind,
          /* required= */ required,
          /* tensor_name= */ &tensor_name,
          /* tensor_datatype= */ &tensor_dtype)) {
    DEBUG_TRACE_ERROR(
        "Error validating typed sequence control for model instance \""
        << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
    THROW_TRITON_EXCEPTION(
        err, "Failed to validate typed sequence control for model instance \""
                 << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
  }

  bool have_control{!tensor_name.empty()};
  if (have_control) {
    int input_index{0};
    int start_pos{static_cast<int>(tensor_name.find(DELIMINATOR))};

    if (start_pos == -1) {
      DEBUG_TRACE_ERROR(
          "Input \""
          << tensor_name
          << "\" does not follow <name>__<index> naming convention.");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Input \""
              << tensor_name
              << "\" does not follow <name>__<index> naming convention.");
    }

    // Check if the index part of the name is not an integer.
    std::string index_str{tensor_name.substr(start_pos + 2)};
    for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
      if (std::isdigit(*itr) == 0) {
        DEBUG_TRACE_ERROR(
            "Input \""
            << tensor_name
            << "\" does not follow <name>__<index> naming convention.");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ERROR_INTERNAL,
            "Input \""
                << tensor_name
                << "\" does not follow <name>__<index> naming convention.");
      }
    }

    // Check if the data type is supported by PyTorch.
    if (!ModelConfigDataTypeToTorchType(tensor_dtype).first) {
      DEBUG_TRACE_ERROR(
          "Unsupported datatype "
          << tensor_dtype << " for typed sequence control input \""
          << tensor_name << "\" for model instance \"" << Name() << "\".");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Unsupported datatype "
              << tensor_dtype << " for typed sequence control input \""
              << tensor_name << "\" for model instance \"" << Name() << "\".");
    }

    input_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
  }

  return have_control;
}
}  // namespace triton::backend::pytorch::pt2
