// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdint.h>
#include <exception>
#include "libtorch_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

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
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

//
// PyTorch C++ (LibTorch) Backend that implements the TRITONBACKEND API.
//

namespace triton { namespace backend { namespace pytorch {

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Load a TorchScript model using 'artifact_name' as the name for the
  // TorchScript file. Return in 'model_path' the full path to the
  // TorchScript file, return in 'torch_model' the Torch Module
  // representing the model.
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name, const torch::Device device,
      std::string* model_path,
      std::unique_ptr<torch::jit::script::Module>* torch_model);

  bool EnabledOptimizedExecution() { return enable_optimized_execution_; }
  const std::pair<bool, bool>& EnabledTensorExprFuser() const
  {
    return enable_tensor_fuser_pair_;
  }
  const std::pair<bool, bool>& EnabledJitProfiling() const
  {
    return enable_jit_profiling_pair_;
  }
  const std::pair<bool, bool>& EnabledJitExecutor() const
  {
    return enable_jit_executor_pair_;
  }
  bool EnabledInferenceMode() { return enable_inference_mode_; }
  const std::pair<bool, bool>& EnabledNvfuserPair() const
  {
    return enable_nvfuser_pair_;
  }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();

  // Parses and validates parameters in config
  TRITONSERVER_Error* ParseParameters();

  // Flag to indicate whether optimized execution is enabled. Defaults to true.
  bool enable_optimized_execution_;

  // Flag to indicate whether inference mode is enabled. Defaults to false.
  bool enable_inference_mode_;

  // Flag pairs to indicate if various JIT settings are set and
  // enabled respectively. Defaults to (false, true). Default behavior
  // is to do nothing if not explicitly set. Tensor fuser flag is
  // ignore if nvfuser is explicitly set.
  std::pair<bool, bool> enable_tensor_fuser_pair_;
  std::pair<bool, bool> enable_jit_profiling_pair_;
  std::pair<bool, bool> enable_jit_executor_pair_;

  // Flag pair to indicate whether nvfuser is set and enabled respectively.
  // Defaults to (false, false).
  std::pair<bool, bool> enable_nvfuser_pair_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());

    triton::common::TritonJson::WriteBuffer json_buffer;
    (*state)->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message* message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, 1 /* config_version */, message));
  }

  RETURN_IF_ERROR((*state)->ParseParameters());

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), enable_optimized_execution_(true),
      enable_inference_mode_(false), enable_tensor_fuser_pair_({false, true}),
      enable_jit_profiling_pair_({false, true}),
      enable_jit_executor_pair_({false, true}),
      enable_nvfuser_pair_({false, false})
{
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& artifact_name, const torch::Device device,
    std::string* model_path,
    std::unique_ptr<torch::jit::script::Module>* torch_model)
{
  // Find the TorchScript file that describes the model. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.pt").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.pt";
  }

  *model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(*model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + *model_path +
            "' for model instance '" + Name() + "'");
  }

  // Serialize the torch model to string
  std::string model_data_str;
  RETURN_IF_ERROR(ReadTextFile(*model_path, &model_data_str));

  // InferenceMode should be used to guard all tensors operations including
  // model loading: https://pytorch.org/cppdocs/notes/inference_mode.html
  torch::InferenceMode infer_guard(EnabledInferenceMode());

  try {
    std::istringstream model_stream(model_data_str);
    torch_model->reset(
        new torch::jit::Module(torch::jit::load(model_stream, device)));
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("failed to load model '" + Name() + "': " + ex.what()).c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Auto-complete configuration is not supported since PyTorch does not
  // store/capture sufficient model metadata so just log error instead.
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("skipping model configuration auto-complete for '") +
       Name() + "': not supported for pytorch backend")
          .c_str());

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ParseParameters()
{
  triton::common::TritonJson::Value params;
  bool status = model_config_.Find("parameters", &params);
  if (status) {
    // If 'DISABLE_OPTIMIZED_EXECUTION' is not present in 'parameters' then no
    // update is made to 'enable_optimized_execution_'.
    bool disable_optimized_execution = false;
    TRITONSERVER_Error* err = ParseParameter(
        params, "DISABLE_OPTIMIZED_EXECUTION", &disable_optimized_execution);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }
    enable_optimized_execution_ = !disable_optimized_execution;

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Optimized execution is ") +
         (enable_optimized_execution_ ? "enabled" : "disabled") +
         " for model instance '" + Name() + "'")
            .c_str());

    // If 'INFERENCE_MODE' is not present in 'parameters' then no update is made
    // to 'enable_inference_mode_'.
    err = ParseParameter(params, "INFERENCE_MODE", &enable_inference_mode_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Inference Mode is ") +
         (enable_inference_mode_ ? "enabled" : "disabled") +
         " for model instance '" + Name() + "'")
            .c_str());

    // If 'ENABLE_TENSOR_FUSER' is not present in 'parameters' then no
    // update is made to 'enable_tensor_fuser'.
    bool enable_tensor_fuser = false;
    err = ParseParameter(params, "ENABLE_TENSOR_FUSER", &enable_tensor_fuser);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      enable_tensor_fuser_pair_ = {true, enable_tensor_fuser};
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Tensor fuser is ") +
           (enable_tensor_fuser ? "enabled" : "disabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // If 'ENABLE_JIT_PROFILING' is not present in 'parameters' then no update
    // is made to 'enable_jit_profiling'.
    bool enable_jit_profiling = false;
    err = ParseParameter(params, "ENABLE_JIT_PROFILING", &enable_jit_profiling);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      enable_jit_profiling_pair_ = {true, enable_jit_profiling};
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Jit profiling is ") +
           (enable_jit_profiling ? "enabled" : "disabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // If 'ENABLE_JIT_EXECUTOR' is not present in 'parameters' then no update is
    // made to 'enable_jit_executor'.
    bool enable_jit_executor = false;
    err = ParseParameter(params, "ENABLE_JIT_EXECUTOR", &enable_jit_executor);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      enable_jit_executor_pair_ = {true, enable_jit_executor};
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Jit executor is ") +
           (enable_jit_executor ? "enabled" : "disabled") +
           " for model instance '" + Name() + "'")
              .c_str());
    }

    // TODO Re-enable NvFuser once fixed
    // If 'ENABLE_NVFUSER' is not present in 'parameters' then no
    // update is made to 'enable_nvfuser'.
    bool enable_nvfuser = false;
    err = ParseParameter(params, "ENABLE_NVFUSER", &enable_nvfuser);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO, (std::string("NvFuser is not specified") +
                                    " for model instance '" + Name() + "'")
                                       .c_str());
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      // Override, disable NvFuser till fixed
      enable_nvfuser_pair_ = {true, false /* enable_nvfuser */};
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO, (std::string("NvFuser is ") +
                                  (enable_nvfuser ? "enabled" : "disabled") +
                                  " for model instance '" + Name() + "'")
                                     .c_str());
    }
  }

  return nullptr;
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  TRITONSERVER_Error* ValidateBooleanSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateTypedSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();
  void Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count,
      std::vector<torch::jit::IValue>* input_tensors,
      std::vector<torch::Tensor>* output_tensors);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      std::vector<torch::jit::IValue>* input_tensors,
      std::vector<BackendMemory*>* input_memories, bool* cuda_copy);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, const std::vector<const char*>& output_names,
      const std::vector<torch::Tensor>& output_tensors,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      uint64_t* compute_end_ns);

  ModelState* model_state_;

  // The full path to the TorchScript model file.
  std::string model_path_;

  std::unique_ptr<torch::jit::script::Module> torch_model_;
  torch::Device device_;

  // Map from configuration name for an input to the index of
  // that input in the model.
  std::unordered_map<std::string, int> input_index_map_;

  // Map from configuration name for an output to the index of
  // that output in the model.
  std::unordered_map<std::string, int> output_index_map_;
  std::unordered_map<std::string, TRITONSERVER_DataType> output_dtype_map_;

  // If the input to the tensor is a dictionary of tensors.
  bool is_dict_input_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
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

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), device_(torch::kCPU), is_dict_input_(false)
{
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    device_ = torch::Device(torch::kCUDA, DeviceId());
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), device_, &model_path_, &torch_model_));

  size_t expected_input_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (model_state->ModelConfig().Find("input", &inputs)) {
      expected_input_cnt = inputs.ArraySize();
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
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs(expected_input_cnt));
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());
}

ModelInstanceState::~ModelInstanceState()
{
  torch_model_.reset();
#ifdef TRITON_ENABLE_GPU
  if (device_.is_cuda()) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
#endif  // TRITON_ENABLE_GPU
}

TRITONSERVER_Error*
ModelInstanceState::ValidateBooleanSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
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
    try {
      int start_pos = tensor_name.find(deliminator);
      if (start_pos == -1) {
        throw std::invalid_argument("input must follow naming convention");
      }
      ip_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
      input_index_map_[tensor_name] = ip_index;
    }
    catch (std::exception& ex) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("input '" + tensor_name +
           "' does not follow naming convention i.e. <name>__<index>.")
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateTypedSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetTypedSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    std::string deliminator = "__";
    int ip_index = 0;
    try {
      int start_pos = tensor_name.find(deliminator);
      if (start_pos == -1) {
        throw std::invalid_argument("input must follow naming convention");
      }
      ip_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
      input_index_map_[tensor_name] = ip_index;
    }
    catch (std::exception& ex) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("input '" + tensor_name +
           "' does not follow naming convention i.e. <name>__<index>.")
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs(const size_t expected_input_cnt)
{
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
      if (arguments.at(i).type()->kind() != c10::TypeKind::TensorType) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("An input of type '") + arguments.at(i).type()->str() +
             "' was detected in the model. Only a single input of type "
             "Dict(str, Tensor) or input(s) of type Tensor are supported.")
                .c_str());
      }
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
  std::string deliminator = "__";
  int ip_index = 0;

  if (ios.ArraySize() == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "model configuration must contain at least one input, none were "
        "specified.");
  }

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Validate name
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    if (is_dict_input_) {
      // If dictionary, index is irrelevant but we use the map to store the
      // input names since they are the keys for the dictionary
      input_index_map_[io_name] = i;
    } else {
      try {
        int start_pos = io_name.find(deliminator);
        if (start_pos == -1) {
          throw std::invalid_argument("input must follow naming convention");
        }
        ip_index = std::atoi(io_name.substr(start_pos + 2).c_str());
        input_index_map_[io_name] = ip_index;
      }
      catch (std::exception& ex) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("input '" + io_name +
             "' does not follow naming convention i.e. <name>__<index>.")
                .c_str());
      }
    }

    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTorchType(io_dtype);
    if (!pr.first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for input '" + io_name +
           "' for model '" + model_state_->Name() + "'")
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
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

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    // Validate name
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    try {
      int start_pos = io_name.find(deliminator);
      if (start_pos == -1) {
        throw std::invalid_argument("output must follow naming convention");
      }
      op_index = std::atoi(io_name.substr(start_pos + 2).c_str());
    }
    catch (std::exception& ex) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("output '" + io_name +
           "' does not follow naming convention i.e. <name>__<index>.")
              .c_str());
    }

    // Validate data type
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    const auto pr = ModelConfigDataTypeToTorchType(io_dtype);
    if (!pr.first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unsupported datatype " + io_dtype + " for output '" + io_name +
           "' for model '" + model_state_->Name() + "'")
              .c_str());
    }
    output_index_map_[io_name] = op_index;
    output_dtype_map_[io_name] = ConvertTorchTypeToDataType(pr.second);
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

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
      // supports batching, the first dimension size is batch size
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
  std::vector<BackendMemory*> input_memories;
  bool cuda_copy = false;
  std::unique_ptr<BackendInputCollector> collector;

  if (!all_response_failed) {
    collector.reset(new BackendInputCollector(
        requests, request_count, &responses,
        model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
        CudaStream(), nullptr, nullptr, 0, HostPolicyName().c_str()));
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        SetInputTensors(
            total_batch_size, requests, request_count, &responses,
            collector.get(), &input_names, &input_tensors, &input_memories,
            &cuda_copy));
  }

  // Request to retrieve all model outputs. 'output_names' and
  // 'output_tensors' are parallel vectors and so must be kept in
  // sync.
  std::vector<const char*> output_names;
  std::vector<torch::Tensor> output_tensors;
  if (!all_response_failed) {
    triton::common::TritonJson::Value ios;
    TRITONSERVER_Error* err =
        model_state_->ModelConfig().MemberAsArray("output", &ios);
    if (err == nullptr) {
      for (size_t i = 0; i < ios.ArraySize(); i++) {
        triton::common::TritonJson::Value io;
        err = ios.IndexAsObject(i, &io);
        if (err != nullptr) {
          break;
        }

        // Use names from ModelConfig by reference since the model
        // config will persist longer than this inference execution.
        const char* io_name;
        size_t io_name_len;
        err = io.MemberAsString("name", &io_name, &io_name_len);
        if (err != nullptr) {
          break;
        }

        output_names.emplace_back(io_name);
      }
    }

    if (err != nullptr) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed, err);
      output_names.clear();
    }
  }

// Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Run...
  if (!all_response_failed) {
    Execute(&responses, request_count, &input_tensors, &output_tensors);
  }

  // Free BackendMemory used for inputs
  for (BackendMemory* mem : input_memories) {
    if (mem != nullptr) {
      delete mem;
    }
  }
  input_memories.clear();

  // Verify output indices are valid with number of outputs after execution
  bool invalid_index = false;
  int max_index = output_tensors.size() - 1;

  if (!all_response_failed) {
    for (const auto& name : output_names) {
      int op_index = output_index_map_[name];
      if ((op_index < 0) || (op_index > max_index)) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "The output " + std::string(name) +
                    " in the model configuration refers to an output index "
                    "which"
                    " doesn't exist. This model has " +
                    std::to_string(max_index + 1) + " outputs")
                    .c_str()));
        invalid_index = true;
        break;
      }
    }
  }

  uint64_t compute_end_ns = 0;

  if (!all_response_failed) {
    if (!invalid_index) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          ReadOutputTensors(
              total_batch_size, output_names, output_tensors, requests,
              request_count, &responses, &compute_end_ns));
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

void
ModelInstanceState::Execute(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count,
    std::vector<torch::jit::IValue>* input_tensors,
    std::vector<torch::Tensor>* output_tensors)
{
  torch::jit::IValue model_outputs_;

  try {
    // enable/disable optimized execution
    torch::jit::setGraphExecutorOptimize(
        model_state_->EnabledOptimizedExecution());

    // enable/disable inference mode - supersedes NoGradGuard
    torch::InferenceMode infer_guard(model_state_->EnabledInferenceMode());

    // JIT. No change is made unless parameter is explicitly set.
    if (std::get<0>(model_state_->EnabledJitProfiling())) {
      torch::jit::getProfilingMode() =
          std::get<1>(model_state_->EnabledJitProfiling());
    }

    if (std::get<0>(model_state_->EnabledJitExecutor())) {
      torch::jit::getExecutorMode() =
          std::get<1>(model_state_->EnabledJitExecutor());
    }

    // Fuser. Parameter is ignored if NVFuser parameter is explicitily
    // set (either enabled or disabled). No change is made unless
    // fuser is explicitly set in parameters.
    if (!std::get<0>(model_state_->EnabledNvfuserPair()) &&
        std::get<0>(model_state_->EnabledTensorExprFuser())) {
      torch::jit::setTensorExprFuserEnabled(
          std::get<1>(model_state_->EnabledTensorExprFuser()));
    }

    // NV-Fuser. No change is made unless parameter is explicitly set.
    if (std::get<0>(model_state_->EnabledNvfuserPair())) {
      if (std::get<1>(model_state_->EnabledNvfuserPair()) &&
          (device_.type() != torch::kCPU)) {
        torch::jit::overrideCanFuseOnCPU(false);
        torch::jit::overrideCanFuseOnGPU(false);
        torch::jit::setTensorExprFuserEnabled(false);
        torch::jit::RegisterCudaFuseGraph::registerPass(true);
      } else {
        torch::jit::overrideCanFuseOnCPU(true);
        torch::jit::overrideCanFuseOnGPU(true);
        torch::jit::setTensorExprFuserEnabled(true);
        torch::jit::RegisterCudaFuseGraph::registerPass(false);
      }
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
      for (auto& m_op : model_outputs_tuple->elements()) {
        output_tensors->push_back(m_op.toTensor());
      }
    } else {
      try {
        auto model_output_tensor = model_outputs_.toTensor();
        output_tensors->push_back(model_output_tensor);
      }
      catch (std::exception& exx) {
        throw std::invalid_argument(
            "Output of torch model should be tensor or a tuple of tensors, not "
            "a list / dictionary of tensors or a scalar: " +
            std::string(exx.what()));
      }
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

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<torch::jit::IValue>* input_tensors,
    std::vector<BackendMemory*>* input_memories, bool* cuda_copy)
{
  const int max_batch_size = model_state_->MaxBatchSize();

  // InferenceMode should be used to guard all tensors operations
  torch::InferenceMode infer_guard(model_state_->EnabledInferenceMode());

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  input_tensors->resize(input_count);
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

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(
        input_shape, input_shape + input_dims_count);
    if (max_batch_size != 0) {
      batchn_shape[0] = total_batch_size;
    }

    // The input must be in contiguous CPU/GPU memory.
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;
    if (device_.is_cpu()) {
      alloc_perference = {{TRITONSERVER_MEMORY_CPU_PINNED, 0},
                          {TRITONSERVER_MEMORY_CPU, 0}};
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

    // Create Torch tenor
    const auto torch_dtype = ConvertDataTypeToTorchType(input_datatype);
    torch::TensorOptions options{torch_dtype.second};
    auto updated_options = (memory_type == TRITONSERVER_MEMORY_GPU)
                               ? options.device(torch::kCUDA, device_.index())
                               : options.device(torch::kCPU);

    // Remove constness to align with the signature of torch::from_blob()
    torch::Tensor input_tensor = torch::from_blob(
        const_cast<char*>(input_buffer), batchn_shape, updated_options);
    (*input_tensors)[input_index_map_[input_name]] = input_tensor;
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<const char*>& output_names,
    const std::vector<torch::Tensor>& output_tensors,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses, uint64_t* compute_end_ns)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->MaxBatchSize() > 0, model_state_->EnablePinnedInput(),
      CudaStream());

  bool cuda_copy = false;
  std::vector<std::vector<char>> string_buffers;
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    std::string name = output_names[idx];
    int op_index = output_index_map_[name];
    torch::Tensor output_flat;

    try {
      output_flat = output_tensors[op_index].contiguous().flatten();
    }
    catch (std::exception& ex) {
      RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("output tensor '") + name + "' is not found").c_str()));
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

    //  Set output shape
    std::vector<int64_t> batchn_shape;
    auto shape = output_tensors[op_index].sizes();
    for (auto itr = shape.begin(); itr != shape.end(); itr++) {
      batchn_shape.push_back(*itr);
    }

    if (batchn_shape.size() == 0) {
      RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("output '") + name +
           "' is a scalar which is not supported.")
              .c_str()));
    }

    responder.ProcessTensor(
        name, output_dtype, batchn_shape, output_buffer,
        (tensor_device.type() == torch::kCPU) ? TRITONSERVER_MEMORY_CPU
                                              : TRITONSERVER_MEMORY_GPU,
        (tensor_device.type() == torch::kCPU) ? 0 : tensor_device.index());

    // PyTorch uses asynchronous execution to run the model. Setting the compute
    // end timestamp immediately after Execute() does not capture the complete
    // model execution time. When the first output buffer is accessed/copied by
    // ProcessTensor(), there is a synchronization that is done to ensure the
    // data is correctly copied from the output tensor. To avoid overheads of
    // additional synchronization, we continue to use the default cuda stream.
    // However the drawback of this is that the compute infer time reported
    // would be slightly later than it is in reality and the compute output time
    // reported would be smaller than it is in reality. We allow this because
    // synchronizing manually negatively impacts performance.
    if (idx == 0) {
      SET_TIMESTAMP(*compute_end_ns);
    }
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU

  return nullptr;
}

/////////////

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
          .c_str());

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::pytorch
