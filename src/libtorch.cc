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
#include "triton/common/nvtx.h"
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
#include <c10/cuda/CUDAGuard.h>
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
      std::shared_ptr<torch::jit::script::Module>* torch_model);

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
  bool EnabledCacheCleaning() { return enable_cache_cleaning_; }

  bool EnabledWeightSharing() { return enable_weight_sharing_; }
  const std::vector<const char*>& ModelOutputs() { return output_names_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();

  // Parses and validates parameters in config
  TRITONSERVER_Error* ParseParameters();

  // Flag to indicate whether optimized execution is enabled. Defaults to true.
  bool enable_optimized_execution_;

  // Flag to indicate whether inference mode is enabled. Defaults to false.
  bool enable_inference_mode_;

  // Flag to indicate whether cache clearning after each run is enabled.
  // Defaults to false.
  bool enable_cache_cleaning_;

  // Flag to indicate whether weight sharing is enabled. Defaults to false.
  bool enable_weight_sharing_;

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

  // Model mapping for shared TorchScript model across all instances on the
  // same device. The key is a pair of isGPU and device index.
  std::map<
      std::pair<bool, int64_t>, std::shared_ptr<torch::jit::script::Module>>
      torch_models_;

  // List of all the outputs specified in the output section of model
  // configuration.
  std::vector<const char*> output_names_;
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
    RETURN_IF_ERROR((*state)->SetModelConfig());
  }

  RETURN_IF_ERROR((*state)->ParseParameters());

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), enable_optimized_execution_(true),
      enable_inference_mode_(false), enable_cache_cleaning_(false),
      enable_weight_sharing_(false), enable_tensor_fuser_pair_({false, true}),
      enable_jit_profiling_pair_({false, true}),
      enable_jit_executor_pair_({false, true}),
      enable_nvfuser_pair_({false, false})
{
  output_names_.clear();

  triton::common::TritonJson::Value ios;
  THROW_IF_BACKEND_INSTANCE_ERROR(ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    THROW_IF_BACKEND_INSTANCE_ERROR(ios.IndexAsObject(i, &io));

    // Use names from ModelConfig by reference since the model
    // config will persist longer than this inference execution.
    const char* io_name;
    size_t io_name_len;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        io.MemberAsString("name", &io_name, &io_name_len));
    output_names_.emplace_back(io_name);
  }
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& artifact_name, const torch::Device device,
    std::string* model_path,
    std::shared_ptr<torch::jit::script::Module>* torch_model)
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

  // If weight sharing is enabled, skip loading model if
  // it is already available on the target device
  std::pair<bool, int> device_pair;
  if (enable_weight_sharing_) {
    device_pair = std::make_pair(!device.is_cpu(), device.index());
    auto mit = torch_models_.find(device_pair);
    if (mit != torch_models_.end()) {
      *torch_model = mit->second;
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Reusing TorchScript model for instance '") + Name() +
           "'")
              .c_str());
      return nullptr;  // success
    }
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

  if (enable_weight_sharing_) {
    if (!((torch_models_.emplace(device_pair, *torch_model)).second)) {
      std::string type = device.is_cpu() ? "CPU" : "GPU";
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("Model already found on target ") + type + " device " +
           "(id " + std::to_string(device.index()) + ") for '" + Name() + "'")
              .c_str());
    }
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

    // If 'ENABLE_CACHE_CLEANING' is not present in 'parameters' then
    // no update is made to 'enable_cache_cleaning_'.
    err = ParseParameter(
        params, "ENABLE_CACHE_CLEANING", &enable_cache_cleaning_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Cache Cleaning is ") +
         (enable_cache_cleaning_ ? "enabled" : "disabled") +
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

    // If 'ENABLE_WEIGHT_SHARING' is not present in 'parameters' then no
    // update is made to 'enable_weight_sharing'.
    err = ParseParameter(
        params, "ENABLE_WEIGHT_SHARING", &enable_weight_sharing_);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        return err;
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Weight sharing is ") +
           (enable_weight_sharing_ ? "enabled" : "disabled") +
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
      enable_nvfuser = false;
      enable_nvfuser_pair_ = {true, enable_nvfuser};
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN, (std::string("NvFuser is ") +
                                  (enable_nvfuser ? "enabled" : "disabled") +
                                  " for model instance '" + Name() + "'")
                                     .c_str());
    }
  }

  return nullptr;
}

// The naming convention followed for inputs/outputs in the model configuration.
// Outputs don't support FORWARD_ARGUMENT.
enum class NamingConvention {
  NAMED_INDEX,
  FORWARD_ARGUMENT,
  STRICT_CONFIG_ORDERING
};

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

  // Clear CUDA cache
  void ClearCache();

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
      std::vector<torch::jit::IValue>* output_tensors);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      std::vector<torch::jit::IValue>* input_tensors,
      std::vector<BackendMemory*>* input_memories, bool* cuda_copy);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size,
      const std::vector<torch::jit::IValue>& output_tensors,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  // Get the naming convention for inputs/outputs from the model configuration
  TRITONSERVER_Error* GetNamingConvention(
      NamingConvention* naming_convention,
      const std::vector<std::string>& allowed_io);

  ModelState* model_state_;

  // The full path to the TorchScript model file.
  std::string model_path_;

  std::shared_ptr<torch::jit::script::Module> torch_model_;
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

  // If the model supports batching.
  bool supports_batching_;

#ifdef TRITON_ENABLE_GPU
  // PyTorch stream used for execution of inferences.
  at::cuda::CUDAStream pytorch_stream_;
  cudaEvent_t compute_input_event_;
  cudaEvent_t compute_infer_event_;
  cudaEvent_t compute_output_event_;
  cudaEvent_t compute_output_end_event_;
#endif
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
#ifdef TRITON_ENABLE_GPU
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), device_(torch::kCPU), is_dict_input_(false),
      pytorch_stream_(at::cuda::getDefaultCUDAStream())
#else
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), device_(torch::kCPU), is_dict_input_(false)
#endif
{
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    device_ = torch::Device(torch::kCUDA, DeviceId());
#ifdef TRITON_ENABLE_GPU
    pytorch_stream_ =
        at::cuda::getStreamFromPool(false /* is_high_priority */, DeviceId());
    if (stream_ != nullptr) {
      cudaError_t err = cudaStreamDestroy(stream_);
      if (err != cudaSuccess) {
        TRITONSERVER_LogMessage(
            TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
            (std::string("~BackendModelInstance: ") + name_ +
             " failed to destroy cuda stream: " + cudaGetErrorString(err))
                .c_str());
      }
      stream_ = pytorch_stream_.stream();
    }
    THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
        cudaEventCreate(&compute_input_event_), TRITONSERVER_ERROR_INTERNAL,
        "Failed to create cuda event"));
    THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
        cudaEventCreate(&compute_infer_event_), TRITONSERVER_ERROR_INTERNAL,
        "Failed to create cuda event"));
    THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
        cudaEventCreate(&compute_output_event_), TRITONSERVER_ERROR_INTERNAL,
        "Failed to create cuda event"));
    THROW_IF_BACKEND_INSTANCE_ERROR(ConvertCUDAStatusToTritonError(
        cudaEventCreate(&compute_output_end_event_),
        TRITONSERVER_ERROR_INTERNAL, "Failed to create cuda event"));
#endif
  } else {
#ifdef TRITON_ENABLE_GPU
    pytorch_stream_ = at::cuda::getDefaultCUDAStream();
#endif
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
  supports_batching_ = model_state_->MaxBatchSize() > 0;

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs(expected_input_cnt));
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());
}

void
ModelInstanceState::ClearCache()
{
#ifdef TRITON_ENABLE_GPU
  if (device_.is_cuda()) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
#endif  // TRITON_ENABLE_GPU
}

ModelInstanceState::~ModelInstanceState()
{
  torch_model_.reset();
#ifdef TRITON_ENABLE_GPU
  stream_ = nullptr;
#endif
  ClearCache();
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
  std::string deliminator = "__";
  int ip_index = 0;

  if (ios.ArraySize() == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "model configuration must contain at least one input, none were "
        "specified.");
  }

  NamingConvention naming_convention;
  RETURN_IF_ERROR(GetNamingConvention(&naming_convention, allowed_inputs));

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
      switch (naming_convention) {
        case NamingConvention::FORWARD_ARGUMENT: {
          auto itr =
              std::find(allowed_inputs.begin(), allowed_inputs.end(), io_name);
          if (itr != allowed_inputs.end()) {
            input_index_map_[io_name] =
                std::distance(allowed_inputs.begin(), itr);
          }
          break;
        }
        case NamingConvention::NAMED_INDEX: {
          int start_pos = io_name.find(deliminator);
          ip_index = std::atoi(io_name.substr(start_pos + 2).c_str());
          input_index_map_[io_name] = ip_index;
          break;
        }
        case NamingConvention::STRICT_CONFIG_ORDERING: {
          input_index_map_[io_name] = i;
          break;
        }
      }
    }

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
  std::vector<BackendMemory*> input_memories;
  bool cuda_copy = false;
  std::unique_ptr<BackendInputCollector> collector;
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventRecord(compute_input_event_, stream_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
#endif
  }

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

  // If the instance kind is not GPU, we need to synchronize the CUDA stream
  if (Kind() != TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    if (cuda_copy) {
      cudaStreamSynchronize(stream_);
      cuda_copy = false;
    }
#endif
  } else {
#ifdef TRITON_ENABLE_GPU
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventRecord(compute_infer_event_, stream_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
#endif
  }

  std::vector<torch::jit::IValue> output_tensors;
  if (!all_response_failed) {
  }

  uint64_t compute_start_ns = 0;
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventRecord(compute_infer_event_, stream_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
#endif
  } else {
    SET_TIMESTAMP(compute_start_ns);
  }


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
    for (const auto& name : model_state_->ModelOutputs()) {
      int op_index = output_index_map_[name];
      if ((op_index < 0) || (op_index > max_index)) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "The output " + std::string(name) +
                    " in the model configuration refers to an output index "
                    "which doesn't exist. This model has " +
                    std::to_string(max_index + 1) + " outputs")
                    .c_str()));
        invalid_index = true;
        break;
      }
    }
  }

  uint64_t compute_end_ns = 0;
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventRecord(compute_output_event_, stream_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
#endif
  } else {
    SET_TIMESTAMP(compute_end_ns);
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
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventRecord(compute_output_end_event_, stream_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record the event"));
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventSynchronize(compute_output_end_event_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to synchronize event"));
    float compute_input_duration = 0;
    float compute_infer_duration = 0;
    float compute_output_duration = 0;
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventElapsedTime(
                &compute_input_duration, compute_input_event_,
                compute_infer_event_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to capture elapsed time"));

    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventElapsedTime(
                &compute_infer_duration, compute_infer_event_,
                compute_output_event_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to capture elapsed time"));

    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventElapsedTime(
                &compute_output_duration, compute_output_event_,
                compute_output_end_event_),
            TRITONSERVER_ERROR_INTERNAL, "Failed to capture elapsed time"));
    compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
    compute_end_ns = compute_start_ns + (compute_infer_duration * 1e6);
    exec_end_ns = compute_end_ns + (compute_output_duration * 1e6);
#endif
  } else {
    SET_TIMESTAMP(exec_end_ns);
  }

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
    std::vector<torch::jit::IValue>* output_tensors)
{
#ifdef TRITON_ENABLE_GPU
  at::cuda::CUDAStreamGuard device_guard{pytorch_stream_};
#endif
  NVTX_RANGE(nvtx_, "Execute " + Name());

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

TRITONSERVER_Error*
ModelInstanceState::GetNamingConvention(
    NamingConvention* naming_convention,
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

  return nullptr;  // success
}

// This function will return a tensor's contents as a contiguous
// chunk in system memory. In some cases this will require copying the data.
// If that  happens, 'contiguous_buffer' will be set to hold the contiguous
// chunk and 'cuda_copy' will be set to indicate whether CUDA copy is
// conducted.  The data copy can be avoided if the input is already in
// a contiguous chunk and the input is located in memory type and id
// specified.
TRITONSERVER_Error*
GetContiguousInputContent(
    TRITONBACKEND_Input* rinput, const uint32_t buffer_count,
    const char** content, size_t* content_byte_size,
    std::vector<char>* contiguous_buffer, cudaStream_t stream, bool* cuda_copy)
{
  *cuda_copy = false;

  // Check input buffers to see if data copy is necessary
  size_t chunk_count = 0;
  bool type_mismatch = false;
  uint64_t total_byte_size = 0;
  for (size_t idx = 0; idx < buffer_count; ++idx) {
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;
    size_t src_byte_size;
    const void* src_ptr;

    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        rinput, idx, &src_ptr, &src_byte_size, &src_memory_type,
        &src_memory_type_id));

    if (src_ptr != nullptr) {
      chunk_count++;
      total_byte_size += src_byte_size;
      type_mismatch |= (src_memory_type == TRITONSERVER_MEMORY_GPU);
    }
  }

  if (chunk_count == 0) {
    *content = nullptr;
    *content_byte_size = 0;
  } else if ((chunk_count == 1) && !type_mismatch) {
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        rinput, 0, (const void**)content, content_byte_size, &src_memory_type,
        &src_memory_type_id));
  } else {
    contiguous_buffer->resize(total_byte_size);

    size_t offset = 0;
    for (size_t i = 0; i < chunk_count; i++) {
      bool cuda_used;
      TRITONSERVER_MemoryType src_memory_type;
      int64_t src_memory_type_id;
      size_t src_byte_size;
      const void* src_ptr;

      RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
          rinput, i, &src_ptr, &src_byte_size, &src_memory_type,
          &src_memory_type_id));
      RETURN_IF_ERROR(CopyBuffer(
          "Contiguous input", src_memory_type, src_memory_type_id,
          TRITONSERVER_MEMORY_CPU, 0, src_byte_size, src_ptr,
          contiguous_buffer->data() + offset, stream, &cuda_used));
      *cuda_copy |= cuda_used;
      offset += src_byte_size;
    }

    *content = contiguous_buffer->data();
    *content_byte_size = total_byte_size;
  }

  return nullptr;  // success
}

void
FillStringTensor(torch::List<std::string>* input_list, const size_t cnt)
{
  for (size_t c = 0; c < cnt; ++c) {
    input_list->push_back("");
  }
}

bool
SetStringInputTensor(
    torch::List<std::string>* input_list, TRITONBACKEND_Input* input,
    const char* name, const uint32_t buffer_count,
    const size_t request_element_cnt, TRITONBACKEND_Response** response,
    cudaStream_t stream, const char* host_policy_name)
{
  bool cuda_copy = false;
  size_t element_idx = 0;

  // For string data type, we always need to have the data on CPU so
  // that we can read string length and construct the string
  // properly. So if the request's input tensor is not in CPU need to
  // copy it there.
  const char* content = nullptr;
  size_t content_byte_size = 0;

  std::vector<char> contiguous_buffer;
  auto err = GetContiguousInputContent(
      input, buffer_count, &content, &content_byte_size, &contiguous_buffer,
      stream, &cuda_copy);
  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    FillStringTensor(input_list, request_element_cnt - element_idx);
    return cuda_copy;
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream);
    cuda_copy = false;
  }
#endif  // TRITON_ENABLE_GPU

  // Parse content and assign to 'tensor'. Each string in 'content'
  // is a 4-byte length followed by the string itself with no
  // null-terminator.
  while (content_byte_size >= sizeof(uint32_t)) {
    if (element_idx >= request_element_cnt) {
      RESPOND_AND_SET_NULL_IF_ERROR(
          response,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              std::string(
                  "unexpected number of string elements " +
                  std::to_string(element_idx + 1) + " for inference input '" +
                  name + "', expecting " + std::to_string(request_element_cnt))
                  .c_str()));
      return cuda_copy;
    }

    const uint32_t len = *(reinterpret_cast<const uint32_t*>(content));
    content += sizeof(uint32_t);
    content_byte_size -= sizeof(uint32_t);

    if (content_byte_size < len) {
      RESPOND_AND_SET_NULL_IF_ERROR(
          response,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              std::string(
                  "incomplete string data for inference input '" +
                  std::string(name) + "', expecting string of length " +
                  std::to_string(len) + " but only " +
                  std::to_string(content_byte_size) + " bytes available")
                  .c_str()));
      FillStringTensor(input_list, request_element_cnt - element_idx);
      return cuda_copy;
    }

    // Set string value
    input_list->push_back(std::string(content, len));

    content += len;
    content_byte_size -= len;
    element_idx++;
  }

  if ((*response != nullptr) && (element_idx != request_element_cnt)) {
    RESPOND_AND_SET_NULL_IF_ERROR(
        response, TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INTERNAL,
                      std::string(
                          "expected " + std::to_string(request_element_cnt) +
                          " strings for inference input '" + name + "', got " +
                          std::to_string(element_idx))
                          .c_str()));
    if (element_idx < request_element_cnt) {
      FillStringTensor(input_list, request_element_cnt - element_idx);
    }
  }

  return cuda_copy;
}

bool
SetStringOutputBuffer(
    torch::List<torch::jit::IValue>* tensor, TRITONBACKEND_Response** response,
    TRITONBACKEND_Output* response_output, const size_t tensor_element_count,
    cudaStream_t stream, std::string* serialized)
{
  bool cuda_copy = false;

  // Serialize the output tensor strings. Each string is serialized as
  // a 4-byte length followed by the string itself with no
  // null-terminator.
  serialized->clear();
  for (size_t e = 0; e < tensor_element_count; ++e) {
    std::string str = tensor->get(e).to<std::string>();
    const char* cstr = str.c_str();
    size_t len = str.length();
    serialized->append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    if (len > 0) {
      serialized->append(cstr, len);
    }
  }

  // Allocate a buffer large enough to hold the serialized tensor.
  TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t actual_memory_type_id = 0;

  void* buffer;
  auto err = TRITONBACKEND_OutputBuffer(
      response_output, &buffer, serialized->size(), &actual_memory_type,
      &actual_memory_type_id);
  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    return cuda_copy;
  }

  // Copy the serialized tensor into the allocated buffer.
  bool cuda_used = false;
  err = CopyBuffer(
      "String output", TRITONSERVER_MEMORY_CPU /* src_memory_type */,
      0 /* src_memory_type_id */, actual_memory_type, actual_memory_type_id,
      serialized->size(), reinterpret_cast<const void*>(serialized->c_str()),
      buffer, stream, &cuda_used);
  cuda_copy |= cuda_used;

  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    return cuda_copy;
  }

  return cuda_copy;
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
    if (supports_batching_) {
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

        const int64_t batch_element_cnt = GetElementCount(shape, dims_count);

        *cuda_copy |= SetStringInputTensor(
            &input_list, input, input_name, buffer_count, batch_element_cnt,
            &((*responses)[idx]), CudaStream(), HostPolicyName().c_str());
      }

      (*input_tensors)[input_index_map_[input_name]] = input_list;
    } else {
      // Remove constness to align with the signature of torch::from_blob()
      torch::Tensor input_tensor = torch::from_blob(
          const_cast<char*>(input_buffer), batchn_shape, updated_options);
      (*input_tensors)[input_index_map_[input_name]] = input_tensor;
    }
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size,
    const std::vector<torch::jit::IValue>& output_tensors,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  NVTX_RANGE(nvtx_, "ReadOutputTensors " + Name());

  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->MaxBatchSize() > 0, model_state_->EnablePinnedInput(),
      CudaStream());

  bool cuda_copy = false;
  // The serialized string buffer must be valid until output copies are done
  std::vector<std::unique_ptr<std::string>> string_buffer;
  for (size_t idx = 0; idx < model_state_->ModelOutputs().size(); idx++) {
    std::string name = model_state_->ModelOutputs()[idx];
    int op_index = output_index_map_[name];

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

      responder.ProcessTensor(
          name, output_dtype, batchn_shape, output_buffer,
          (tensor_device.type() == torch::kCPU) ? TRITONSERVER_MEMORY_CPU
                                                : TRITONSERVER_MEMORY_GPU,
          (tensor_device.type() == torch::kCPU) ? 0 : tensor_device.index());

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

        const size_t tensor_element_cnt = GetElementCount(batchn_shape);

        // Only need an response tensor for requested outputs.
        if (response != nullptr) {
          TRITONBACKEND_Output* response_output;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_ResponseOutput(
                             response, &response_output, name.c_str(),
                             TRITONSERVER_TYPE_BYTES, batchn_shape.data(),
                             batchn_shape.size()));
          string_buffer.emplace_back(new std::string());
          cuda_copy |= SetStringOutputBuffer(
              &output_list, &response, response_output, tensor_element_cnt,
              CudaStream(), string_buffer.back().get());
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

  if (Kind() != TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    if (cuda_copy) {
      cudaStreamSynchronize(stream_);
      cuda_copy = false;
    }
#endif
  }

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

  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
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

  if (model_state->EnabledCacheCleaning()) {
    instance_state->ClearCache();
  }

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::pytorch
