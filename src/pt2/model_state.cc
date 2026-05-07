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

#include "model_state.hh"

#include <mutex>

#include "../libtorch.hh"
#include "../triton_utils.hh"
#include "call_spec.hh"

#ifdef TRITON_ENABLE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#endif

#ifdef ENABLE_DEBUG_TRACE_INFO
#include <sstream>
#endif

namespace {
std::once_flag pytorch_interop_threads_flag;
std::once_flag pytorch_intraop_threads_flag;
}  // namespace

namespace triton::backend::pytorch::pt2 {
using TritonBatchOutput = triton::backend::BatchOutput;
using TritonBatchInput = triton::backend::BatchInput;
using TritonBackendModel = triton::backend::BackendModel;
using TritonJsonValue = triton::common::TritonJson::Value;

ModelState::ModelState(TRITONBACKEND_Model* backend_model)
    : BackendModel{backend_model}
{
  if (!backend_model)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Argument `backend_model` cannot be `null`.");
}

void
ModelState::AutoCompleteConfig()
{
  TRITON_LOG_WARN(
      "Auto-complete configuration is not supported for Inductor models. "
      "Skipping auto-complete for model \""
      << Name() << "\".");
}

const std::vector<triton::backend::BatchInput>&
ModelState::BatchInputs() const
{
  return TritonBackendModel::BatchInputs();
}

const std::vector<triton::backend::BatchOutput>&
ModelState::BatchOutputs() const
{
  return TritonBackendModel::BatchOutputs();
}

bool
ModelState::CacheCleaningEnabled() const
{
  return cache_cleaning_enabled_;
}

void
ModelState::CacheCleaningEnabled(bool value)
{
  cache_cleaning_enabled_ = value;
}

ModelState*
ModelState::Create(TRITONBACKEND_Model* triton_model)
{
  if (!triton_model)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Argument `triton_model` cannot be `null`.");

  auto aoti_model = new ModelState{triton_model};
  bool auto_complete_config = false;

  if (auto err = TRITONBACKEND_ModelAutoCompleteConfig(
          triton_model, &auto_complete_config)) {
    DEBUG_TRACE_ERROR(
        "{ model: \"" << aoti_model->Name() << "\", error: \""
                      << TRITONSERVER_ErrorMessage(err) << "\" }");
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to check if auto-complete configuration is requested for model "
        "\"" << aoti_model->Name()
             << "\": " << TRITONSERVER_ErrorMessage(err) << ".");
  }

  if (auto_complete_config) {
    aoti_model->AutoCompleteConfig();
    aoti_model->SetModelConfig();
  }

  auto& model_outputs = aoti_model->model_outputs_;
  TritonJsonValue sequence_batching;

  if (aoti_model->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    TritonJsonValue states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i += 1) {
        TritonJsonValue state;
        if (auto err = states.IndexAsObject(i, &state)) {
          DEBUG_TRACE_ERROR(
              "{ model: \"" << aoti_model->Name() << "\""
                            << ", error: \"" << TRITONSERVER_ErrorMessage(err)
                            << "\" }");
          THROW_TRITON_EXCEPTION(
              TRITONSERVER_ERROR_INTERNAL,
              "Failed to get sequence batching state object at index "
                  << i << ": " << TRITONSERVER_ErrorMessage(err) << ".");
        }

        std::string output_state_name;
        if (auto err =
                state.MemberAsString("output_name", &output_state_name)) {
          DEBUG_TRACE_ERROR(
              "{ model: \"" << aoti_model->Name() << "\", error: \""
                            << TRITONSERVER_ErrorMessage(err) << "\" }");
          THROW_TRITON_EXCEPTION(
              TRITONSERVER_ERROR_INTERNAL,
              "Failed to get sequence batching state output name at index "
                  << i << ": " << TRITONSERVER_ErrorMessage(err) << ".");
        }

        DEBUG_TRACE_INFO(
            "{ auto_complete_config: "
            << (auto_complete_config ? "true" : "false") << ", states[" << i
            << "]: { output_state_name: " << output_state_name << " } }");

        auto it = model_outputs.find(output_state_name);
        if (it == model_outputs.end()) {
          model_outputs.insert({output_state_name, std::make_pair(-1, i)});
        } else {
          it->second.second = i;
        }
      }
    }
  }

  TritonJsonValue outputs;
  if (auto err = aoti_model->ModelConfig().MemberAsArray("output", &outputs)) {
    DEBUG_TRACE_ERROR(
        "{ model: \"" << aoti_model->Name() << "\", error: \""
                      << TRITONSERVER_ErrorMessage(err) << "\" }");
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to get model outputs array: " << TRITONSERVER_ErrorMessage(err)
                                              << ".");
  }

  for (size_t i = 0; i < outputs.ArraySize(); i += 1) {
    TritonJsonValue output;
    if (auto err = outputs.IndexAsObject(i, &output)) {
      DEBUG_TRACE_ERROR(
          "{ model: \"" << aoti_model->Name() << "\", error: \""
                        << TRITONSERVER_ErrorMessage(err) << "\" }");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Failed to get model output object at index "
              << i << ": " << TRITONSERVER_ErrorMessage(err) << ".");
    }

    std::string output_name;
    if (auto err = output.MemberAsString("name", &output_name)) {
      DEBUG_TRACE_ERROR(
          "{ model: \"" << aoti_model->Name() << "\", error: \""
                        << TRITONSERVER_ErrorMessage(err) << "\" }");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Failed to get model output name at index "
              << i << ": " << TRITONSERVER_ErrorMessage(err) << ".");
    }
    DEBUG_TRACE_INFO(
        "{ auto_complete_config: " << (auto_complete_config ? "true" : "false")
                                   << ", outputs[" << i << "]: { output_name: "
                                   << output_name << " } }");

    auto it = model_outputs.find(output_name);
    if (it == model_outputs.end()) {
      model_outputs.insert({output_name, std::make_pair(i, -1)});
    } else {
      it->second.first = i;
    }
  }

  aoti_model->ParseParameters();

  return aoti_model;
}

bool
ModelState::CudnnEnabled() const
{
  return cudnn_enabled_;
}

void
ModelState::CudnnEnabled(bool value)
{
  cudnn_enabled_ = value;
}

bool
ModelState::EnablePinnedInput() const
{
  return TritonBackendModel::EnablePinnedInput();
}

bool
ModelState::EnablePinnedOutput() const
{
  return TritonBackendModel::EnablePinnedOutput();
}

const triton::backend::BatchOutput*
ModelState::FindBatchOutput(const std::string& output_name) const
{
  return TritonBackendModel::FindBatchOutput(output_name);
}

std::vector<torch::Tensor>
ModelState::Forward(
    const std::vector<torch::Tensor>& inputs, void* stream_handle)
{
  DEBUG_TRACE_INFO(
      "{ len(inputs): " << inputs.size() << ", stream_handle: "
                        << (stream_handle ? "pointer" : "null") << " }");
  if (!model_loader_) {
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Model \""
            << Name()
            << "\" not loaded. Use `LoadModel` before calling `Forward`.");
  }

  auto output_tensors = model_loader_->get_runner()->run(inputs, stream_handle);

  return output_tensors;
}

bool
ModelState::InferenceModeEnabled() const
{
  return inference_mode_enabled_;
}

std::unordered_map<std::string, uint32_t>&
ModelState::InputMap()
{
  return map_input_index_;
}

bool
ModelState::IsDictionaryInput() const
{
  return is_dictionary_input_;
}

bool
ModelState::IsInputRagged(const std::string& input_name) const
{
  return TritonBackendModel::IsInputRagged(input_name);
}

bool
ModelState::IsInputOptional(const std::string& input_name) const
{
  return TritonBackendModel::IsInputOptional(input_name);
}

void
ModelState::LoadModel(
    const std::string& model_file_name, const torch::Device& device,
    uint32_t device_count, TRITONSERVER_InstanceGroupKind kind)
{
  if (kind != TRITONSERVER_INSTANCEGROUPKIND_MODEL && !device.is_cpu() &&
      device_count == 0)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Argument `device_count` must be greater than zero.");

  DEBUG_TRACE_INFO(
      "{ model_file_name: \""
      << model_file_name << "\""
      << ", device: " << device.str() << ", device_count: " << device_count
      << ", kind: \"" << TRITONSERVER_InstanceGroupKindString(kind) << "\""
      << " }");

  std::string local_file_name{model_file_name};
  std::string local_file_path{};
  if (local_file_name.empty()) {
    local_file_name = INDUCTOR_MODEL_ARTIFACT_NAME_DEFAULT;
  }

  std::string local_name{INDUCTOR_MODEL_NAME_DEFAULT};
  auto repository_path = RepositoryPath();
  auto repository_version = std::to_string(Version());

  local_file_path = triton::backend::JoinPath(
      {repository_path, repository_version, local_file_name});
  DEBUG_TRACE_INFO(
      "{ local_file_name: \""
      << local_file_name << "\""
      << ", local_name: \"" << local_name << "\""
      << ", repository_path: \"" << repository_path << "\""
      << ", repository_version: \"" << repository_version << "\""
      << ", local_file_path: \"" << local_file_path << "\""
      << " }");

  bool exists{false};
  THROW_IF_BACKEND_MODEL_ERROR(FileExists(local_file_path, &exists));

  if (!exists) {
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "PyTorch inductor model file \""
            << local_file_path << "\" is unreachable or inaccessible.");
  }

  std::pair<bool, int> device_pair{false, 0};
  if (weight_sharing_enabled_) {
    device_pair = std::make_pair(!device.is_cpu(), device.index());
    auto mit = model_package_loaders_.find(device_pair);

    if (mit != model_package_loaders_.end()) {
      // Since the model package loader is already created, reuse it.
      model_loader_ = mit->second;
      TRITON_LOG_INFO(
          "Reusing Inductor model loader for instance \"" << Name() << "\".");
      return;
    }
  }

  std::string model_data_string;
  THROW_IF_BACKEND_MODEL_ERROR(
      triton::backend::ReadTextFile(local_file_path, &model_data_string));

  DEBUG_TRACE_INFO(
      "{ local_file_name: \""
      << local_file_name << "\""
      << ", local_name: \"" << local_name << "\""
      << ", repository_path: \"" << repository_path << "\""
      << ", repository_version: \"" << repository_version << "\""
      << ", local_file_path: \"" << local_file_path << "\""
      << ", len(model_data_string): " << model_data_string.size() << " }");

  torch::InferenceMode infer_guard{InferenceModeEnabled()};

  TorchModelLoader* model_loader{nullptr};
  if (device.is_cuda()) {
    DEBUG_TRACE_INFO(
        "Creating model loader for GPU device " << device.str() << ".");
    model_loader = new TorchModelLoader{
        /*model_package_path=*/local_file_path,
        /*model_name=*/local_name,
        /*run_single_threaded=*/false,
        /*num_runners=*/1,
        /*device_index=*/device.index()};
  } else {
    DEBUG_TRACE_INFO(
        "Creating model loader for CPU device " << device.str() << ".");
    model_loader = new TorchModelLoader{
        /*model_package_path=*/local_file_path,
        /*model_name=*/local_name,
        /*run_single_threaded=*/false,
        /*num_runners=*/1,
        /*device_index=*/-1};
  }

  model_loader_.reset(model_loader);

  if (weight_sharing_enabled_) {
    if (!model_package_loaders_.emplace(device_pair, model_loader_).second) {
      TRITON_LOG_WARN(
          "Model already found on target "
          << (device.is_cpu() ? "CPU" : "GPU") << " device " << device.index()
          << " for model \"" << Name() << "\".");
    }
  }

  auto call_spec = model_loader_->get_call_spec();

  pt2::call_spec input_spec;
  if (!call_spec::try_parse(call_spec[0], input_spec))
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to parse input call specification for model \"" << Name()
                                                                << "\".");

  if (input_spec.type() != call_spec_type::builtins_tuple)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Unexpected input call specification type \""
            << input_spec.type() << "\" for model \"" << Name()
            << "\". Expected type is \"builtins_tuple\".");

  pt2::call_spec output_spec;
  if (!call_spec::try_parse(call_spec[1], output_spec))
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to parse output call specification for model \"" << Name()
                                                                 << "\".");

  auto input_children = input_spec.children();
  if (input_children.size() != 2) {
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Unexpected number of children in input call specification for model \""
            << Name()
            << "\". Expected structure to contain two children, but got "
            << input_children.size() << ".");
  }
  if (input_children[0].type() != call_spec_type::builtins_tuple ||
      input_children[1].type() != call_spec_type::builtins_dict) {
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Unexpected input call specification structure for model \""
            << Name() << "\". Expected structure to be tuple["
            << call_spec_type::builtins_tuple << ", "
            << call_spec_type::builtins_dict << "], stead got ["
            << input_children[0].type() << ", " << input_children[1].type()
            << "].");
  }

  auto args_names = input_spec.children()[0].get_names();
  auto kwargs_names = input_spec.children()[1].dictionary_keys();

  const char ARGS_PREFIX[] = "ARGS";
  const char KWARGS_PREFIX[] = "KWARGS";

  std::stringstream logbuf;
  logbuf << "(";

  for (size_t i = 0; i < args_names.size(); i += 1) {
    auto arg_name = ARGS_PREFIX + args_names[i];
    map_input_index_[arg_name] = i;

    logbuf << (i > 0 ? ", " : "") << arg_name;
  }
  for (size_t i = 0; i < kwargs_names.size(); i += 1) {
    auto kwarg_name = KWARGS_PREFIX + kwargs_names[i];
    map_input_index_[kwarg_name] = i + args_names.size();

    logbuf << (i > 0 ? ", " : "") << kwarg_name;
  }
  logbuf << ")";

  const char RESULT_PREFIX[] = "RESULT";

  logbuf << " -> (";

  // When the output call specification looks like `forward(...) ->
  // torch.Tensor` (i.e. there's no structure), assume a single unnamed output
  // and map it to index 0. The default output name of "RESULT" is used to
  // specify a single torch.Tensor as output.
  if (output_spec.children().empty()) {
    // If the output call specification does not contain any names, assume a
    // single unnamed output.
    map_output_index_[RESULT_PREFIX] = 0;

    logbuf << RESULT_PREFIX;
  } else {
    auto result_names = output_spec.get_names();

    for (size_t i = 0; i < result_names.size(); i += 1) {
      auto result_name = RESULT_PREFIX + result_names[i];
      map_output_index_[result_name] = i;

      logbuf << (i > 0 ? ", " : "") << result_name;
    }
  }

  logbuf << ")";

  TRITON_LOG_VERBOSE(
      "Model " << repository_path
               << " loaded with the following interface: " << logbuf.str());
}

int
ModelState::MaxBatchSize() const
{
  return TritonBackendModel::MaxBatchSize();
}

const std::map<std::string, std::pair<int64_t, int64_t>>&
ModelState::ModelOutputs() const
{
  return model_outputs_;
}

const std::string&
ModelState::ModelPath() const
{
  return model_path_;
}

const std::string&
ModelState::Name() const
{
  return TritonBackendModel::Name();
}

bool
ModelState::OptimizedExecutionEnabled() const
{
  return optimized_execution_enabled_;
}

void
ModelState::OptimizedExecutionEnabled(bool value)
{
  optimized_execution_enabled_ = value;
}

std::unordered_map<std::string, uint32_t>&
ModelState::OutputMap()
{
  return map_output_index_;
}

void
ModelState::ParseParameters()
{
  TritonJsonValue parameters;
  if (!ModelConfig().Find("parameters", &parameters)) {
    bool disable_optimized_execution{false};
    if (auto err = ParseParameter(
            parameters, "DISABLE_OPTIMIZED_EXECUTION",
            &disable_optimized_execution)) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        DEBUG_TRACE_ERROR(
            "{ model: \"" << Name() << "\", error: \""
                          << TRITONSERVER_ErrorMessage(err) << "\" }");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ErrorCode(err),
            "Failed to parse 'DISABLE_OPTIMIZED_EXECUTION' parameter for model "
            "\"" << Name()
                 << "\": " << TRITONSERVER_ErrorMessage(err) << ".");
      }
      TRITONSERVER_ErrorDelete(err);
    }

    optimized_execution_enabled_ = !disable_optimized_execution;
    TRITON_LOG_INFO(
        "Optimized execution is "
        << (optimized_execution_enabled_ ? "enabled" : "disabled")
        << " for model instance \"" << Name() << "\".");

    if (auto err = ParseParameter(
            parameters, "CACHE_CLEANING_ENABLED", &cache_cleaning_enabled_)) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        DEBUG_TRACE_ERROR(
            "{ model: \"" << Name() << "\", error: \""
                          << TRITONSERVER_ErrorMessage(err) << "\" }");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ErrorCode(err),
            "Failed to parse 'CACHE_CLEANING_ENABLED' parameter for model \""
                << Name() << "\": " << TRITONSERVER_ErrorMessage(err) << ".");
      }
      TRITONSERVER_ErrorDelete(err);
    }

    TRITON_LOG_INFO(
        "Cache cleaning is "
        << (cache_cleaning_enabled_ ? "enabled" : "disabled")
        << " for model instance \"" << Name() << "\".");

    if (auto err = ParseParameter(
            parameters, "INFERENCE_MODE", &inference_mode_enabled_)) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        DEBUG_TRACE_ERROR(
            "{ model: \"" << Name() << "\", error: \""
                          << TRITONSERVER_ErrorMessage(err) << "\" }");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ErrorCode(err),
            "Failed to parse 'INFERENCE_MODE' parameter for model \""
                << Name() << "\": " << TRITONSERVER_ErrorMessage(err) << ".");
      }
      TRITONSERVER_ErrorDelete(err);
    }

    TRITON_LOG_INFO(
        "Inference mode is "
        << (inference_mode_enabled_ ? "enabled" : "disabled")
        << " for model instance \"" << Name() << "\".");

    if (auto err =
            ParseParameter(parameters, "DISABLE_CUDNN", &cudnn_enabled_)) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        DEBUG_TRACE_ERROR(
            "{ model: \"" << Name() << "\", error: \""
                          << TRITONSERVER_ErrorMessage(err) << "\" }");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ErrorCode(err),
            "Failed to parse 'DISABLE_CUDNN' parameter for model \""
                << Name() << "\": " << TRITONSERVER_ErrorMessage(err) << ".");
      }
      TRITONSERVER_ErrorDelete(err);
    }

    TRITON_LOG_INFO(
        "cuDNN is " << (cudnn_enabled_ ? "enabled" : "disabled")
                    << " for model instance \"" << Name() << "\".");

    if (auto err = ParseParameter(
            parameters, "ENABLE_WEIGHT_SHARING", &weight_sharing_enabled_)) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        DEBUG_TRACE_ERROR(
            "{ model: \"" << Name() << "\""
                          << ", error: \"" << TRITONSERVER_ErrorMessage(err)
                          << "\" }");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ErrorCode(err),
            "Failed to parse 'ENABLE_WEIGHT_SHARING' parameter for model \""
                << Name() << "\": " << TRITONSERVER_ErrorMessage(err) << ".");
      }
      TRITONSERVER_ErrorDelete(err);
    }

    TRITON_LOG_INFO(
        "Weight sharing is "
        << (weight_sharing_enabled_ ? "enabled" : "disabled")
        << " for model instance \"" << Name() << "\".");

    int intra_op_thread_count{-1};
    if (auto err = ParseParameter(
            parameters, "INTRA_OP_THREAD_COUNT", &intra_op_thread_count)) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        DEBUG_TRACE_ERROR(
            "{ model: \"" << Name() << "\""
                          << ", error: \"" << TRITONSERVER_ErrorMessage(err)
                          << "\" }");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ErrorCode(err),
            "Failed to parse 'INTRA_OP_THREAD_COUNT' parameter for model \""
                << Name() << "\": " << TRITONSERVER_ErrorMessage(err) << ".");
      }
      TRITONSERVER_ErrorDelete(err);
    }

    if (intra_op_thread_count > 0) {
      std::call_once(pytorch_intraop_threads_flag, [intra_op_thread_count]() {
        at::set_num_threads(intra_op_thread_count);
      });

      TRITON_LOG_INFO(
          "Intra op thread count is set to " << intra_op_thread_count
                                             << " for model instance "
                                             << "\"" << Name() << "\".");
    }

    int inter_op_thread_count{-1};
    if (auto err = ParseParameter(
            parameters, "INTER_OP_THREAD_COUNT", &inter_op_thread_count)) {
      if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND) {
        DEBUG_TRACE_ERROR(
            "{ model: \"" << Name() << "\", error: \""
                          << TRITONSERVER_ErrorMessage(err) << "\" }");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ErrorCode(err),
            "Failed to parse 'INTER_OP_THREAD_COUNT' parameter for model \""
                << Name() << "\": " << TRITONSERVER_ErrorMessage(err) << ".");
      }
      TRITONSERVER_ErrorDelete(err);
    }
    DEBUG_TRACE_INFO(
        "{ disable_optimized_execution: "
        << (disable_optimized_execution ? "true" : "false")
        << ", cache_cleaning_enabled_: "
        << (cache_cleaning_enabled_ ? "true" : "false")
        << ", inference_mode_enabled_: "
        << (inference_mode_enabled_ ? "true" : "false") << ", cudnn_enabled_: "
        << (cudnn_enabled_ ? "true" : "false") << ", weight_sharing_enabled_: "
        << (weight_sharing_enabled_ ? "true" : "false")
        << ", intra_op_thread_count: " << intra_op_thread_count
        << ", inter_op_thread_count: " << inter_op_thread_count << " }");

    if (inter_op_thread_count > 0) {
      std::call_once(pytorch_interop_threads_flag, [inter_op_thread_count]() {
        at::set_num_interop_threads(inter_op_thread_count);
      });
      TRITON_LOG_INFO(
          "Inter op thread count is set to " << inter_op_thread_count
                                             << " for model instance "
                                             << "\"" << Name() << "\".");
    }
  }
}

const std::string&
ModelState::RepositoryPath() const
{
  return TritonBackendModel::RepositoryPath();
}

void
ModelState::SetMaxBatchSize(int value)
{
  TritonBackendModel::SetMaxBatchSize(value);
}

TRITONSERVER_Error*
ModelState::SupportsFirstDimBatching(bool* value_out)
{
  return TritonBackendModel::SupportsFirstDimBatching(value_out);
}

TRITONBACKEND_MemoryManager*
ModelState::TritonMemoryManager()
{
  return TritonBackendModel::TritonMemoryManager();
}

TRITONBACKEND_Model*
ModelState::TritonModel()
{
  return TritonBackendModel::TritonModel();
}

TRITONSERVER_Server*
ModelState::TritonServer()
{
  return TritonBackendModel::TritonServer();
}

uint64_t
ModelState::Version() const
{
  return TritonBackendModel::Version();
}

bool
ModelState::WeightSharingEnabled() const
{
  return weight_sharing_enabled_;
}
}  // namespace triton::backend::pytorch::pt2
