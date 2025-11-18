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

#include "model_state.hh"

#include <mutex>

namespace
{
  std::once_flag pytorch_interop_threads_flag;
  std::once_flag pytorch_intraop_threads_flag;
} // namespace

namespace triton::backend::pytorch
{
  ModelState::ModelState(
      TRITONBACKEND_Model* triton_model)
      : BackendModel(triton_model)
      , enable_optimized_execution_(true)
      , enable_inference_mode_(true)
      , enable_cudnn_(true)
      , enable_cache_cleaning_(false)
      , enable_weight_sharing_(false)
      , enable_tensor_fuser_pair_({false, true})
      , enable_jit_profiling_pair_({false, true})
      , enable_jit_executor_pair_({false, true})
  {
  }

  TRITONSERVER_Error*
  ModelState::AutoCompleteConfig()
  {
    // Auto-complete configuration is not supported since PyTorch does not store/capture sufficient model metadata
    // so just log error instead.
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("skipping model configuration auto-complete for '") + Name()
         + "': not supported for pytorch backend")
            .c_str());

    return nullptr; // success
  }

  TRITONSERVER_Error*
  ModelState::Create(
      TRITONBACKEND_Model* triton_model, ModelState** state)
  {
    try
    {
      *state = new ModelState(triton_model);
    }
    catch (const BackendModelException& ex)
    {
      RETURN_ERROR_IF_TRUE(
          ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL, std::string("unexpected nullptr in BackendModelException"));
      RETURN_IF_ERROR(ex.err_);
    }

    // Auto-complete the configuration if requested...
    bool auto_complete_config = false;
    RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(triton_model, &auto_complete_config));
    if (auto_complete_config)
    {
      RETURN_IF_ERROR((*state)->AutoCompleteConfig());
      RETURN_IF_ERROR((*state)->SetModelConfig());
    }

    auto& model_outputs = (*state)->model_outputs_;
    // Parse the output states in the model configuration
    triton::common::TritonJson::Value sequence_batching;
    if ((*state)->ModelConfig().Find("sequence_batching", &sequence_batching))
    {
      triton::common::TritonJson::Value states;
      if (sequence_batching.Find("state", &states))
      {
        for (size_t i = 0; i < states.ArraySize(); i++)
        {
          triton::common::TritonJson::Value state;
          RETURN_IF_ERROR(states.IndexAsObject(i, &state));
          std::string output_state_name;
          RETURN_IF_ERROR(state.MemberAsString("output_name", &output_state_name));
          auto it = model_outputs.find(output_state_name);
          if (it == model_outputs.end())
          {
            model_outputs.insert({output_state_name, std::make_pair(-1, i)});
          }
          else
          {
            it->second.second = i;
          }
        }
      }
    }

    // Parse the output names in the model configuration
    triton::common::TritonJson::Value outputs;
    RETURN_IF_ERROR((*state)->ModelConfig().MemberAsArray("output", &outputs));
    for (size_t i = 0; i < outputs.ArraySize(); i++)
    {
      triton::common::TritonJson::Value output;
      THROW_IF_BACKEND_INSTANCE_ERROR(outputs.IndexAsObject(i, &output));

      // Use names from ModelConfig by reference since the model config will persist longer than this inference execution.
      std::string output_name;
      THROW_IF_BACKEND_INSTANCE_ERROR(output.MemberAsString("name", &output_name));

      auto it = model_outputs.find(output_name);
      if (it == model_outputs.end())
      {
        model_outputs.insert({output_name, std::make_pair(i, -1)});
      }
      else
      {
        it->second.first = i;
      }
    }

    RETURN_IF_ERROR((*state)->ParseParameters());

    return nullptr; // success
  }

  bool
  ModelState::EnabledCacheCleaning()
  {
    return enable_cache_cleaning_;
  }

  bool
  ModelState::EnabledCudnn()
  {
    return enable_cudnn_;
  }

  bool
  ModelState::EnabledInferenceMode()
  {
    return enable_inference_mode_;
  }

  const std::pair<bool, bool>&
  ModelState::EnabledJitExecutor() const
  {
    return enable_jit_executor_pair_;
  }

  const std::pair<bool, bool>&
  ModelState::EnabledJitProfiling() const
  {
    return enable_jit_profiling_pair_;
  }

  bool
  ModelState::EnabledOptimizedExecution()
  {
    return enable_optimized_execution_;
  }

  const std::pair<bool, bool>&
  ModelState::EnabledTensorExprFuser() const
  {
    return enable_tensor_fuser_pair_;
  }

  bool
  ModelState::EnabledWeightSharing()
  {
    return enable_weight_sharing_;
  }

  TRITONSERVER_Error*
  ModelState::LoadModel(
      const std::string& artifact_name,
      const torch::Device device,
      std::string* model_path,
      const TRITONSERVER_InstanceGroupKind& kind,
      std::shared_ptr<torch::jit::script::Module>* torch_model)
  {
    // Find the TorchScript file that describes the model.
    // If the model configuration doesn't have an explicit model file specified then use the default name ("model.pt").
    std::string cc_model_filename = artifact_name;
    if (cc_model_filename.empty())
    {
      cc_model_filename = "model.pt";
    }

    *model_path = JoinPath({RepositoryPath(), std::to_string(Version()), cc_model_filename});

    {
      bool exists;
      RETURN_IF_ERROR(FileExists(*model_path, &exists));
      RETURN_ERROR_IF_FALSE(
          exists,
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("unable to find '") + *model_path + "' for model instance '" + Name() + "'");
    }

    // If weight sharing is enabled, skip loading model if  it is already available on the target device
    std::pair<bool, int> device_pair;
    if (enable_weight_sharing_)
    {
      device_pair = std::make_pair(!device.is_cpu(), device.index());
      auto mit = torch_models_.find(device_pair);
      if (mit != torch_models_.end())
      {
        *torch_model = mit->second;
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO, (std::string("Reusing TorchScript model for instance '") + Name() + "'").c_str());
        return nullptr; // success
      }
    }

    // Serialize the torch model to string
    std::string model_data_str;
    RETURN_IF_ERROR(ReadTextFile(*model_path, &model_data_str));

    // InferenceMode should be used to guard all tensors operations including model loading: https://pytorch.org/cppdocs/notes/inference_mode.html
    torch::InferenceMode infer_guard(EnabledInferenceMode());

    try
    {
      std::istringstream model_stream(model_data_str);
      if (kind == TRITONSERVER_INSTANCEGROUPKIND_MODEL)
      {
        // Load the model without selecting a device.
        torch_model->reset(new torch::jit::Module(torch::jit::load(model_stream)));
      }
      else
      {
        torch_model->reset(new torch::jit::Module(torch::jit::load(model_stream, device)));
      }
    }
    catch (const std::exception& ex)
    {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, ("failed to load model '" + Name() + "': " + ex.what()).c_str());
    }

    if (enable_weight_sharing_)
    {
      if (!((torch_models_.emplace(device_pair, *torch_model)).second))
      {
        std::string type = device.is_cpu() ? "CPU" : "GPU";
        LOG_MESSAGE(
            TRITONSERVER_LOG_WARN,
            (std::string("Model already found on target ") + type + " device " + "(id " + std::to_string(device.index())
             + ") for '" + Name() + "'")
                .c_str());
      }
    }

    return nullptr; // success
  }

  const std::map<std::string, std::pair<int64_t, int64_t>>&
  ModelState::ModelOutputs()
  {
    return model_outputs_;
  }

  TRITONSERVER_Error*
  ModelState::ParseParameters()
  {
    triton::common::TritonJson::Value params;
    bool status = model_config_.Find("parameters", &params);
    if (status)
    {
      // If 'DISABLE_OPTIMIZED_EXECUTION' is not present in 'parameters' then no update is made to 'enable_optimized_execution_'.
      bool disable_optimized_execution = false;
      TRITONSERVER_Error* err = ParseParameter(params, "DISABLE_OPTIMIZED_EXECUTION", &disable_optimized_execution);
      if (err != nullptr)
      {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
        {
          return err;
        }
        else
        {
          TRITONSERVER_ErrorDelete(err);
        }
      }
      enable_optimized_execution_ = !disable_optimized_execution;

      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Optimized execution is ") + (enable_optimized_execution_ ? "enabled" : "disabled")
           + " for model instance '" + Name() + "'")
              .c_str());

      // If 'ENABLE_CACHE_CLEANING' is not present in 'parameters' then no update is made to 'enable_cache_cleaning_'.
      err = ParseParameter(params, "ENABLE_CACHE_CLEANING", &enable_cache_cleaning_);
      if (err != nullptr)
      {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
        {
          return err;
        }
        else
        {
          TRITONSERVER_ErrorDelete(err);
        }
      }

      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Cache Cleaning is ") + (enable_cache_cleaning_ ? "enabled" : "disabled")
           + " for model instance '" + Name() + "'")
              .c_str());

      // If 'INFERENCE_MODE' is not present in 'parameters' then no update is made to 'enable_inference_mode_'.
      err = ParseParameter(params, "INFERENCE_MODE", &enable_inference_mode_);
      if (err != nullptr)
      {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
        {
          return err;
        }
        else
        {
          TRITONSERVER_ErrorDelete(err);
        }
      }
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Inference Mode is ") + (enable_inference_mode_ ? "enabled" : "disabled")
           + " for model instance '" + Name() + "'")
              .c_str());

      // If 'DISABLE_CUDNN' is not present in 'parameters' then no update is made to 'enable_cudnn_'.
      bool disable_cudnn = false;
      err = ParseParameter(params, "DISABLE_CUDNN", &disable_cudnn);
      if (err != nullptr)
      {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
        {
          return err;
        }
        else
        {
          TRITONSERVER_ErrorDelete(err);
        }
      }
      enable_cudnn_ = !disable_cudnn;
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("cuDNN is ") + (enable_cudnn_ ? "enabled" : "disabled") + " for model instance '" + Name() + "'")
              .c_str());

      // If 'ENABLE_TENSOR_FUSER' is not present in 'parameters' then no update is made to 'enable_tensor_fuser'.
      bool enable_tensor_fuser = false;
      err = ParseParameter(params, "ENABLE_TENSOR_FUSER", &enable_tensor_fuser);
      if (err != nullptr)
      {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
        {
          return err;
        }
        else
        {
          TRITONSERVER_ErrorDelete(err);
        }
      }
      else
      {
        enable_tensor_fuser_pair_ = {true, enable_tensor_fuser};
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Tensor fuser is ") + (enable_tensor_fuser ? "enabled" : "disabled") + " for model instance '"
             + Name() + "'")
                .c_str());
      }

      // If 'ENABLE_WEIGHT_SHARING' is not present in 'parameters' then no update is made to 'enable_weight_sharing'.
      err = ParseParameter(params, "ENABLE_WEIGHT_SHARING", &enable_weight_sharing_);
      if (err != nullptr)
      {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
        {
          return err;
        }
        else
        {
          TRITONSERVER_ErrorDelete(err);
        }
      }
      else
      {
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Weight sharing is ") + (enable_weight_sharing_ ? "enabled" : "disabled")
             + " for model instance '" + Name() + "'")
                .c_str());
      }

      // If 'ENABLE_JIT_PROFILING' is not present in 'parameters' then no update is made to 'enable_jit_profiling'.
      bool enable_jit_profiling = false;
      err = ParseParameter(params, "ENABLE_JIT_PROFILING", &enable_jit_profiling);
      if (err != nullptr)
      {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
        {
          return err;
        }
        else
        {
          TRITONSERVER_ErrorDelete(err);
        }
      }
      else
      {
        enable_jit_profiling_pair_ = {true, enable_jit_profiling};
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Jit profiling is ") + (enable_jit_profiling ? "enabled" : "disabled")
             + " for model instance '" + Name() + "'")
                .c_str());
      }

      // If 'ENABLE_JIT_EXECUTOR' is not present in 'parameters' then no update is made to 'enable_jit_executor'.
      bool enable_jit_executor = false;
      err = ParseParameter(params, "ENABLE_JIT_EXECUTOR", &enable_jit_executor);
      if (err != nullptr)
      {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
        {
          return err;
        }
        else
        {
          TRITONSERVER_ErrorDelete(err);
        }
      }
      else
      {
        enable_jit_executor_pair_ = {true, enable_jit_executor};
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Jit executor is ") + (enable_jit_executor ? "enabled" : "disabled") + " for model instance '"
             + Name() + "'")
                .c_str());
      }

      // If 'INTRA_OP_THREAD_COUNT' is not present in 'parameters' then no update is made to 'intra_op_thread_count',
      // which by default will take all threads
      int intra_op_thread_count = -1;
      err = ParseParameter(params, "INTRA_OP_THREAD_COUNT", &intra_op_thread_count);
      if (err != nullptr)
      {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
        {
          return err;
        }
        else
        {
          TRITONSERVER_ErrorDelete(err);
        }
      }
      else
      {
        if (intra_op_thread_count > 0)
        {
          // at::set_num_threads() does not throw if called more than once, but issues warnings. std::call_once() is useful to limit these.
          std::call_once(
              pytorch_intraop_threads_flag,
              [intra_op_thread_count]()
              {
                at::set_num_threads(intra_op_thread_count);
              });
          LOG_MESSAGE(
              TRITONSERVER_LOG_INFO,
              (std::string("Intra op thread count is set to ") + std::to_string(at::get_num_threads())
               + " for model instance '" + Name() + "'")
                  .c_str());
        }
      }

      // If 'INTER_OP_THREAD_COUNT' is not present in 'parameters' then no update  is made to 'inter_op_thread_count',
      // which by default will take all threads
      int inter_op_thread_count = -1;
      err = ParseParameter(params, "INTER_OP_THREAD_COUNT", &inter_op_thread_count);
      if (err != nullptr)
      {
        if (TRITONSERVER_ErrorCode(err) != TRITONSERVER_ERROR_NOT_FOUND)
        {
          return err;
        }
        else
        {
          TRITONSERVER_ErrorDelete(err);
        }
      }
      else
      {
        if (inter_op_thread_count > 0)
        {
          // at::set_num_interop_threads() throws if called more than once.
          // std::call_once() should prevent this, but try/catch is additionally used for safety.
          std::call_once(
              pytorch_interop_threads_flag,
              [inter_op_thread_count]()
              {
                try
                {
                  at::set_num_interop_threads(inter_op_thread_count);
                }
                catch (const c10::Error& e)
                {
                  // do nothing
                }
              });
          LOG_MESSAGE(
              TRITONSERVER_LOG_INFO,
              (std::string("Inter op thread count is set to ") + std::to_string(at::get_num_interop_threads())
               + " for model instance '" + Name() + "'")
                  .c_str());
        }
      }
    }

    return nullptr;
  }
}
