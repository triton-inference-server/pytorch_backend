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

#pragma once

#include <stdint.h>

#include <cstdint>
#include <exception>
#include <mutex>

#include "libtorch_utils.h"
#include "naming_convention.hh"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/common/nvtx.h"
#include "triton/core/tritonbackend.h"

// for thread control
// https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#runtime-api
// https://github.com/pytorch/pytorch/blob/v2.2.1-rc3/aten/src/ATen/Parallel.h#L133
#include <ATen/Parallel.h>


namespace triton::backend::pytorch
{

  class ModelState : public triton::backend::BackendModel
  {
  private:
    // Flag to indicate whether optimized execution is enabled. Defaults to true.
    bool enable_optimized_execution_;

    // Flag to indicate whether inference mode is enabled. Defaults to false.
    bool enable_inference_mode_;

    // Flag to indicate whether cudnn is enabled. Defaults to true.
    bool enable_cudnn_;

    // Flag to indicate whether cache cleaning after each run is enabled.
    // Defaults to false.
    bool enable_cache_cleaning_;

    // Flag to indicate whether weight sharing is enabled. Defaults to false.
    bool enable_weight_sharing_;

    // Flag pairs to indicate if various JIT settings are set and
    // enabled respectively. Defaults to (false, true). Default behavior
    // is to do nothing if not explicitly set.
    std::pair<bool, bool> enable_tensor_fuser_pair_;
    std::pair<bool, bool> enable_jit_profiling_pair_;
    std::pair<bool, bool> enable_jit_executor_pair_;

    // Model mapping for shared TorchScript model across all instances on the
    // same device. The key is a pair of isGPU and device index.
    std::map<std::pair<bool, int64_t>, std::shared_ptr<torch::jit::script::Module>> torch_models_;

    // model_outputs is a map that contains unique outputs that the model must
    // provide. The first pair is the model output index and the second is
    // the index in the model state, -1 is used if one is not required.
    // In the model configuration, the output in the state configuration
    // can have intersection with the outputs section of the model. If an output
    // is specified both in the output section and state section, it indicates
    // that the backend must return the output state to the client too.
    std::map<std::string, std::pair<int64_t, int64_t>> model_outputs_;

  public:
    virtual ~ModelState() = default;

    static TRITONSERVER_Error*
    Create(TRITONBACKEND_Model* triton_model, ModelState** state);

    bool
    EnabledCacheCleaning();

    bool
    EnabledCudnn();

    bool
    EnabledInferenceMode();

    const std::pair<bool, bool>&
    EnabledJitExecutor() const;

    const std::pair<bool, bool>&
    EnabledJitProfiling() const;

    bool
    EnabledOptimizedExecution();

    const std::pair<bool, bool>&
    EnabledTensorExprFuser() const;

    bool
    EnabledWeightSharing();

    TRITONSERVER_Error*
    LoadModel(
        const std::string& artifact_name,
        const torch::Device device,
        std::string* model_path,
        const TRITONSERVER_InstanceGroupKind& kind,
        std::shared_ptr<torch::jit::script::Module>* torch_model);

    const std::map<std::string, std::pair<int64_t, int64_t>>&
    ModelOutputs();

  private:
    ModelState(TRITONBACKEND_Model* triton_model);

    TRITONSERVER_Error*
    AutoCompleteConfig();

    TRITONSERVER_Error*
    ParseParameters();
  };

} // namespace triton::backend::pytorch
