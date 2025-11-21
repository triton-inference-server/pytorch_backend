// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/core/tritonserver.h"

#include <exception>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma warning(push, 0)
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/script.h>
#pragma warning(pop)
#pragma GCC diagnostic pop

// For thread control
// https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#runtime-api
// https://github.com/pytorch/pytorch/blob/v2.2.1-rc3/aten/src/ATen/Parallel.h#L133
#include <ATen/Parallel.h>

namespace triton::backend::pytorch
{
  constexpr char INDUCTOR_MODEL_ARTIFACT_NAME_DEFAULT[] = "model.pt2";
  constexpr char INDUCTOR_MODEL_NAME_DEFAULT[] = "model";

  using inductor_loader = torch::inductor::AOTIModelPackageLoader;

  class InductorModel
    : public triton::backend::BackendModel
  {
    private:

      torch::Device device_{torch::kCPU};
      int32_t device_count_{0};
      bool cache_cleaning_enabled_{false};
      bool cudnn_enabled_{true};
      bool inference_mode_enabled_{true};
      bool is_dictionary_input_{false};
      std::shared_ptr<inductor_loader> model_loader_{nullptr};
      std::string model_name_;
      std::map<std::string, std::pair<int64_t, int64_t>> model_outputs_;
      std::string model_path_;
      std::map<std::pair<bool, int64_t>, std::shared_ptr<inductor_loader>> model_package_loaders_;
      bool optimized_execution_enabled_{true};
      bool weight_sharing_enabled_{false};

    public:

      InductorModel() = delete;

      virtual ~InductorModel() = default;

      [[nodiscard]]
      bool
      CacheCleaningEnabled() const;

      void
      CacheCleaningEnabled(
        bool value);

      [[nodiscard]]
      std::shared_ptr<InductorModel>
      Create(
        TRITONBACKEND_Model* triton_model);

      [[nodiscard]]
      bool
      CudnnEnabled() const;

      void
      CudnnEnabled(
        bool value);

      [[nodiscard]]
      std::vector<torch::Tensor>
      Forward(
        const std::vector<torch::Tensor>& inputs,
        void* stream_handle = nullptr);

      [[nodiscard]]
      std::vector<std::string>
      GetModelCallSpec();

      [[nodiscard]]
      bool
      InferenceModeEnabled() const;

      [[nodiscard]]
      bool
      IsDictionaryInput() const;

      void
      LoadModel(
        const std::string& model_file_name,
        const torch::Device& device,
        uint32_t device_count,
        TRITONSERVER_InstanceGroupKind kind);

      [[nodiscard]]
      const std::map<std::string, std::pair<int64_t, int64_t>>&
      ModelOutputs() const;

      [[nodiscard]]
      bool
      OptimizedExecutionEnabled() const;

      void
      OptimizedExecutionEnabled(
        bool value);

      [[nodiscard]]
      bool
      WeightSharingEnabled() const;

      /** triton::backend::BackendModel implementation **/

      [[nodiscard]]
      const std::vector<BatchInput>&
      BatchInputs() const;

      [[nodiscard]]
      const std::vector<BatchOutput>&
      BatchOutputs() const;

      [[nodiscard]]
      bool
      EnablePinnedInput() const;

      [[nodiscard]]
      bool
      EnablePinnedOutput() const;

      [[nodiscard]]
      const BatchOutput*
      FindBatchOutput(
        const std::string& output_name) const;

      [[nodiscard]]
      bool
      IsInputRagged(
        const std::string& input_name) const;

      [[nodiscard]]
      bool
      IsInputOptional(
        const std::string& input_name) const;

      [[nodiscard]]
      int
      MaxBatchSize() const;

      [[nodiscard]]
      const std::string&
      ModelPath() const;

      [[nodiscard]]
      const std::string&
      Name() const;

      [[nodiscard]]
      const std::string&
      RepositoryPath() const;

      void
      SetMaxBatchSize(
        int value);

      [[nodiscard]]
      TRITONSERVER_Error*
      SupportsFirstDimBatching(
        bool* value_out);

      [[nodiscard]]
      TRITONBACKEND_MemoryManager*
      TritonMemoryManager();

      [[nodiscard]]
      TRITONBACKEND_Model*
      TritonModel();

      [[nodiscard]]
      TRITONSERVER_Server*
      TritonServer();

      [[nodiscard]]
      uint64_t
       Version() const;

    private:

      InductorModel(
        TRITONBACKEND_Model* backend_model,
        bool allow_optional = false);

      void
      AutoCompleteConfig();

      void
      ParseParameters();
  };
}
