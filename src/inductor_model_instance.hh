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

#include "inductor_model.hh"
#include "naming_convention.hh"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/common/triton_json.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#include <memory>
#include <string>
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma warning(push, 0)
#include <torch/script.h>
#pragma warning(pop)
#pragma GCC diagnostic pop

namespace triton::backend::pytorch
{
  using TritonInductorModel = triton::backend::pytorch::InductorModel;
  using TritonJsonValue = triton::common::TritonJson::Value;
  using TritonNamingConvention = triton::backend::pytorch::NamingConvention;

  class InductorModelInstance
    : public triton::backend::BackendModelInstance
  {
    private:

      uint32_t batch_input_count_{0};
#ifdef TRITON_ENABLE_GPU
      triton::backend::cudaEvent_t compute_infer_start_event_;
      triton::backend::cudaEvent_t compute_input_start_event_;
      triton::backend::cudaEvent_t compute_output_start_event_;
#endif
      torch::Device device_{torch::kCPU};
      int device_count_{0};
      std::unordered_map<std::string, int> input_index_map_;
      bool is_batching_supported_{false};
      bool is_dictionary_input_;
      std::shared_ptr<TritonInductorModel> model_{nullptr};
      std::string model_path_;
      std::unordered_map<std::string, TRITONSERVER_DataType> output_dtype_map_;
      std::unordered_map<std::string, int> output_index_map_;
#ifdef TRITON_ENABLE_GPU
      std::vector<triton::backend::cudaStream_t> stream_vector_;
#endif

    public:

      InductorModelInstance(
        std::shared_ptr<TritonInductorModel> model,
        TRITONBACKEND_ModelInstance* triton_model_instance);

      InductorModelInstance() = delete;

      ~InductorModelInstance() override;

      void
      ClearCache();

      [[nodiscard]]
      static std::shared_ptr<InductorModelInstance>
      Create(
        std::shared_ptr<TritonInductorModel> model,
        TRITONBACKEND_ModelInstance* triton_model_instance);

      void
      ProcessRequests(
        TRITONBACKEND_Request** requests,
        const uint32_t request_count);

      [[nodiscard]]
      std::shared_ptr<TritonInductorModel>
      InductorModel() const;

      /** triton::backend::BackendModelInstance implementation **/

      [[nodiscard]]
      const std::string&
      ArtifactFilename() const;

      [[nodiscard]]
      triton::backend::cudaStream_t
      CudaStream();

      [[nodiscard]]
      int32_t
      DeviceId() const;

      [[nodiscard]]
      const std::string&
      HostPolicyName() const;

      [[nodiscard]]
      TRITONSERVER_InstanceGroupKind
      Kind() const;

      [[nodiscard]]
      triton::backend::BackendModel*
      Model() const;

      [[nodiscard]]
      const std::string&
      Name() const;

      [[nodiscard]]
      TRITONBACKEND_ModelInstance*
      TritonModelInstance();

    private:

      void
      AddInputToMap(
        TritonNamingConvention naming_convention,
        const std::vector<std::string>& allowed_inputs,
        const std::string& io_name,
        uint32_t index);

      void
      CreateCudaEvents(
        int32_t device_id);

      void
      Execute(
        std::vector<TRITONBACKEND_Response*>* responses,
        uint32_t response_count,
        std::vector<torch::Tensor>& input_tensors,
        std::vector<torch::Tensor>& output_tensors);

      float
      GetCudaEventElapsedTime(
        const triton::backend::cudaEvent_t& start_event,
        const triton::backend::cudaEvent_t& end_event);

      [[nodiscard]]
      triton::backend::cudaStream_t
      GetCudaStreamByInstanceKind();

      [[nodiscard]]
      TritonNamingConvention
      GetNamingConvention(
        const std::vector<std::string>& allowed_ios);

      void
      ReadOutputTensors(
        size_t total_batch_size,
        const std::vector<torch::Tensor>& output_tensors,
        TRITONBACKEND_Request** requests,
        uint32_t request_count,
        std::vector<TRITONBACKEND_Response*>& responses);

      [[nodiscard]]
      TRITONSERVER_Error*
      RecordBackendTimestamp(
        uint64_t* timestamp,
        void* cuda_event);

      void
      SetCurrentCudaStream(
        const triton::backend::cudaStream_t& stream,
        const int& device_id);

      TRITONSERVER_Error*
      SetInputTensors(
        size_t total_batch_size,
        TRITONBACKEND_Request** requests,
        uint32_t request_count,
        std::vector<TRITONBACKEND_Response*>* responses,
        triton::backend::BackendInputCollector* collector,
        std::vector<const char*>* input_names,
        std::vector<torch::Tensor>* input_tensors,
        bool* cuda_copy);

      [[nodiscard]]
      bool
      ValidateBooleanSequenceControl(
        TritonJsonValue& sequence_batching,
        const std::string& control_kind,
        bool required);

      void
      ValidateInputs(
        const size_t expected_input_count);

      void
      ValidateOutputs();

      [[nodiscard]]
      bool
      ValidateTypedSequenceControl(
        TritonJsonValue& sequence_batching,
        const std::string& control_kind,
        bool required);
  };
}
