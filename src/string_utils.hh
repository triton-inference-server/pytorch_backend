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

// for thread control
// https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#runtime-api
// https://github.com/pytorch/pytorch/blob/v2.2.1-rc3/aten/src/ATen/Parallel.h#L133
#include <ATen/Parallel.h>


namespace triton::backend::pytorch {

void FillStringTensor(torch::List<std::string>* input_list, const size_t cnt);

// This function will return a tensor's contents as a contiguous
// chunk in system memory. In some cases this will require copying the data.
// If that  happens, 'contiguous_buffer' will be set to hold the contiguous
// chunk and 'cuda_copy' will be set to indicate whether CUDA copy is
// conducted.  The data copy can be avoided if the input is already in
// a contiguous chunk and the input is located in memory type and id
// specified.
TRITONSERVER_Error* GetContiguousInputContent(
    TRITONBACKEND_Input* rinput, const uint32_t buffer_count,
    const char** content, size_t* content_byte_size,
    std::vector<char>* contiguous_buffer, cudaStream_t stream, bool* cuda_copy);

bool SetStringBuffer(
    torch::List<torch::jit::IValue>* tensor, TRITONBACKEND_Response** response,
    TRITONBACKEND_Output* response_output, TRITONBACKEND_State* response_state,
    const size_t tensor_element_count, cudaStream_t stream,
    std::string* serialized, bool state);

    torch::List<std::string>* input_list, TRITONBACKEND_Input* input,
    const char* name, const uint32_t buffer_count,
    const size_t request_element_cnt, TRITONBACKEND_Response** response,
    cudaStream_t stream, const char* host_policy_name);

bool SetStringOutputBuffer(
    torch::List<torch::jit::IValue>* tensor, TRITONBACKEND_Response** response,
    TRITONBACKEND_Output* response_output, const size_t tensor_element_count,
    cudaStream_t stream, std::string* serialized);

bool SetStringStateBuffer(
    torch::List<torch::jit::IValue>* tensor, TRITONBACKEND_Response** response,
    TRITONBACKEND_State* response_state, const size_t tensor_element_count,
    cudaStream_t stream, std::string* serialized);

}  // namespace triton::backend::pytorch
