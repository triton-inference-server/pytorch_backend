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

#include "triton_utils.hh"

namespace triton::backend::pytorch
{
  TRITONSERVER_Error_Code
  __to_error_code__(
    TRITONSERVER_Error_Code value)
  {
    return value;
  }

  TRITONSERVER_Error_Code
  __to_error_code__(
    TRITONSERVER_Error* error)
  {
    return (error) ? TRITONSERVER_ErrorCode(error) : TRITONSERVER_ERROR_UNKNOWN;
  }

  TRITONSERVER_Error_Code
  __to_error_code__(
    uint64_t value)
  {
    switch (value)
    {
      case 0: return TRITONSERVER_ERROR_UNKNOWN;
      case 1: return TRITONSERVER_ERROR_INTERNAL;
      case 2: return TRITONSERVER_ERROR_NOT_FOUND;
      case 3: return TRITONSERVER_ERROR_INVALID_ARG;
      case 4: return TRITONSERVER_ERROR_UNAVAILABLE;
      case 5: return TRITONSERVER_ERROR_UNSUPPORTED;
      case 6: return TRITONSERVER_ERROR_ALREADY_EXISTS;
      case 7: return TRITONSERVER_ERROR_CANCELLED;
      default: return TRITONSERVER_ERROR_UNKNOWN;
    }
  }

  TRITONSERVER_Error_Code
  __to_error_code__(
    triton::common::Error::Code value)
  {
    return StatusCodeToTritonCode(value);
  }
}
