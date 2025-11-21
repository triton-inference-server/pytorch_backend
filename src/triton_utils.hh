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
#include "triton/core/tritonserver.h"

#include <exception>
#include <sstream>

namespace triton::backend::pytorch
{
  class BackendException
    : std::runtime_error
  {
    private:

      TRITONSERVER_Error_Code error_code_;

    public:

      BackendException(
          TRITONSERVER_Error_Code error_code,
          const std::string &message)
        : runtime_error{message}
        , error_code_{error_code}
      { }

      BackendException() = delete;

      TRITONSERVER_Error_Code
      error_code() const noexcept
      {
        return error_code_;
      }

      virtual const char*
      what() const noexcept override
      {
        return runtime_error::what();
      }
  };

  TRITONSERVER_Error_Code
  __to_error_code__(
    TRITONSERVER_Error_Code value);

  TRITONSERVER_Error_Code
  __to_error_code__(
    TRITONSERVER_Error* error);

  TRITONSERVER_Error_Code
  __to_error_code__(
    uint64_t value);

  TRITONSERVER_Error_Code
  __to_error_code__(
    triton::common::Error::Code value);

#define THROW_TRITON_EXCEPTION(error_code, message) \
  do { \
    TRITONSERVER_Error_Code ec = triton::backend::pytorch::__to_error_code__(error_code); \
    std::stringstream buf; \
    buf << message; \
    throw triton::backend::pytorch::BackendException{ec, buf.str()}; \
  } while (false);

#define TRITON_LOG_ERROR(message) \
  do { \
    std::stringstream buf; \
    buf << message; \
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, buf.str().c_str()); \
  } while (false);

#define TRITON_LOG_INFO(message) \
  do { \
    std::stringstream buf; \
    buf << message; \
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, buf.str().c_str()); \
  } while (false);

#define TRITON_LOG_VERBOSE(message) \
  do { \
    std::stringstream buf; \
    buf << message; \
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, buf.str().c_str()); \
  } while (false);

#define TRITON_LOG_WARN(message) \
  do { \
    std::stringstream buf; \
    buf << message; \
    LOG_MESSAGE(TRITONSERVER_LOG_WARN, buf.str().c_str()); \
  } while (false);
}
