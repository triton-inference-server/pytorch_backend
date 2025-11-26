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

#include "model_instance_state.hh"
#include "model_state.hh"
#include "naming_convention.hh"
#include "string_utils.hh"

//
// PyTorch C++ (LibTorch) Backend that implements the TRITONBACKEND API.
//

namespace triton::backend::pytorch {

extern "C" {

TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend);

TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model);

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model);

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance);

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance);

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count);

}  // extern "C"


}  // namespace triton::backend::pytorch

#ifndef ENABLE_DEBUG_TRACE_ERROR
#define ENABLE_DEBUG_TRACE_ERROR 0
#endif

#ifndef ENABLE_DEBUG_TRACE_FUNCTION_CALL
#define ENABLE_DEBUG_TRACE_FUNCTION_CALL 0
#endif

#ifndef ENABLE_DEBUG_TRACE_INFO
#define ENABLE_DEBUG_TRACE_INFO 0
#endif

#if ENABLE_DEBUG_TRACE_ERROR || ENABLE_DEBUG_TRACE_FUNCTION_CALL || ENABLE_DEBUG_TRACE_INFO

#include <cstring>
#include <iostream>
#include <string>
#include <sstream>
#include <thread>

struct __debug_trace_prefix__
{
  std::string _prefix;

  __debug_trace_prefix__(
      const char* func_name)
  {
    auto idx1 = ::strchr(func_name, ' ');
    auto idx2 = ::strchr(func_name, '(');

    if (idx1 && idx2 && idx1 < idx2)
    {
      func_name = idx1 + 1;
      if (*func_name == '*')
      {
        func_name += 1;
      }
    }

    std::stringstream ss;
    ss << "  <debug> "
       << " ["
       << std::this_thread::get_id()
       << "] "
       << func_name
       << ": ";

    _prefix = ss.str();
  }

  std::string print() { return _prefix; }
};

struct __debug_func_printer__
{
  const char* _func_name;

  __debug_func_printer__(
      const char* func_name)
    : _func_name{func_name}
  {
    std::stringstream out;
    out << std::endl
        << __debug_trace_prefix__(_func_name).print()
        << "entered."
        << std::endl;
    std::cerr << out.str();
  }

  ~__debug_func_printer__()
  {
    std::stringstream out;
    out << std::endl
        << __debug_trace_prefix__(_func_name).print()
        << "exited."
        << std::endl;
    std::cerr << out.str();
  }
};
#endif

#if ENABLE_DEBUG_TRACE_ERROR
#define DEBUG_TRACE_ERROR(string) { \
  std::stringstream out; \
  out << __debug_trace_prefix__(__PRETTY_FUNCTION__).print() << "ERROR: " << string << std::endl; \
  std::cerr << out.str(); \
}
#define DEBUG_TRACE_ERROR_WHEN(condition,string) { \
  if (condition) { \
    std::stringstream out; \
    out << __debug_trace_prefix__(__PRETTY_FUNCTION__).print() << "ERROR: " << string << std::endl; \
    std::cerr << out.str(); \
  } \
}
#else
#define DEBUG_TRACE_ERROR(string) { }
#define DEBUG_TRACE_ERROR_WHEN(condition,string) { }
#endif

#if ENABLE_DEBUG_TRACE_FUNCTION_CALL
#define DEBUG_TRACE_FUNCTION_CALL() __debug_func_printer__ __H__(__PRETTY_FUNCTION__);
#else
#define DEBUG_TRACE_FUNCTION_CALL() { }
#endif

#if ENABLE_DEBUG_TRACE_INFO
#define DEBUG_TRACE_INFO(string) { \
  std::stringstream out; \
  out << __debug_trace_prefix__(__PRETTY_FUNCTION__).print() << "INFO: " << string << std::endl; \
  std::cerr << out.str(); \
}
#define DEBUG_TRACE_INFO_WHEN(condition,string) { \
  if (condition) { \
    std::stringstream out; \
    out << __debug_trace_prefix__(__PRETTY_FUNCTION__).print() << "INFO: " << string << std::endl; \
    std::cerr << out.str(); \
  } \
}
#else
#define DEBUG_TRACE_INFO(string) { }
#define DEBUG_TRACE_INFO_WHEN(condition,string) { }
#endif
