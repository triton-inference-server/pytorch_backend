// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "call_spec_type.hh"

namespace triton::backend::pytorch::pt2
{
  static const std::string CALLSPEC_TYPE_DICT{"dict"};
  static const std::string CALLSPEC_TYPE_LIST{"list"};
  static const std::string CALLSPEC_TYPE_TUPLE{"tuple"};
  static const std::string CALLSPEC_TYPE_VALUE_TENSOR{"tensor"};
  static const std::string CALLSPEC_TYPE_UNSPECIFIED{"<unspecified>"};

  const std::string&
  name_of(
      call_spec_type value) noexcept
  {
    switch (value)
    {
      case call_spec_type::builtins_dict:
        return CALLSPEC_TYPE_DICT;
      case call_spec_type::builtins_list:
        return CALLSPEC_TYPE_LIST;
      case call_spec_type::builtins_tuple:
        return CALLSPEC_TYPE_TUPLE;
      case call_spec_type::value_tensor:
        return CALLSPEC_TYPE_VALUE_TENSOR;

      default:
      case call_spec_type::unspecified:
        return CALLSPEC_TYPE_UNSPECIFIED;
    }
  }

  std::ostream& operator<<(
      std::ostream& writable,
      const call_spec_type& value) noexcept
  {
    writable << name_of(value);
    return writable;
  }
}
