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

#pragma once

#include <string>
#include <vector>

#include "call_spec_type.hh"
#include "triton/backend/backend_common.h"

namespace triton::backend::pytorch::pt2
{
  class call_spec
  {
    private:

      std::vector<call_spec> children_;
      std::vector<std::string> dictionary_keys_;
      call_spec* parent_{nullptr};
      call_spec_type type_{call_spec_type::unspecified};

    public:

      call_spec() = default;

      void
      add_child(
          call_spec& child);

      [[nodiscard]]
      const std::vector<call_spec>&
      children() const;

      [[nodiscard]]
      bool
      is_leaf_tensor() const;

      void
      dictionary_keys(
          const std::vector<std::string>& keys);

      [[nodiscard]]
      const std::vector<std::string>&
      dictionary_keys() const;

      [[nodiscard]]
      std::vector<std::string>
      get_names() const;

      void
      type(
          call_spec_type type);

      [[nodiscard]]
      call_spec_type
      type() const;

      [[nodiscard]]
      static bool
      try_parse(
          const std::string& definition,
          call_spec& call_spec_out);

      void
      write_to(
          std::ostream& writable) const;

    protected:

      void
      set_parent(
          call_spec& parent);

    private:

      [[nodiscard]]
      static bool
      try_parse_spec_object(
          common::TritonJson::Value& spec_object,
          call_spec& call_spec_out);

      void
      write_to(
          std::ostream& writable,
          size_t indent) const;
  };

  inline std::ostream&
  operator<<(
      std::ostream& os,
      const call_spec& call_spec)
  {
    call_spec.write_to(os);
    return os;
  }
}
