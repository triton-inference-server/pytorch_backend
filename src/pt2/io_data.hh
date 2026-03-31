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

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "../libtorch.hh"
#include "triton/backend/backend_common.h"

namespace triton::backend::pytorch::pt2
{
  class io_data
  {
    public:

      class data
      {
        private:

          uint32_t model_index_{0};
          uint32_t server_index_{0};
          std::vector<int64_t> shape_;
          torch::ScalarType torch_dtype_{torch::kFloat32};
          TRITONSERVER_DataType triton_dtype_{TRITONSERVER_TYPE_FP32};

        public:

          [[nodiscard]]
          const uint32_t&
          model_index() const;

          [[nodiscard]]
          uint32_t&
          model_index();

          [[nodiscard]]
          const uint32_t&
          server_index() const;

          [[nodiscard]]
          uint32_t&
          server_index();

          [[nodiscard]]
          const std::vector<int64_t>&
          shape() const;

          [[nodiscard]]
          std::vector<int64_t>&
          shape();

          [[nodiscard]]
          const torch::ScalarType&
          torch_dtype() const;

          [[nodiscard]]
          torch::ScalarType&
          torch_dtype();

          [[nodiscard]]
          const TRITONSERVER_DataType&
          triton_dtype() const;

          [[nodiscard]]
          TRITONSERVER_DataType&
          triton_dtype();

          void
          write_to(
              std::ostream& writable) const;
      };


    private:

      std::unordered_map<std::string, size_t> map_;
      std::vector<data> values_;

    public:

      bool
      contains(
          const std::string& name) const;

      size_t
      count() const;

      void
      emplace(
          const std::string& name,
          const std::string& alternate_name);

      [[nodiscard]]
      const data&
      get(
          const std::string& name) const;

      [[nodiscard]]
      data&
      get(
          const std::string& name);

      [[nodiscard]]
      std::vector<std::string>
      names() const;

      [[nodiscard]]
      size_t
      size() const;

      [[nodiscard]]
      bool
      try_get(
          const std::string& name,
          data& info_out) const;

      [[nodiscard]]
      const std::vector<data>&
      values() const;

      [[nodiscard]]
      const data&
      operator[](
          const std::string& name) const;

      [[nodiscard]]
      data&
      operator[](
          const std::string& name);
  };

  std::ostream&
  operator<<(
      std::ostream& writable,
      const io_data::data& value);
}
