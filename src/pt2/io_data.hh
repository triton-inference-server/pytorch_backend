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

namespace triton::backend::pytorch::pt2 {
/// @brief Represents the I/O data specifications for a model.
///        This includes the shapes and data types of the inputs and outputs of
///        the model.
class io_data {
 public:
  class data {
   private:
    uint32_t model_index_{0};
    uint32_t server_index_{0};
    std::vector<int64_t> shape_;
    torch::ScalarType torch_dtype_{torch::kFloat32};
    TRITONSERVER_DataType triton_dtype_{TRITONSERVER_TYPE_FP32};

   public:
    /// @brief Gets the input/output vector index associated with this data
    /// specification.
    [[nodiscard]] const uint32_t& model_index() const;

    /// @brief Gets the input/output vector index associated with this data
    /// specification.
    [[nodiscard]] uint32_t& model_index();

    /// @brief Gets the server input/output index associated with this data
    /// specification.
    [[nodiscard]] const uint32_t& server_index() const;

    /// @brief Gets the server input/output index associated with this data
    /// specification.
    [[nodiscard]] uint32_t& server_index();

    /// @brief Gets the shape of the input/output associated with this data
    /// specification.
    [[nodiscard]] const std::vector<int64_t>& shape() const;

    /// @brief Gets the shape of the input/output associated with this data
    /// specification.
    [[nodiscard]] std::vector<int64_t>& shape();

    /// @brief Gets the PyTorch data type of the input/output associated with
    /// this data specification.
    [[nodiscard]] const torch::ScalarType& torch_dtype() const;

    /// @brief Gets the PyTorch data type of the input/output associated with
    /// this data specification.
    [[nodiscard]] torch::ScalarType& torch_dtype();

    /// @brief Gets the Triton data type of the input/output associated with
    /// this data specification.
    [[nodiscard]] const TRITONSERVER_DataType& triton_dtype() const;

    /// @brief Gets the Triton data type of the input/output associated with
    /// this data specification.
    [[nodiscard]] TRITONSERVER_DataType& triton_dtype();

    /// @brief Writes a human-readable representation of this data specification
    /// to the provided output stream.
    /// @param writable The output stream to write to.
    void write_to(std::ostream& writable) const;
  };


 private:
  std::unordered_map<std::string, size_t> map_;
  std::vector<data> values_;

 public:
  /// @brief Checks if a data specification with the given name exists.
  /// @param name The name of the data specification to check for.
  /// @returns True when a data specification with the given name exists; false
  /// otherwise.
  [[nodiscard]] bool contains(const std::string& name) const;

  /// @brief Gets the number of data specifications stored in this object.
  [[nodiscard]] size_t count() const;

  /// @brief Adds a new data specification with the given name and ordinal name
  /// to this object.
  /// @param name The name of the data specification to add.
  ///             This value is predetermined based on the call specification
  ///             provided by PyTorch which is determined by the model's forward
  ///             method signature. For example, "ARGS[0]" or "KWARGS[key]".
  /// @param ordinal_name The ordinal based name of the specification to add.
  ///                     For example, "INPUT__0" or "OUTPUT__0".
  void emplace(const std::string& name, const std::string& ordinal_name);

  /// @brief Gets the data specification associated with the given name.
  /// @param name The name of the data specification to get.
  ///             Either the name or ordinal name can be used to retrieve the
  ///             data specification.
  /// @returns The data specification associated with the given name.
  [[nodiscard]] const data& get(const std::string& name) const;

  /// @brief Gets the data specification associated with the given name.
  /// @param name The name of the data specification to get.
  ///             Either the name or ordinal name can be used to retrieve the
  ///             data specification.
  /// @returns The data specification associated with the given name.
  [[nodiscard]] data& get(const std::string& name);

  /// @brief Gets the names of all data specifications stored in this object.
  /// @returns The names of all data specifications stored in this object.
  [[nodiscard]] std::vector<std::string> names() const;

  /// @brief Gets the data specifications stored in this object.
  [[nodiscard]] size_t size() const;

  /// @brief Attempts to get the data specification associated with the given
  /// name.
  /// @param name The name of the data specification to get.
  ///             Either the name or ordinal name can be used to retrieve the
  ///             data specification.
  /// @param info_out The output parameter to store the data specification if
  /// found.
  /// @returns True when a data specification with the given name was found and
  /// stored in info_out; false otherwise.
  [[nodiscard]] bool try_get(const std::string& name, data& info_out) const;

  /// @brief Gets the data specifications stored in this object.
  [[nodiscard]] const std::vector<data>& values() const;

  /// @brief Gets the data specification associated with the given name.
  /// @param name The name of the data specification to get.
  ///             Either the name or ordinal name can be used to retrieve the
  ///             data specification.
  /// @returns The data specification associated with the given name.
  [[nodiscard]] const data& operator[](const std::string& name) const;

  /// @brief Gets the data specification associated with the given name.
  /// @param name The name of the data specification to get.
  ///             Either the name or ordinal name can be used to retrieve the
  ///             data specification.
  /// @returns The data specification associated with the given name.
  [[nodiscard]] data& operator[](const std::string& name);
};

std::ostream& operator<<(std::ostream& writable, const io_data::data& value);
}  // namespace triton::backend::pytorch::pt2
