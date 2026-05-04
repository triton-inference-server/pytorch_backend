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

#include "triton/backend/backend_common.h"

namespace triton::backend::pytorch::pt2
{
  enum class call_spec_type
  {
    unspecified,

    /// @brief Represents a dictionary of call specifications in the call specification tree.
    builtins_dict,

    /// @brief Represents a list of call specifications in the call specification tree.
    builtins_list,

    /// @brief Represents a tuple of call specifications in the call specification tree.
    builtins_tuple,

    /// @brief Represents a leaf tensor in the call specification tree.
    ///        A value tensor is a tensor that is an input or output of the model and does not have any children in the call specification tree.
    value_tensor,
  };

  /// @brief Gets the name of a call specification type.
  const std::string&
  name_of(
      call_spec_type value) noexcept;

  std::ostream&
  operator<<(
      std::ostream& writable,
      const call_spec_type& value) noexcept;

  /// @brief Represents the I/O call specification for a single model.
  ///        The call specification is a tree structure that describes how to construct the inputs and how to extract the outputs from the model's forward method.
  ///        Call specifications are provided by PyTorch.
  class call_spec
  {
    private:

      std::vector<call_spec> children_;
      std::vector<std::string> dictionary_keys_;
      call_spec* parent_{nullptr};
      call_spec_type type_{call_spec_type::unspecified};

    public:

      call_spec() = default;

      /// @brief Adds a child call specification to this call specification.
      /// @param child The child call specification to add.
      void
      add_child(
          call_spec& child);

      /// @brief Gets the child call specifications of this call specification.
      [[nodiscard]]
      const std::vector<call_spec>&
      children() const;

      /// @brief Checks if this call specification is a leaf tensor (i.e., it has no children and is of type value_tensor).
      [[nodiscard]]
      bool
      is_leaf_tensor() const;

      /// @brief Sets the dictionary keys for this call specification.
      ///        This is only applicable if the type of this call specification is builtins_dict.
      /// @param keys The dictionary keys to set.
      void
      dictionary_keys(
          const std::vector<std::string>& keys);

      /// @brief Gets the dictionary keys for this call specification.
      ///        This is only applicable if the type of this call specification is builtins_dict.
      [[nodiscard]]
      const std::vector<std::string>&
      dictionary_keys() const;

      /// @brief Gets the names of the call specifications in this call specification tree.
      [[nodiscard]]
      std::vector<std::string>
      get_names() const;

      /// @brief Sets the type of this call specification.
      void
      type(
          call_spec_type type);

      /// @brief Gets the type of this call specification.
      [[nodiscard]]
      call_spec_type
      type() const;

      /// @brief Tries to parse a call specification from a JSON string.
      /// @param definition The JSON string defining the call specification.
      /// @param call_spec_out The output call specification if parsing is successful.
      /// @return True if parsing was successful, false otherwise.
      [[nodiscard]]
      static bool
      try_parse(
          const std::string& definition,
          call_spec& call_spec_out);

      /// @brief Writes a human-readable representation of this call specification to the provided output stream.
      /// @param writable The output stream to write to.
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
