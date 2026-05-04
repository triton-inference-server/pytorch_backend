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

#include "io_data.hh"
#include "../triton_utils.hh"

namespace triton::backend::pytorch::pt2
{
  bool
  io_data::contains(
      const std::string& name) const
  {
    return map_.find(name) != map_.end();
  }

  size_t
  io_data::count() const
  {
    return map_.size();
  }

  void
  io_data::emplace(
      const std::string& name,
      const std::string& ordinal_name)
  {
    if (contains(name) || (!ordinal_name.empty() && contains(ordinal_name)))
      THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_ALREADY_EXISTS,
                             "io_data already contains name: \"" + name + "\"");

    map_[name] = values_.size();
    if (!ordinal_name.empty())
    {
      map_[ordinal_name] = values_.size();
    }
    values_.push_back(pt2::io_data::data{});
  }

  const io_data::data&
  io_data::get(
      const std::string& name) const
  {
    auto iter = map_.find(name);
    if (iter == map_.end())
      THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_NOT_FOUND,
                             "io_data does not contain name: \"" + name + "\"");

    return values_[iter->second];
  }

  io_data::data&
  io_data::get(
      const std::string& name)
  {
    auto iter = map_.find(name);
    if (iter == map_.end())
      THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_NOT_FOUND,
                             "io_data does not contain name: \"" + name + "\"");

    return values_[iter->second];
  }

  std::vector<std::string>
  io_data::names() const
  {
    std::vector<std::string> result;
    result.reserve(map_.size());

    for (const auto& pair : map_)
    {
      result.push_back(pair.first);
    }

    return result;
  }

  size_t
  io_data::size() const
  {
    return values_.size();
  }

  bool
  io_data::try_get(
      const std::string& name,
      data& info_out) const
  {
    auto iter = map_.find(name);
    if (iter == map_.end())
      return false;

    info_out = values_[iter->second];
    return true;
  }

  const std::vector<io_data::data>&
  io_data::values() const
  {
    return values_;
  }

  const io_data::data&
  io_data::operator[](
      const std::string& name) const
  {
    return get(name);
  }

  io_data::data&
  io_data::operator[](
      const std::string& name)
  {
    return get(name);
  }

  /*** io_data::Data ***/

  const uint32_t&
  io_data::data::model_index() const
  {
    return model_index_;
  }

  uint32_t&
  io_data::data::model_index()
  {
    return model_index_;
  }

  const uint32_t&
  io_data::data::server_index() const
  {
    return server_index_;
  }

  uint32_t&
  io_data::data::server_index()
  {
    return server_index_;
  }

  const std::vector<int64_t>&
  io_data::data::shape() const
  {
    return shape_;
  }

  std::vector<int64_t>&
  io_data::data::shape()
  {
    return shape_;
  }

  const torch::ScalarType&
  io_data::data::torch_dtype() const
  {
    return torch_dtype_;
  }

  torch::ScalarType&
  io_data::data::torch_dtype()
  {
    return torch_dtype_;
  }

  const TRITONSERVER_DataType&
  io_data::data::triton_dtype() const
  {
    return triton_dtype_;
  }

  TRITONSERVER_DataType&
  io_data::data::triton_dtype()
  {
    return triton_dtype_;
  }

  void
  io_data::data::write_to(
      std::ostream& writable) const
  {
    writable << "{ model_index: " << model_index_
             << ", server_index: " << server_index_
             << ", shape: " << ShapeToString(shape_)
             << ", torch_dtype: " << torch_dtype_
             << ", triton_dtype: " << TRITONSERVER_DataTypeString(triton_dtype_)
             << " }";
  }

  std::ostream&
  operator<<(
      std::ostream& writable,
      const io_data::data& value)
  {
    value.write_to(writable);
    return writable;
  }
}
