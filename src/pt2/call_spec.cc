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

#include "call_spec.hh"

#include <ostream>
#include <string>

#include "../libtorch.hh"
#include "../triton_utils.hh"

namespace triton::backend::pytorch::pt2
{
  static const std::string CALLSPEC_TYPE_DICT{"builtins.dict"};
  static const std::string CALLSPEC_TYPE_LIST{"builtins.list"};
  static const std::string CALLSPEC_TYPE_TUPLE{"builtins.tuple"};
  static const std::string CALLSPEC_TYPE_VALUE_TENSOR{"value.tensor"};
  static const std::string CALLSPEC_TYPE_UNSPECIFIED{"unspecified"};

  void
  call_spec::add_child(
      call_spec& child)
  {
    DEBUG_TRACE_FUNCTION_CALL();
    children_.push_back(child);
    child.set_parent(*this);
  }

  const std::vector<call_spec>&
  call_spec::children() const
  {
    DEBUG_TRACE_FUNCTION_CALL();
    return children_;
  }

  bool
  call_spec::is_leaf_tensor() const
  {
    DEBUG_TRACE_FUNCTION_CALL();
    return type_ == call_spec_type::value_tensor
        && children_.empty();
  }

  void
  call_spec::dictionary_keys(
      const std::vector<std::string>& keys)
  {
    DEBUG_TRACE_FUNCTION_CALL();
    dictionary_keys_ = keys;
  }

  const std::vector<std::string>&
  call_spec::dictionary_keys() const
  {
    DEBUG_TRACE_FUNCTION_CALL();
    return dictionary_keys_;
  }

  std::vector<std::string>
  call_spec::get_names() const
  {
    DEBUG_TRACE_FUNCTION_CALL();
    std::vector<std::string> names;

    // When the type is unspecified or a tensor, return an empty vector.
    if (type_ == call_spec_type::unspecified || type_ == call_spec_type::value_tensor)
      return names;

    if (type_ == call_spec_type::builtins_dict)
    {
      if (dictionary_keys_.size() != children_.size())
      {
        TRITON_LOG_ERROR("call_spec node has type `" << CALLSPEC_TYPE_DICT << "` "
                         << "but number of keys (" << dictionary_keys_.size() << ") "
                         << "does not match number of children (" << children_.size() << ").");
        return names;
      }

      for (size_t i = 0; i < dictionary_keys_.size(); i += 1)
      {
        std::string dictionary_entry = "[" + dictionary_keys_.at(i) + "]";

        auto child_names = children_[i].get_names();
        if (child_names.empty())
        {
          names.push_back(dictionary_entry);
        }
        else
        {
          for (const auto& child_name : child_names)
          {
            auto name = dictionary_entry + child_name;
            names.push_back(name);
          }
        }
      }
    }
    else
    {
      for (size_t i = 0; i < children_.size(); i += 1)
      {
        std::string tuple_entry = "[" + std::to_string(i) + "]";

        auto child_names = children_[i].get_names();
        if (child_names.empty())
        {
          names.push_back(tuple_entry);
        }
        else
        {
          for (const auto& child_name : child_names)
          {
            auto name = tuple_entry + child_name;
            names.push_back(name);
          }
        }
      }
    }

    return names;
  }

  void
  call_spec::set_parent(
      call_spec& parent)
  {
    DEBUG_TRACE_FUNCTION_CALL();
    parent_ = &parent;
  }

  void
  call_spec::type(
      call_spec_type type)
  {
    DEBUG_TRACE_FUNCTION_CALL();
    type_ = type;
  }

  call_spec_type
  call_spec::type() const
  {
    DEBUG_TRACE_FUNCTION_CALL();
    return type_;
  }

  constexpr char PARSE_FAILURE_PREAMBLE[] = "Failed to parse PyTorch model's PT2 call specification: ";

  bool
  call_spec::try_parse(
    const std::string& definition,
    call_spec& call_spec_out)
  {
    DEBUG_TRACE_FUNCTION_CALL();
    if (definition.empty())
      return false;

    common::TritonJson::Value definition_json;

    if (auto err = definition_json.Parse(definition.c_str(), definition.size()))
    {
      TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                       << "{ "
                       << "error: \"" << TRITONSERVER_ErrorMessage(err) << "\""
                       << ", specification: \"" << definition << "\" "
                       << "}.");
      TRITONSERVER_ErrorDelete(err);
      return false;
    }

    if (!definition_json.IsArray() || definition_json.ArraySize() != 2)
    {
      TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                       << "{ "
                       << "error: \"Expected [version, spec] array.\""
                       << ", specification: \"" << definition << "\" "
                       << "}.");
      return false;
    }

    common::TritonJson::Value spec_object;
    if (auto err = definition_json.IndexAsObject(1, &spec_object))
    {
      TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                       << "{ "
                       << "error: \"" << TRITONSERVER_ErrorMessage(err) << "\""
                       << ", specification: \"" << definition << "\" "
                       << "}.");
      TRITONSERVER_ErrorDelete(err);
      return false;
    }

    if (spec_object.IsNull() || !spec_object.IsObject())
    {
      TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                       << "{ "
                       << "error: \"[1] != typeof(object<specification>)\""
                       << ", specification: \"" << definition << "\" "
                       << "}.");
      return false;
    }

    call_spec spec{};
    if (!try_parse_spec_object(spec_object, spec))
      return false;

    call_spec_out = spec;

    return true;
  }

  bool
  call_spec::try_parse_spec_object(
      common::TritonJson::Value& spec_object,
      call_spec& call_spec_out)
  {
    DEBUG_TRACE_FUNCTION_CALL();
    if (spec_object.IsNull() || !spec_object.IsObject())
      return false;

    constexpr char type_property_name[] = "type";
    constexpr char context_property_name[] = "context";
    constexpr char children_spec_property_name[] = "children_spec";

    call_spec result;

    common::TritonJson::WriteBuffer buffer;
    spec_object.PrettyWrite(&buffer);
    std::string spec_object_as_string{buffer.Contents()};
    buffer.Clear();

    if (spec_object.Find(type_property_name)
        && !spec_object.MemberIsNull(type_property_name)
        && spec_object.MemberIsString(type_property_name))
    {
      std::string type_string;
      if (auto err = spec_object.MemberAsString(type_property_name, &type_string))
      {
        TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                         << "{ "
                         << "details: \"Required property 'type' invalid.\""
                         << ", error: \"" << TRITONSERVER_ErrorMessage(err) << "\""
                         << ", specification: \"" << spec_object_as_string << "\" "
                         << "}.");
        TRITONSERVER_ErrorDelete(err);
        return false;
      }

      if (type_string == CALLSPEC_TYPE_DICT)
      {
        result.type(call_spec_type::builtins_dict);
      }
      else
      if (type_string == CALLSPEC_TYPE_LIST)
      {
        result.type(call_spec_type::builtins_list);
      }
      else
      if (type_string == CALLSPEC_TYPE_TUPLE)
      {
        result.type(call_spec_type::builtins_tuple);
      }
      else
      {
        TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                         << "{ "
                         << "details: \"Required property 'type' invalid or unavailable.\""
                         << ", error: \"Value of `type` property ('" << type_string << "') is unsupported.\""
                         << ", specification: \"" << spec_object_as_string << "\" "
                         << "}.");
        return false;
      }
    }
    else
    {
      result.type(call_spec_type::value_tensor);
      call_spec_out = result;
      return true;
    }

    common::TritonJson::Value context_array;

    if (spec_object.Find(context_property_name)
        && !spec_object.MemberIsNull(context_property_name)
        && (spec_object.MemberIsArray(context_property_name)
            || spec_object.MemberIsString(context_property_name)))
    {
      if (spec_object.MemberIsArray(context_property_name))
      {
        if (auto err = spec_object.MemberAsArray(context_property_name, &context_array))
        {
          TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                           << "{ "
                           << "details: \"Property 'context' is not an array.\""
                           << ", error: \"" << TRITONSERVER_ErrorMessage(err) << "\""
                           << ", specification: \"" << spec_object_as_string << "\" "
                           << "}.");
          TRITONSERVER_ErrorDelete(err);
          return false;
        }
      }
      else
      if (spec_object.MemberIsString(context_property_name))
      {
        std::string context_string;
        if (auto err = spec_object.MemberAsString(context_property_name, &context_string))
        {
          TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                           << "{ "
                           << "details: \"Property 'context' is not an array.\""
                           << ", error: \"" << TRITONSERVER_ErrorMessage(err) << "\""
                           << ", specification: \"" << spec_object_as_string << "\" "
                           << "}.");
          TRITONSERVER_ErrorDelete(err);
          return false;
        }

        if (context_string != "null")
        {
          if (auto err = context_array.Parse(context_string.c_str(), context_string.size()))
          {
            TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                             << "{ "
                             << "details: \"Failed to parse 'context' property string as JSON array.\""
                             << ", error: \"" << TRITONSERVER_ErrorMessage(err) << "\""
                             << ", specification: \"" << spec_object_as_string << "\" "
                             << "}.");
            TRITONSERVER_ErrorDelete(err);
            return false;
          }

          if (!context_array.IsArray())
          {
            TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                             << "{ "
                             << "details: \"Failed to parse 'context' property string as JSON array.\""
                             << ", specification: \"" << spec_object_as_string << "\" "
                             << "}.");
            return false;
          }
        }
      }

      context_array.PrettyWrite(&buffer);
      std::string context_array_as_string{buffer.Contents()};
      buffer.Clear();

      std::vector<std::string> context_entries;

      for (auto i = 0; i < context_array.ArraySize(); i += 1)
      {
        std::string context_entry;
        if (auto err = context_array.IndexAsString(i, &context_entry))
        {
          TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                           << "{ "
                           << "details: \"Context entry at index " << i << " is not a string.\""
                           << ", error: \"" << TRITONSERVER_ErrorMessage(err) << "\""
                           << ", specification: \"" << context_array_as_string << "\" "
                           << "}.");
          TRITONSERVER_ErrorDelete(err);
          return false;
        }

        context_entries.push_back(context_entry);
      }

      result.dictionary_keys(context_entries);
    }

    if (spec_object.Find(children_spec_property_name)
        && !spec_object.MemberIsNull(children_spec_property_name)
        && spec_object.MemberIsArray(children_spec_property_name))
    {
      common::TritonJson::Value children_array;

      if (auto err = spec_object.MemberAsArray(children_spec_property_name, &children_array))
      {
        TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                         << "{ "
                         << "details: \"Required property 'children_spec' invalid or unavailable.\""
                         << ", error: \"" << TRITONSERVER_ErrorMessage(err) << "\""
                         << ", specification: \"" << spec_object_as_string << "\" "
                         << "}.");
        TRITONSERVER_ErrorDelete(err);
        return false;
      }

      children_array.PrettyWrite(&buffer);
      std::string children_array_as_string{buffer.Contents()};
      buffer.Clear();

      switch (result.type())
      {
        case call_spec_type::builtins_dict:
        {
          if (children_array.ArraySize() != result.dictionary_keys().size())
          {
            TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                             << "{ "
                             << "details: \"Number of entries in 'children_spec' does not match number of entries in 'context'.\""
                             << ", specification: \"" << spec_object_as_string << "\" "
                             << "}.");
            return false;
          }
        }
        break;

        case call_spec_type::builtins_list:
        case call_spec_type::builtins_tuple:
        {
          if (children_array.ArraySize() == 0)
          {
            TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                             << "{ "
                             << "details: \"Property 'children_spec' must have at least one entry when 'type' is 'builtins.list' or 'builtins.tuple'.\""
                             << ", specification: \"" << spec_object_as_string << "\" "
                             << "}.");
            return false;
          }
        }
        break;

        case call_spec_type::value_tensor:
        case call_spec_type::unspecified:
          break;
      }

      if (result.type() == call_spec_type::builtins_dict && children_array.ArraySize() != result.dictionary_keys().size())
      {
        TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                         << "{ "
                         << "details: \"Number of entries in 'children_spec' does not match number of entries in 'context'.\""
                         << ", specification: \"" << spec_object_as_string << "\" "
                         << "}.");
        return false;
      }

      for (auto i = 0; i < children_array.ArraySize(); i += 1)
      {
        common::TritonJson::Value child_object;
        if (auto err = children_array.IndexAsObject(i, &child_object))
        {
          TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                           << "{ "
                           << "details: \"Child spec at index " << i << " is not an object.\""
                           << ", error: \"" << TRITONSERVER_ErrorMessage(err) << "\""
                           << ", specification: \"" << children_array_as_string << "\" "
                           << "}.");
          TRITONSERVER_ErrorDelete(err);
          return false;
        }

        if (!child_object.IsObject())
          continue;

        call_spec child_spec{};
        if (!try_parse_spec_object(child_object, child_spec))
          return false;

        result.add_child(child_spec);
      }
    }

    call_spec_out = result;
    return true;
  }

  void
  call_spec::write_to(
      std::ostream& writable) const
  {
    DEBUG_TRACE_FUNCTION_CALL();
    write_to(writable, 0);
  }

  constexpr size_t INDENT_SIZE = 2;

  void
  call_spec::write_to(
      std::ostream& writable,
      size_t indent) const
  {
    DEBUG_TRACE_FUNCTION_CALL();
    std::string indent_str(static_cast<long unsigned int>(indent * INDENT_SIZE), ' ');
    writable << indent_str << "{\n";
    writable << indent_str << "  type: \"" << type_ << "\",\n";

    if (dictionary_keys_.empty())
    {
      writable << indent_str << "  context: [],\n";
    }
    else
    {
      writable << indent_str << "  context: [\n";
      for (const auto& key : dictionary_keys_)
      {
        writable << indent_str << "    \"" << key << "\",\n";
      }
      writable << indent_str << "  ],\n";
    }
    if (children_.empty())
    {
      writable << indent_str << "  children: []\n";
    }
    else
    {
      writable << indent_str << "  children: [\n";
      for (const auto& child : children_)
      {
        child.write_to(writable, indent + 2);
        writable << ",\n";
      }
      writable << indent_str << "  ]\n";
    }

    writable << indent_str << "}";
  }
}
