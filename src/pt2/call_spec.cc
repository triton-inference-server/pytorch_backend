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

  static const std::string CALLSPEC_DICT{"builtins.dict"};
  static const std::string CALLSPEC_LIST{"builtins.list"};
  static const std::string CALLSPEC_TUPLE{"builtins.tuple"};
  static const std::string CALLSPEC_VALUE_TENSOR{"value.tensor"};
  static const std::string CALLSPEC_UNSPECIFIED{"unspecified"};

  void
  call_spec::add_child(
      call_spec& child)
  {
    children_.push_back(child);
    child.set_parent(*this);
  }

  const std::vector<call_spec>&
  call_spec::children() const
  {
    return children_;
  }

  bool
  call_spec::is_leaf_tensor() const
  {
    return type_ == call_spec_type::value_tensor
        && children_.empty();
  }

  void
  call_spec::dictionary_keys(
      const std::vector<std::string>& keys)
  {
    dictionary_keys_ = keys;
  }

  const std::vector<std::string>&
  call_spec::dictionary_keys() const
  {
    return dictionary_keys_;
  }

  std::vector<std::string>
  call_spec::get_names() const
  {
    std::vector<std::string> names;

    // When the type is unspecified or a tensor, return an empty vector.
    if (type_ == call_spec_type::unspecified || type_ == call_spec_type::value_tensor)
      return names;

    if (type_ == call_spec_type::builtins_dict)
    {
      if (dictionary_keys_.size() != children_.size())
      {
        TRITON_LOG_ERROR("call_spec node has type `" << CALLSPEC_DICT << "` "
                         << "but number of keys (" << dictionary_keys_.size() << ") "
                         << "does not match number of children (" << children_.size() << ").");
        return names;
      }

      // Walk the list of dictionary keys, prepending them as nodes in the format "[key]" to the names of the children nodes.
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
    if (type_ == call_spec_type::builtins_list || type_ == call_spec_type::builtins_tuple)
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
    else
    {
      TRITON_LOG_ERROR("Invalid type \"" << type_ << "\" found in call specification.");
    }

    return names;
  }

  void
  call_spec::set_parent(
      call_spec& parent)
  {
    parent_ = &parent;
  }

  void
  call_spec::type(
      call_spec_type type)
  {
    type_ = type;
  }

  call_spec_type
  call_spec::type() const
  {
    return type_;
  }

  constexpr char PARSE_FAILURE_PREAMBLE[] = "Failed to parse PyTorch model's PT2 call specification: ";

  bool
  call_spec::try_parse(
    const std::string& definition,
    call_spec& call_spec_out)
  {
    if (definition.empty())
      return false;

    common::TritonJson::Value definition_json;

    /*
      This function is expecting a JSON string with the following format:
      [
        version, // currently expected to be 1
        {
          type: "builtins.dict" | "builtins.list" | "builtins.tuple" | "value.tensor",
          context: [key1, key2, ...], // only applicable when type is builtins.dict; otherwise expected to be an empty array
          children_spec: [spec1, spec2, ...] // only applicable when type is builtins.dict, builtins.list, or builtins.tuple; otherwise expected to be an empty array
        }
      ]

      Its responsibility is to convert the JSON string (provided by PyTorch based on the model's forward method signature) into a call_spec object model which
      Triton can use to understand the structure of the inputs and outputs of the model's forward method.
    */

    if (auto err = definition_json.Parse(definition.c_str(),
                                         definition.size()))
    {
      TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                       << "{ "
                       << "error: \"" << TRITONSERVER_ErrorMessage(err) << "\""
                       << ", specification: \"" << definition << "\" "
                       << "}.");
      TRITONSERVER_ErrorDelete(err);
      return false;
    }

    // Ensure the JSON is an array with two entries: [version, spec]
    if (!definition_json.IsArray() || definition_json.ArraySize() != 2)
    {
      TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                       << "{ "
                       << "error: \"Expected [version, spec] array.\""
                       << ", specification: \"" << definition << "\" "
                       << "}.");
      return false;
    }

    // Ensure [1] is an object representing the call specification.
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

    // Ensure the spec object is a JSON object.
    if (spec_object.IsNull() || !spec_object.IsObject())
    {
      TRITON_LOG_ERROR(PARSE_FAILURE_PREAMBLE
                       << "{ "
                       << "error: \"[1] != typeof(object<specification>)\""
                       << ", specification: \"" << definition << "\" "
                       << "}.");
      return false;
    }

    // Parse the spec object into a call_spec instance.
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
    if (spec_object.IsNull() || !spec_object.IsObject())
      return false;

    /*
      This functions resposibility to parse the specification JSON object (which is the second entry in the top-level array parsed by try_parse) into a call_spec instance.
      The expected format of the specification JSON object is as follows:

      {
        type: "builtins.dict" | "builtins.list" | "builtins.tuple" | "value.tensor",
        context: [key1, key2, ...], // only applicable when type is builtins.dict; otherwise expected to be an empty array
        children_spec: [spec1, spec2, ...] // only applicable when type is builtins.dict, builtins.list, or builtins.tuple; otherwise expected to be an empty array
      }

      When the type is "builtins.dict", each entry in the context array corresponds to a key for the dictionary entry at that position in the children_spec array. When the type is "builtins.list" or "builtins.tuple", the context array is expected to be empty and each entry in the children_spec array corresponds to an entry in the list or tuple. When the type is "value.tensor", both the context and children_spec arrays are expected to be empty since tensors are leaf nodes in the call specification tree.
    */

    call_spec result;

    common::TritonJson::WriteBuffer buffer;
    // Create a buffer for pretty-printing JSON values in error messages.
    // Be sure to clear the buffer after each use to avoid appending to previous contents when pretty-printing multiple JSON values during the parsing of a single specification object.
    spec_object.PrettyWrite(&buffer);
    std::string spec_object_as_string{buffer.Contents()};
    buffer.Clear();

    constexpr char type_property_name[] = "type";

    // Determine if the object has a type property, and if so, set the type of the call_spec accordingly.
    // When the type property is not provided, the type of the call_spec is assumed to be value_tensor.
    // When the type property is provided but has an unrecognized value, return an error.
    if (spec_object.Find(type_property_name)
        && !spec_object.MemberIsNull(type_property_name)
        && spec_object.MemberIsString(type_property_name))
    {
      // Parse the value of the type property as a string.
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

      // Depending on the value of the type property, set the type of this call_spec node.
      // When the type is unrecognized, return an error.
      if (type_string == CALLSPEC_DICT)
      {
        result.type(call_spec_type::builtins_dict);
      }
      else
      if (type_string == CALLSPEC_LIST)
      {
        result.type(call_spec_type::builtins_list);
      }
      else
      if (type_string == CALLSPEC_TUPLE)
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
    // The the type property is not provided, the object is assumed to be a value tensor.
    else
    {
      result.type(call_spec_type::value_tensor);
      call_spec_out = result;
      return true;
    }

    constexpr char context_property_name[] = "context";

    /*
      Attempt to parse the context property if it exists.
      When the context property exists but is not an array or string, return an error.
      When the context property is a string, attempt to parse it as JSON; if parsing fails or the result is not an array, return an error.
      When the type of this call specification is builtins.dict, the context property is expected to be an array of strings representing the keys for each dictionary entry in this call specification.
      When the type of this call specification is builtins.list or builtins.tuple, the context property is expected to be an empty array since lists and tuples do not have keys.
    */
    common::TritonJson::Value context_array;
    if (spec_object.Find(context_property_name)
        && !spec_object.MemberIsNull(context_property_name)
        && (spec_object.MemberIsArray(context_property_name)
            || spec_object.MemberIsString(context_property_name)))
    {
      /*
        Because the context property can provided as `null`, a JSON array, or a string (i.e. "null"), we need to handle all cases when parsing the context.
      */

      // When the context property is a JSON array, we can parse it directly.
      // When the context property is a string, we need to first parse the string as JSON and then ensure the result is a JSON array.
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
      // When the context property is a string, attempt to parse it as JSON; if parsing fails or the result is not an array, return an error.
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

        // Only attempt to parse the context string as JSON if it is not equal to "null".
        // This allows us to support both a null value and a string value of "null" to represent an empty context, which provides flexibility for how the JSON specification can be defined on the PyTorch side.
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

          // The result of parsing the context string must be a JSON array; if it is not, return an error.
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

      // Create a buffer for pretty-printing JSON values in error messages.
      // Be sure to clear the buffer after each use to avoid appending to previous contents when pretty-printing multiple JSON values during the parsing of a single specification object.
      context_array.PrettyWrite(&buffer);
      std::string context_array_as_string{buffer.Contents()};
      buffer.Clear();

      // Parse the entries in the context array as strings and store them in the call_spec instance.
      std::vector<std::string> context_entries;
      for (auto i = 0; i < context_array.ArraySize(); i += 1)
      {
        // Read the value as a string.
        // When the value is not a JSON string, return an error.
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

    constexpr char children_spec_property_name[] = "children_spec";

    // Attempt to parse the children_spec property if it exists.
    if (spec_object.Find(children_spec_property_name)
        && !spec_object.MemberIsNull(children_spec_property_name)
        && spec_object.MemberIsArray(children_spec_property_name))
    {
      common::TritonJson::Value children_array;

      // When the children_spec property exists but is not an array, return an error.
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

      // Create a buffer for pretty-printing JSON values in error messages.
      // Be sure to clear the buffer after each use to avoid appending to previous contents when pretty-printing multiple JSON values during the parsing of a single specification object.
      children_array.PrettyWrite(&buffer);
      std::string children_array_as_string{buffer.Contents()};
      buffer.Clear();

      // Depending on the resulting type of this call specification, validate that the children_spec array has the expected number of entries.
      switch (result.type())
      {
        // Validate that when the type of this call specification is builtins.dict, the number of entries in the children_spec array matches the number of dictionary keys specified in the context property since each entry in the children_spec array corresponds to a dictionary entry with a key specified in the context array.
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

        // Validate that when the type of this call specification is builtins.list or builtins.tuple, the children_spec array has at least one entry since an empty list or tuple would not be valid.
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

      // Parse each entry in the children_spec array as a call_spec object and add it as a child of this call_spec instance.
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

        // Parse the child specification object into a call_spec instance.
        // RECURSIVE CALL
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
    write_to(writable, 0);
  }

  constexpr size_t INDENT_SIZE = 2;

  void
  call_spec::write_to(
      std::ostream& writable,
      size_t indent) const
  {
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
