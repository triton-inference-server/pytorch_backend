// Copyright 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "string_utils.hh"

#include "triton_utils.hh"

namespace triton::backend::pytorch {
TRITONSERVER_Error*
GetContiguousInputContent(
    TRITONBACKEND_Input* input, const uint32_t buffer_count,
    const char** content, size_t* content_byte_size,
    std::vector<char>* contiguous_buffer, cudaStream_t cuda_stream,
    bool* cuda_copy)
{
  /*
  This function will return a tensor's contents as a contiguous chunk in system
  memory. In some cases this will require copying the data. If that  happens,
  'contiguous_buffer' will be set to hold the contiguous chunk and 'cuda_copy'
  will be set to to `true` to indicate whether CUDA copy has been conducted. The
  data copy can be avoided if the input is already in a contiguous chunk and the
  input is located in memory type and memory-type-id specified.
  */
  if (!input)
    throw std::invalid_argument("Argument 'input' cannot be null.");
  if (!content)
    throw std::invalid_argument("Argument 'content' cannot be null.");
  if (!content_byte_size)
    throw std::invalid_argument("Argument 'content_byte_size' cannot be null.");
  if (!contiguous_buffer)
    throw std::invalid_argument("Argument 'contiguous_buffer' cannot be null.");
  if (!cuda_copy)
    throw std::invalid_argument("Argument 'cuda_copy' cannot be null.");

  *cuda_copy = false;

  // Check input buffers to see if data copy is necessary
  size_t chunk_count{0};
  bool type_mismatch{false};
  uint64_t total_byte_size{0};

  for (size_t index = 0; index < buffer_count; index += 1) {
    TRITONSERVER_MemoryType src_memory_type{TRITONSERVER_MEMORY_CPU};
    int64_t src_memory_type_id{0};
    size_t src_byte_size{0};
    const void* src_ptr{nullptr};

    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        /* input= */ input,
        /* index= */ index,
        /* buffer= */ &src_ptr,
        /* buffer_byte_size= */ &src_byte_size,
        /* memory_type= */ &src_memory_type,
        /* memory_type_id= */ &src_memory_type_id));

    if (src_ptr != nullptr) {
      chunk_count++;
      total_byte_size += src_byte_size;
      type_mismatch |= (src_memory_type == TRITONSERVER_MEMORY_GPU);
    }
  }

  if (chunk_count == 0) {
    *content = nullptr;
    *content_byte_size = 0;
  } else if (chunk_count == 1 && !type_mismatch) {
    TRITONSERVER_MemoryType src_memory_type{TRITONSERVER_MEMORY_CPU};
    int64_t src_memory_type_id{0};
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        /* input= */ input,
        /* index= */ 0,
        /* buffer= */ (const void**)content,
        /* buffer_byte_size= */ content_byte_size,
        /* memory_type= */ &src_memory_type,
        /* memory_type_id= */ &src_memory_type_id));
  } else {
    contiguous_buffer->resize(total_byte_size);

    size_t offset{0};
    for (size_t i = 0; i < chunk_count; i += 1) {
      bool cuda_used{false};
      TRITONSERVER_MemoryType src_memory_type{TRITONSERVER_MEMORY_CPU};
      int64_t src_memory_type_id{0};
      size_t src_byte_size{0};
      const void* src_ptr{nullptr};

      RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
          /* input= */ input,
          /* index= */ i,
          /* buffer= */ &src_ptr,
          /* buffer_byte_size= */ &src_byte_size,
          /* memory_type= */ &src_memory_type,
          /* memory_type_id= */ &src_memory_type_id));
      RETURN_IF_ERROR(CopyBuffer(
          /* msg= */ "Contiguous input",
          /* src_memory_type= */ src_memory_type,
          /* src_memory_type_id= */ src_memory_type_id,
          /* dst_memory_type= */ TRITONSERVER_MEMORY_CPU,
          /* dst_memory_type_id= */ 0,
          /* byte_size= */ src_byte_size,
          /* src= */ src_ptr,
          /* dst= */ contiguous_buffer->data() + offset,
          /* cuda_used= */ cuda_stream,
          /* copy_on_stream= */ &cuda_used));
      *cuda_copy |= cuda_used;
      offset += src_byte_size;
    }

    *content = contiguous_buffer->data();
    *content_byte_size = total_byte_size;
  }

  return nullptr;  // success
}

void
FillStringTensor(
    torch::List<std::string>* input_list, const size_t input_list_count)
{
  if (!input_list)
    throw std::invalid_argument("Argument 'input_list' cannot be null.");

  for (size_t c = 0; c < input_list_count; c += 1) {
    input_list->push_back("");
  }
}

bool
SetStringBuffer(
    torch::List<torch::jit::IValue>* tensor, TRITONBACKEND_Response** response,
    TRITONBACKEND_Output* response_output, TRITONBACKEND_State* response_state,
    const size_t tensor_element_count, cudaStream_t cuda_stream,
    std::string* serialized, bool state)
{
  if (!tensor)
    throw std::invalid_argument("Argument 'tensor' cannot be null.");
  if (!serialized)
    throw std::invalid_argument("Argument 'serialized' cannot be null.");

  bool cuda_copy{false};

  // Serialize the output tensor strings. Each string is serialized as a 4-byte
  // length followed by the string itself with no null-terminator.
  serialized->clear();

  for (size_t e = 0; e < tensor_element_count; e += 1) {
    std::string str{tensor->get(e).to<std::string>()};
    const char* cstr{str.c_str()};
    size_t clen{str.length()};

    serialized->append(reinterpret_cast<const char*>(&clen), sizeof(uint32_t));

    if (clen > 0) {
      serialized->append(cstr, clen);
    }
  }

  // Allocate a buffer large enough to hold the serialized tensor.
  TRITONSERVER_MemoryType actual_memory_type{TRITONSERVER_MEMORY_CPU};
  int64_t actual_memory_type_id{0};
  void* buffer{nullptr};

  if (!state) {
    if (auto err = TRITONBACKEND_OutputBuffer(
            /* output= */ response_output,
            /* buffer= */ &buffer,
            /* buffer_byte_size= */ serialized->size(),
            /* memory_type= */ &actual_memory_type,
            /* memory_type_id= */ &actual_memory_type_id)) {
      RESPOND_AND_SET_NULL_IF_ERROR(response, err);
      return cuda_copy;
    }
  } else {
    if (auto err = TRITONBACKEND_StateBuffer(
            /* state= */ response_state,
            /* buffer= */ &buffer,
            /* buffer_byte_size= */ serialized->size(),
            /* memory_type= */ &actual_memory_type,
            /* memory_type_id= */ &actual_memory_type_id)) {
      RESPOND_AND_SET_NULL_IF_ERROR(response, err);
      return cuda_copy;
    }
  }
  // Copy the serialized tensor into the allocated buffer.
  bool cuda_used{false};

  if (auto err = CopyBuffer(
          /* msg= */ "String output",
          /* src_memory_type= */ TRITONSERVER_MEMORY_CPU,
          /* src_memory_type_id=0 */ 0,
          /* dst_memory_type= */ actual_memory_type,
          /* dst_memory_type_id= */ actual_memory_type_id, serialized->size(),
          /* src= */ reinterpret_cast<const void*>(serialized->c_str()),
          /* dst= */ buffer,
          /* cuda_stream= */ cuda_stream,
          /* copy_on_stream= */ &cuda_used)) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    cuda_copy |= cuda_used;
    return cuda_copy;
  }

  cuda_copy |= cuda_used;

  if (state) {
    RESPOND_AND_SET_NULL_IF_ERROR(
        response, TRITONBACKEND_StateUpdate(/* state= */ response_state));
  }

  return cuda_copy;
}

bool
SetStringInputTensor(
    torch::List<std::string>* input_list, TRITONBACKEND_Input* input,
    const char* name, const uint32_t buffer_count,
    const size_t request_element_cnt, TRITONBACKEND_Response** response,
    cudaStream_t cuda_stream, const char* host_policy_name)
{
  if (!input_list)
    throw std::invalid_argument("Argument 'input_list' cannot be null.");

  bool cuda_copy{false};

  // For string data type, we always need to have the data on CPU so that we can
  // read string length and construct the string properly.
  // So if the request's input tensor is not in CPU need to copy it there.
  const char* content{nullptr};
  size_t content_byte_size{0};
  std::vector<char> contiguous_buffer{};

  if (auto err = GetContiguousInputContent(
          /* input= */ input,
          /* buffer_count= */ buffer_count,
          /* content= */ &content,
          /* content_byte_size= */ &content_byte_size,
          /* contiguous_buffer= */ &contiguous_buffer,
          /* cuda_stream= */ cuda_stream,
          /* cuda_copy= */ &cuda_copy)) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    FillStringTensor(
        /* input_list= */ input_list,
        /* input_list_count= */ request_element_cnt);
    return cuda_copy;
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(cuda_stream);
    cuda_copy = false;
  }
#endif  // TRITON_ENABLE_GPU

  std::vector<std::pair<const char*, const uint32_t>> str_list{};
  auto err = ValidateStringBuffer(
      /* buffer= */ content,
      /* buffer_byte_size= */ content_byte_size,
      /* expected_element_cnt= */ request_element_cnt,
      /* input_name= */ name,
      /* str_list= */ &str_list);
  // Set string values.
  for (const auto& [cstr, clen] : str_list) {
    input_list->push_back(std::string(cstr, clen));
  }

  size_t element_cnt{str_list.size()};
  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    FillStringTensor(
        /* input_list= */ input_list,
        /* input_list_count= */ (request_element_cnt - element_cnt));
  }

  return cuda_copy;
}

bool
SetStringOutputBuffer(
    torch::List<torch::jit::IValue>* tensor, TRITONBACKEND_Response** response,
    TRITONBACKEND_Output* response_output, const size_t tensor_element_count,
    cudaStream_t cuda_stream, std::string* serialized)
{
  return SetStringBuffer(
      /* tensor= */ tensor,
      /* response= */ response,
      /* response_output= */ response_output,
      /* response_state= */ nullptr,
      /* tensor_element_count= */ tensor_element_count,
      /* cuda_stream= */ cuda_stream,
      /* serialized= */ serialized,
      /* state= */ false);
}

bool
SetStringStateBuffer(
    torch::List<torch::jit::IValue>* tensor, TRITONBACKEND_Response** response,
    TRITONBACKEND_State* response_state, const size_t tensor_element_count,
    cudaStream_t cuda_stream, std::string* serialized)
{
  return SetStringBuffer(
      /* tensor= */ tensor,
      /* response= */ response,
      /* response_output= */ nullptr,
      /* response_state= */ response_state,
      /* tensor_element_count= */ tensor_element_count,
      /* cuda_stream= */ cuda_stream,
      /* serialized= */ serialized,
      /* state= */ true);
}
}  // namespace triton::backend::pytorch
