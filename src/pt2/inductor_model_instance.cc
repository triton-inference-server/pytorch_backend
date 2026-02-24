// Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "inductor_model_instance.hh"

#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../libtorch.hh"
#include "../libtorch_utils.h"
#include "../string_utils.hh"
#include "triton/common/nvtx.h"
#include "triton_exception.hh"
#include "triton_utils.hh"

#ifdef TRITON_ENABLE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#endif

namespace triton::backend::pytorch::pt2 {
using TritonInductorModel = triton::backend::pytorch::pt2::InductorModel;
using TritonJsonValue = triton::common::TritonJson::Value;
using TritonNamingConvention = triton::backend::pytorch::NamingConvention;

static const std::string DELIMINATOR{"__"};

InductorModelInstance::InductorModelInstance(
    TritonInductorModel* model,
    TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance{model, triton_model_instance}, model_{model}
{
  DEBUG_TRACE_FUNCTION_CALL();
  if (!model)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL, "Argument `model` cannot be `null`.")
  if (!triton_model_instance)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Argument `triton_model_instance` cannot be `null`.")

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    DEBUG_TRACE_INFO(
        "device = torch::Device(torch::kCUDA, " << DeviceId() << ")");
    device_ = torch::Device{torch::kCUDA, static_cast<int8_t>(DeviceId())};
    CreateCudaEvents(DeviceId());
  }

  device_count_ = torch::cuda::device_count();
#endif

  model_->LoadModel(ArtifactFilename(), device_, device_count_, Kind());

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    DEBUG_TRACE_INFO(
        "{ Kind(): TRITONSERVER_INSTANCEGROUPKIND_GPU, len(device_count_): "
        << device_count_ << " }");
    // Since we cannot determine the exact devices used by the model, we create
    // a CUDA stream for every available device to ensure proper synchronization
    // of CUDA streams. This approach may have implications when a timestamp is
    // captured on a device that is not used by the model. Currently, this issue
    // is addressed by synchronizing the CUDA streams before recording
    // timestamps to prevent timestamp skewing. However, in the future, any
    // modifications to the CUDA stream synchronization logic should be handled
    // with caution.
    for (int i = 0; i < device_count_; i++) {
      cudaStream_t stream;
      if (auto err = CreateCudaStream(i, 0 /* cuda_stream_priority */, &stream))
        throw triton::backend::BackendModelInstanceException(err);

      stream_vector_.push_back(stream);
    }

    if (!stream_vector_.empty()) {
      // Create CUDA events on the first device that will be used for collecting
      // inputs/outputs.
      CreateCudaEvents(0);
    }
  }
#endif

  size_t expected_input_count{0};

  TritonJsonValue inputs;
  if (model_->ModelConfig().Find("input", &inputs)) {
    expected_input_count = inputs.ArraySize();
  }

  TritonJsonValue config_batch_inputs;
  if (model_->ModelConfig().Find("batch_input", &config_batch_inputs)) {
    batch_input_count_ = config_batch_inputs.ArraySize();
    expected_input_count += batch_input_count_;
  }

  // If this is a sequence model then make sure that the required inputs are
  // present in the model and have the correct shape and datatype.
  TritonJsonValue sequence_batching;
  if (model_->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    if (ValidateBooleanSequenceControl(
            sequence_batching, "CONTROL_SEQUENCE_START",
            false /* required */)) {
      DEBUG_TRACE_INFO(
          "ValidateBooleanSequenceControl(sequence_batching,"
          " \"CONTROL_SEQUENCE_START\", false /* required */) -> true");
      expected_input_count += 1;
    }
    if (ValidateBooleanSequenceControl(
            sequence_batching, "CONTROL_SEQUENCE_END", false /* required */)) {
      DEBUG_TRACE_INFO(
          "ValidateBooleanSequenceControl(sequence_batching,"
          " \"CONTROL_SEQUENCE_END\", false /* required */) -> true");
      expected_input_count += 1;
    }
    if (ValidateBooleanSequenceControl(
            sequence_batching, "CONTROL_SEQUENCE_READY",
            false /* required */)) {
      DEBUG_TRACE_INFO(
          "ValidateBooleanSequenceControl(sequence_batching,"
          " \"CONTROL_SEQUENCE_READY\", false /* required */) -> true");
      expected_input_count += 1;
    }
    if (ValidateBooleanSequenceControl(
            sequence_batching, "CONTROL_SEQUENCE_CORRID",
            false /* required */)) {
      DEBUG_TRACE_INFO(
          "ValidateBooleanSequenceControl(sequence_batching,"
          " \"CONTROL_SEQUENCE_CORRID\", false /* required */) -> true");
      expected_input_count += 1;
    }

    // Add the state inputs to the expected count.
    TritonJsonValue states;
    if (sequence_batching.Find("state", &states)) {
      expected_input_count += states.ArraySize();
    }
  }

  is_batching_supported_ = model_->MaxBatchSize() > 0;

  DEBUG_TRACE_INFO(
      "{ device_count: " << device_count_
                         << ", expected_input_count: " << expected_input_count
                         << ", batch_input_count_: " << batch_input_count_
                         << ", is_batching_supported: "
                         << (is_batching_supported_ ? "true" : "false")
                         << " }");

  ValidateInputs(expected_input_count);
  ValidateOutputs();
}

InductorModelInstance::~InductorModelInstance()
{
  DEBUG_TRACE_FUNCTION_CALL();
  ClearCache();

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
    DEBUG_TRACE_INFO(
        "{ Kind(): TRITONSERVER_INSTANCEGROUPKIND_MODEL, len(stream_vector_): "
        << stream_vector_.size() << " }");
    for (size_t i = 0; i < stream_vector_.size(); i += 1) {
      if (auto err = ConvertCUDAStatusToTritonError(
              /* cuda_error= */ cudaSetDevice(i),
              /* code= */ TRITONSERVER_ERROR_INTERNAL,
              /* msg= */ "Failed to set device for stream destruction")) {
        TRITON_LOG_ERROR(
            "Failed to set the device while destroying streams for instance \""
            << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
      }

      if (auto err = ConvertCUDAStatusToTritonError(
              /* cuda_error= */ cudaStreamDestroy(stream_vector_[i]),
              /* code= */ TRITONSERVER_ERROR_INTERNAL,
              /* msg= */ "Failed to destroy cuda stream")) {
        TRITON_LOG_ERROR(
            "Failed to destroy cuda stream for instance \""
            << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
      }

      stream_vector_[i] = nullptr;
    }
  }
#endif
}

void
InductorModelInstance::AddInputToMap(
    TritonNamingConvention naming_convention,
    const std::vector<std::string>& allowed_inputs, const std::string& io_name,
    uint32_t index)
{
  DEBUG_TRACE_FUNCTION_CALL();
  if (is_dictionary_input_) {
    input_index_map_[io_name] = index;
  } else {
    switch (naming_convention) {
      case TritonNamingConvention::FORWARD_ARGUMENT: {
        auto it =
            std::find(allowed_inputs.begin(), allowed_inputs.end(), io_name);
        if (it != allowed_inputs.end()) {
          input_index_map_[io_name] = std::distance(allowed_inputs.begin(), it);
          DEBUG_TRACE_INFO(
              "{ name: \"" << Name() << "\""
                           << ", naming_convention: \"" << naming_convention
                           << "\", io_name: \"" << io_name << "\""
                           << ", input_index: " << input_index_map_[io_name]
                           << " }");
        }
      } break;

      case TritonNamingConvention::NAMED_INDEX: {
        int start_pos = io_name.find(DELIMINATOR);
        int ip_index = std::atoi(io_name.substr(start_pos + 2).c_str());
        input_index_map_[io_name] = ip_index;
        DEBUG_TRACE_INFO(
            "{ name: \"" << Name() << "\""
                         << ", naming_convention: \"" << naming_convention
                         << "\", io_name: \"" << io_name << "\""
                         << ", input_index: " << input_index_map_[io_name]
                         << " }");
      } break;

      case TritonNamingConvention::STRICT_CONFIG_ORDERING: {
        input_index_map_[io_name] = index;
        DEBUG_TRACE_INFO(
            "{ name: \"" << Name() << "\""
                         << ", naming_convention: \"" << naming_convention
                         << "\", io_name: \"" << io_name << "\""
                         << ", input_index: " << input_index_map_[io_name]
                         << " }");
      } break;

      default: {
        DEBUG_TRACE_ERROR(
            "Invalid naming convention value: " << naming_convention);
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ERROR_INVALID_ARG,
            "Argument 'naming_convention' value of \""
                << naming_convention << "\" is invalid or unsupported.");
      } break;
    }
  }
}

const std::string&
InductorModelInstance::ArtifactFilename() const
{
  DEBUG_TRACE_FUNCTION_CALL();
  return BackendModelInstance::ArtifactFilename();
}

void
InductorModelInstance::ClearCache()
{
  DEBUG_TRACE_FUNCTION_CALL();
#ifdef TRITON_ENABLE_GPU
  if (device_.is_cuda() ||
      (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0)) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
#endif
}

InductorModelInstance*
InductorModelInstance::Create(
    TritonInductorModel* model,
    TRITONBACKEND_ModelInstance* triton_model_instance)
{
  DEBUG_TRACE_FUNCTION_CALL();
  try {
    return new InductorModelInstance(model, triton_model_instance);
  }
  catch (const triton::backend::pytorch::BackendException& exception) {
    DEBUG_TRACE_ERROR(
        "{ model_name: \"" << model->Name() << "\", error: \""
                           << exception.what() << "\" }");
    THROW_TRITON_EXCEPTION(
        exception.error_code(),
        "Failed to create InductorModelInstance for model \""
            << model->Name() << "\": " << exception.what());
  }
  catch (const std::exception& exception) {
    DEBUG_TRACE_ERROR(
        "{ model_name: \"" << model->Name()
                           << "\""
                              ", error: \""
                           << exception.what() << "\" }");
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to create InductorModelInstance for model \""
            << model->Name() << "\": " << exception.what());
  }
}

void
InductorModelInstance::CreateCudaEvents(int32_t device_id)
{
  DEBUG_TRACE_FUNCTION_CALL();
#ifdef TRITON_ENABLE_GPU
  // Need to set the CUDA context so that the context that events are created on
  // match with contexts that events are recorded with.
  DEBUG_TRACE_INFO("cudaSetDevice(" << device_id << ")");
  if (auto err = cudaSetDevice(device_id)) {
    DEBUG_TRACE_ERROR(
        "cudaSetDevice(" << device_id
                         << ") failed with error: " << cudaGetErrorString(err));
    THROW_TRITON_EXCEPTION(
        err,
        "When creating CUDA events, failed to set the device for "
        "model instance "
        "\"" << Name()
             << "\" : " << cudaGetErrorString(err));
  }

  DEBUG_TRACE_INFO("cudaEventCreate(&compute_input_start_event_)");
  if (auto err = cudaEventCreate(&compute_input_start_event_)) {
    DEBUG_TRACE_ERROR(
        "cudaEventCreate(&compute_input_start_event_) failed with error: "
        << cudaGetErrorString(err));
    THROW_TRITON_EXCEPTION(
        err,
        "When creating CUDA events, failed to create compute input start event "
        "for model instance "
        "\"" << Name()
             << "\" : " << cudaGetErrorString(err));
  }

  DEBUG_TRACE_INFO("cudaEventCreate(&compute_infer_start_event_)");
  if (auto err = cudaEventCreate(&compute_infer_start_event_)) {
    DEBUG_TRACE_ERROR(
        "cudaEventCreate(&compute_infer_start_event_) failed with error: "
        << cudaGetErrorString(err));
    THROW_TRITON_EXCEPTION(
        err,
        "When creating CUDA events, failed to create compute infer start event "
        "for model instance "
        "\"" << Name()
             << "\" : " << cudaGetErrorString(err));
  }

  DEBUG_TRACE_INFO("cudaEventCreate(&compute_output_start_event_)");
  if (auto err = cudaEventCreate(&compute_output_start_event_)) {
    DEBUG_TRACE_ERROR(
        "cudaEventCreate(&compute_output_start_event_) failed with error: "
        << cudaGetErrorString(err));
    THROW_TRITON_EXCEPTION(
        err,
        "When creating CUDA events, failed to create compute output start "
        "event for model instance "
        "\"" << Name()
             << "\" : " << cudaGetErrorString(err));
  }
#endif
}

cudaStream_t
InductorModelInstance::CudaStream()
{
  DEBUG_TRACE_FUNCTION_CALL();
  return BackendModelInstance::CudaStream();
}

int32_t
InductorModelInstance::DeviceId() const
{
  DEBUG_TRACE_FUNCTION_CALL();
  return BackendModelInstance::DeviceId();
}

void
InductorModelInstance::Execute(
    std::vector<TRITONBACKEND_Response*>* responses, uint32_t response_count,
    std::vector<torch::IValue>& input_tensors,
    std::vector<torch::IValue>& output_tensors)
{
  DEBUG_TRACE_FUNCTION_CALL();
  NVTX_RANGE(nvtx_, "Execute " + Name());

  std::vector<torch::IValue> model_outputs;

  try {
    // Enable/disable inference mode based on the model setting.
    // Supersedes NoGradGuard.
    torch::InferenceMode guard{model_->InferenceModeEnabled()};

    // Enable/disable cuDNN.
    at::globalContext().setUserEnabledCuDNN(model_->CudnnEnabled());

    torch::NoGradGuard no_grad_guard;

    if (is_dictionary_input_) {
      // DEBUG_TRACE_ERROR("Dictionary inputs are not supported yet for model
      // instance \"" << Name() << "\"."); NO SUPPORT FOR DICTIONARY INPUTS YET
      // torch::Dict<std::string, torch::Tensor> dict_input;
      // for (auto& input_index : input_index_map_)
      // {
      //   const std::string& input_name = input_index.first;
      //   uint32_t model_input_index = input_index.second;

      //   dict_input.insert(input_name, input_tensors.at(model_input_index));
      // }

      // std::vector<torch::IValue> model_inputs = {dict_input};
      // model_outputs = model_->Forward(model_inputs);
      torch::Dict<std::string, torch::Tensor> input_dictionary{};
      for (auto& input_index : input_index_map_) {
        const std::string& input_name = input_index.first;
        uint32_t model_input_index = input_index.second;

        input_dictionary.insert(
            input_name, input_tensors.at(model_input_index).toTensor());
      }

      std::vector<torch::IValue> model_inputs{input_dictionary};
      DEBUG_TRACE_INFO(
          "calling Forward() for model instance \""
          << Name() << "\" with " << model_inputs.size() << " input tensors.");
      model_outputs = model_->Forward(model_inputs);
      DEBUG_TRACE_INFO(
          "Forward() for model instance \"" << Name()
                                            << "\""
                                               " returned "
                                            << model_outputs.size()
                                            << " output tensors.");
    } else {
      DEBUG_TRACE_INFO(
          "calling Forward() for model instance \""
          << Name() << "\" with " << input_tensors.size() << " input tensors.");
      model_outputs = model_->Forward(input_tensors);
      DEBUG_TRACE_INFO(
          "Forward() for model instance \"" << Name() << "\" returned "
                                            << model_outputs.size()
                                            << " output tensors.");
    }

    for (auto& output_tensor : model_outputs) {
      output_tensors.push_back(output_tensor);
    }
  }
  catch (const std::exception& exception) {
    DEBUG_TRACE_ERROR(
        "{ model_name: \"" << model_->Name()
                           << "\""
                              ", error: \""
                           << exception.what() << "\" }");
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            TOSTRING(
                "Inductor model instance \""
                << Name() << "\" execution failure: " << exception.what())
                .c_str()));
  }
}

float
InductorModelInstance::GetCudaEventElapsedTime(
    const cudaEvent_t& start_event, const cudaEvent_t& end_event)
{
  DEBUG_TRACE_FUNCTION_CALL();
  float duration{0};
#ifdef TRITON_ENABLE_GPU
  if (auto err = ConvertCUDAStatusToTritonError(
          cudaEventElapsedTime(&duration, start_event, end_event),
          TRITONSERVER_ERROR_INTERNAL, "Failed to capture elapsed time")) {
    TRITON_LOG_ERROR(
        "Failed to capture elapsed time"
        << " for instance \"" << Name()
        << "\": " << TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
  }
#endif
  return duration;
}

cudaStream_t
InductorModelInstance::GetCudaStreamByInstanceKind()
{
  DEBUG_TRACE_FUNCTION_CALL();
#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    DEBUG_TRACE_INFO(
        "Returning CUDA stream for GPU instance kind for model instance \""
        << Name() << "\".");
    return stream_;
  }

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL &&
      !stream_vector_.empty()) {
    DEBUG_TRACE_INFO(
        "Returning CUDA stream for MODEL instance kind for model instance \""
        << Name() << "\".");
    return stream_vector_[0];
  }
#endif
  DEBUG_TRACE_INFO(
      "No CUDA stream available for model instance \"" << Name() << "\".");
  return nullptr;
}

TritonNamingConvention
InductorModelInstance::GetNamingConvention(
    const std::vector<std::string>& allowed_ios)
{
  DEBUG_TRACE_FUNCTION_CALL();
  // Rules for (non-Dictionary) input tensor names:
  // 1. Must be in 'allowed_inputs' (arguments in the forward function)
  // 2. Must follow the naming convention i.e. <name>__<index>
  // 3. If neither of the above conditions are satisfied, enforce strict
  // ordering of model inputs.
  //
  // Rules for output tensor names:
  // 1. Must follow the naming convention i.e. <name>__<index>
  // 2. If not, we enforce strict ordering of model outputs.
  std::string io_kind{"input"};
  TritonNamingConvention naming_convention{
      TritonNamingConvention::FORWARD_ARGUMENT};

  if (allowed_ios.size() == 0) {
    io_kind = "output";
    naming_convention = TritonNamingConvention::NAMED_INDEX;
  }

  TritonJsonValue ios;
  if (auto err = model_->ModelConfig().MemberAsArray(io_kind.c_str(), &ios)) {
    THROW_TRITON_EXCEPTION(
        err, "Failed to get \"" << io_kind
                                << "\" array from model config for model \""
                                << model_->Name() << "\".");
  }

  if (io_kind == "input") {
    for (size_t i = 0; i < ios.ArraySize(); i += 1) {
      TritonJsonValue io;
      if (auto err = ios.IndexAsObject(i, &io)) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", io_kind: \"" << io_kind << "\""
                               << ", index: " << i << ", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\""
                               << " }");
        THROW_TRITON_EXCEPTION(
            err, "Failed to get \"" << io_kind << "\" object at index " << i
                                    << " from model config for model \""
                                    << model_->Name() << "\".");
      }

      // Validate name.
      std::string io_name;
      if (auto err = io.MemberAsString("name", &io_name)) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", io_kind: \"" << io_kind << "\""
                               << ", index: " << i << ", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\""
                               << " }");
        THROW_TRITON_EXCEPTION(
            err, "Failed to get \"" << io_kind << "\" name at index " << i
                                    << " from model config for model \""
                                    << model_->Name() << "\".");
      }
      DEBUG_TRACE_INFO(
          "{ model_name: \"" << model_->Name() << "\", io_name: \"" << io_name
                             << "\" }");

      auto it = std::find(allowed_ios.begin(), allowed_ios.end(), io_name);
      if (it == allowed_ios.end()) {
        naming_convention = TritonNamingConvention::NAMED_INDEX;
        break;
      }
    }
  }

  if (naming_convention == TritonNamingConvention::NAMED_INDEX) {
    for (size_t i = 0; i < ios.ArraySize(); i += 1) {
      TritonJsonValue io;
      if (auto err = ios.IndexAsObject(i, &io)) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", naming_convention: \"" << naming_convention
                               << "\", io_kind: \"" << io_kind << "\""
                               << ", index: " << i << ", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\""
                               << " }");
        THROW_TRITON_EXCEPTION(
            err, "Failed to get " << io_kind << " object at index " << i
                                  << " from model config for model \""
                                  << model_->Name() << "\".");
      }

      // Validate name.
      std::string io_name;
      if (auto err = io.MemberAsString("name", &io_name)) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", naming_convention: \"" << naming_convention
                               << "\", io_kind: \"" << io_kind << "\""
                               << ", index: " << i << ", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\""
                               << " }");
        THROW_TRITON_EXCEPTION(
            err, "Failed to get \"" << io_kind << "\" name at index " << i
                                    << " from model config for model \""
                                    << model_->Name() << "\".");
      }
      DEBUG_TRACE_INFO(
          "{ model_name: \"" << model_->Name() << "\""
                             << ", index: " << i << ", naming_convention: \""
                             << naming_convention << "\""
                             << ", io_kind: \"" << io_kind << "\""
                             << ", io_name: \"" << io_name << "\" }");

      int start_pos = io_name.find(DELIMINATOR);
      if (start_pos == -1) {
        naming_convention = TritonNamingConvention::STRICT_CONFIG_ORDERING;
        DEBUG_TRACE_INFO(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", index: " << i << ", naming_convention: \""
                               << naming_convention << "\" }");
        break;
      } else {
        // check if the index part of the name is not an integer
        std::string index_str = io_name.substr(start_pos + 2);
        bool is_int{true};
        for (auto it = index_str.begin(); it != index_str.end(); it++) {
          if (std::isdigit(*it) == 0) {
            is_int = false;
          }
        }

        if (!is_int) {
          if (io_kind == "input") {
            TRITON_LOG_WARN(
                "Input \""
                << io_name
                << "\" or previous input(s) are neither an input argument to "
                   "the model \""
                << model_->Name()
                << "\" nor follow the <name>__<index> naming convention."
                   " Defaulting to strict ordering of model inputs.");
          } else {
            TRITON_LOG_WARN(
                "Output \""
                << io_name << "\" or previous output(s) of the model \""
                << model_->Name()
                << "\" do not follow the <name>__<index> naming convention."
                   " Defaulting to strict ordering of model outputs.");
          }

          naming_convention = TritonNamingConvention::STRICT_CONFIG_ORDERING;
          DEBUG_TRACE_INFO(
              "{ model_name: \"" << model_->Name() << "\""
                                 << ", index: " << i
                                 << ", naming_convention: \""
                                 << naming_convention << "\" }");
          break;
        }
      }
    }
  }

  return naming_convention;
}

const std::string&
InductorModelInstance::HostPolicyName() const
{
  DEBUG_TRACE_FUNCTION_CALL();
  return BackendModelInstance::HostPolicyName();
}

TritonInductorModel*
InductorModelInstance::InductorModel() const
{
  DEBUG_TRACE_FUNCTION_CALL();
  return model_;
}

TRITONSERVER_InstanceGroupKind
InductorModelInstance::Kind() const
{
  DEBUG_TRACE_FUNCTION_CALL();
  return BackendModelInstance::Kind();
}

triton::backend::BackendModel*
InductorModelInstance::Model() const
{
  DEBUG_TRACE_FUNCTION_CALL();
  return BackendModelInstance::Model();
}

const std::string&
InductorModelInstance::Name() const
{
  DEBUG_TRACE_FUNCTION_CALL();
  return BackendModelInstance::Name();
}

void
InductorModelInstance::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  DEBUG_TRACE_FUNCTION_CALL();
  TRITON_LOG_VERBOSE(
      "TRITONBACKEND_ModelExecute: Running model \""
      << Name() << "\" with " << request_count << " requests");

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    SetCurrentCudaStream(stream_, DeviceId());
  } else if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
    // Replace the default stream of each device with the one we created.
    for (size_t i = 0; i < stream_vector_.size(); i += 1) {
      SetCurrentCudaStream(stream_vector_[i], DeviceId());
    }
  }
#endif

  NVTX_RANGE(nvtx_, "ProcessRequests " + Name());

  uint64_t exec_start_ns{0};
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_->MaxBatchSize();
  DEBUG_TRACE_INFO(
      "{ model: \"" << Name() << "\", max_batch_size: " << max_batch_size
                    << " }");

  // For each request collect the total batch size for this inference execution.
  // The batch-size, number of inputs, and size of each input has already been
  // checked so don't need to do that here.
  size_t total_batch_size{0};
  for (size_t i = 0; i < request_count; i += 1) {
    // If we get a nullptr request then something is badly wrong. Fail and
    // release all requests.
    if (!requests[i]) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              TOSTRING(
                  "NULL request given to PyTorch backend for model \"" << Name()
                                                                       << "\".")
                  .c_str()));
      return;
    }
  }

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed{false};

  for (size_t i = 0; i < request_count; i += 1) {
    TRITONBACKEND_Response* response;
    if (auto err = TRITONBACKEND_ResponseNew(&response, requests[i])) {
      responses.emplace_back(nullptr);
      TRITON_LOG_ERROR(
          "Failed to create response for request "
          << i << " of model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
    } else {
      responses.emplace_back(response);
    }
  }

  for (size_t i = 0; i < request_count; i += 1) {
    if (max_batch_size > 0) {
      TRITONBACKEND_Input* input{nullptr};
      auto err = TRITONBACKEND_RequestInputByIndex(requests[i], 0, &input);
      if (!err) {
        const int64_t* shape{nullptr};
        err = TRITONBACKEND_InputProperties(
            /* input= */ input,
            /* name= */ nullptr,
            /* datatype= */ nullptr,
            /* shape= */ &shape,
            /* dims_count= */ nullptr,
            /* byte_size= */ nullptr,
            /* buffer_count= */ nullptr);
        total_batch_size += shape[0];
      }

      if (err) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed, err);
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      total_batch_size += 1;
    }
  }

  if (total_batch_size == 0) {
    DEBUG_TRACE_INFO("{ model: \"" << Name() << "\", batch_size: 0 }");
    return;
  }

  if (!all_response_failed) {
    if (total_batch_size != 1 && total_batch_size > (size_t)max_batch_size) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              TOSTRING(
                  "batch size " << total_batch_size << " for \"" << Name()
                                << "\", max allowed is " << max_batch_size)
                  .c_str()));
    }
  }

  std::vector<const char*> input_names;
  std::vector<torch::IValue> input_tensors;
  bool cuda_copy{false};
  std::unique_ptr<BackendInputCollector> collector;

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU ||
      (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0)) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ConvertCUDAStatusToTritonError(
            cudaEventRecord(
                compute_input_start_event_, GetCudaStreamByInstanceKind()),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record the event."));
  }
#endif

  if (!all_response_failed) {
    collector.reset(new BackendInputCollector(
        requests, request_count, &responses, model_->TritonMemoryManager(),
        model_->EnablePinnedInput(), GetCudaStreamByInstanceKind(), nullptr,
        nullptr, 0, HostPolicyName().c_str()));

    triton_exception triton_ex;
    TRITONSERVER_Error* err{nullptr};
    try {
      SetInputTensors(
          total_batch_size, requests, request_count, &responses,
          collector.get(), &input_names, &input_tensors, &cuda_copy);
    }
    catch (const triton_exception& e) {
      triton_ex = e;
    }
    catch (const std::exception& e) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          TOSTRING(
              "Failed to set input tensors for model instance \""
              << Name() << "\": " << e.what())
              .c_str());
    }
    err = err ? err : triton_ex.get_error();
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed, err);
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(GetCudaStreamByInstanceKind());
    cuda_copy = false;
  }
#endif

  std::vector<torch::IValue> output_tensors;
  uint64_t compute_start_ns{0};
  uint64_t compute_infer_start{0};

  triton_exception triton_ex;
  TRITONSERVER_Error* err{nullptr};
  try {
    RecordBackendTimestamp(
        &compute_start_ns,
        reinterpret_cast<void*>(&compute_infer_start_event_));
  }
  catch (const triton_exception& e) {
    triton_ex = e;
  }
  catch (const std::exception& e) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        TOSTRING(
            "Failed record compute start timestamp for \""
            << Name() << "\": " << e.what())
            .c_str());
  }
  err = err ? err : triton_ex.get_error();

  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed, err);

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0) {
    SET_TIMESTAMP(compute_infer_start);
  }

  if (!all_response_failed) {
    DEBUG_TRACE_INFO(
        "Execute() for { model: \""
        << Name() << "\""
        << ", request_count: " << request_count
        << ", len(input_tensors): " << input_tensors.size()
        << ", len(output_tensors): " << output_tensors.size() << " }");

    Execute(
        /* responses= */ &responses,
        /* response_count= */ request_count,
        /* input_tensors= */ input_tensors,
        /* output_tensors= */ output_tensors);
  }

  bool invalid_index{false};
  int max_index{static_cast<int>(output_tensors.size() - 1)};

  if (!all_response_failed) {
    for (const auto& name : model_->ModelOutputs()) {
      int op_index = output_index_map_[name.first];
      if (op_index < 0 || op_index > max_index) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                TOSTRING(
                    "The output \"" << name.first
                                    << "\" in the model configuration refers to"
                                       " an output index which doesn't exist."
                                       " This model has "
                                    << max_index << " outputs.")
                    .c_str()));
        invalid_index = true;
        break;
      }
    }
  }

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL) {
    // For 'KIND_MODEL', multiple streams will be involved, so we need to call
    // 'cudaStreamSynchronize' before reading the output tensors.
    for (auto& stream : stream_vector_) {
      cudaStreamSynchronize(stream);
    }
  }
#endif

  uint64_t compute_end_ns{0};
  uint64_t compute_output_start{0};

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0) {
    SET_TIMESTAMP(compute_output_start);
  } else
#endif
  {
    triton_exception triton_ex;
    TRITONSERVER_Error* err{nullptr};
    try {
      RecordBackendTimestamp(
          &compute_end_ns,
          reinterpret_cast<void*>(&compute_output_start_event_));
    }
    catch (const triton_exception& e) {
      triton_ex = e;
    }
    catch (const std::exception& e) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          TOSTRING(
              "Failed to record compute end timestamp for \""
              << Name() << "\": " << e.what())
              .c_str());
    }
    err = err ? err : triton_ex.get_error();
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed, err);
  }

  if (!all_response_failed) {
    if (!invalid_index) {
      TRITONSERVER_Error* err{nullptr};
      try {
        DEBUG_TRACE_INFO(
            "ReadOutputTensors() for {"
            << " model: \"" << Name() << "\""
            << ", total_batch_size: " << total_batch_size
            << ", len(output_tensors): " << output_tensors.size()
            << ", request_count: " << request_count
            << ", responses_count: " << responses.size() << " }");
        ReadOutputTensors(
            total_batch_size, output_tensors, requests, request_count,
            responses);
      }
      catch (const triton::backend::pytorch::BackendException& exception) {
        TRITON_LOG_ERROR(
            "Failed to read output tensors for model instance \""
            << Name() << "\": " << exception.what());

        err = TRITONSERVER_ErrorNew(
            exception.error_code(),
            TOSTRING(
                "Failed to read output tensors for model instance \""
                << Name() << "\": " << exception.what())
                .c_str());
      }
      catch (const std::exception& exception) {
        TRITON_LOG_ERROR(
            "Failed to read output tensors for model instance \""
            << Name() << "\": " << exception.what());

        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            TOSTRING(
                "Failed to read output tensors for model instance \""
                << Name() << "\": " << exception.what())
                .c_str());
      }
      catch (const std::exception& exception) {
        TRITON_LOG_ERROR(
            "Failed to read output tensors for model instance \""
            << Name() << "\": " << exception.what());
      }

      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed, err);
    }
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of an earlier
  // error. Note that the responses are not set to nullptr here as we need that
  // indication below to determine if the request we successful or not.
  for (auto& response : responses) {
    if (response) {
      if (auto err = TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr)) {
        TRITON_LOG_ERROR(
            "Failed to send PyTorch backend response for model instance \""
            << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
      }
    }
  }

#ifdef TRITON_ENABLE_GPU
  // We don't need an explicit CUDA synchronization here since we have already
  // synchronized the stream in the ReadOutputTensors function.
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    float compute_input_duration = GetCudaEventElapsedTime(
        compute_input_start_event_, compute_infer_start_event_);
    float compute_infer_duration = GetCudaEventElapsedTime(
        compute_infer_start_event_, compute_output_start_event_);

    compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
    compute_end_ns = compute_start_ns + (compute_infer_duration * 1e6);
  } else if (
      Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0) {
    float compute_input_duration = GetCudaEventElapsedTime(
        compute_input_start_event_, compute_infer_start_event_);
    uint64_t compute_infer_duration =
        compute_output_start - compute_infer_start;

    compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
    compute_end_ns = compute_start_ns + compute_infer_duration;
  }
#endif

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; r += 1) {
    auto& request = requests[r];
    if (auto err = TRITONBACKEND_ModelInstanceReportStatistics(
            /* instance= */ TritonModelInstance(),
            /* request= */ request,
            /* success= */ (responses[r] != nullptr),
            /* exec_start_ns= */ exec_start_ns,
            /* exec_end_ns= */ exec_end_ns,
            /* compute_start_ns= */ compute_start_ns,
            /* compute_end_ns= */ compute_end_ns)) {
      TRITON_LOG_ERROR(
          "Failed to report statistics for request "
          << r << " of model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
    }

    if (auto err = TRITONBACKEND_RequestRelease(
            request, TRITONSERVER_REQUEST_RELEASE_ALL)) {
      TRITON_LOG_ERROR(
          "Failed to release request "
          << r << " of model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
    }
  }

  if (!all_response_failed) {
    if (auto err = TRITONBACKEND_ModelInstanceReportBatchStatistics(
            /* instance= */ TritonModelInstance(),
            /* batch_size= */ total_batch_size,
            /* exec_start_ns= */ exec_start_ns,
            /* compute_start_ns= */ compute_start_ns,
            /* compute_end_ns= */ compute_end_ns,
            /* exec_end_ns= */ exec_end_ns)) {
      TRITON_LOG_ERROR(
          "Failed to report batch statistics for model instance \""
          << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
InductorModelInstance::ReadOutputTensors(
    size_t total_batch_size, const std::vector<torch::IValue>& output_tensors,
    TRITONBACKEND_Request** requests, uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>& responses)
{
  DEBUG_TRACE_FUNCTION_CALL();
  if (!requests)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Invalid nullptr requests pointer for reading output tensors"
        " for model instance \""
            << Name() << "\".");

  NVTX_RANGE(nvtx_, "ReadOutputTensors " + Name());

  DEBUG_TRACE_INFO(
      "{ model: \"" + Name() + "\""
      << ", total_batch_size: " << total_batch_size << ", request_count: "
      << request_count << ", max_batch_size: " << model_->MaxBatchSize()
      << ", enable_pinned_input: " << model_->EnablePinnedInput()
      << ", supports_batching: " << is_batching_supported_ << " }");

  triton::backend::BackendOutputResponder responder{
      /* requests= */ requests,
      /* request_count= */ request_count,
      /* responses= */ &responses,
      /* memory_manager= */ model_->TritonMemoryManager(),
      /* first_dim_batching= */ false,
      /* pinned_enabled= */ model_->EnablePinnedInput(),
      /* stream= */ GetCudaStreamByInstanceKind()};

  bool cuda_copy{false};
  std::vector<std::shared_ptr<std::string>> string_buffers;

  for (auto& output : model_->ModelOutputs()) {
    int op_index = output_index_map_[output.first];
    auto output_name = output.first;
    auto output_tensor_pair = output.second;

    DEBUG_TRACE_INFO(
        "{ model_name: \"" << model_->Name() << "\""
                           << ", output_name: \"" << output_name << "\""
                           << ", op_index: " << op_index << ", is_tensor: true"
                           << ", output_tensor_pair: ("
                           << output_tensor_pair.first << ", "
                           << output_tensor_pair.second << ")"
                           << " }");

    torch::Tensor output_flat;

    try {
      output_flat = output_tensors[op_index].toTensor().contiguous().flatten();
    }
    catch (const std::exception& exception) {
      DEBUG_TRACE_ERROR(
          "{ model_name: \"" << model_->Name() << "\", output_name: \""
                             << output_name << "\", error: \""
                             << exception.what() << "\" }");

      if (auto err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              TOSTRING(
                  "Inductor model instance \""
                  << Name() << "\" output tensor \"" << output_name
                  << "\" not found: " << exception.what())
                  .c_str())) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", output_name: \"" << output_name << "\""
                               << ", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\""
                               << " }");
        THROW_TRITON_EXCEPTION(err, TRITONSERVER_ErrorMessage(err));
      }
    }

    auto scalar_type = output_flat.scalar_type();
    auto output_dtype = ConvertTorchTypeToDataType(scalar_type);
    auto config_dtype = output_dtype_map_[output_name];

    DEBUG_TRACE_INFO(
        "{ model_name: \"" << model_->Name() << "\""
                           << ", output_name: \"" << output_name << "\""
                           << ", scalar_type: \""
                           << torch::toString(scalar_type) << "\""
                           << ", output_dtype: \"TYPE_"
                           << TRITONSERVER_DataTypeString(output_dtype) << "\""
                           << ", config_dtype: \"TYPE_"
                           << TRITONSERVER_DataTypeString(config_dtype) << "\""
                           << " }");

    if (output_dtype != config_dtype) {
      DEBUG_TRACE_ERROR(
          "{ model_name: \"" << model_->Name() << "\""
                             << ", output_name: \"" << output_name << "\""
                             << ", scalar_type: \""
                             << torch::toString(scalar_type) << "\""
                             << ", output_dtype: TYPE_"
                             << TRITONSERVER_DataTypeString(output_dtype)
                             << ", config_dtype: TYPE_"
                             << TRITONSERVER_DataTypeString(config_dtype)
                             << ", error: \"datatype mismatch\""
                             << " }");
      if (auto err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              TOSTRING(
                  "Inductor model instance \""
                  << Name() << "\" output tensor \"" << output_name << "\""
                  << " has datatype TYPE_"
                  << TRITONSERVER_DataTypeString(output_dtype)
                  << " but model configuration expects TYPE_"
                  << TRITONSERVER_DataTypeString(config_dtype) << ".")
                  .c_str())) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\""
                               << ", output_name: \"" << output_name << "\""
                               << ", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\""
                               << " }");
        THROW_TRITON_EXCEPTION(err, TRITONSERVER_ErrorMessage(err));
      }
    }

    const char* output_buffer =
        static_cast<const char*>(output_flat.data_ptr());

    // Output tensors might not reside on the same device as the model instance.
    torch::Device output_device = output_flat.device();
    const auto memory_type = (output_device.type() == torch::kCPU)
                                 ? TRITONSERVER_MEMORY_CPU
                                 : TRITONSERVER_MEMORY_GPU;
    const auto memory_type_id =
        (output_device.type() == torch::kCPU) ? 0 : output_device.index();

    DEBUG_TRACE_INFO(
        "{ model_name: \"" << model_->Name() << "\""
                           << ", output_name: \"" << output_name << "\""
                           << ", output_device: \"" << output_device << "\""
                           << ", memory_type: \""
                           << TRITONSERVER_MemoryTypeString(memory_type) << "\""
                           << ", memory_type_id: " << memory_type_id << " }");

    // Batch output doesn't support string data type yet, as it is not trivial
    // to parse string output.
    const triton::backend::BatchOutput* batch_output =
        model_->FindBatchOutput(output_name);

    if (!batch_output) {
      DEBUG_TRACE_INFO(
          "{ model_name: \"" << model_->Name() << "\", output_name: \""
                             << output_name << "\", batch_output: NULL }");
      std::vector<int64_t> batch_n_shape;
      auto shape = output_tensors[op_index].toTensor().sizes();
      for (auto it = shape.begin(); it != shape.end(); it++) {
        batch_n_shape.push_back(*it);
      }

      if (batch_n_shape.size() == 0) {
        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\", output_name: \""
                               << output_name
                               << "\", error: \"scalar output not supported\""
                                  " }");
        auto error_message = TOSTRING(
            "Inductor model instance \""
            << Name() << "\" output tensor \"" << output_name
            << "\" is a scalar which is not supported.");
        auto err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, error_message.c_str());

        DEBUG_TRACE_ERROR(
            "{ model_name: \"" << model_->Name() << "\", output_name: \""
                               << output_name << "\", error: \""
                               << TRITONSERVER_ErrorMessage(err) << "\" }");
        THROW_TRITON_EXCEPTION(err, error_message);
      }

      if (output_tensor_pair.first != -1) {
        DEBUG_TRACE_INFO(
            "responder.ProcessTensor { model_name: \""
            << model_->Name() << "\""
            << ", output_name: \"" << output_name << "\""
            << ", output_dtype: TYPE_"
            << TRITONSERVER_DataTypeString(output_dtype) << ", batch_n_shape: "
            << triton::backend::ShapeToString(batch_n_shape)
            << ", memory_type: \"" << TRITONSERVER_MemoryTypeString(memory_type)
            << "\", memory_type_id: " << memory_type_id << " }");
        responder.ProcessTensor(
            /* output_name= */ output_name,
            /* datatype= */ output_dtype,
            /* batchn_shape= */ batch_n_shape,
            /* buffer= */ output_buffer,
            /* memory_type= */ memory_type,
            /* memory_type_id= */ memory_type_id);
      }

      if (output_tensor_pair.second != -1) {
        auto states = responder.ProcessStateTensor(
            /* output_name= */ output_name,
            /* datatype= */ output_dtype,
            /* batchn_shape= */ batch_n_shape,
            /* buffer= */ output_buffer,
            /* memory_type= */ memory_type,
            /* memory_type_id= */ memory_type_id);

        for (auto& state : states) {
          if (auto err = TRITONBACKEND_StateUpdate(state)) {
            DEBUG_TRACE_ERROR(
                "{ model_name: \"" << model_->Name() << "\", output_name: \""
                                   << output_name
                                   << "\", error: \"failed to update state\""
                                      " }");
            THROW_TRITON_EXCEPTION(
                err, "Failed to update state for model instance \""
                         << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
          }
        }
      }
    } else {
      responder.ProcessBatchOutput(
          /* name= */ output_name,
          /* batch_output= */ *batch_output,
          /* buffer= */ output_buffer,
          /* memory_type= */ memory_type,
          /* memory_type_id= */ memory_type_id);
    }
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  cudaStreamSynchronize(GetCudaStreamByInstanceKind());
#endif
}

void
InductorModelInstance::RecordBackendTimestamp(
    uint64_t* timestamp, void* cuda_event_ptr)
{
  DEBUG_TRACE_FUNCTION_CALL();
#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU ||
      (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0)) {
    cudaEvent_t& cuda_event = *(reinterpret_cast<cudaEvent_t*>(cuda_event_ptr));
    if (auto err = ConvertCUDAStatusToTritonError(
            cudaEventRecord(cuda_event, GetCudaStreamByInstanceKind()),
            TRITONSERVER_ERROR_INTERNAL, "Failed to record CUDA event")) {
      DEBUG_TRACE_ERROR(
          "{ model_name: \"" << model_->Name() << "\""
                             << ", error: \"" << TRITONSERVER_ErrorMessage(err)
                             << "\" }");
      throw triton_exception{err};
    }
  } else
#endif
  {
    SET_TIMESTAMP(*timestamp);
  }
}

void
InductorModelInstance::SetCurrentCudaStream(
    const cudaStream_t& stream, int device_id)
{
  DEBUG_TRACE_FUNCTION_CALL();
#ifdef TRITON_ENABLE_GPU
  DEBUG_TRACE_INFO(
      "Setting current CUDA stream on device "
      << device_id << " for model instance \"" << Name() << "\".");
  at::cuda::CUDAStream torch_stream{
      at::cuda::getStreamFromExternal(stream, device_id)};
  // This function replaces the default stream with the stream we created.
  // It is not necessary to change the current device to the desired device when
  // replacing the default stream for that device. See the documentation here:
  // https://pytorch.org/cppdocs/api/function_namespacec10_1_1cuda_1a6ed50cc0fc16cc7014d9c2f4c3bd098d.html
  at::cuda::setCurrentCUDAStream(torch_stream);
#endif
}

void
InductorModelInstance::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::vector<torch::IValue>* input_tensors, bool* cuda_copy)
{
  DEBUG_TRACE_FUNCTION_CALL();
  if (!requests)
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Argument 'requests' cannot be nullptr.");

  torch::InferenceMode guard{model_->InferenceModeEnabled()};

  uint32_t input_count{0};
  if (auto err = TRITONBACKEND_RequestInputCount(requests[0], &input_count)) {
    TRITON_LOG_ERROR(
        "Failed to get input count"
        << " for request 0 of model instance \"" << Name()
        << "\": " << TRITONSERVER_ErrorMessage(err));
    throw triton_exception{err};
  }

  input_tensors->resize(input_count + batch_input_count_);

  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_preference;
  if (device_.is_cpu()) {
    alloc_preference = {
        {TRITONSERVER_MEMORY_CPU_PINNED, 0},
        {TRITONSERVER_MEMORY_CPU, 0},
    };

    DEBUG_TRACE_INFO(
        "{ model: \"" << Name()
                      << "\", allocation_preference: ["
                         " { type: \""
                      << TRITONSERVER_MemoryTypeString(
                             TRITONSERVER_MEMORY_CPU_PINNED)
                      << "\", id: 0 },"
                         " { type: \""
                      << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_CPU)
                      << "\""
                         ", id: 0 } ] }");
  } else {
    alloc_preference = {
        {TRITONSERVER_MEMORY_GPU, device_.index()},
    };

    DEBUG_TRACE_INFO(
        "{ model: \"" << Name()
                      << "\", allocation_preference: ["
                         " { type: \""
                      << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_GPU)
                      << "\", id: " << device_.index() << " } ] }");
  }

  for (size_t input_idx = 0; input_idx < input_count; input_idx += 1) {
    TRITONBACKEND_Input* input{nullptr};
    if (auto err =
            TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input)) {
      TRITON_LOG_ERROR(
          "Failed to get input "
          << input_idx << " for request 0 of model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      throw triton_exception{err};
    }

    const char* input_name{nullptr};
    TRITONSERVER_DataType input_datatype{TRITONSERVER_TYPE_INVALID};
    const int64_t* input_shape{nullptr};
    uint32_t input_dims_count{0};

    if (auto err = TRITONBACKEND_InputProperties(
            input, &input_name, &input_datatype, &input_shape,
            &input_dims_count, nullptr, nullptr)) {
      TRITON_LOG_ERROR(
          "Failed to get properties for input "
          << input_idx << " for request 0 of model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      throw triton_exception{err};
    }

    input_names->emplace_back(input_name);

    std::vector<int64_t> batch_n_shape{};

    if (model_->IsInputRagged(input_name)) {
      DEBUG_TRACE_INFO(
          "{ model: \"" << Name() << "\", input: \"" << input_name
                        << "\", is_ragged: true }");

      batch_n_shape = std::vector<int64_t>{0};
      for (size_t request_index = 0; request_index < request_count;
           request_index += 1) {
        TRITONBACKEND_Input* input{nullptr};
        if (auto err = TRITONBACKEND_RequestInput(
                requests[request_index], input_name, &input)) {
          RESPOND_AND_SET_NULL_IF_ERROR(
              &((*responses)[request_index]),
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  TOSTRING(
                      "Failed to get input \""
                      << input_name << "\" for request " << request_index
                      << " of model instance \"" << Name()
                      << "\": " << TRITONSERVER_ErrorMessage(err))
                      .c_str()));
          TRITONSERVER_ErrorDelete(err);
        }

        const int64_t* input_shape{nullptr};
        uint32_t input_dims_count{0};
        if (auto err = TRITONBACKEND_InputProperties(
                /* input= */ input,
                /* name= */ nullptr,
                /* datatype= */ nullptr,
                /* shape= */ &input_shape,
                /* dims_count= */ &input_dims_count,
                /* byte_size= */ nullptr,
                /* buffer_count= */ nullptr)) {
          RESPOND_AND_SET_NULL_IF_ERROR(
              &((*responses)[request_index]),
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  TOSTRING(
                      "Failed to get properties for input \""
                      << input_name << "\" for request " << request_index
                      << " of model instance \"" << Name()
                      << "\": " << TRITONSERVER_ErrorMessage(err))
                      .c_str()));
          TRITONSERVER_ErrorDelete(err);
        }

        int64_t element_cnt{0};
        if (auto err =
                GetElementCount(input_shape, input_dims_count, &element_cnt)) {
          RESPOND_AND_SET_NULL_IF_ERROR(
              &((*responses)[request_index]),
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  TOSTRING(
                      "Failed to get element count for input \""
                      << input_name << "\" for request " << request_index
                      << " of model instance \"" << Name()
                      << "\": " << TRITONSERVER_ErrorMessage(err))
                      .c_str()));
          TRITONSERVER_ErrorDelete(err);
        }

        batch_n_shape[0] += element_cnt;
      }
    } else {
      batch_n_shape =
          std::vector<int64_t>{input_shape, input_shape + input_dims_count};
      if (is_batching_supported_) {
        DEBUG_TRACE_INFO(
            "{ model: \"" << Name() << "\""
                          << ", input: \"" << input_name << "\""
                          << ", original_shape: "
                          << triton::backend::ShapeToString(batch_n_shape)
                          << ", batch_n_shape[0]=total_batch_size: "
                          << total_batch_size << " }");
        batch_n_shape[0] = total_batch_size;
      }
    }

    // The input must be in contiguous CPU/GPU memory.
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_preference;
    // For 'KIND_MODEL', input will always be in CPU as we don't have a way to
    // query the input types.
    if (device_.is_cpu() || (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL)) {
      alloc_preference = {
          {TRITONSERVER_MEMORY_CPU_PINNED, 0},
          {TRITONSERVER_MEMORY_CPU, 0},
      };

      DEBUG_TRACE_INFO(
          "{ model: \""
          << Name() << "\", input: \"" << input_name
          << "\", allocation_preference: [ { type: \""
          << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_CPU_PINNED)
          << "\", id: 0 }, { type: \""
          << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_CPU)
          << "\", id: 0 } ] }");
    } else {
      alloc_preference = {
          {TRITONSERVER_MEMORY_GPU, device_.index()},
      };

      DEBUG_TRACE_INFO(
          "{ model: \"" << Name() << "\", input: \"" << input_name
                        << "\", allocation_preference: [ { type: \""
                        << TRITONSERVER_MemoryTypeString(
                               TRITONSERVER_MEMORY_GPU)
                        << "\", id: " << device_.index() << " } ] }");
    }

    const char* input_buffer{nullptr};
    size_t batch_n_byte_size{0};
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id{0};

    if (auto err = collector->ProcessTensor(
            /* input_name= */ input_name,
            /* buffer= */ nullptr,
            /* buffer_byte_size= */ 0,
            /* allowed_input_types= */ alloc_preference,
            /* dst_buffer= */ &input_buffer,
            /* dst_buffer_byte_size= */ &batch_n_byte_size,
            /* dst_memory_type= */ &memory_type,
            /* dst_memory_type_id= */ &memory_type_id)) {
      TRITON_LOG_ERROR(
          "Failed to process input \""
          << input_name << "\" for model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      throw triton_exception{err};
    }

    const std::pair<bool, torch::ScalarType> torch_dtype =
        ConvertDataTypeToTorchType(input_datatype);
    torch::TensorOptions options{torch_dtype.second};
    auto updated_options = (memory_type == TRITONSERVER_MEMORY_GPU)
                               ? options.device(torch::kCUDA, device_.index())
                               : options.device(torch::kCPU);

    if (input_datatype == TRITONSERVER_TYPE_BYTES) {
      DEBUG_TRACE_INFO(
          "{ model: \"" << Name() << "\", input: \"" << input_name
                        << "\", input_datatype: TYPE_BYTES"
                           " }");

      // Create the PyTorch list to hold the strings.
      torch::List<std::string> input_list{};
      input_list.reserve(batch_n_shape[0]);

      for (size_t idx = 0; idx < request_count; idx += 1) {
        TRITONBACKEND_Input* input{nullptr};
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));

        const int64_t* shape{nullptr};
        uint32_t dims_count{0};
        uint32_t buffer_count{0};

        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_InputPropertiesForHostPolicy(
                /* input= */ input,
                /* host_policy_name= */ HostPolicyName().c_str(),
                /* name= */ nullptr,
                /* datatype= */ nullptr,
                /* shape= */ &shape,
                /* dims_count= */ &dims_count,
                /* byte_size= */ nullptr,
                /* buffer_count= */ &buffer_count));

        int64_t batch_element_count = 0;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            GetElementCount(shape, dims_count, &batch_element_count));

        *cuda_copy |= SetStringInputTensor(
            &input_list, input, input_name, buffer_count, batch_element_count,
            &((*responses)[idx]), GetCudaStreamByInstanceKind(),
            HostPolicyName().c_str());
      }

      (*input_tensors)[input_index_map_[input_name]] = input_list;
    } else {
      DEBUG_TRACE_INFO(
          "{ model: \"" << Name() << "\""
                        << ", input: \"" << input_name << "\""
                        << ", input_datatype: " << torch_dtype
                        << ", batch_n_byte_size: " << batch_n_byte_size
                        << " }");

      if (batch_n_byte_size > 0) {
        (*input_tensors)[input_index_map_[input_name]] = torch::from_blob(
            const_cast<char*>(input_buffer), torch::IntArrayRef(batch_n_shape),
            updated_options);
      } else {
        // Create an empty tensor for zero-sized input.
        (*input_tensors)[input_index_map_[input_name]] =
            torch::zeros(batch_n_shape, updated_options);
      }
    }
  }

  for (const auto& batch_input : model_->BatchInputs()) {
    std::vector<int64_t> shape{};
    collector->BatchInputShape(batch_input, &shape);

    for (const auto& input_name : batch_input.TargetNames()) {
      input_names->emplace_back(input_name.c_str());

      const char* dst_buffer{nullptr};
      size_t dst_buffer_byte_size{0};
      TRITONSERVER_MemoryType dst_memory_type{TRITONSERVER_MEMORY_CPU};
      int64_t dst_memory_type_id{0};

      RESPOND_ALL_AND_SET_NULL_IF_ERROR(
          (*responses), responses->size(),
          collector->ProcessBatchInput(
              batch_input, nullptr, 0, alloc_preference, &dst_buffer,
              &dst_buffer_byte_size, &dst_memory_type, &dst_memory_type_id));

      const auto torch_dtype =
          ConvertDataTypeToTorchType(batch_input.DataType());
      torch::TensorOptions options{torch_dtype.second};
      auto updated_options = (dst_memory_type == TRITONSERVER_MEMORY_GPU)
                                 ? options.device(torch::kCUDA, device_.index())
                                 : options.device(torch::kCPU);

      if (dst_buffer_byte_size) {
        (*input_tensors)[input_index_map_[input_name]] = torch::from_blob(
            const_cast<char*>(dst_buffer), shape, updated_options);
      } else {
        // special handle when input has zero size
        (*input_tensors)[input_index_map_[input_name]] =
            torch::zeros(shape, updated_options);
      }
    }
  }

  *cuda_copy |= collector->Finalize();
}

TRITONBACKEND_ModelInstance*
InductorModelInstance::TritonModelInstance()
{
  return BackendModelInstance::TritonModelInstance();
}

bool
InductorModelInstance::ValidateBooleanSequenceControl(
    TritonJsonValue& sequence_batching, const std::string& control_kind,
    bool required)
{
  DEBUG_TRACE_FUNCTION_CALL();
  std::string tensor_name;
  std::string tensor_dtype;
  if (auto err = GetBooleanSequenceControlProperties(
          sequence_batching, model_->Name(), control_kind, required,
          &tensor_name, &tensor_dtype, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr)) {
    DEBUG_TRACE_ERROR(
        "Failed to validate boolean sequence control for model instance \""
        << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
    THROW_TRITON_EXCEPTION(
        err, "Failed to validate boolean sequence control for model instance \""
                 << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
  }

  bool have_control{!tensor_name.empty()};

  if (have_control) {
    int input_index{0};
    int start_pos{static_cast<int>(tensor_name.find(DELIMINATOR))};

    if (start_pos == -1) {
      DEBUG_TRACE_ERROR(
          "Input \""
          << tensor_name
          << "\" does not follow <name>__<index> naming convention.");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Input \""
              << tensor_name
              << "\" does not follow <name>__<index> naming convention.");
    }

    // Check if the index part of the name is not an integer.
    std::string index_str = tensor_name.substr(start_pos + 2);
    for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
      if (std::isdigit(*itr) == 0) {
        DEBUG_TRACE_ERROR(
            "Input \""
            << tensor_name
            << "\" does not follow <name>__<index> naming convention.");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ERROR_INTERNAL,
            "Input \""
                << tensor_name
                << "\" does not follow <name>__<index> naming convention.");
      }
    }

    input_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
    input_index_map_[tensor_name] = input_index;
    DEBUG_TRACE_INFO(
        "{ model_name: \"" << model_->Name() << "\""
                           << ", control_kind: \"" << control_kind << "\""
                           << ", tensor_name: \"" << tensor_name << "\""
                           << ", tensor_dtype: \"" << tensor_dtype << "\""
                           << ", input_index: " << input_index_map_[tensor_name]
                           << " }");
  }

  return have_control;
}

void
InductorModelInstance::ValidateInputs(const size_t expected_input_count)
{
  DEBUG_TRACE_FUNCTION_CALL();
  std::vector<std::string> allowed_inputs{model_->GetModelCallSpec()};

  if (allowed_inputs.size() != expected_input_count) {
    DEBUG_TRACE_ERROR(
        "Failed to load model \""
        << Name() << "\" configuration expects " << expected_input_count
        << " inputs, but model expects " << allowed_inputs.size()
        << " inputs.");
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to load model \""
            << Name() << "\" configuration expects " << expected_input_count
            << " inputs, but model expects " << allowed_inputs.size()
            << " inputs.");
  }

  /* CANNOT VALIDATE INPUTS BY DTYPE DUE TO LACK OF INFORMATION FROM MODEL */

  TritonJsonValue ios;
  if (auto err = model_->ModelConfig().MemberAsArray("input", &ios)) {
    DEBUG_TRACE_ERROR(
        "Failed to get model \"" << Name() << "\" input configuration: "
                                 << TRITONSERVER_ErrorMessage(err));
    THROW_TRITON_EXCEPTION(
        err, "Failed to get model \"" << Name() << "\" input configuration: "
                                      << TRITONSERVER_ErrorMessage(err));
  }

  if (ios.ArraySize() != expected_input_count) {
    DEBUG_TRACE_ERROR(
        "Failed to load model \""
        << Name() << "\" configuration expects " << expected_input_count
        << " inputs, but model configuration has " << ios.ArraySize()
        << " inputs.");
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to load model \""
            << Name() << "\" configuration expects " << expected_input_count
            << " inputs, but model configuration has " << ios.ArraySize()
            << " inputs.");
  }

  auto naming_convention = GetNamingConvention(allowed_inputs);

  for (size_t i = 0; i < ios.ArraySize(); i += 1) {
    TritonJsonValue io;
    if (auto err = ios.IndexAsObject(i, &io)) {
      DEBUG_TRACE_ERROR(
          "Failed to get input " << i << " for model instance \"" << Name()
                                 << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get input "
                   << i << " for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    std::string io_name;
    if (auto err = io.MemberAsString("name", &io_name)) {
      DEBUG_TRACE_ERROR(
          "Failed to get name for input "
          << i << " for model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get name for input "
                   << i << " for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    AddInputToMap(
        /* naming_convention= */ naming_convention,
        /* allowed_inputs= */ allowed_inputs,
        /* io_name= */ io_name,
        /* index= */ i);

    // Validate dtype
    std::string io_dtype;
    if (auto err = io.MemberAsString("data_type", &io_dtype)) {
      DEBUG_TRACE_ERROR(
          "Failed to get data type for input \""
          << io_name << "\" for model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get data type for input \""
                   << io_name << "\" for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    const auto pr = ModelConfigDataTypeToTorchType(io_dtype);
    if (!pr.first && (io_dtype != "TYPE_STRING")) {
      DEBUG_TRACE_ERROR(
          "Unsupported datatype " << io_dtype << " for input \"" << io_name
                                  << "\" for model instance \"" << Name()
                                  << "\".");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Unsupported datatype " << io_dtype << " for input \"" << io_name
                                  << "\" for model instance \"" << Name()
                                  << "\".");
    }

    // Validate shape for String inputs. Only allow 1 dimension.
    if (io_dtype == "TYPE_STRING") {
      // If a reshape is provided for the input then use that when validating
      // the model shapes.
      std::vector<int64_t> dims;
      TritonJsonValue reshape;
      if (io.Find("reshape", &reshape)) {
        if (auto err = ParseShape(reshape, "shape", &dims)) {
          DEBUG_TRACE_ERROR(
              "Failed to parse reshape dims for input \""
              << io_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to parse reshape shape for input \""
                       << io_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }
      } else {
        if (auto err = ParseShape(io, "dims", &dims)) {
          DEBUG_TRACE_ERROR(
              "Failed to parse dims for input \""
              << io_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to parse dims for input \""
                       << io_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }
      }
    }
  }

  TritonJsonValue sequence_batching;
  if (model_->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    TritonJsonValue states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i += 1) {
        TritonJsonValue state;
        if (auto err = states.IndexAsObject(i, &state)) {
          DEBUG_TRACE_ERROR(
              "Failed to get sequence state "
              << i << " for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to get sequence state "
                       << i << " for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        std::string state_name;
        if (auto err = state.MemberAsString("input_name", &state_name)) {
          DEBUG_TRACE_ERROR(
              "Failed to get input name for sequence state "
              << i << " for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to get input name for sequence state "
                       << i << " for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        AddInputToMap(naming_convention, allowed_inputs, state_name, i);

        // Validate dtype
        std::string state_dtype;
        if (auto err = state.MemberAsString("data_type", &state_dtype)) {
          DEBUG_TRACE_ERROR(
              "Failed to get data type for sequence state input \""
              << state_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to get data type for sequence state input \""
                       << state_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        DEBUG_TRACE_INFO(
            "{ name: \"" << Name() << "\""
                         << ", state_name: \"" << state_name << "\""
                         << ", state_dtype: \"" << state_dtype << "\""
                         << " }");

        const auto pr = ModelConfigDataTypeToTorchType(state_dtype);
        if (!pr.first) {
          DEBUG_TRACE_ERROR(
              "Unsupported datatype "
              << state_dtype << " for sequence state input \"" << state_name
              << "\" for model instance \"" << Name() << "\".");
          THROW_TRITON_EXCEPTION(
              TRITONSERVER_ERROR_INTERNAL,
              "Unsupported datatype "
                  << state_dtype << " for sequence state input \"" << state_name
                  << "\" for model instance \"" << Name() << "\".");
        }

        // Validate shape for String inputs. Only allow 1 dimension.
        if (state_dtype == "TYPE_STRING") {
          if (is_batching_supported_) {
            DEBUG_TRACE_ERROR(
                "Triton only supports 1-dimensional string inputs for sequence "
                "state input \""
                << state_name << "\" for model \"" << Name()
                << "\" when batching is enabled.");
            THROW_TRITON_EXCEPTION(
                TRITONSERVER_ERROR_INVALID_ARG,
                "Triton only supports 1-dimensional string inputs for sequence "
                "state input \""
                    << state_name << "\" for model \"" << Name() << "\".");
          }
        }
      }
    }
  }

  uint32_t i = 0;
  for (const auto& batch_input : model_->BatchInputs()) {
    for (const auto& input_name : batch_input.TargetNames()) {
      DEBUG_TRACE_INFO(
          "{ name: \"" << Name() << "\""
                       << ", batch_input_name: \"" << input_name << "\""
                       << ", index: " << i + ios.ArraySize()
                       << ", naming_convention: " << naming_convention
                       << ", len(allowed_inputs): " << allowed_inputs.size()
                       << " }");

      AddInputToMap(
          naming_convention, allowed_inputs, input_name, i + ios.ArraySize());
      i += 1;
    }
  }

  DEBUG_TRACE_INFO(
      "{ name: \"" << Name() << "\""
                   << ", total_batch_input_target_names: " << i
                   << ", batch_input_count: " << batch_input_count_ << " }");
}

void
InductorModelInstance::ValidateOutputs()
{
  DEBUG_TRACE_FUNCTION_CALL();
  TritonJsonValue ios;
  if (auto err = model_->ModelConfig().MemberAsArray("output", &ios)) {
    DEBUG_TRACE_ERROR(
        "Failed to get model \"" << Name() << "\" output configuration: "
                                 << TRITONSERVER_ErrorMessage(err));
    THROW_TRITON_EXCEPTION(
        err, "Failed to get model \"" << Name() << "\" output configuration: "
                                      << TRITONSERVER_ErrorMessage(err));
  }

  int op_index{0};

  if (ios.ArraySize() == 0) {
    DEBUG_TRACE_ERROR(
        "Configuration for model \""
        << Name()
        << "\" must define at least one output, none were specified.");
    THROW_TRITON_EXCEPTION(
        TRITONSERVER_ERROR_INTERNAL,
        "Configuration for model \""
            << Name()
            << "\" must define at least one output, none were specified.");
  }

  auto naming_convention = GetNamingConvention({});

  for (size_t i = 0; i < ios.ArraySize(); i++) {
    TritonJsonValue io;
    if (auto err = ios.IndexAsObject(i, &io)) {
      DEBUG_TRACE_ERROR(
          "Failed to get output " << i << " for model instance \"" << Name()
                                  << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get output "
                   << i << " for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    // Validate name
    std::string io_name;
    if (auto err = io.MemberAsString("name", &io_name)) {
      DEBUG_TRACE_ERROR(
          "Failed to get name for output "
          << i << " for model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get name for output "
                   << i << " for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    switch (naming_convention) {
      case TritonNamingConvention::NAMED_INDEX: {
        int start_pos = io_name.find(DELIMINATOR);
        op_index = std::atoi(io_name.substr(start_pos + 2).c_str());
      } break;

      case TritonNamingConvention::STRICT_CONFIG_ORDERING: {
        op_index = i;
      } break;

      default:
        break;
    }

    DEBUG_TRACE_INFO(
        "{ name: \"" << Name() << "\""
                     << ", io_name: \"" << io_name << "\""
                     << ", index: " << i << ", naming_convention: "
                     << naming_convention << ", result: NAMED_INDEX"
                     << ", op_index: " << op_index << " }");

    // Validate data type.
    std::string io_dtype;
    if (auto err = io.MemberAsString("data_type", &io_dtype)) {
      DEBUG_TRACE_ERROR(
          "Failed to get data type for output \""
          << io_name << "\" for model instance \"" << Name()
          << "\": " << TRITONSERVER_ErrorMessage(err));
      THROW_TRITON_EXCEPTION(
          err, "Failed to get data type for output \""
                   << io_name << "\" for model instance \"" << Name()
                   << "\": " << TRITONSERVER_ErrorMessage(err));
    }

    const auto torch_type = ModelConfigDataTypeToTorchType(io_dtype);
    if (!torch_type.first && (io_dtype != "TYPE_STRING")) {
      DEBUG_TRACE_ERROR(
          "Unsupported datatype " << io_dtype << " for output \"" << io_name
                                  << "\" for model instance \"" << Name()
                                  << "\".");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Unsupported datatype " << io_dtype << " for output \"" << io_name
                                  << "\" for model instance \"" << Name()
                                  << "\".");
    }

    // Validate shape for String outputs. Only allow 1 dimension.
    if (io_dtype == "TYPE_STRING") {
      std::vector<int64_t> dims;
      TritonJsonValue reshape;

      if (io.Find("reshape", &reshape)) {
        if (auto err = ParseShape(reshape, "shape", &dims)) {
          DEBUG_TRACE_ERROR(
              "Failed to parse reshape shape for output \""
              << io_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to parse reshape shape for output \""
                       << io_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        DEBUG_TRACE_INFO(
            "{ name: \"" << Name() << "\""
                         << ", io_name: \"" << io_name << "\""
                         << ", op_index: " << op_index << ", io_dtype: \""
                         << io_dtype << "\""
                         << ", reshape: true, len(dims): " << dims.size()
                         << " }");
      } else {
        if (auto err = ParseShape(io, "dims", &dims)) {
          DEBUG_TRACE_ERROR(
              "Failed to parse dims for output \""
              << io_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to parse dims for output \""
                       << io_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        DEBUG_TRACE_INFO(
            "{ name: \"" << Name() << "\""
                         << ", io_name: \"" << io_name << "\""
                         << ", op_index: " << op_index << ", io_dtype: \""
                         << io_dtype << "\""
                         << ", reshape: false, len(dims): " << dims.size()
                         << " }");
      }

      if ((dims.size() + (is_batching_supported_ ? 1 : 0)) > 1) {
        DEBUG_TRACE_ERROR(
            "Triton only supports 1 dimensional List of String as output for \""
            << io_name << "\" for model instance \"" << Name() << "\".");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ERROR_INTERNAL,
            "Triton only supports 1 dimensional List of String as output for"
            " \""
                << io_name << "\" for model instance \"" << Name() << "\".");
      }
    }

    output_index_map_[io_name] = op_index;
    output_dtype_map_[io_name] = ConvertTorchTypeToDataType(torch_type.second);

    DEBUG_TRACE_INFO(
        "{ name: \"" << Name() << "\""
                     << ", io_name: \"" << io_name << "\""
                     << ", op_index: " << output_index_map_[io_name]
                     << ", output_dtype: \""
                     << TRITONSERVER_DataTypeString(output_dtype_map_[io_name])
                     << "\" }");
  }

  TritonJsonValue sequence_batching;
  if (model_->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    TritonJsonValue states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i += 1) {
        TritonJsonValue state;
        if (auto err = states.IndexAsObject(i, &state)) {
          DEBUG_TRACE_ERROR(
              "Failed to get sequence state "
              << i << " for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to get sequence state "
                       << i << " for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        std::string state_name;
        if (auto err = state.MemberAsString("output_name", &state_name)) {
          DEBUG_TRACE_ERROR(
              "Failed to get output name for sequence state "
              << i << " for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to get output name for sequence state "
                       << i << " for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        std::string state_dtype;
        if (auto err = state.MemberAsString("data_type", &state_dtype)) {
          DEBUG_TRACE_ERROR(
              "Failed to get data type for sequence state output \""
              << state_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to get data type for sequence state output \""
                       << state_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        std::vector<int64_t> dims;
        if (auto err = ParseShape(state, "dims", &dims)) {
          DEBUG_TRACE_ERROR(
              "Failed to parse dims for sequence state output \""
              << state_name << "\" for model instance \"" << Name()
              << "\": " << TRITONSERVER_ErrorMessage(err));
          THROW_TRITON_EXCEPTION(
              err, "Failed to parse dims for sequence state output \""
                       << state_name << "\" for model instance \"" << Name()
                       << "\": " << TRITONSERVER_ErrorMessage(err));
        }

        DEBUG_TRACE_INFO(
            "{ name: \"" << Name() << "\""
                         << ", op_index: " << op_index << ", states[" << i
                         << "]: { state_name: \"" << state_name << "\""
                         << ", state_dtype: \"" << state_dtype << "\""
                         << ", len(dims): " << dims.size() << " } }");

        int start_pos = state_name.find(DELIMINATOR);
        op_index = std::atoi(state_name.substr(start_pos + 2).c_str());

        const auto pr = ModelConfigDataTypeToTorchType(state_dtype);
        if (!pr.first && state_dtype != "TYPE_STRING") {
          DEBUG_TRACE_ERROR(
              "Unsupported datatype "
              << state_dtype << " for sequence state output \"" << state_name
              << "\" for model instance \"" << Name() << "\".");
          THROW_TRITON_EXCEPTION(
              TRITONSERVER_ERROR_INTERNAL, "Unsupported datatype "
                                               << state_dtype
                                               << " for sequence state output "
                                                  "\""
                                               << state_name
                                               << "\" for model instance \""
                                               << Name() << "\".");
        }

        // Validate shape for String outputs. Only allow 1 dimension.
        if (state_dtype == "TYPE_STRING") {
          if ((dims.size() + (is_batching_supported_ ? 1 : 0)) > 1) {
            DEBUG_TRACE_ERROR(
                "Triton only supports 1-dimensional string outputs for "
                "sequence state output \""
                << state_name << "\" for model instance \"" << Name() << "\".");
            THROW_TRITON_EXCEPTION(
                TRITONSERVER_ERROR_INTERNAL,
                "Triton only supports 1-dimensional string outputs for "
                "sequence state output \""
                    << state_name << "\" for model instance \"" << Name()
                    << "\".");
          }
        }

        output_index_map_[state_name] = op_index;
        output_dtype_map_[state_name] = ConvertTorchTypeToDataType(pr.second);
        DEBUG_TRACE_INFO(
            "{ name: \"" << Name() << "\""
                         << ", state_name: \"" << state_name << "\""
                         << ", op_index: " << output_index_map_[state_name]
                         << ", torch_type: " << static_cast<int>(pr.second)
                         << ", output_dtype: \""
                         << TRITONSERVER_DataTypeString(
                                output_dtype_map_[state_name])
                         << "\" }");
      }
    }
  }
}

bool
InductorModelInstance::ValidateTypedSequenceControl(
    TritonJsonValue& sequence_batching, const std::string& control_kind,
    bool required)
{
  DEBUG_TRACE_FUNCTION_CALL();

  std::string tensor_name;
  std::string tensor_dtype;

  if (auto err = GetTypedSequenceControlProperties(
          sequence_batching, model_->Name(), control_kind, required,
          &tensor_name, &tensor_dtype)) {
    DEBUG_TRACE_ERROR(
        "Error validating typed sequence control for model instance \""
        << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
    THROW_TRITON_EXCEPTION(
        err, "Failed to validate typed sequence control for model instance \""
                 << Name() << "\": " << TRITONSERVER_ErrorMessage(err));
  }

  bool have_control{!tensor_name.empty()};
  DEBUG_TRACE_INFO(
      "{ name: \"" << Name() << "\", control_kind: \"" << control_kind
                   << "\", tensor_name: \"" << tensor_name
                   << "\", tensor_dtype: \"" << tensor_dtype
                   << "\", have_control: " << (have_control ? "true" : "false")
                   << " }");
  if (have_control) {
    int input_index{0};
    int start_pos{static_cast<int>(tensor_name.find(DELIMINATOR))};

    if (start_pos == -1) {
      DEBUG_TRACE_ERROR(
          "Input \""
          << tensor_name
          << "\" does not follow <name>__<index> naming convention.");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Input \""
              << tensor_name
              << "\" does not follow <name>__<index> naming convention.");
    }

    // Check if the index part of the name is not an integer.
    std::string index_str{tensor_name.substr(start_pos + 2)};
    for (auto itr = index_str.begin(); itr != index_str.end(); itr++) {
      if (std::isdigit(*itr) == 0) {
        DEBUG_TRACE_ERROR(
            "Input \""
            << tensor_name
            << "\" does not follow <name>__<index> naming convention.");
        THROW_TRITON_EXCEPTION(
            TRITONSERVER_ERROR_INTERNAL,
            "Input \""
                << tensor_name
                << "\" does not follow <name>__<index> naming convention.");
      }
    }

    // Check if the data type is supported by PyTorch.
    if (!ModelConfigDataTypeToTorchType(tensor_dtype).first) {
      DEBUG_TRACE_ERROR(
          "Unsupported datatype " << tensor_dtype
                                  << " for typed sequence control input"
                                     " \""
                                  << tensor_name << "\" for model instance \""
                                  << Name() << "\".");
      THROW_TRITON_EXCEPTION(
          TRITONSERVER_ERROR_INTERNAL,
          "Unsupported datatype " << tensor_dtype
                                  << " for typed sequence control input"
                                     " \""
                                  << tensor_name << "\" for model instance \""
                                  << Name() << "\".");
    }

    input_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
    input_index_map_[tensor_name] = input_index;
    DEBUG_TRACE_INFO(
        "{ name: \"" << Name() << "\", control_kind: \"" << control_kind
                     << "\", tensor_name: \"" << tensor_name
                     << "\", tensor_dtype: \"" << tensor_dtype
                     << "\", have_control: true, index: "
                     << input_index_map_[tensor_name] << " }");
  }

  return have_control;
}
}  // namespace triton::backend::pytorch
