// Copyright 2025, NVIDIA CORPORATION&  AFFILIATES. All rights reserved.
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
#include "libtorch_utils.h"
#include "string_utils.hh"
#include "triton_utils.hh"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/common/nvtx.h"

namespace triton::backend::pytorch
{
  InductorModelInstance::InductorModelInstance(
      std::shared_ptr<pytorch::InductorModel> model,
      TRITONBACKEND_ModelInstance *triton_model_instance)
    : BackendModelInstance{model.get(), triton_model_instance}
    , model_{model}
  {
#ifdef TRITON_ENABLE_GPU
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU)
    {
      device_ = torch::Device(torch::kCUDA, DeviceId());
      CreateCudaEvents(DeviceId());
      device_count_ = model_->DeviceCount();
    }
#endif

    model_->LoadModel(ArtifactFilename(), device_, device_count_, Kind());

#ifdef TRITON_ENABLE_GPU
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU)
    {
      // Since we cannot determine the exact devices used by the model, we create
      // a CUDA stream for every available device to ensure proper synchronization
      // of CUDA streams. This approach may have implications when a timestamp is
      // captured on a device that is not used by the model. Currently, this issue
      // is addressed by synchronizing the CUDA streams before recording
      // timestamps to prevent timestamp skewing. However, in the future, any
      // modifications to the CUDA stream synchronization logic should be handled
      // with caution.
      for (int i = 0; i < device_cnt_; i++)
      {
        triton::backend::cudaStream_t stream;
        if (auto err = CreateCudaStream(i, 0 /* cuda_stream_priority */,& stream))
          throw triton::backend::BackendModelInstanceException(err);

        stream_vec_.push_back(stream);
      }

      if (!stream_vec_.empty())
      {
        // Create CUDA events on the first device that will be used for collecting
        // inputs/outputs.
        CreateCudaEvents(0);
      }
    }
#endif

    size_t expected_input_count{0};

    triton::common::TritonJson::Value inputs;
    if (model_->ModelConfig().Find("input", &inputs))
    {
      expected_input_count = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (model_->ModelConfig().Find("batch_input", &config_batch_inputs))
    {
      batch_input_count_ = config_batch_inputs.ArraySize();
      expected_input_count += batch_input_count_;
    }

    // If this is a sequence model then make sure that the required inputs are
    // present in the model and have the correct shape and datatype.
    triton::common::TritonJson::Value sequence_batching;
    if (model_->ModelConfig().Find("sequence_batching", &sequence_batching))
    {
      if (ValidateBooleanSequenceControl(sequence_batching, "CONTROL_SEQUENCE_START", false /* required */))
      {
        expected_input_count += 1;
      }
      if (ValidateBooleanSequenceControl(sequence_batching, "CONTROL_SEQUENCE_END", false /* required */))
      {
        expected_input_count += 1;
      }
      if (ValidateBooleanSequenceControl(sequence_batching, "CONTROL_SEQUENCE_READY", false /* required */))
      {
        expected_input_count += 1;
      }
      if (ValidateBooleanSequenceControl(sequence_batching, "CONTROL_SEQUENCE_CORRID", false /* required */))
      {
        expected_input_count += 1;
      }

      // Add the state inputs to the expected count.
      triton::common::TritonJson::Value states;
      if (sequence_batching.Find("state", &states))
      {
        expected_input_count += states.ArraySize();
      }
    }

    is_batching_supported_ = model_->MaxBatchSize() > 0;

    ValidateInputs(expected_input_count);
    ValidateOutputs();
  }

  InductorModelInstance::~InductorModelInstance()
  {
    model_.reset();
    ClearCache();

#ifdef TRITON_ENABLE_GPU
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL)
    {
      for (size_t i = 0; i < stream_vec_.size(); i += 1)
      {
        if (auto err = ConvertCUDAStatusToTritonError(cudaSetDevice(i),
                                                      TRITONSERVER_ERROR_INTERNAL,
                                                      "Failed to set device for stream destruction"))
        {
          TRITON_LOG_ERROR("Failed to set the device" << " while destroying streams for instance '" << Name()
                           << "': " << TRITONSERVER_ErrorMessage(err));
        }

        if (auto err = ConvertCUDAStatusToTritonError(cudaStreamDestroy(stream_vec_[i]),
                                                      TRITONSERVER_ERROR_INTERNAL,
                                                      "Failed to destroy cuda stream"))
        {
          TRITON_LOG_ERROR("Failed to destroy cuda stream" << " for instance '" << Name()
                           << "': " << TRITONSERVER_ErrorMessage(err));
        }

        stream_vec_[i] = nullptr;
      }
    }
#endif
  }

  void
  InductorModelInstance::AddInputToMap(
      pytorch::NamingConvention naming_convention,
      const std::vector<std::string>& allowed_inputs,
      const std::string& io_name,
      uint32_t index)
  {
    std::string deliminator{"__"};

    if (is_dictionary_input_)
    {
      input_index_map_[io_name] = index;
    }
    else
    {
      switch (naming_convention)
      {
        case pytorch::NamingConvention::FORWARD_ARGUMENT:
        {
          auto it = std::find(allowed_inputs.begin(), allowed_inputs.end(), io_name);
          if (it != allowed_inputs.end())
          {
            input_index_map_[io_name] = std::distance(allowed_inputs.begin(), it);
          }
        }
        break;

        case pytorch::NamingConvention::NAMED_INDEX:
        {
          int start_pos = io_name.find(deliminator);
          int ip_index = std::atoi(io_name.substr(start_pos + 2).c_str());
          input_index_map_[io_name] = ip_index;
        }
        break;

        case pytorch::NamingConvention::STRICT_CONFIG_ORDERING:
        {
          input_index_map_[io_name] = index;
        }
        break;

        default:
          THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INVALID_ARG,
                                "Argument 'naming_convention' value of " << static_cast<uint32_t>(naming_convention)
                                 << " is invalid or unsupported.");
      }
    }
  }

  const std::string&
  InductorModelInstance::ArtifactFilename() const
  {
    return BackendModelInstance::ArtifactFilename();
  }

  void
  InductorModelInstance::ClearCache()
  {
#ifdef TRITON_ENABLE_GPU
    if (device_.is_cuda() ||
        (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0))
    {
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
#endif
  }

  std::shared_ptr<pytorch::InductorModelInstance>
  InductorModelInstance::Create(
      std::shared_ptr<pytorch::InductorModel> model,
      TRITONBACKEND_ModelInstance *triton_model_instance)
  {
    try
    {
      return std::make_shared<InductorModelInstance>(model, triton_model_instance);
    }
    catch (const triton::backend::pytorch::BackendException& exception)
    {
      THROW_TRITON_EXCEPTION(exception.error_code(),
                             "Failed to create InductorModelInstance for model '" << model->Name()
                             << "': " << exception.what());
    }
    catch (const std::exception& exception)
    {
      THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                             "Failed to create InductorModelInstance for model '" << model->Name()
                             << "': " << exception.what());
    }
  }

  void
  InductorModelInstance::CreateCudaEvents(
      int32_t device_id)
  {
#ifdef TRITON_ENABLE_GPU
    // Need to set the CUDA context so that the context that events are created on match with contexts that events
    // are recorded with.
    if (auto err = cudaSetDevice(device_id))
    {
      THROW_TRITON_EXCEPTION(err,
                             "When creating CUDA events, failed to set the device for model instance '" << Name()
                             << "' : " << cudaGetErrorString(err));
    }

    if (auto err = cudaEventCreate(&compute_input_start_event_))
    {
      THROW_TRITON_EXCEPTION(err,
                             "When creating CUDA events, failed to create compute input start event for model instance '"
                             << Name() << "' : " << cudaGetErrorString(err));
    }

    if (auto err = cudaEventCreate(&compute_infer_start_event_))
    {
      THROW_TRITON_EXCEPTION(err,
                             "When creating CUDA events, failed to create compute infer start event for model instance '"
                             << Name() << "' : " << cudaGetErrorString(err));
    }

    if (auto err = cudaEventCreate(&compute_output_start_event_))
    {
      THROW_TRITON_EXCEPTION(err,
                             "When creating CUDA events, failed to create compute output start event for model instance '"
                             << Name() << "' : " << cudaGetErrorString(err));
    }
#endif
  }

  triton::backend::cudaStream_t
  InductorModelInstance::CudaStream()
  {
    return BackendModelInstance::CudaStream();
  }

  int32_t
  InductorModelInstance::DeviceId() const
  {
    return BackendModelInstance::DeviceId();
  }

  void
  InductorModelInstance::Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      uint32_t response_count,
      std::vector<torch::Tensor>& input_tensors,
      std::vector<torch::Tensor>& output_tensors)
  {
    NVTX_RANGE(nvtx_, "Execute " + Name());

    std::vector<torch::Tensor> model_outputs;

    try
    {
      // Enable/disable inference mode based on the model setting.
      // Supersedes NoGradGuard.
      torch::InferenceMode guard{model_->InferenceModeEnabled()};

      // Enable/disable cuDNN.
      at::globalContext().setUserEnabledCuDNN(model_->CudnnEnabled());

      torch::NoGradGuard no_grad_guard;

      if (is_dictionary_input_)
      {
        // NO SUPPORT FOR DICTIONARY INPUTS YET
        // torch::Dict<std::string, torch::Tensor> dict_input;
        // for (auto& input_index : input_index_map_)
        // {
        //   const std::string& input_name = input_index.first;
        //   uint32_t model_input_index = input_index.second;

        //   dict_input.insert(input_name, input_tensors.at(model_input_index));
        // }

        // std::vector<torch::Tensor> model_inputs = {dict_input};
        // model_outputs = model_->Forward(model_inputs);
      }
      else
      {
        model_outputs = model_->Forward(input_tensors);
      }

      if (model_outputs.size() == 1)
      {
        output_tensors.push_back(model_outputs[0]);
      }
      else
      {
        for (auto& output_tensor : model_outputs)
        {
          output_tensors.push_back(output_tensor);
        }
      }
    }
    catch (const std::exception& exception)
    {
      SendErrorForResponses(responses,
                            response_count,
                            TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                                  std::string("Inductor model instance '" + Name()
                                                  + "' execution failure: " + exception.what()).c_str()));
    }
  }

  float
  InductorModelInstance::GetCudaEventElapsedTime(
      const triton::backend::cudaEvent_t& start_event,
      const triton::backend::cudaEvent_t& end_event)
  {
    float duration{0};
#ifdef TRITON_ENABLE_GPU
    if (auto err = ConvertCUDAStatusToTritonError(cudaEventElapsedTime(&duration, start_event, end_event),
                                                                       TRITONSERVER_ERROR_INTERNAL,
                                                                       "Failed to capture elapsed time"))
   {
      TRITON_LOG_ERROR("Failed to capture elapsed time" << " for instance '" << Name()
                       << "': " << TRITONSERVER_ErrorMessage(err));
    }
#endif

    return duration;
  }

  triton::backend::cudaStream_t
  InductorModelInstance::GetCudaStreamByInstanceKind()
  {
#ifdef TRITON_ENABLE_GPU
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU)
      return stream_;

    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && !stream_vec_.empty())
      return stream_vec_[0];
#endif

    return nullptr;
  }

  pytorch::NamingConvention
  InductorModelInstance::GetNamingConvention(
      const std::vector<std::string>& allowed_ios)
  {
    // Rules for (non-Dictionary) input tensor names:
    // 1. Must be in 'allowed_inputs' (arguments in the forward function)
    // 2. Must follow the naming convention i.e. <name>__<index>
    // 3. If neither of the above conditions are satisfied, enforce strict
    // ordering of model inputs.
    //
    // Rules for output tensor names:
    // 1. Must follow the naming convention i.e. <name>__<index>
    // 2. If not, we enforce strict ordering of model outputs.
    std::string deliminator{"__"};
    std::string io_kind{"input"};
    NamingConvention naming_convention{NamingConvention::FORWARD_ARGUMENT};

    if (allowed_ios.size() == 0)
    {
      io_kind = "output";
      naming_convention = NamingConvention::NAMED_INDEX;
    }

    triton::common::TritonJson::Value ios;
    if (auto err = model_->ModelConfig().MemberAsArray(io_kind.c_str(),& ios))
    {
      THROW_TRITON_EXCEPTION(err,
                             "Failed to get " << io_kind << " array from model config for model '"
                             << model_->Name() << "'.");
    }

    if (io_kind == "input")
    {
      for (size_t i = 0; i < ios.ArraySize(); i += 1)
      {
        triton::common::TritonJson::Value io;
        if (auto err = ios.IndexAsObject(i,& io))
        {
          THROW_TRITON_EXCEPTION(err,
                                "Failed to get " << io_kind << " object at index " << i
                                << " from model config for model '" << model_->Name() << "'.");
        }

        // Validate name.
        std::string io_name;
        if (auto err = io.MemberAsString("name", &io_name))
        {
          THROW_TRITON_EXCEPTION(err,
                                 "Failed to get " << io_kind << " name at index " << i
                                 << " from model config for model '" << model_->Name() << "'.");
        }
        auto it = std::find(allowed_ios.begin(), allowed_ios.end(), io_name);
        if (it == allowed_ios.end())
        {
          naming_convention = NamingConvention::NAMED_INDEX;
          break;
        }
      }
    }

    if (naming_convention == NamingConvention::NAMED_INDEX)
    {
      for (size_t i = 0; i < ios.ArraySize(); i += 1)
      {
        triton::common::TritonJson::Value io;
        if (auto err = ios.IndexAsObject(i,& io))
        {
          THROW_TRITON_EXCEPTION(err,
                                 "Failed to get " << io_kind << " object at index " << i
                                 << " from model config for model '" << model_->Name() << "'.");
        }

        // Validate name.
        std::string io_name;
        if (auto err = io.MemberAsString("name",& io_name))
        {
          THROW_TRITON_EXCEPTION(err,
                                 "Failed to get " << io_kind << " name at index " << i
                                 << " from model config for model '" << model_->Name() << "'.");
        }

        int start_pos = io_name.find(deliminator);
        if (start_pos == -1)
        {
          naming_convention = NamingConvention::STRICT_CONFIG_ORDERING;
          break;
        } else
        {
          // check if the index part of the name is not an integer
          std::string index_str = io_name.substr(start_pos + 2);
          bool is_int{true};
          for (auto it = index_str.begin(); it != index_str.end(); it++)
          {
            if (std::isdigit(*it) == 0)
            {
              is_int = false;
            }
          }

          if (!is_int)
          {
            if (io_kind == "input")
            {
              TRITON_LOG_WARN("Input '" << io_name << "' or previous input(s) are neither an input argument to "
                              "the model '" << model_->Name() << "' nor follow the <name>__<index> naming convention. "
                              "Defaulting to strict ordering of model inputs.");
            }
            else
            {
              TRITON_LOG_WARN("Output '" << io_name << "' or previous output(s) of the model '" << model_->Name()
                              << "' do not follow the <name>__<index> naming convention. "
                              "Defaulting to strict ordering of model outputs.");
            }

            naming_convention = NamingConvention::STRICT_CONFIG_ORDERING;
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
    return BackendModelInstance::HostPolicyName();
  }

  std::shared_ptr<pytorch::InductorModel>
  InductorModelInstance::InductorModel() const
  {
    return model_;
  }

  TRITONSERVER_InstanceGroupKind
  InductorModelInstance::Kind() const
  {
    return BackendModelInstance::Kind();
  }

  triton::backend::BackendModel*
  InductorModelInstance::Model() const
  {
    return BackendModelInstance::Model();
  }

  const std::string&
  InductorModelInstance::Name() const
  {
    return BackendModelInstance::Name();
  }

  void
  InductorModelInstance::ProcessRequests(
      TRITONBACKEND_Request **requests,
      const uint32_t request_count)
  {
    TRITON_LOG_VERBOSE("TRITONBACKEND_ModelExecute: Running " << Name() << " with " << request_count << " requests");

#ifdef TRITON_ENABLE_GPU
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU)
    {
      SetCurrentCudaStream(stream_, DeviceId());
    }
    else if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL)
    {
      // Replace the default stream of each device with the one we created.
      for (size_t i = 0; i < stream_vector_.size(); i += 1)
      {
        SetCurrentCudaStream(stream_vector_[i], DeviceId());
      }
    }
#endif

    NVTX_RANGE(nvtx_, "ProcessRequests " + Name());

    uint64_t exec_start_ns{0};
    SET_TIMESTAMP(exec_start_ns);

    const int max_batch_size = model_->MaxBatchSize();

    // For each request collect the total batch size for this inference execution.
    // The batch-size, number of inputs, and size of each input has already been
    // checked so don't need to do that here.
    size_t total_batch_size{0};
    for (size_t i = 0; i < request_count; i += 1)
    {
      // If we get a nullptr request then something is badly wrong. Fail and
      // release all requests.
      if (!requests[i])
      {
        RequestsRespondWithError(requests,
                                 request_count,
                                 TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                                       std::string("NULL request given to PyTorch backend for '"
                                                       + Name() + "'.").c_str()));
        return;
      }
    }

    std::vector<TRITONBACKEND_Response*> responses;
    responses.reserve(request_count);
    bool all_response_failed{false};

    for (size_t i = 0; i < request_count; i += 1)
    {
      TRITONBACKEND_Response *response;
      if (auto err = TRITONBACKEND_ResponseNew(&response, requests[i]))
      {
        responses.emplace_back(nullptr);
        TRITON_LOG_ERROR("Failed to create response" << " for request " << i << " of model instance '" << Name()
                         << "': " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
      }
      else
      {
        responses.emplace_back(response);
      }
    }

    for (size_t i = 0; i < request_count; i += 1)
    {
      if (max_batch_size > 0)
      {
        TRITONBACKEND_Input *input{nullptr};
        auto err = TRITONBACKEND_RequestInputByIndex(requests[i], 0,& input);
        if (!err)
        {
          const int64_t *shape;
          err = TRITONBACKEND_InputProperties(input, nullptr, nullptr,& shape, nullptr, nullptr, nullptr);
          total_batch_size += shape[0];
        }

        if (err)
        {
          RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count, all_response_failed, err);
        }
      }
      else
      {
        total_batch_size += 1;
      }
    }

    if (total_batch_size == 0)
      return;

    if (!all_response_failed)
    {
      if (total_batch_size != 1 && total_batch_size > (size_t)max_batch_size)
      {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses,
                                          request_count,
                                          all_response_failed,
                                          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                                                std::string("batch size "
                                                                            + std::to_string(total_batch_size)
                                                                            + " for '" + Name() + "', max allowed is "
                                                                            + std::to_string(max_batch_size)).c_str()));
      }
    }

    std::vector<const char *> input_names;
    std::vector<torch::Tensor> input_tensors;
    bool cuda_copy{false};
    std::unique_ptr<BackendInputCollector> collector;

#ifdef TRITON_ENABLE_GPU
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU ||
        (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0))
    {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses,
                                        request_count,
                                        all_response_failed,
                                        ConvertCUDAStatusToTritonError(cudaEventRecord(compute_input_start_event_,
                                                                                       GetCudaStreamByInstanceKind()),
                                                                       TRITONSERVER_ERROR_INTERNAL,
                                                                       "Failed to record the event."));
    }
#endif

    if (!all_response_failed)
    {
      collector.reset(new BackendInputCollector(requests,
                                                request_count,
                                                &responses,
                                                model_->TritonMemoryManager(),
                                                model_->EnablePinnedInput(),
                                                GetCudaStreamByInstanceKind(),
                                                nullptr,
                                                nullptr,
                                                0,
                                                HostPolicyName().c_str()));
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses,
                                        request_count,
                                        all_response_failed,
                                        SetInputTensors(total_batch_size,
                                                        requests,
                                                        request_count,
                                                        &responses,
                                                        collector.get(),
                                                        &input_names,
                                                        &input_tensors,
                                                        &cuda_copy));
    }

#ifdef TRITON_ENABLE_GPU
    if (cuda_copy)
    {
      cudaStreamSynchronize(GetCudaStreamByInstanceKind());
      cuda_copy = false;
    }
#endif

    std::vector<torch::Tensor> output_tensors;
    uint64_t compute_start_ns{0};
    uint64_t compute_infer_start{0};

    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses,
                                      request_count,
                                      all_response_failed,
                                      RecordBackendTimestamp(&compute_start_ns,
                                                             reinterpret_cast<void *>(&compute_infer_start_event_)));

    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0)
    {
      SET_TIMESTAMP(compute_infer_start);
    }

    if (!all_response_failed)
    {
      Execute(&responses, request_count, input_tensors, output_tensors);
    }

    bool invalid_index{false};
    int max_index{output_tensors.size() - 1};

    if (!all_response_failed)
    {
      for (const auto& name : model_->ModelOutputs())
      {
        int op_index = output_index_map_[name.first];
        if (op_index < 0 || op_index > max_index)
        {
          RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses,
                                            request_count,
                                            all_response_failed,
                                            TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                                                  std::string("The output " + std::string(name.first) + " in the model configuration refers to "
                                                                              " an output indexwhich doesn't exist. This model has " + std::to_string(max_index)
                                                                              + " outputs.").c_str()));
          invalid_index = true;
          break;
        }
      }
    }

#ifdef TRITON_ENABLE_GPU
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL)
    {
      // For 'KIND_MODEL', multiple streams will be involved, so we need to call
      // 'cudaStreamSynchronize' before reading the output tensors.
      for (auto& stream : stream_vector_)
      {
        cudaStreamSynchronize(stream);
      }
    }
#endif

    uint64_t compute_end_ns{0};
    uint64_t compute_output_start{0};

#ifdef TRITON_ENABLE_GPU
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0)
    {
      SET_TIMESTAMP(compute_output_start);
    }
    else
#endif
    {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses,
                                        request_count,
                                        all_response_failed,
                                        RecordBackendTimestamp(&compute_end_ns,
                                                               reinterpret_cast<void *>(&compute_output_start_event_)));
    }

    if (!all_response_failed)
    {
      if (!invalid_index)
      {
        TRITONSERVER_Error* err{nullptr};
        try
        {
          ReadOutputTensors(total_batch_size, output_tensors, requests, request_count, responses);
        }
        catch (const triton::backend::pytorch::BackendException& exception)
        {
          TRITON_LOG_ERROR("Failed to read output tensors" << " for model instance '" << Name()
                           << "': " << exception.what());

          err = TRITONSERVER_ErrorNew(exception.error_code(),
                                      std::string("Failed to read output tensors for model instance '" + Name()
                                                  + "': " + exception.what()).c_str());
        }
        catch (const std::exception& exception)
        {
          TRITON_LOG_ERROR("Failed to read output tensors" << " for model instance '" << Name()
                           << "': " << exception.what());

          err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                      std::string("Failed to read output tensors for model instance '" + Name()
                                                  + "': " + exception.what()).c_str());
        }

        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses,
                                          request_count,
                                          all_response_failed,
                                          err);
      }
    }

    uint64_t exec_end_ns{0};
    SET_TIMESTAMP(exec_end_ns);

    // Send all the responses that haven't already been sent because of an earlier
    // error. Note that the responses are not set to `nullptr` here as we need
    // that indication below to determine if the request was successful or not.
    for (auto& response : responses)
    {
      if (response != nullptr)
      {
        if (auto err = TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr))
        {
          TRITON_LOG_ERROR("Failed to send response for model instance '" << Name()
                           << "': " << TRITONSERVER_ErrorMessage(err));
          TRITONSERVER_ErrorDelete(err);
        }
      }
    }

    // We don't need an explicit CUDA synchronization here since we have already
    // synchronized the stream in the `ReadOutputTensors` function.
#ifdef TRITON_ENABLE_GPU
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU)
    {
      float compute_input_duration = GetCudaEventElapsedTime(compute_input_start_event_, compute_infer_start_event_);
      float compute_infer_duration = GetCudaEventElapsedTime(compute_infer_start_event_, compute_output_start_event_);

      compute_start_ns = exec_start_ns + (compute_input_duration * 1e6);
      compute_end_ns = compute_start_ns + (compute_infer_duration * 1e6);
    }
    else if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0)
    {
      float compute_input_duration = GetCudaEventElapsedTime(compute_input_start_event_, compute_infer_start_event_);
      float compute_infer_duration = GetCudaEventElapsedTime(compute_infer_start_event_, compute_output_start_event_);
    }
#endif

    // Report statistics for each request.
    for (uint32_t r = 0; r < request_count; r += 1)
    {
      auto& request = requests[r];
      if (auto err = TRITONBACKEND_ModelInstanceReportStatistics(TritonModelInstance(),
                                                                 request,
                                                                 (responses[r] != nullptr) /* success */,
                                                                 exec_start_ns,
                                                                 exec_end_ns,
                                                                 compute_start_ns,
                                                                 compute_end_ns))
      {
        TRITON_LOG_ERROR("Failed to report statistics for request " << r << " of model instance '"
                         << Name() << "': " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
      }

      if (auto err = TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL))
      {
        TRITON_LOG_ERROR("Failed to release request " << r << " of model instance '" << Name()
                         << "': " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
      }
    }

    if (!all_response_failed)
    {
      if (auto err = TRITONBACKEND_ModelInstanceReportBatchStatistics(TritonModelInstance(),
                                                                      total_batch_size,
                                                                      exec_start_ns,
                                                                      compute_start_ns,
                                                                      compute_end_ns,
                                                                      exec_end_ns))
      {
        TRITON_LOG_ERROR("Failed to report batch statistics for model instance '" << Name()
                         << "': " << TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
      }
    }
  }

  void
  InductorModelInstance::ReadOutputTensors(
      size_t total_batch_size,
      const std::vector<torch::Tensor>& output_tensors,
      TRITONBACKEND_Request** requests,
      uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>& responses)
  {
    NVTX_RANGE(nvtx_, "ReadOutputTensors " + Name());

    triton::backend::BackendOutputResponder responder{requests,
                                                      request_count,
                                                      &responses,
                                                      model_->MaxBatchSize(),
                                                      model_->TritonMemoryManager(),
                                                      model_->EnablePinnedOutput(),
                                                      GetCudaStreamByInstanceKind()};

    bool cuda_copy{false};
    std::vector<std::shared_ptr<std::string>> string_buffers;

    for (auto& output : model_->ModelOutputs())
    {
      int op_index = output_index_map_[output.first];
      auto name = output.first;
      auto output_tensor_pair = output.second;

      // if (output_tensors[op_index].isTensor())
      // {
        torch::Tensor output_flat;
        try
        {
          output_flat = output_tensors[op_index].contiguous().flatten();
        }
        catch (const std::exception& exception)
        {
          if (auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                               std::string("Inductor model instance '" + Name() + "' output tensor '"
                                                           + name + "' not found: " + exception.what()).c_str()))
          {
            THROW_TRITON_EXCEPTION(err, TRITONSERVER_ErrorMessage(err));
          }
        }

        auto output_dtype = ConvertTorchTypeToDataType(output_flat.scalar_type());
        auto config_dtype = output_dtype_map_[name];
        if (output_dtype != config_dtype)
        {
          if (auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                               std::string("Inductor model instance '" + Name() + "' output tensor '"
                                               + name + "' has datatype TYPE_"
                                               + TRITONSERVER_DataTypeString(output_dtype)
                                               + " but model configuration expects TYPE_"
                                               + TRITONSERVER_DataTypeString(config_dtype) + ".").c_str()))
          {
            THROW_TRITON_EXCEPTION(err, TRITONSERVER_ErrorMessage(err));
          }
        }

        const char *output_buffer = static_cast<const char *>(output_flat.data_ptr());

        // Output tensors might not reside on the same device as the model instance.
        torch::Device tensor_device = output_flat.device();
        const auto memory_type = (tensor_device.type() == torch::kCPU)
                                    ? TRITONSERVER_MEMORY_CPU
                                    : TRITONSERVER_MEMORY_GPU;
        const auto memory_type_id = (tensor_device.type() == torch::kCPU)
                                    ? 0
                                    : tensor_device.index();

        // Batch output doesn't support string data type yet, as it is not trivial to parse string output.
        const BatchOutput *batch_output = model_->FindBatchOutput(name);
        if (batch_output != nullptr)
        {
          std::vector<int64_t> batch_n_shape;
          auto shape = output_tensors[op_index].sizes();
          for (auto it = shape.begin(); it != shape.end(); it++)
          {
            batch_n_shape.push_back(*it);
          }

          if (batch_n_shape.size() == 0)
          {
            if (auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                                 std::string("Inductor model instance '" + Name() + "' output tensor '"
                                                             + name + "' is a scalar which is not supported.").c_str()))
            {
              THROW_TRITON_EXCEPTION(err, TRITONSERVER_ErrorMessage(err));
            }
          }

          if (output_tensor_pair.first != -1)
          {
            responder.ProcessTensor(name, output_dtype, batch_n_shape, output_buffer, memory_type, memory_type_id);
          }

          if (output_tensor_pair.second != -1)
          {
            auto states = responder.ProcessStateTensor(name,
                                                       output_dtype,
                                                       batch_n_shape,
                                                       output_buffer,
                                                       memory_type,
                                                       memory_type_id);

            for (auto& state : states)
            {
              if (auto err = TRITONBACKEND_StateUpdate(state))
              {
                THROW_TRITON_EXCEPTION(err,
                                       "Failed to update state for model instance '" << Name()
                                       << "': " << TRITONSERVER_ErrorMessage(err));
              }
            }
          }
        }
        else
        {
          responder.ProcessBatchOutput(name, *batch_output, output_buffer, memory_type, memory_type_id);
        }
      // }
      // else if (output_tensors[op_index].isList())
      // {
      //   // Custom handling for string/bytes tensor:
      //   torch::List<torch::Tensor> output_list = output_tensors[op_index].toList();
      //   std::vector<int64_t> batch_n_shape{(int64_t)output_list.size()};

      //   for (size_t idx = 0; idx < responses->size(); idx += 1)
      //   {
      //     auto request = requests[idx];
      //     auto response = (*responses)[idx];

      //     if (is_batching_supported_) {
      //       TRITONBACKEND_Input *input{nullptr};
      //       TRITONBACKEND_RequestInputByIndex(request, 0, &input);

      //       const int64_t *shape{nullptr};
      //       TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);

      //       batch_n_shape[0] = shape[0];
      //     }

      //     int64_t tensor_element_count{0};
      //     if (auto err = GetElementCount(batch_n_shape,& tensor_element_count))
      //     {
      //       THROW_TRITON_EXCEPTION(err,
      //                              "Failed to get element count for output '" << name << "' for request " << idx
      //                              << " of model instance '" << Name() << "': " << TRITONSERVER_ErrorMessage(err));
      //     }

      //     if (response)
      //     {
      //       if (output_tensor_pair.first != -1)
      //       {
      //         TRITONBACKEND_Output *output{nullptr};
      //         RESPOND_AND_SET_NULL_IF_ERROR(&response,
      //                                       TRITONBACKEND_ResponseOutput(response,
      //                                                                    &response_output,
      //                                                                    name.c_str(),
      //                                                                    TRITONSERVER_TYPE_BYTES,
      //                                                                    batch_n_shape.data(),
      //                                                                    batch_n_shape.size()));

      //         string_buffers.emplace_back(new std::string{});
      //         cuda_copy |= SetStringOutputBuffer(&output_list,
      //                                            &response,
      //                                            response_output,
      //                                            tensor_element_count,
      //                                            GetCudaStreamByInstanceKind(),
      //                                            string_buffers.back().get());
      //       }
      //     }
      //   }
      // }
      // else
      // {
      //   if (auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
      //                                        std::string("Inductor model instance '" + Name() + "' output '" + name
      //                                                    + "' must be of type Tensor or List[str].").c_str()))
      //   {
      //     THROW_TRITON_EXCEPTION(err, TRITONSERVER_ErrorMessage(err));
      //   }
      // }
    }

    // Finalize and wait for any pending buffer copies.
    cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
    cudaStreamSynchronize(GetCudaStreamByInstanceKind());
#endif
  }

  TRITONSERVER_Error*
  InductorModelInstance::RecordBackendTimestamp(
      uint64_t* timestamp,
      void* cuda_event_ptr)
  {
#ifdef TRITON_ENABLE_GPU
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU ||
        (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL && device_count_ > 0))
    {
      triton::backend::cudaEvent_t& cuda_event = *(reinterpret_cast<triton::backend::cudaEvent_t*>(cuda_event_ptr));
      if (auto err = ConvertCUDAStatusToTritonError(cudaEventRecord(cuda_event, GetCudaStreamByInstanceKind()),
                                                    TRITONSERVER_ERROR_INTERNAL,
                                                    "Failed to record CUDA event"))
        return err;
    } else
#endif
    {
      SET_TIMESTAMP(*timestamp);
    }

    return nullptr;
  }

  void
  InductorModelInstance::SetCurrentCudaStream(
      const triton::backend::cudaStream_t& stream,
      const int& device_id)
  {
#ifdef TRITON_ENABLE_GPU
    at::cuda::CUDAStream torch_stream{at::cuda::getStreamFromExternal(stream, device_id`)};
    // This function replaces the default stream with the stream we created.
    // It is not necessary to change the current device to the desired device when
    // replacing the default stream for that device. See the documentation here:
    // https://pytorch.org/cppdocs/api/function_namespacec10_1_1cuda_1a6ed50cc0fc16cc7014d9c2f4c3bd098d.html
    at::cuda::setCurrentCUDAStream(torch_stream);
#endif
  }

  TRITONSERVER_Error*
  InductorModelInstance::SetInputTensors(
      size_t total_batch_size,
      TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector,
      std::vector<const char*>* input_names,
      std::vector<torch::Tensor>* input_tensors,
      bool* cuda_copy)
  {
    if (!requests)
      THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INVALID_ARG,
                            "Argument 'requests' cannot be nullptr.");

    torch::InferenceMode guard{model_->InferenceModeEnabled()};

    uint32_t input_count{0};
    if (auto err = TRITONBACKEND_RequestInputCount(requests[0], &input_count))
    {
      TRITON_LOG_ERROR("Failed to get input count" << " for request 0 of model instance '"
                       << Name() << "': " << TRITONSERVER_ErrorMessage(err));
      return err;
    }

    input_tensors->resize(input_count + batch_input_count_);

    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_preference;
    if (device_.is_cpu())
    {
      alloc_preference = {
        {TRITONSERVER_MEMORY_CPU_PINNED, 0},
        {TRITONSERVER_MEMORY_CPU, 0},
      };
    }
    else
    {
      alloc_preference = {
        {TRITONSERVER_MEMORY_GPU, device_.index()},
      };
    }

    for (size_t input_idx = 0; input_idx < input_count; input_idx += 1)
    {
      TRITONBACKEND_Input *input{nullptr};
      if (auto err = TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input))
      {
        TRITON_LOG_ERROR("Failed to get input " << input_idx << " for request 0 of model instance '"
                         << Name() << "': " << TRITONSERVER_ErrorMessage(err));
        return err;
      }

      const char *input_name{nullptr};
      TRITONSERVER_DataType input_datatype;
      const int64_t *input_shape{nullptr};
      uint32_t input_dims_count{0};
      if (auto err = TRITONBACKEND_InputProperties(input,
                                                   &input_name,
                                                   &input_datatype,
                                                   &input_shape,
                                                   &input_dims_count,
                                                   nullptr,
                                                   nullptr))
      {
        TRITON_LOG_ERROR("Failed to get properties for input " << input_idx << " for request 0 of model instance '"
                         << Name() << "': " << TRITONSERVER_ErrorMessage(err));
        return err;
      }

      input_names->emplace_back(input_name);

      std::vector<int64_t> batch_n_shape;
      if (model_->IsInputRagged(input_name))
      {
        batch_n_shape = std::vector<int64_t>{0};
        for (size_t idx = 0; idx < request_count; idx += 1)
        {
          TRITONBACKEND_Input *input{nullptr};
          if (auto err = TRITONBACKEND_RequestInput(requests[idx], input_name, &input))
          {
            RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]),
                                          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                                                std::string("Failed to get input '"
                                                                            + std::string(input_name)
                                                                            + "' for request "
                                                                            + std::to_string(idx)
                                                                            + " of model instance '"
                                                                            + Name() + "': "
                                                                            + TRITONSERVER_ErrorMessage(err)).c_str()));
          }

          const int64_t* input_shape{nullptr};
          uint32_t input_dims_count{0};
          if (auto err = TRITONBACKEND_InputProperties(input,
                                                       nullptr,
                                                       nullptr,
                                                       &input_shape,
                                                       &input_dims_count,
                                                       nullptr,
                                                       nullptr))
          {
            RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]),
                                          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                                                std::string("Failed to get properties for input '"
                                                                            + std::string(input_name)
                                                                            + "' for request "
                                                                            + std::to_string(idx)
                                                                            + " of model instance '"
                                                                            + Name()
                                                                            + "': "
                                                                            + TRITONSERVER_ErrorMessage(err)).c_str()));
          }

          int64_t element_cnt{0};
          if (auto err = GetElementCount(input_shape, input_dims_count,& element_cnt))
          {
            RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]),
                                          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                                                std::string(std::string(input_name)
                                                                            + "' for request "
                                                                            + std::to_string(idx)
                                                                            + " of model instance '"
                                                                            + "Failed to get element count for input '"
                                                                            + Name()
                                                                            + "': "
                                                                            + TRITONSERVER_ErrorMessage(err)).c_str()));
          }

          batch_n_shape[0] += element_cnt;
        }
      }
      else
      {
        batch_n_shape = std::vector<int64_t>(input_shape, input_shape + input_dims_count);
        if (is_batching_supported_)
        {
          batch_n_shape[0] = total_batch_size;
        }
      }

      // The input must be in contiguous CPU/GPU memory.
      std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_preference;
      // For 'KIND_MODEL', input will always be in CPU as we don't have a way to
      // query the input types.
      if (device_.is_cpu() || (Kind() == TRITONSERVER_INSTANCEGROUPKIND_MODEL))
      {
        alloc_preference = {
            {TRITONSERVER_MEMORY_CPU_PINNED, 0},
            {TRITONSERVER_MEMORY_CPU, 0},
        };
      }
      else
      {
        alloc_preference = {
            {TRITONSERVER_MEMORY_GPU, device_.index()},
        };
      }

      const char *input_buffer{nullptr};
      size_t batch_n_byte_size{0};
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id{0};
      if (auto err = collector->ProcessTensor(input_name,
                                              nullptr,
                                              0,
                                              alloc_preference,
                                              &input_buffer,
                                              &batch_n_byte_size,
                                              &memory_type,
                                              &memory_type_id))
      {
        TRITON_LOG_ERROR("Failed to process input '" << input_name << "' for model instance '"
                         << Name() << "': " << TRITONSERVER_ErrorMessage(err));
        return err;
      }

      const std::pair<bool, torch::ScalarType> torch_dtype = ConvertDataTypeToTorchType(input_datatype);
      torch::TensorOptions options{torch_dtype.second};
      auto updated_options = (memory_type == TRITONSERVER_MEMORY_GPU)
                                ? options.device(torch::kCUDA, device_.index())
                                : options.device(torch::kCPU);

      // if (input_datatype == TRITONSERVER_TYPE_BYTES)
      // {
      //   // Create the PyTorch list to hold the strings.
      //   torch::List<std::string> input_list;
      //   input_list.reserve(batch_n_shape[0]);

      //   for (size_t idx = 0; idx < request_count; idx += 1)
      //   {
      //     TRITONBACKEND_Input *input{nullptr};
      //     RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]),
      //                                   TRITONBACKEND_RequestInput(requests[idx], input_name,& input));

      //     const int64_t *shape;
      //     uint32_t dims_count;
      //     uint32_t buffer_count;
      //     RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]),
      //                                   TRITONBACKEND_InputPropertiesForHostPolicy(input,
      //                                                                              HostPolicyName().c_str(),
      //                                                                              nullptr,
      //                                                                              nullptr,
      //                                                                              &shape,
      //                                                                              &dims_count,
      //                                                                              nullptr,
      //                                                                              &buffer_count));

      //     int64_t batch_element_count = 0;
      //     RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]),
      //                                   GetElementCount(shape, dims_count,& batch_element_count));

      //     *cuda_copy |= SetStringInputTensor(&input_list,
      //                                        input,
      //                                        input_name,
      //                                        buffer_count,
      //                                        batch_element_count,
      //                                        &((*responses)[idx]),
      //                                        GetCudaStreamByInstanceKind(),
      //                                        HostPolicyName().c_str());
      //   }

      //   (*input_tensors)[input_index_map_[input_name]] = torch::tensor(input_list);
      // }
      // else
      // {
        if (batch_n_byte_size > 0)
        {
          (*input_tensors)[input_index_map_[input_name]] = torch::from_blob(const_cast<char *>(input_buffer),
                                                                            torch::IntArrayRef(batch_n_shape),
                                                                            updated_options);
        }
        else
        {
          // Create an empty tensor for zero-sized input.
          (*input_tensors)[input_index_map_[input_name]] = torch::zeros(batch_n_shape, updated_options);
        }
      // }
    }

    for (const auto& batch_input : model_->BatchInputs())
    {
      std::vector<int64_t> shape;
      collector->BatchInputShape(batch_input,& shape);

      for (const auto& input_name : batch_input.TargetNames())
      {
        input_names->emplace_back(input_name.c_str());

        const char *dst_buffer{nullptr};
        size_t dst_buffer_byte_size{0};
        TRITONSERVER_MemoryType dst_memory_type;
        int64_t dst_memory_type_id{0};

        RESPOND_ALL_AND_SET_NULL_IF_ERROR((*responses),
                                          responses->size(),
                                          collector->ProcessBatchInput(batch_input,
                                                                       nullptr,
                                                                       0,
                                                                       alloc_preference,
                                                                       &dst_buffer,
                                                                       &dst_buffer_byte_size,
                                                                       &dst_memory_type,
                                                                       &dst_memory_type_id));

        const auto torch_dtype = ConvertDataTypeToTorchType(batch_input.DataType());
        torch::TensorOptions options{torch_dtype.second};
        auto updated_options = (dst_memory_type == TRITONSERVER_MEMORY_GPU)
                                  ? options.device(torch::kCUDA, device_.index())
                                  : options.device(torch::kCPU);

        if (dst_buffer_byte_size)
        {
          (*input_tensors)[input_index_map_[input_name]] = torch::from_blob(
              const_cast<char *>(dst_buffer), shape, updated_options);
        }
        else
        {
          // special handle when input has zero size
          (*input_tensors)[input_index_map_[input_name]] =
              torch::zeros(shape, updated_options);
        }
      }
    }

    *cuda_copy |= collector->Finalize();

    return nullptr;
  }

  TRITONBACKEND_ModelInstance*
  InductorModelInstance::TritonModelInstance()
  {
    return BackendModelInstance::TritonModelInstance();
  }

  bool
  InductorModelInstance::ValidateBooleanSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind,
      bool required)
  {
    std::string tensor_name;
    std::string tensor_dtype;
    if (auto err = GetBooleanSequenceControlProperties(sequence_batching,
                                                       model_->Name(),
                                                       control_kind,
                                                       required,
                                                       &tensor_name,
                                                       &tensor_dtype,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr))
     {
      THROW_TRITON_EXCEPTION(err,
                             "Failed to validate boolean sequence control for model instance '" << Name()
                             << "': " << TRITONSERVER_ErrorMessage(err));
    }

    bool have_control{!tensor_name.empty()};

    if (have_control)
    {
      std::string deliminator{"__"};
      int ip_index{0};
      int start_pos{tensor_name.find(deliminator)};

      if (start_pos == -1)
      {
        THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                               "Input '" << tensor_name << "' does not follow <name>__<index> naming convention.");
      }

      // Check if the index part of the name is not an integer.
      std::string index_str = tensor_name.substr(start_pos + 2);
      for (auto itr = index_str.begin(); itr != index_str.end(); itr++)
      {
        if (std::isdigit(*itr) == 0)
        {
          THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                                 "Input '" << tensor_name << "' does not follow <name>__<index> naming convention.");
        }
      }

      ip_index = std::atoi(tensor_name.substr(start_pos + 2).c_str());
      input_index_map_[tensor_name] = ip_index;
    }

    return have_control;
  }

  void
  InductorModelInstance::ValidateInputs(
      const size_t expected_input_count)
  {
    std::vector<std::string> allowed_inputs = model_->GetModelCallSpec();

    if (allowed_inputs.size() != expected_input_count)
    {
      THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                             "Failed to load model '" << Name() << "' configuration expects " << expected_input_count
                             << " inputs, but model expects " << allowed_inputs.size() << " inputs.");
    }

    /* CANNOT VALIDATE INPUTS BY DTYPE DUE TO LACK OF INFORMATION FROM MODEL */

    triton::common::TritonJson::Value ios;
    if (auto err = model_->ModelConfig().MemberAsArray("input", &ios))
    {
      THROW_TRITON_EXCEPTION(err,
                             "Failed to get model '" << Name() << "' input configuration: "
                             << TRITONSERVER_ErrorMessage(err));
    }

    if (ios.ArraySize() != expected_input_count)
    {
      THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                             "Failed to load model '" << Name() << "' configuration expects " << expected_input_count
                             << " inputs, but model configuration has " << ios.ArraySize() << " inputs.");
    }

    auto naming_convention = GetNamingConvention(allowed_inputs);

    for (size_t i = 0; i < ios.ArraySize(); i += 1)
    {
      triton::common::TritonJson::Value io;
      if (auto err = ios.IndexAsObject(i,& io))
      {
        THROW_TRITON_EXCEPTION(err,
                               "Failed to get input " << i << " for model instance '" << Name() << "': "
                               << TRITONSERVER_ErrorMessage(err));
      }

      std::string io_name;
      if (auto err = io.MemberAsString("name", &io_name))
      {
        THROW_TRITON_EXCEPTION(err,
                               "Failed to get name for input " << i << " for model instance '" << Name() << "': "
                               << TRITONSERVER_ErrorMessage(err));
      }

      AddInputToMap(naming_convention, allowed_inputs, io_name, i);

      // Validate dtype
      std::string io_dtype;
      if (auto err = io.MemberAsString("data_type", &io_dtype))
      {
        THROW_TRITON_EXCEPTION(err,
                               "Failed to get data type for input '" << io_name << "' for model instance '" << Name()
                               << "': " << TRITONSERVER_ErrorMessage(err));
      }

      const auto pr = ModelConfigDataTypeToTorchType(io_dtype);
      if (!pr.first && (io_dtype != "TYPE_STRING"))
      {
        THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                               "Unsupported datatype " << io_dtype << " for input '" << io_name
                               << "' for model instance '" << Name() << "'.");
      }

      // Validate shape for String inputs. Only allow 1 dimension.
      if (io_dtype == "TYPE_STRING")
      {
        // If a reshape is provided for the input then use that when validating
        // the model shapes.
        std::vector<int64_t> dims;
        triton::common::TritonJson::Value reshape;
        if (io.Find("reshape", &reshape))
        {
          if (auto err = ParseShape(reshape, "shape", &dims))
          {
            THROW_TRITON_EXCEPTION(err,
                                   "Failed to parse reshape shape for input '" << io_name << "' for model instance '"
                                   << Name() << "': " << TRITONSERVER_ErrorMessage(err));
          }
        } else {
          if (auto err = ParseShape(io, "dims",& dims))
          {
            THROW_TRITON_EXCEPTION(err,
                                   "Failed to parse dims for input '" << io_name << "' for model instance '"
                                   << Name() << "': " << TRITONSERVER_ErrorMessage(err));
          }
        }
      }
    }

    triton::common::TritonJson::Value sequence_batching;
    if (model_->ModelConfig().Find("sequence_batching",& sequence_batching))
    {
      triton::common::TritonJson::Value states;
      if (sequence_batching.Find("state", &states))
      {
        for (size_t i = 0; i < states.ArraySize(); i += 1)
        {
          triton::common::TritonJson::Value state;
          if (auto err = states.IndexAsObject(i, &state))
          {
            THROW_TRITON_EXCEPTION(err,
                                   "Failed to get sequence state " << i << " for model instance '" << Name()
                                   << "': " << TRITONSERVER_ErrorMessage(err));
          }

          std::string state_name;
          if (auto err = state.MemberAsString("input_name",& state_name))
          {
            THROW_TRITON_EXCEPTION(err,
                                   "Failed to get input name for sequence state " << i << " for model instance '"
                                   << Name() << "': " << TRITONSERVER_ErrorMessage(err));
          }

          AddInputToMap(naming_convention, allowed_inputs, state_name, i);

          // Validate dtype
          std::string state_dtype;
          if (auto err = state.MemberAsString("data_type", &state_dtype))
          {
            THROW_TRITON_EXCEPTION(err,
                                   "Failed to get data type for sequence state input '" << state_name <<
                                   "' for model instance '" << Name() << "': " << TRITONSERVER_ErrorMessage(err));
          }

          const auto pr = ModelConfigDataTypeToTorchType(state_dtype);
          if (!pr.first)
          {
            THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                                   "Unsupported datatype " << state_dtype << " for sequence state input '"
                                   << state_name << "' for model instance '" << Name() << "'.");
          }

          // Validate shape for String inputs. Only allow 1 dimension.
          if (state_dtype == "TYPE_STRING")
          {
            std::vector<int64_t> dims;
            if ((dims.size() + (is_batching_supported_ ? 1 : 0)) > 1)
            {
              THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INVALID_ARG,
                                     "Triton only supports 1-dimensional string inputs for sequence state input '"
                                     << state_name << "' for model '" << Name() << "'.");
            }
          }
        }
      }
    }

    triton::common::TritonJson::Value batch_inputs;
    if (auto err = model_->ModelConfig().MemberAsArray("batch_input", &batch_inputs))
    {
      THROW_TRITON_EXCEPTION(err,
                             "Failed to get batch input configuration for model instance '" << Name()
                             << "': " << TRITONSERVER_ErrorMessage(err));
    }

    size_t i = 0;
    for (const auto& batch_input : model_->BatchInputs())
    {
      for (const auto& input_name : batch_input.TargetNames())
      {
        AddInputToMap(naming_convention, allowed_inputs, input_name, i + ios.ArraySize());
        i += 1;
      }
    }
  }

  void
  InductorModelInstance::ValidateOutputs()
  {
    triton::common::TritonJson::Value ios;
    if (auto err = model_->ModelConfig().MemberAsArray("output", &ios))
    {
      THROW_TRITON_EXCEPTION(err,
                             "Failed to get model '" << Name() << "' output configuration: "
                             << TRITONSERVER_ErrorMessage(err));
    }

    std::string deliminator{"__"};
    int op_index{0};

    if (ios.ArraySize() == 0)
    {
      THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                             "Configuration for model '" << Name()
                             << "' must define at least one output, none were specified.");
    }

    pytorch::NamingConvention naming_convention = GetNamingConvention({});

    for (size_t i = 0; i < ios.ArraySize(); i++)
    {
      triton::common::TritonJson::Value io;
      if (auto err = ios.IndexAsObject(i, &io))
      {
        THROW_TRITON_EXCEPTION(err,
                               "Failed to get output " << i << " for model instance '" << Name()
                               << "': " << TRITONSERVER_ErrorMessage(err));
      }

      // Validate name
      std::string io_name;
      if (auto err = io.MemberAsString("name", &io_name))
      {
        THROW_TRITON_EXCEPTION(err,
                               "Failed to get name for output " << i << " for model instance '" << Name()
                               << "': " << TRITONSERVER_ErrorMessage(err));
      }

      switch (naming_convention)
      {
        case NamingConvention::NAMED_INDEX:
        {
          int start_pos = io_name.find(deliminator);
          op_index = std::atoi(io_name.substr(start_pos + 2).c_str());
        }
        break;

        case NamingConvention::STRICT_CONFIG_ORDERING:
        {
          op_index = i;
        }
        break;

        default: break;
      }

      // Validate data type.
      std::string io_dtype;
      if (auto err = io.MemberAsString("data_type", &io_dtype))
      {
        THROW_TRITON_EXCEPTION(err,
                               "Failed to get data type for output '" << io_name << "' for model instance '" << Name()
                               << "': " << TRITONSERVER_ErrorMessage(err));
      }

      const auto pr = ModelConfigDataTypeToTorchType(io_dtype);
      if (!pr.first && (io_dtype != "TYPE_STRING"))
      {
        THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                               "Unsupported datatype " << io_dtype << " for output '" << io_name
                               << "' for model instance '" << Name() << "'.");
      }

      // Validate shape for String outputs. Only allow 1 dimension.
      if (io_dtype == "TYPE_STRING")
      {
        std::vector<int64_t> dims;
        triton::common::TritonJson::Value reshape;

        if (io.Find("reshape", &reshape))
        {
          if (auto err = ParseShape(reshape, "shape", &dims))
          {
            THROW_TRITON_EXCEPTION(err,
                                   "Failed to parse reshape shape for output '" << io_name << "' for model instance '"
                                   << Name() << "': " << TRITONSERVER_ErrorMessage(err));
          }
        }
        else
        {
          if (auto err = ParseShape(io, "dims", &dims))
          {
            THROW_TRITON_EXCEPTION(err,
                                   "Failed to parse dims for output '" << io_name << "' for model instance '"
                                   << Name() << "': " << TRITONSERVER_ErrorMessage(err));
          }
        }

        if ((dims.size() + (is_batching_supported_ ? 1 : 0)) > 1)
        {
          THROW_TRITON_EXCEPTION(TRITONSERVER_ERROR_INTERNAL,
                                 "Triton only supports 1 dimensional List of String as output for '" << io_name
                                 << "' for model instance '" << Name() << "'.");
        }
      }

      output_index_map_[io_name] = op_index;
      output_dtype_map_[io_name] = ConvertTorchTypeToDataType(pr.second);
    }

    // CONTINUE HERE <-- model_instance_state.cc @ line 1557
  }
}
