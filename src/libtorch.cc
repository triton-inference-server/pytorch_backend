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

#include "libtorch.hh"

#include "inductor_model.hh"
#include "inductor_model_instance.hh"
#include "model_instance_state.hh"
#include "model_state.hh"
#include "triton/backend/backend_common.h"
#include "triton_utils.hh"

#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>

// _CRT_INTERNAL_NONSTDC_NAMES 1 before including Microsoft provided C Runtime
// library to expose declarations without "_" prefix to match POSIX style.
#define _CRT_INTERNAL_NONSTDC_NAMES 1
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif
#include <sys/stat.h>

#include <mutex>
#include <sstream>
#include <unordered_map>

//
// PyTorch C++ (LibTorch) Backend that implements the TRITONBACKEND API.
//

namespace triton::backend::pytorch {
static std::unordered_map<void*, bool> model_is_inductor_map{};
static std::mutex model_is_inductor_map_mutex{};

TRITONSERVER_Error*
GetModelConfigPlatform(
    TRITONBACKEND_Model* model, const std::string model_name,
    std::string& platform)
{
  TRITONSERVER_Message* config_msg{nullptr};
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(model, 1, &config_msg));

  const char* json_buffer{nullptr};
  size_t json_buffer_size{0};
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      config_msg, &json_buffer, &json_buffer_size));

  common::TritonJson::Value model_config;
  auto err = model_config.Parse(json_buffer, json_buffer_size);
  TRITONSERVER_MessageDelete(config_msg);
  RETURN_IF_ERROR(err);

  platform.clear();

  const char* platform_cstr{nullptr};
  size_t platform_size{0};
  err = model_config.MemberAsString("platform", &platform_cstr, &platform_size);
  if (err) {
    TRITON_LOG_WARN(
        "Failed to get platform from model config for '"
        << model_name << "': " << TRITONSERVER_ErrorMessage(err) << ".");
    return err;
  }

  platform = std::string{platform_cstr, platform_size};
  TRITON_LOG_INFO(
      "Model platform for '" << model_name << "': '" << platform << "'.")

  return nullptr;  // success
}

TRITONSERVER_Error*
GetModelInfo(
    TRITONBACKEND_Model* model, std::string& model_name,
    uint64_t& model_version, TRITONBACKEND_ArtifactType& artifact_type,
    bool& is_inductor)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));

  model_name = std::string{cname};

  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &model_version));

  const char* location{nullptr};
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &location));

  TRITON_LOG_INFO(
      "TRITONBACKEND_ModelInitialize: " << model_name << " (version "
                                        << model_version
                                        << ") location=" << location)

  try {
    std::lock_guard lock{model_is_inductor_map_mutex};
    auto it = model_is_inductor_map.find(static_cast<void*>(model));
    if (it != model_is_inductor_map.end()) {
      is_inductor = it->second;
    } else {
      std::string platform;
      if (auto err = GetModelConfigPlatform(model, model_name, platform)) {
        model_is_inductor_map[static_cast<void*>(model)] = is_inductor = false;
        return err;
      }

      is_inductor = (platform == "torch_aoti");
      model_is_inductor_map[static_cast<void*>(model)] = is_inductor;
    }
  }
  catch (const std::exception& ex) {
    DEBUG_TRACE_ERROR(
        "{ model: \"" << model_name << "\""
                      << ", version: " << model_version
                      << ", error: " << ex.what() << " }");
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        TOSTRING(
            "Failed to inspect model \"" << model_name << "\""
                                         << ", version " << model_version
                                         << ": " << ex.what())
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
GetModelInstanceInfo(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Model** model,
    std::string& model_name, uint64_t& model_version, bool& is_inductor,
    std::string& instance_name, int32_t& device_id,
    TRITONSERVER_InstanceGroupKind& kind)
{
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, model));

  TRITONBACKEND_ArtifactType artifact_type{TRITONBACKEND_ARTIFACT_FILESYSTEM};
  RETURN_IF_ERROR(GetModelInfo(
      *model, model_name, model_version, artifact_type, is_inductor));

  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  instance_name = std::string{cname};

  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  TRITON_LOG_INFO(
      "TRITONBACKEND_ModelInstanceInitialize: "
      << instance_name << " (" << TRITONSERVER_InstanceGroupKindString(kind)
      << " device " << device_id << ")");

  return nullptr;  // success
}

extern "C" {
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));

  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      TOSTRING("TRITONBACKEND_Initialize: " << name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        TOSTRING(
            "Triton TRITONBACKEND API version: "
            << api_version_major << "." << api_version_minor
            << " does not support \"" << name << "\""
            << " TRITONBACKEND API version: " << TRITONBACKEND_API_VERSION_MAJOR
            << "." << TRITONBACKEND_API_VERSION_MINOR)
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  std::string model_name{};
  uint64_t model_version{0};
  TRITONBACKEND_ArtifactType artifact_type{TRITONBACKEND_ARTIFACT_FILESYSTEM};
  bool is_inductor{false};

  RETURN_IF_ERROR(GetModelInfo(
      model, model_name, model_version, artifact_type, is_inductor));

  if (is_inductor) {
    // Create an InductorModel object and associate it with the
    // TRITONBACKEND_Model.
    try {
      auto aoti_model = InductorModel::Create(model);
      RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(
          model, reinterpret_cast<void*>(aoti_model)));
    }
    catch (const std::exception& ex) {
      RETURN_ERROR_IF_TRUE(
          true, TRITONSERVER_ERROR_INTERNAL,
          TOSTRING(
              "failed to create InductorModel for model \""
              << model_name << "\""
              << ", version " << model_version << ": " << ex.what()));
    }
  } else {
    // Create a ModelState object and associate it with the
    // TRITONBACKEND_Model.
    ModelState* model_state;
    RETURN_IF_ERROR(ModelState::Create(model, &model_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(
        model, reinterpret_cast<void*>(model_state)));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  std::string model_name{};
  uint64_t model_version{0};
  TRITONBACKEND_ArtifactType artifact_type{TRITONBACKEND_ARTIFACT_FILESYSTEM};
  bool is_inductor{false};

  RETURN_IF_ERROR(GetModelInfo(
      model, model_name, model_version, artifact_type, is_inductor));

  void* vmodel;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodel));

  if (is_inductor) {
    InductorModel* aoti_model = reinterpret_cast<InductorModel*>(vmodel);

    delete aoti_model;
  } else {
    ModelState* model_state = reinterpret_cast<ModelState*>(vmodel);

    delete model_state;
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      TOSTRING(
          "TRITONBACKEND_ModelFinalize: model state for"
          " model \""
          << model_name << "\" version " << model_version << " deleted.")
          .c_str());

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  TRITONBACKEND_Model* model;
  std::string model_name{};
  uint64_t model_version{0};
  bool is_inductor{false};
  std::string instance_name{};
  int32_t device_id{0};
  TRITONSERVER_InstanceGroupKind kind;

  RETURN_IF_ERROR(GetModelInstanceInfo(
      instance, &model, model_name, model_version, is_inductor, instance_name,
      device_id, kind));

  void* vmodel;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodel));

  if (is_inductor) {
    auto aoti_model = reinterpret_cast<InductorModel*>(vmodel);

    try {
      // Create an InductorModelInstance object and associate it with the
      // TRITONBACKEND_ModelInstance.
      auto aoti_instance = InductorModelInstance::Create(aoti_model, instance);
      RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
          instance, reinterpret_cast<void*>(aoti_instance)));
    }
    catch (const triton::backend::pytorch::BackendException& ex) {
      RETURN_ERROR_IF_TRUE(
          true, ex.error_code(),
          TOSTRING(
              "failed to create instance \""
              << instance_name << "\" for inductor model \"" << model_name
              << "\": " << ex.what()));
    }
    catch (const std::exception& ex) {
      RETURN_ERROR_IF_TRUE(
          true, TRITONSERVER_ERROR_INTERNAL,
          TOSTRING(
              "failed to create instance \""
              << instance_name << "\" for inductor model \"" << model_name
              << "\": " << ex.what()));
    }

    return nullptr;  // success
  } else {
    ModelState* model_state = reinterpret_cast<ModelState*>(vmodel);

    // Create a ModelInstanceState object and associate it with the
    // TRITONBACKEND_ModelInstance.
    ModelInstanceState* instance_state;
    RETURN_IF_ERROR(
        ModelInstanceState::Create(model_state, instance, &instance_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
        instance, reinterpret_cast<void*>(instance_state)));

    return nullptr;  // success
  }
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  TRITONBACKEND_Model* model;
  std::string model_name{};
  uint64_t model_version{0};
  bool is_inductor{false};
  std::string instance_name{};
  int32_t device_id{0};
  TRITONSERVER_InstanceGroupKind kind;

  RETURN_IF_ERROR(GetModelInstanceInfo(
      instance, &model, model_name, model_version, is_inductor, instance_name,
      device_id, kind));

  void* vmodel;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vmodel));

  if (is_inductor) {
    auto aoti_instance = reinterpret_cast<InductorModelInstance*>(vmodel);
    delete aoti_instance;
  } else {
    auto instance_state = reinterpret_cast<ModelInstanceState*>(vmodel);
    delete instance_state;
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      TOSTRING(
          "TRITONBACKEND_ModelInstanceFinalize: model instance state for model "
          "\""
          << model_name << "\" version " << model_version << " deleted.")
          .c_str());

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  TRITONBACKEND_Model* model;
  std::string model_name{};
  uint64_t model_version{0};
  bool is_inductor{false};
  std::string instance_name{};
  int32_t device_id{0};
  TRITONSERVER_InstanceGroupKind kind;

  RETURN_IF_ERROR(GetModelInstanceInfo(
      instance, &model, model_name, model_version, is_inductor, instance_name,
      device_id, kind));

  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.

  if (is_inductor) {
    InductorModelInstance* aoti_instance{nullptr};
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
        instance, reinterpret_cast<void**>(&aoti_instance)));

    auto aoti_model = aoti_instance->InductorModel();

    // This backend specifies BLOCKING execution policy. That means that
    // we should not return from this function until execution is
    // complete. Triton will automatically release 'instance' on return
    // from this function so that it is again available to be used for
    // another call to TRITONBACKEND_ModelInstanceExecute.

    TRITON_LOG_VERBOSE(
        "model \"" << aoti_model->Name() << "\", instance \""
                   << aoti_instance->Name() << "\", executing " << request_count
                   << " requests");

    // At this point we accept ownership of 'requests', which means that
    // even if something goes wrong we must still return success from
    // this function. If something does go wrong in processing a
    // particular request then we send an error response just for the
    // specific request.
    aoti_instance->ProcessRequests(requests, request_count);

    if (aoti_model->CacheCleaningEnabled()) {
      aoti_instance->ClearCache();
    }
  } else {
    ModelInstanceState* instance_state{nullptr};
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
        instance, reinterpret_cast<void**>(&instance_state)));

    ModelState* model_state = instance_state->StateForModel();

    // This backend specifies BLOCKING execution policy. That means that
    // we should not return from this function until execution is
    // complete. Triton will automatically release 'instance' on return
    // from this function so that it is again available to be used for
    // another call to TRITONBACKEND_ModelInstanceExecute.

    TRITON_LOG_VERBOSE(
        "model \"" << model_name << "\", instance \"" << instance_state->Name()
                   << "\", executing " << request_count << " requests");

    // At this point we accept ownership of 'requests', which means that
    // even if something goes wrong we must still return success from
    // this function. If something does go wrong in processing a
    // particular request then we send an error response just for the
    // specific request.
    instance_state->ProcessRequests(requests, request_count);

    if (model_state->EnabledCacheCleaning()) {
      instance_state->ClearCache();
    }
  }

  return nullptr;  // success
}
}  // extern "C"
}  // namespace triton::backend::pytorch
