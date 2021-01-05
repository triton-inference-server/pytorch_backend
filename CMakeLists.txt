# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 3.18)

project(tritonpytorchbackend LANGUAGES C CXX)

#
# Options
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
set(TRITON_PYTORCH_INCLUDE_PATHS "" CACHE PATH "Paths to Torch includes")
set(TRITON_PYTORCH_LIB_PATHS "" CACHE PATH "Paths to Torch libraries")

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Python.h needed by torch headers.
find_package(Python3 REQUIRED COMPONENTS Development)

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
endif() # TRITON_ENABLE_GPU

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_pytorch.ldscript libtriton_pytorch.ldscript COPYONLY)

add_library(
  triton-pytorch-backend SHARED
  src/libtorch.cc
  src/libtorch_utils.cc
  src/libtorch_utils.h
)

add_library(
  TritonPyTorchBackend::triton-pytorch-backend ALIAS triton-pytorch-backend
)

target_include_directories(
  triton-pytorch-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRITON_PYTORCH_INCLUDE_PATHS}
    ${Python3_INCLUDE_DIRS}
)

target_compile_features(triton-pytorch-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-pytorch-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-pytorch-backend
    PRIVATE TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU

set_target_properties(
  triton-pytorch-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_pytorch
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_pytorch.ldscript
    LINK_FLAGS "-Wl,--no-as-needed,--version-script libtriton_pytorch.ldscript"
)

# Need to turn off unused-but-set-variable due to Torchvision
# Need to turn off unknown-pragmas due to ATen OpenMP
set_target_properties(
  triton-pytorch-backend
  PROPERTIES COMPILE_FLAGS
    "-Wno-unknown-pragmas -Wno-unused-but-set-variable"
)

set(TRITON_PYTORCH_LDFLAGS "")
FOREACH(p ${TRITON_PYTORCH_LIB_PATHS})
  set(TRITON_PYTORCH_LDFLAGS ${TRITON_PYTORCH_LDFLAGS} "-L${p}")
ENDFOREACH(p)

target_link_libraries(
  triton-pytorch-backend
  PRIVATE
    triton-core-serverapi  # from repo-core
    triton-core-backendapi # from repo-core
    triton-core-serverstub # from repo-core
    triton-backend-utils   # from repo-backend
    ${TRITON_PYTORCH_LDFLAGS}
    -ltorch
    -ltorch_cpu
    -ltorch_cuda
    -ltorchvision
    -lc10
    -lc10_cuda
    -lmkl_core
    -lmkl_gnu_thread
    -lmkl_intel_lp64
    -lmkl_avx2
    -lmkl_def
    -liomp5
    -lmkl_intel_thread
)

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-pytorch-backend
    PRIVATE
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonPyTorchBackend)

install(
  TARGETS
    triton-pytorch-backend
  EXPORT
    triton-pytorch-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/pytorch
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/pytorch
)

install(
  EXPORT
    triton-pytorch-backend-targets
  FILE
    TritonPyTorchBackendTargets.cmake
  NAMESPACE
    TritonPyTorchBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonPyTorchBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonPyTorchBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonPyTorchBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-pytorch-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonPyTorchBackendTargets.cmake
  NAMESPACE TritonPyTorchBackend::
)

export(PACKAGE TritonPyTorchBackend)
