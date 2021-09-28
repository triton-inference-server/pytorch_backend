# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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
# To build the PyTorch backend you must either:
#
#   - Point to the already built PyTorch and Torchvision using
#     TRITON_PYTORCH_INCLUDE_PATHS and TRITON_PYTORCH_LIB_PATHS
#
#   or:
#
#   - Set TRITON_PYTORCH_DOCKER_IMAGE to use the docker image of
#     PyTorch to base the build off.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
set(TRITON_PYTORCH_DOCKER_IMAGE "" CACHE STRING "Docker image containing the PyTorch build required by backend.")
set(TRITON_PYTORCH_INCLUDE_PATHS "" CACHE PATH "Paths to Torch includes")
set(TRITON_PYTORCH_LIB_PATHS "" CACHE PATH "Paths to Torch libraries")

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TRITON_PYTORCH_DOCKER_BUILD OFF)
if(TRITON_PYTORCH_LIB_PATHS STREQUAL "")
  if(TRITON_PYTORCH_DOCKER_IMAGE STREQUAL "")
    message(FATAL_ERROR "Using the PyTorch docker based build requires TRITON_PYTORCH_DOCKER_IMAGE")
  endif()
  set(TRITON_PYTORCH_DOCKER_BUILD ON)
endif()
message(STATUS "Using PyTorch docker: ${TRITON_PYTORCH_DOCKER_IMAGE}")

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

if (${TRITON_PYTORCH_DOCKER_BUILD})
  if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(LIBS_ARCH "aarch64")
    set(CONDA_LIBS
        "libopenblas.so.0"
    )
  else()
    set(LIBS_ARCH "x86_64")
    set(CONDA_LIBS
        "libmkl_core.so"
        "libmkl_gnu_thread.so"
        "libmkl_intel_lp64.so"
        "libmkl_intel_thread.so"
        "libmkl_def.so"
        "libmkl_vml_def.so"
        "libmkl_rt.so"
        "libmkl_avx2.so"
        "libmkl_avx512.so"
        "libmkl_sequential.so"
        "libomp.so"
    )
  endif()
  set(PT_LIBS
      ${CONDA_LIBS}
      "libc10.so"
      "libc10_cuda.so"
      "libtorch.so"
      "libtorch_cpu.so"
      "libtorch_cuda.so"
      "libtorchvision.so"
  )
  set(OPENCV_LIBS
      "libopencv_video.so"
      "libopencv_videoio.so"
      "libopencv_highgui.so"
      "libopencv_imgcodecs.so"
      "libopencv_imgproc.so"
      "libopencv_core.so"
      "libpng16.so"
  )

  string(REPLACE ";" " " CONDA_LIBS_STR "${CONDA_LIBS}")

  add_custom_command(
    OUTPUT
      ${PT_LIBS}
      ${OPENCV_LIBS}
      LICENSE.pytorch
      include/torch
      include/torchvision
    COMMAND ${CMAKE_COMMAND} -E make_directory "include/torchvision"
    COMMAND docker pull ${TRITON_PYTORCH_DOCKER_IMAGE}
    COMMAND docker rm pytorch_backend_ptlib || echo "error ignored..." || true
    COMMAND docker create --name pytorch_backend_ptlib ${TRITON_PYTORCH_DOCKER_IMAGE}
    COMMAND /bin/sh -c "for i in ${CONDA_LIBS_STR} ; do echo copying $i && docker cp -L pytorch_backend_ptlib:/opt/conda/lib/$i $i ; done"
    COMMAND docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libc10.so libc10.so
    COMMAND docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libc10_cuda.so libc10_cuda.so
    COMMAND docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libtorch.so libtorch.so
    COMMAND docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so libtorch_cpu.so
    COMMAND docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so libtorch_cuda.so
    COMMAND docker cp pytorch_backend_ptlib:/opt/pytorch/vision/build/libtorchvision.so libtorchvision.so
    COMMAND docker cp pytorch_backend_ptlib:/opt/pytorch/pytorch/LICENSE LICENSE.pytorch
    COMMAND docker cp pytorch_backend_ptlib:/opt/conda/lib/python3.8/site-packages/torch/include include/torch
    COMMAND docker cp pytorch_backend_ptlib:/opt/pytorch/vision/torchvision/csrc include/torchvision/torchvision
    COMMAND docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_videoio.so.3.4.11 libopencv_videoio.so
    COMMAND docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_highgui.so.3.4.11 libopencv_highgui.so
    COMMAND docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_video.so.3.4.11 libopencv_video.so
    COMMAND docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_imgcodecs.so.3.4.11 libopencv_imgcodecs.so
    COMMAND docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_imgproc.so.3.4.11 libopencv_imgproc.so
    COMMAND docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libopencv_core.so.3.4.11 libopencv_core.so
    COMMAND docker cp pytorch_backend_ptlib:/usr/lib/${LIBS_ARCH}-linux-gnu/libpng16.so.16.37.0 libpng16.so
    COMMAND /bin/sh -c "if [ -f libmkl_def.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_def.so; fi"
    COMMAND /bin/sh -c "if [ -f libmkl_def.so ]; then patchelf --add-needed libmkl_core.so libmkl_def.so; fi"
    COMMAND /bin/sh -c "if [ -f libmkl_avx2.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_avx2.so; fi"
    COMMAND /bin/sh -c "if [ -f libmkl_avx2.so ]; then patchelf --add-needed libmkl_core.so libmkl_avx2.so; fi"
    COMMAND /bin/sh -c "if [ -f libmkl_avx512.so ]; then patchelf --add-needed libmkl_gnu_thread.so libmkl_avx512.so; fi"
    COMMAND /bin/sh -c "if [ -f libmkl_avx512.so ]; then patchelf --add-needed libmkl_core.so libmkl_avx512.so; fi"
    COMMAND docker rm pytorch_backend_ptlib
    COMMENT "Extracting pytorch and torchvision libraries and includes from ${TRITON_PYTORCH_DOCKER_IMAGE}"
    VERBATIM
  )
  add_custom_target(ptlib_target DEPENDS ${PT_LIBS} ${OPENCV_LIBS})
  add_library(ptlib SHARED IMPORTED GLOBAL)
  add_dependencies(ptlib ptlib_target)

  # Just one of the libs are enough to ensure the docker build
  set_target_properties(
    ptlib
    PROPERTIES
      IMPORTED_LOCATION libtorch.so
  )
endif() # TRITON_PYTORCH_DOCKER_BUILD

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
    ${Python3_INCLUDE_DIRS}
)

if (${TRITON_PYTORCH_DOCKER_BUILD})
  target_include_directories(
    triton-pytorch-backend
    PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}/include/torch
      ${CMAKE_CURRENT_BINARY_DIR}/include/torch/torch/csrc/api/include
      ${CMAKE_CURRENT_BINARY_DIR}/include/torchvision
  )
else()
  target_include_directories(
    triton-pytorch-backend
    PRIVATE ${TRITON_PYTORCH_INCLUDE_PATHS}
  )
endif() # TRITON_PYTORCH_DOCKER_BUILD

# Need to turn off -Werror due to Torchvision vision.h extern initialization
# Unfortunately gcc does not provide a specific flag to ignore the specific
# warning: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=45977
target_compile_features(triton-pytorch-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-pytorch-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits>
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

if (${TRITON_PYTORCH_DOCKER_BUILD})
  add_dependencies(
    triton-pytorch-backend
    ptlib
  )
endif() # TRITON_PYTORCH_DOCKER_BUILD

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
    -ltorchvision
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

if (${TRITON_PYTORCH_DOCKER_BUILD})
  set(PT_LIB_PATHS "")
  FOREACH(plib ${PT_LIBS} ${OPENCV_LIBS})
    set(PT_LIB_PATHS ${PT_LIB_PATHS} "${CMAKE_CURRENT_BINARY_DIR}/${plib}")
  ENDFOREACH(plib)

  install(
    FILES
      ${PT_LIB_PATHS}
      ${CMAKE_CURRENT_BINARY_DIR}/LICENSE.pytorch
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/pytorch
  )

  FOREACH(plib ${PT_LIBS} ${OPENCV_LIBS})
    install(
      CODE
        "EXECUTE_PROCESS(
          COMMAND patchelf --set-rpath \$ORIGIN ${plib}
          RESULT_VARIABLE PATCHELF_STATUS
          WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/backends/pytorch)
        if(PATCHELF_STATUS AND NOT PATCHELF_STATUS EQUAL 0)
          message(FATAL_ERROR \"FAILED: to run patchelf\")
        endif()"
    )
  ENDFOREACH(plib)

  set(OPENCV_VERSION "3.4")
  install(
    CODE
      "EXECUTE_PROCESS(
        COMMAND ln -sf libopencv_video.so libopencv_video.so.${OPENCV_VERSION}
        COMMAND ln -sf libopencv_videoio.so libopencv_videoio.so.${OPENCV_VERSION}
        COMMAND ln -sf libopencv_highgui.so libopencv_highgui.so.${OPENCV_VERSION}
        COMMAND ln -sf libopencv_imgcodecs.so libopencv_imgcodecs.so.${OPENCV_VERSION}
        COMMAND ln -sf libopencv_imgproc.so libopencv_imgproc.so.${OPENCV_VERSION}
        COMMAND ln -sf libopencv_core.so libopencv_core.so.${OPENCV_VERSION}
        COMMAND ln -sf libpng16.so libpng16.so.16
        RESULT_VARIABLE LINK_STATUS
        WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/backends/pytorch)
      if(LINK_STATUS AND NOT LINK_STATUS EQUAL 0)
        message(FATAL_ERROR \"FAILED: to create links\")
      endif()"
  )
endif() # TRITON_PYTORCH_DOCKER_BUILD

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
