<!--
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# PyTorch (LibTorch) Backend

The Triton backend for [PyTorch](https://github.com/pytorch/pytorch). 
You can learn more about Triton backends in the [backend
repo](https://github.com/triton-inference-server/backend). Ask
questions or report problems on the [issues
page](https://github.com/triton-inference-server/server/issues).
This backend is designed to run [TorchScript](https://pytorch.org/docs/stable/jit.html)
models using the PyTorch C++ API. All models created in PyTorch
using the python API must be traced/scripted to produce a TorchScript
model.

Where can I ask general questions about Triton and Triton backends?
Be sure to read all the information below as well as the [general
Triton documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the main [server](https://github.com/triton-inference-server/server)
repo. If you don't find your answer there you can ask questions on the
main Triton [issues page](https://github.com/triton-inference-server/server/issues).

## Build the PyTorch Backend

Use a recent cmake to build. First install the required dependencies.

```
$ apt-get install patchelf rapidjson-dev python3-dev
```

An appropriate PyTorch container from [NGC](https://ngc.nvidia.com) must be used.
For example, to build a backend that uses the 21.02 version of the PyTorch
container from NGC:

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_PYTORCH_DOCKER_IMAGE="nvcr.io/nvidia/pytorch:21.02-py3" ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

## Build the PyTorch Backend With Custom PyTorch

Currently, Triton requires that a specially patched version of
PyTorch be used with the PyTorch backend. The full source for
these PyTorch versions are available as Docker images from
[NGC](https://ngc.nvidia.com). For example, the PyTorch version
compatible with the 21.02 release of Triton is available as
nvcr.io/nvidia/pytorch:21.02-py3.

Copy over the LibTorch and Torchvision headers and libraries from the
[PyTorch NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
into local directories. You can see which headers and libraries
are needed/copied from the docker. 

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_PYTORCH_INCLUDE_PATHS="<PATH_PREFIX>/torch;<PATH_PREFIX>/torch/torch/csrc/api/include;<PATH_PREFIX>/torchvision" -DTRITON_PYTORCH_LIB_PATHS="<LIB_PATH_PREFIX>" ..
$ make install
```

## Using the PyTorch Backend

### Parameters

Disabling the optimized execution of the PyTorch models is done through the Parameters section of the model's 'config.pbtxt' file.

* `DISABLE_OPTIMIZED_EXECUTION`: Boolean flag to disable the optimized execution
of TorchScript models. By default the optimized execuiton is always enabled. 

The initial calls to a loaded TorchScript model take extremely long. Due to this longer
model warmup [issue](https://github.com/pytorch/pytorch/issues/57894), Triton also allows 
execution of models without these optimizations. In some models, optimized execution 
does not benefit performance as seen [here](https://github.com/pytorch/pytorch/issues/19978)
and in other cases impacts performance negatively, as seen [here](https://github.com/pytorch/pytorch/issues/53824).

The section of model config file specifying these parameters will look like:

```
parameters: {
key: "DISABLE_OPTIMIZED_EXECUTION"
    value: {
    string_value:"true"
    }
}
```
