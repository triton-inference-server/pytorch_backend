<!--
# Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
For example, to build a backend that uses the 23.04 version of the PyTorch
container from NGC:

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_PYTORCH_DOCKER_IMAGE="nvcr.io/nvidia/pytorch:23.04-py3" ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default, the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

## Build the PyTorch Backend With Custom PyTorch

Currently, Triton requires that a specially patched version of
PyTorch be used with the PyTorch backend. The full source for
these PyTorch versions are available as Docker images from
[NGC](https://ngc.nvidia.com). For example, the PyTorch version
compatible with the 22.12 release of Triton is available as
nvcr.io/nvidia/pytorch:22.12-py3.

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

Triton exposes some flags to control the execution mode of the TorchScript models through
the Parameters section of the model's `config.pbtxt` file.

* `DISABLE_OPTIMIZED_EXECUTION`: Boolean flag to disable the optimized execution
of TorchScript models. By default, the optimized execution is always enabled.

The initial calls to a loaded TorchScript model take extremely long. Due to this longer
model warmup [issue](https://github.com/pytorch/pytorch/issues/57894), Triton also allows
execution of models without these optimizations. In some models, optimized execution
does not benefit performance as seen [here](https://github.com/pytorch/pytorch/issues/19978)
and in other cases impacts performance negatively, as seen [here](https://github.com/pytorch/pytorch/issues/53824).

The section of model config file specifying this parameter will look like:

```
parameters: {
key: "DISABLE_OPTIMIZED_EXECUTION"
    value: {
    string_value: "true"
    }
}
```

* `INFERENCE_MODE`: Boolean flag to enable the Inference Mode execution
of TorchScript models. By default, the inference mode is enabled.

[InferenceMode](https://pytorch.org/cppdocs/notes/inference_mode.html) is a new
RAII guard analogous to NoGradMode to be used when you are certain your operations
will have no interactions with autograd. Compared to NoGradMode, code run under
this mode gets better performance by disabling autograd.

Please note that in some models, InferenceMode might not benefit performance
and in fewer cases might impact performance negatively.

The section of model config file specifying this parameter will look like:

```
parameters: {
key: "INFERENCE_MODE"
    value: {
    string_value: "true"
    }
}
```

* `DISABLE_CUDNN`: Boolean flag to disable the cuDNN library. By default, cuDNN is enabled.

[cuDNN](https://developer.nvidia.com/cudnn) is a GPU-accelerated library of primitives for 
deep neural networks. cuDNN provides highly tuned implementations for standard routines.

Typically, models run with cuDNN enabled are faster. However there are some exceptions
where using cuDNN can be slower, cause higher memory usage or result in errors. 


The section of model config file specifying this parameter will look like:

```
parameters: {
key: "DISABLE_CUDNN"
    value: {
    string_value: "true"
    }
}
```

* `ENABLE_WEIGHT_SHARING`: Boolean flag to enable model instances on the same device to
share weights. This optimization should not be used with stateful models. If not specified,
weight sharing is disabled.

The section of model config file specifying this parameter will look like:

```
parameters: {
key: "ENABLE_WEIGHT_SHARING"
    value: {
    string_value: "true"
    }
}
```

* `ENABLE_CACHE_CLEANING`: Boolean flag to enable CUDA cache cleaning after each model execution.
If not specified, cache cleaning is disabled. This flag has no effect if model is on CPU.
Setting this flag to true will negatively impact the performance due to additional CUDA cache
cleaning operation after each model execution. Therefore, you should only use this flag if you
serve multiple models with Triton and encounter CUDA out of memory issue during model executions.

The section of model config file specifying this parameter will look like:

```
parameters: {
key: "ENABLE_CACHE_CLEANING"
    value: {
    string_value:"true"
    }
}
```

* `INTER_OP_THREAD_COUNT`:

PyTorch allows using multiple CPU threads during TorchScript model inference.
One or more inference threads execute a model’s forward pass on the given
inputs. Each inference thread invokes a JIT interpreter that executes the ops
of a model inline, one by one. This parameter sets the size of this thread
pool. The default value of this setting is the number of cpu cores. Please refer
to [this](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
document on how to set this parameter properly.

The section of model config file specifying this parameter will look like:

```
parameters: {
key: "INTER_OP_THREAD_COUNT"
    value: {
    string_value:"1"
    }
}
```

* `INTRA_OP_THREAD_COUNT`:

In addition to the inter-op parallelism, PyTorch can also utilize multiple threads
within the ops (intra-op parallelism). This can be useful in many cases, including
element-wise ops on large tensors, convolutions, GEMMs, embedding lookups and
others. The default value for this setting is the number of CPU cores. Please refer
to [this](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
document on how to set this parameter properly.

The section of model config file specifying this parameter will look like:

```
parameters: {
key: "INTRA_OP_THREAD_COUNT"
    value: {
    string_value:"1"
    }
}
```

* Additional Optimizations: Three additional boolean parameters are available to disable
certain Torch optimizations that can sometimes cause latency regressions in models with
complex execution modes and dynamic shapes. If not specified, all are enabled by default.

    `ENABLE_JIT_EXECUTOR`

    `ENABLE_JIT_PROFILING`

### Support

#### Model Instance Group Kind

The PyTorch backend supports the following kinds of
[Model Instance Groups](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
where the input tensors are placed as follows:

* `KIND_GPU`: Inputs are prepared on the GPU device associated with the model
instance.

* `KIND_CPU`: Inputs are prepared on the CPU.

* `KIND_MODEL`: Inputs are prepared on the CPU. When loading the model, the
backend does not choose the GPU device for the model; instead, it respects the
device(s) specified in the model and uses them as they are during inference.
This is useful when the model internally utilizes multiple GPUs, as demonstrated
in this
[example model](https://github.com/triton-inference-server/server/blob/main/qa/L0_libtorch_instance_group_kind_model/gen_models.py).
If no device is specified in the model, the backend uses the first available
GPU device. This feature is available starting in the 23.06 release.

### Important Notes

* The execution of PyTorch model on GPU is asynchronous in nature. See
  [here](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
  for more details. Consequently, an error in PyTorch model execution may
  be raised during the next few inference requests to the server. Setting
  environment variable `CUDA_LAUNCH_BLOCKING=1` when launching server will
  help in correctly debugging failing cases by forcing synchronous execution.
  * The PyTorch model in such cases may or may not recover from the failed
    state and a restart of the server may be required to continue serving
    successfully.

* PyTorch does not support Tensor of Strings but it does support models that
accept a List of Strings as input(s) / produces a List of String as output(s).
For these models Triton allows users to pass String input(s)/receive String
output(s) using the String datatype. As a limitation of using List instead of
Tensor for String I/O, only for 1-dimensional input(s)/output(s) are supported
for I/O of String type.

* In a multi-GPU environment, a potential runtime issue can occur when using
[Tracing](https://pytorch.org/docs/stable/generated/torch.jit.trace.html)
to generate a
[TorchScript](https://pytorch.org/docs/stable/jit.html) model. This issue
arises due to a device mismatch between the model instance and the tensor. By
default, Triton creates a single execution instance of the model for each
available GPU. The runtime error occurs when a request is sent to a model
instance with a different GPU device from the one used during the TorchScript
generation process. To address this problem, it is highly recommended to use
[Scripting](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script)
instead of Tracing for model generation in a multi-GPU environment. Scripting
avoids the device mismatch issue and ensures compatibility with different GPUs
when used with Triton. However, if using Tracing is unavoidable, there is a
workaround available. You can explicitly specify the GPU device for the model
instance in the
[model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
to ensure that the model instance and the tensors used for inference are
assigned to the same GPU device as on which the model was traced.

# PyTorch 2.0 Backend \[Experimental\]

> [!WARNING]
> *This feature is subject to change and removal.*

Starting from 24.01, PyTorch models can be served directly via
[Python runtime](src/model.py). By default, Triton will use the
[LibTorch runtime](#pytorch-libtorch-backend) for PyTorch models. To use Python
runtime, provide the following
[runtime setting](https://github.com/triton-inference-server/backend/blob/main/README.md#backend-shared-library)
in the model configuration:

```
runtime: "model.py"
```

## Dependencies

### Python backend dependency

This feature depends on
[Python backend](https://github.com/triton-inference-server/python_backend),
see
[Python-based Backends](https://github.com/triton-inference-server/backend/blob/main/docs/python_based_backends.md)
for more details.

### PyTorch dependency

This feature will take advantage of the
[`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile)
optimization, make sure the
[PyTorch 2.0+ pip package](https://pypi.org/project/torch) is available in the
same Python environment.

Alternatively, a [Python Execution Environment](#using-custom-python-execution-environments)
with the PyTorch dependency may be used. It can be created with the
[provided script](tools/gen_pb_exec_env.sh). The resulting
`pb_exec_env_model.py.tar.gz` file should be placed at the same
[backend shared library](https://github.com/triton-inference-server/backend/blob/main/README.md#backend-shared-library)
directory as the [Python runtime](src/model.py).

## Model Layout

### PyTorch 2.0 models

The model repository should look like:

```
model_repository/
`-- model_directory
    |-- 1
    |   |-- model.py
    |   `-- [model.pt]
    `-- config.pbtxt
```

The `model.py` contains the class definition of the PyTorch model. The class
should extend the
[`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
The `model.pt` may be optionally provided which contains the saved
[`state_dict`](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference)
of the model.

### TorchScript models

The model repository should look like:

```
model_repository/
`-- model_directory
    |-- 1
    |   `-- model.pt
    `-- config.pbtxt
```

The `model.pt` is the TorchScript model file.

## Customization

The following PyTorch settings may be customized by setting parameters on the
`config.pbtxt`.

[`torch.set_num_threads(int)`](https://pytorch.org/docs/stable/generated/torch.set_num_threads.html#torch.set_num_threads)
- Key: NUM_THREADS
- Value: The number of threads used for intraop parallelism on CPU.

[`torch.set_num_interop_threads(int)`](https://pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html#torch.set_num_interop_threads)
- Key: NUM_INTEROP_THREADS
- Value: The number of threads used for interop parallelism (e.g. in JIT
interpreter) on CPU.

[`torch.compile()` parameters](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile)
- Key: TORCH_COMPILE_OPTIONAL_PARAMETERS
- Value: Any of following parameter(s) encoded as a JSON object.
  - fullgraph (*bool*): Whether it is ok to break model into several subgraphs.
  - dynamic (*bool*): Use dynamic shape tracing.
  - backend (*str*): The backend to be used.
  - mode (*str*): Can be either "default", "reduce-overhead" or "max-autotune".
  - options (*dict*): A dictionary of options to pass to the backend.
  - disable (*bool*): Turn `torch.compile()` into a no-op for testing.

For example:
```
parameters: {
    key: "NUM_THREADS"
    value: { string_value: "4" }
}
parameters: {
    key: "TORCH_COMPILE_OPTIONAL_PARAMETERS"
    value: { string_value: "{\"disable\": true}" }
}
```

## Limitations

Following are few known limitations of this feature:
- Python functions optimizable by `torch.compile` may not be served directly in
the `model.py` file, they need to be enclosed by a class extending the
[`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
- Model weights cannot be shared across multiple instances on the same GPU
device.
- When using `KIND_MODEL` as model instance kind, the default device of the
first parameter on the model is used.
