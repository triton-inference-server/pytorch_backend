<!--
# Copyright 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# PyTorch (LibTorch) Backend

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

The Triton backend for
[PyTorch](https://github.com/pytorch/pytorch)
is designed to run
[TorchScript](https://pytorch.org/docs/stable/jit.html)
models using the PyTorch C++ API.
All models created in PyTorch using the python API must be traced/scripted to produce a TorchScript model.

You can learn more about Triton backends in the
[Triton Backend](https://github.com/triton-inference-server/backend)
repository.

Ask questions or report problems using
[Triton Server issues](https://github.com/triton-inference-server/server/issues).

Be sure to read all the information below as well as the
[general Triton documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the [Triton Server](https://github.com/triton-inference-server/server) repository.

## Build the PyTorch Backend

Use a recent cmake to build.
First install the required dependencies.

```bash
apt-get install rapidjson-dev python3-dev python3-pip
pip3 install patchelf==0.18.0
```

An appropriate PyTorch container from [NVIDIA NGC Catalog](https://ngc.nvidia.com) must be used.
For example, to build a backend that uses the 26.04 version of the PyTorch container from NGC:

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_PYTORCH_DOCKER_IMAGE="nvcr.io/nvidia/pytorch:26.04-py3" ..
make -j<N> install
```

The following required Triton repositories will be pulled and used in the build.
By default, the `main` head will be used for each repository but the listed CMake argument can be used to override the value.

* triton-inference-server/backend: `-DTRITON_BACKEND_REPO_TAG=[tag]`
* triton-inference-server/core: `-DTRITON_CORE_REPO_TAG=[tag]`
* triton-inference-server/common: `-DTRITON_COMMON_REPO_TAG=[tag]`

## Build the PyTorch Backend With Custom PyTorch

Currently, Triton requires that a specially patched version of PyTorch be used with the PyTorch backend.
The full source for these PyTorch versions are available as Docker images from
[NGC](https://ngc.nvidia.com).

For example, the PyTorch version compatible with the 26.04 release of Triton is available as `nvcr.io/nvidia/pytorch:26.04-py3` which supports PyTorch version `2.12.0a0`.

> [!NOTE]
> Additional details and version information can be found in the container's
> [release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-04.html#rel-26-04).

Copy over the LibTorch and TorchVision headers and libraries from the
[PyTorch NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
into local directories.
You can see which headers and libraries are needed/copied from the docker.

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_PYTORCH_INCLUDE_PATHS="<PATH_PREFIX>/torch;<PATH_PREFIX>/torch/torch/csrc/api/include;<PATH_PREFIX>/torchvision" -DTRITON_PYTORCH_LIB_PATHS="<LIB_PATH_PREFIX>" ..
make install
```

## Using the PyTorch Backend

## AOT Inductor Support (Beta)

Starting with the 26.03 release of Triton, support for ahead-of-time (AOT) inductor compiled and packaged model archives is available.
The new model archive package (PT2) can be generated using the following example, and generally uses the `.pt2` file extension.

Example Python code for creating a `model.pt2` packaged model archive.

```python
    ep = torch.export.export(model, sample_inputs)
    torch._inductor.compile_and_package(ep, OUTPUT_FOLDER + "/model.pt2")
```

The model repository should look like:

```bash
model_repository/
`-- model_directory
    |-- 1
    |   `-- model.pt2
    `-- config.pbtxt
```

### Model Configuration

Triton will load a model from a PyTorch PT2 model archive when `platform: "torch_aoti"` is provided as part of a model's `config.pbtxt`.
Starting with Triton Server 26.05, Triton will determine the names of the input and output tensors at runtime based on the loaded model's call specification (provided by the PyTorch runtime).
There are two supported naming conventions:

#### Ordinal Base I/O Names

Input and outputs can be addressed using names based on their position in the call specification generated by the PyTorch runtime when a PT2 archive model is loaded.
To address an input by its ordinal, the input must be named `INPUT__` followed by it zero-based index.
For example: `name: "INPUT__0"`

Similarly, to address an output by its ordinal, the output must be named `OUTPUT__` followed by its zero-based index.
For example: `name: "OUTPUT__0"`

#### Forward Based I/O Names

Inputs and outputs can be addressed using names based on the model's forward function's specification and the sample inputs used when compiling the model.

Inputs are broken down into two categories based on how `torch.export.export` handles them.
The first category is "arguments", which is always treated as a tuple and addressed using `ARGS` as the name prefix.
The second category is "keyword arguments", which is always treated as a dictionary and addressed using `KWARGS` as the name prefix.

Outputs can be either a tensor, a tuple, or a dictionary and are addressed using the `RESULT` prefix.
Technically lists supported as well, but they effectively treated as a tuples.

#### Basic Rules for Forward Based I/O Names

* Addressing a tuple uses Pythonic tuple semantics such as `ARGS[0]`, `ARGS[0]`, and `RESULT[0]`.
* Addressing a list is identical to addressing a tuple.
* Addressing a dictionary uses semi-Pythonic dictionary semantics such as `RESULT[key]`, `KWARGS[key]`, and `ARGS[3][key]`.
* When the a model's output is a single tensor, it is addressed as `RESULT` with no suffix accessors.

#### Example

```python
import torch

SHAPE = (1, 16)

# Define a model with complex inputs and outputs.
# Include the eventual name mappings based on the return value of the model and sample inputs.
class ExampleAotiModel(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(
    self,
    hdata: torch.Tensor,
    vdata: torch.Tensor,
    options: dict[str, torch.Tensor],
    *
    kwag1: torch.Tensor,
    kw_arg2: torch.Tensor
  ) -> dict[
    str, torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | dict[str, torch.Tensor]
  ]:
    return {
      # OUTPUT__0 or RESULT[AAA]
      "AAA": hdata + vdata,
      # OUTPUT__1 or RESULT[ZZZ]
      # Notice that the ordinal name of ZZZ is OUTPUT__1 due to its position
      # in the returned dictionary object.
      "ZZZ": hdata - vdata,
      "BBB": (
        # OUTPUT_2 or RESULT[BBB][0]
        hdata,
        # OUTPUT_3 or RESULT[BBB][1]
        vdata,
      ),
      # OUTPUT__4 or RESULT[CCC][keyA]
      # OUTPUT__5 or RESULT[CCC][key2]
      "CCC": options, # `options` is a dictionary.
      "DDD": {
        # OUTPUT__6 or RESULT[DDD][KA]
        "KA": kwarg1,
        # OUTPUT__7 or RESULT[DDD][KB]
        "KB": kw_arg2,
      }
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExampleAotiModel()
model.to(device)
model = model.eval()

sample_args = (
  # INPUT__0 or ARGS[0]
  torch.zeros(SHAPE, dtype=torch.float16, device=device),
  # INPUT__1 or ARGS[1]
  torch.zeros(SHAPE, dtype=torch.float16, device=device),
  {
    # INPUT__2 or ARGS[2][keyA]
    "keyA": torch.zeros(SHAPE, dtype=torch.float16, device=device),
    # INPUT__3 or ARGS[2][key2]
    "key2": torch.zeros(SHAPE, dtype=torch.float16, device=device),
  },
)
sample_kwargs = {
  # INPUT__4 or KWARGS[kwarg1]
  "kwarg1": torch.zeros(SHAPE, dtype=torch.float16, device=device),
  # INPUT__5 or KWARGS[kw_arg2]
  "kw_arg2": torch.zeros(SHAPE, dtype=torch.float16, device=device),
}

# Export and then compile and package the model.
print("Exporting and packaging the model...")
exported_model = torch.export.export(model, args=sample_args, kwargs=sample_kwargs)
torch._inductor.aoti_compile_and_package(exported_model, package_path="model.pt2")

# Load the compiled and packaged model and print its call specification.
compiled_model = torch._inductor.aoti_load_package("model.pt2")

print("Compiled model call spec:")
for spec in compiled_model.loader.get_call_spec():
  print(spec)
```

Using the above example, the call specification of the compiled and packaged model would inform Triton of the following:

* There are 5 inputs.
  * The first 2 inputs are unnamed.
    * Therefore they'll be addressed as `"ARGS[0]"` and `"ARGS[1]"`.
  * The third input is a dictionary with 2 keys: `"keyA"` and `"key2"`.
    * Therefore they'll be addressed as `"ARGS[2][keyA]"` and `"ARGS[2][key2]"`.
  * The fourth and fifth inputs are keyword arguments (kwargs) named `"kwarg1"` and `"kw_arg2"`.
    * Therefore they'll be addressed as `"KWARGS[kwarg1]"` and `"KWARGS[kw_arg2]"`.
* There's a single dictionary returned as the output that contains the following keys: `"AAA"`, `"BBB"`, `"CCC"`, `"DDD"`, and `"ZZZ"`.
  * Keys `"AAA"` and `"ZZZ"` are simple tensors.
    * Therefore they'll be addressed as `"RESULT[AAA]"` and `"RESULT[ZZZ]"` respectively.
  * Key `"BBB"` is a tuple with 2 elements.
    * Therefore they'll be addressed as `"RESULT[BBB][0]"` and `"RESULT[BBB][1]"`.
  * Key `"CCC"` is a dictionary with the following keys: `"keyA"` and `"key2"`.
    * Therefore they'll be addressed as `"RESULT[CCC][keyA]"` and `"RESULT[CCC][key2]"` respectively.
  * Key `"DDD"` is a dictionary with the following keys: `"KA"` and `"KB"`.
    * Therefore they'll be addressed as `"RESULT[DDD][KA]"` and `"RESULT[DDD][KB]"` respectively.

Using the above knowledge, we can create the necessary `config.pbtxt` file to enable Triton to properly load and interact with the model.

> [!NOTE]
> The below example is intended to display how model names function, and is truncated for brevity.
> The configuration is incomplete and cannot be used as-is by Triton Server.

```proto
backend: "pytorch"
platform: "torch_aoti"
input: [
  { name: "ARGS[0]" ... }, // "INPUT__0"
  { name: "ARGS[1]" ... }, // "INPUT__1"
  { name: "ARGS[2][keyA]" ... }, // "INPUT__2"
  { name: "ARGS[2][key2]" ... }, // "INPUT__3",
  { name: "KWARGS[kwarg1]" ... }, // INPUT__4,
  { name: "KWARGS[kw_arg2]" ... } // "INPUT__5"
]
output: [
  // Output results sorted alphabetically instead by ordinal.
  { name: "RESULT[AAA]" ... }, // "OUTPUT__0",
  { name: "RESULT[BBB][0]" ... }, // "OUTPUT__2",
  { name: "RESULT[BBB][1]" ... }, // "OUTPUT__3",
  { name: "RESULT[CCC][keyA]" ... }, // "OUTPUT__4",
  { name: "RESULT[CCC][key2]" ... }, // "OUTPUT__5",
  { name: "RESULT[DDD][KA]" ... }, // "OUTPUT__6",
  { name: "RESULT[DDD][KB]" ... }, // "OUTPUT__7",
  { name: "RESULT[ZZZ]" ... } // "OUTPUT__1"
  // Notice RESULT[ZZZ] would be OUTPUT__1 when using ordinal names.
]
```

> [!IMPORTANT]
> Only `dict`, `tuple`, `list`, and `torch.Tensor` types are supported,
> and all leaf, or value, types *MUST* be `torch.Tensor` values.
> Inputs or outputs of other types are *NOT* supported.
>
> Dictionary keys cannot contain `"`, `[`, or `]`, nor can they contain whitespace or non-printable characters.

> [!WARNING]
> Support for batch sizes greater than 1 and for sequence batching for AOT Inductor compiled models has not be completed.
> These Triton Server features are currently unavailable for PyTorch models compiled using AOT Inductor and packaged as a PT2 model archive.

### PyTorch 2.0 Models

PyTorch 2.0 features are available.
However, Triton's PyTorch backend requires a serialized representation of the model in the form a `model.pt` file.
The serialized representation of the model can be generated using PyTorch's
[`torch.save()`](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#id1)
function to generate the `model.pt` file.
Support for PyTorch's new [PT2 model archive package](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/pt2_archive.html) is currently in-development and can be tested as of Triton's 26.03 or later releases.

Example Python code for creating `model.pt` packaged model archive:

```python
  traced = torch.jit.script(model)
  traced.save(OUTPUT_FOLDER + "/model.pt")
```

The model repository should look like:

```bash
model_repository/
`-- model_directory
    |-- 1
    |   `-- model.pt
    `-- config.pbtxt
```

Where `model.pt` is the serialized representation of the model.

### TorchScript Models

The model repository should look like:

```bash
model_repository/
`-- model_directory
    |-- 1
    |   `-- model.pt
    `-- config.pbtxt
```

The `model.pt` is the TorchScript model file.

## Configuration

Triton exposes some flags to control the execution mode of the TorchScript models through the `Parameters` section of the model's `config.pbtxt` file.

### Configuration Options

* `default_model_name`:
  Instructs the Triton PyTorch backend to load the model from a file of the given name.

  The model config specifying the option would look like:

  ```proto
  default_model_name: "another_file_name.pt2"
  ```

* `platform`:
  Instructs the Triton PyTorch backend to load the model using either the AOT Inductor or legacy LibTorch framework.
  Legacy LibTorch framework is selected using `pytorch_libtorch`.
  AOT Inductor framework is selected using `torch_aoti`.

  The model config specifying the platform would look like:

  ```proto
  platform: "torch_aoti"
  ```

### Parameters

* `DISABLE_OPTIMIZED_EXECUTION`:
  Boolean flag to disable the optimized execution of TorchScript models.
  By default, the optimized execution is always enabled.

  The initial calls to a loaded TorchScript model take a significant amount of time.
  Due to this longer model warmup
  ([pytorch #57894](https://github.com/pytorch/pytorch/issues/57894)),
  Triton also allows execution of models without these optimizations.
  In some models, optimized execution does not benefit performance
  ([pytorch #19978](https://github.com/pytorch/pytorch/issues/19978))
  and in other cases impacts performance negatively
  ([pytorch #53824](https://github.com/pytorch/pytorch/issues/53824)).

  The section of model config file specifying this parameter will look like:

  ```proto
  parameters: {
    key: "DISABLE_OPTIMIZED_EXECUTION"
    value: { string_value: "true" }
  }
  ```

* `INFERENCE_MODE`:

  Boolean flag to enable the Inference Mode execution of TorchScript models.
  By default, the inference mode is enabled.

  [InferenceMode](https://pytorch.org/cppdocs/notes/inference_mode.html) is a new RAII guard analogous to `NoGradMode` to be used when you are certain your operations will have no interactions with autograd.
  Compared to `NoGradMode`, code run under this mode gets better performance by disabling autograd.

  Please note that in some models, InferenceMode might not benefit performance and in fewer cases might impact performance negatively.

  To enable inference mode, use the configuration example below:

  ```proto
  parameters: {
    key: "INFERENCE_MODE"
    value: { string_value: "true" }
  }
  ```

* `DISABLE_CUDNN`:

  Boolean flag to disable the cuDNN library.
  By default, cuDNN is enabled.

  [cuDNN](https://developer.nvidia.com/cudnn) is a GPU-accelerated library of primitives for deep neural networks.
  It provides highly tuned implementations for standard routines.

  Typically, models run with cuDNN enabled execute faster.
  However there are some exceptions where using cuDNN can be slower, cause higher memory usage, or result in errors.

  To disable cuDNN, use the configuration example below:

  ```proto
  parameters: {
    key: "DISABLE_CUDNN"
    value: { string_value: "true" }
  }
  ```

* `ENABLE_WEIGHT_SHARING`:

  Boolean flag to enable model instances on the same device to share weights.
  This optimization should not be used with stateful models.
  If not specified, weight sharing is disabled.

  To enable weight sharing, use the configuration example below:

  ```proto
  parameters: {
    key: "ENABLE_WEIGHT_SHARING"
    value: { string_value: "true" }
  }
  ```

* `ENABLE_CACHE_CLEANING`:

  Boolean flag to enable CUDA cache cleaning after each model execution.
  If not specified, cache cleaning is disabled.
  This flag has no effect if model is on CPU.

  Setting this flag to true will likely negatively impact the performance due to additional CUDA cache cleaning operation after each model execution.
  Therefore, you should only use this flag if you serve multiple models with Triton and encounter CUDA out-of-memory issues during model executions.

  To enable cleaning of the CUDA cache after every execution, use the configuration example below:

  ```proto
  parameters: {
    key: "ENABLE_CACHE_CLEANING"
    value: { string_value: "true" }
  }
  ```

* `INTER_OP_THREAD_COUNT`:

  PyTorch allows using multiple CPU threads during TorchScript model inference.
  One or more inference threads execute a model’s forward pass on the given inputs.
  Each inference thread invokes a JIT interpreter that executes the ops of a model inline, one by one.

  This parameter sets the size of this thread pool.
  The default value of this setting is the number of cpu cores.

  > [!TIP]
  > Refer to
  > [CPU Threading TorchScript](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
  > on how to set this parameter properly.

  To set the inter-op thread count, use the configuration example below:

  ```proto
  parameters: {
    key: "INTER_OP_THREAD_COUNT"
    value: { string_value: "1" }
  }
  ```

> [!NOTE]
> This parameter is set globally for the PyTorch backend.
> The value from the first model config file that specifies this parameter will be used.
> Subsequent values from other model config files, if different, will be ignored.

* `INTRA_OP_THREAD_COUNT`:

  In addition to the inter-op parallelism, PyTorch can also utilize multiple threads within the ops (intra-op parallelism).
  This can be useful in many cases, including element-wise ops on large tensors, convolutions, GEMMs, embedding lookups and others.

  The default value for this setting is the number of CPU cores.

  > [!TIP]
  > Refer to
  > [CPU Threading TorchScript](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
  > on how to set this parameter properly.

  To set the intra-op thread count, use the configuration example below:

  ```proto
  parameters: {
    key: "INTRA_OP_THREAD_COUNT"
    value: { string_value: "1" }
  }
  ```

* **Additional Optimizations**:

  Three additional boolean parameters are available to disable certain Torch optimizations that can sometimes cause latency regressions in models with complex execution modes and dynamic shapes.
  If not specified, all are enabled by default.

    `ENABLE_JIT_EXECUTOR`

    `ENABLE_JIT_PROFILING`

### Model Instance Group Kind

The PyTorch backend supports the following kinds of
[Model Instance Groups](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
where the input tensors are placed as follows:

* `KIND_GPU`:

  Inputs are prepared on the GPU device associated with the model instance.

* `KIND_CPU`:

  Inputs are prepared on the CPU.

* `KIND_MODEL`:

  Inputs are prepared on the CPU.
  When loading the model, the backend does not choose the GPU device for the model;
  instead, it respects the device(s) specified in the model and uses them as they are during inference.

  This is useful when the model internally utilizes multiple GPUs, as demonstrated in
  [this example model](https://github.com/triton-inference-server/server/blob/main/qa/L0_libtorch_instance_group_kind_model/gen_models.py).

  > [!IMPORTANT]
  > If a device is not specified in the model, the backend uses the first available GPU device.

To set the model instance group, use the configuration example below:

```proto
instance_group {
   count: 2
   kind: KIND_GPU
}
```

### Customization

The following PyTorch settings may be customized by setting parameters on the
`config.pbtxt`.

> [!IMPORTANT]
> The following options only apply when `platform: "pytorch_libtorch"` is used.

[`torch.set_num_threads(int)`](https://pytorch.org/docs/stable/generated/torch.set_num_threads.html#torch.set_num_threads)

* Key: `NUM_THREADS`
* Value: The number of threads used for intra-op parallelism on CPU.

[`torch.set_num_interop_threads(int)`](https://pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html#torch.set_num_interop_threads)

* Key: `NUM_INTEROP_THREADS`
* Value: The number of threads used for interop parallelism (e.g. in JIT interpreter) on CPU.

[`torch.compile()` parameters](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile)

* Key: `TORCH_COMPILE_OPTIONAL_PARAMETERS`
* Value: Any of following parameter(s) encoded as a JSON object.
  * `fullgraph` (`bool`): Whether it is ok to break model into several subgraphs.
  * `dynamic` (`bool`): Use dynamic shape tracing.
  * `backend` (`str`): The backend to be used.
  * `mode` (`str`): Can be either `"default"`, `"reduce-overhead"`, or `"max-autotune"`.
  * `options` (`dict`): A dictionary of options to pass to the backend.
  * `disable` (`bool`): Turn `torch.compile()` into a no-op for testing.

For example:

```proto
parameters: {
  key: "NUM_THREADS"
  value: { string_value: "4" }
}
parameters: {
  key: "TORCH_COMPILE_OPTIONAL_PARAMETERS"
  value: { string_value: "{\"disable\": true}" }
}
```

## Important Notes

* The execution of PyTorch model on GPU is asynchronous in nature.
  See
  [CUDA Asynchronous Execution](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
  for additional details.
  Consequently, an error in PyTorch model execution may be raised during the next few inference requests to the server.
  Setting environment variable `CUDA_LAUNCH_BLOCKING=1` when launching server will help in correctly debugging failing cases by forcing synchronous execution.

  * The PyTorch model in such cases may or may not recover from the failed state and a restart of the server may be required to continue serving successfully.

* PyTorch does not support Tensor of Strings but it does support models that accept a List of Strings as input(s) / produces a List of String as output(s).
  For these models Triton allows users to pass String input(s)/receive String output(s) using the String datatype.
  As a limitation of using List instead of Tensor for String I/O, only for 1-dimensional input(s)/output(s) are supported for I/O of String type.

* In a multi-GPU environment, a potential runtime issue can occur when using
  [Tracing](https://pytorch.org/docs/stable/generated/torch.jit.trace.html)
  to generate a
  [TorchScript](https://pytorch.org/docs/stable/jit.html)
  model.
  This issue arises due to a device mismatch between the model instance and the tensor.

  By default, Triton creates a single execution instance of the model for each available GPU.
  The runtime error occurs when a request is sent to a model instance with a different GPU device from the one used during the TorchScript generation process.

  To address this problem, it is highly recommended to use
  [Scripting](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script)
  instead of Tracing for model generation in a multi-GPU environment.
  Scripting avoids the device mismatch issue and ensures compatibility with different GPUs when used with Triton.

  However, if using Tracing is unavoidable, there is a workaround available.
  You can explicitly specify the GPU device for the model instance in the
  [model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
  to ensure that the model instance and the tensors used for inference are assigned to the same GPU device as on which the model was traced.

* When using `KIND_MODEL` as model instance kind, the default device of the first parameter on the model is used.

> [!WARNING]
>
> * Python functions optimizable by `torch.compile` may not be served directly in the `model.py` file, they need to be enclosed by a class extending the
  [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
>
> * Model weights cannot be shared across multiple instances on the same GPU device.
