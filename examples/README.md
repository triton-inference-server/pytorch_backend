# Python BLS based script warmup

Triton Inference Server currently doesn't support adding custom scripts directly for model warmup. However, the Python backend's [Business Logic Scripting(BLS)](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#business-logic-scripting) feature can be used as a workaround to run Python “scripts” that help warm up models. The Python backend is included in all versions of the Triton server docker image.

Note that the PyTorch backend does not unload a model from memory after warmup — once loaded and warmed up, the model remains in memory until the user explicitly sends a model unload request to the server.

With BLS, users can load a Python script as a model in the Python backend. Inside that script, users can create and send inference requests to other models already loaded on the same Triton server.

## Approach 1

This approach is based on the contents in the `warmup_with_requests` folder. 

In this approach, we register a `warmup_helper` model in the same model repository as other PyTorch models. It accepts:

    a list of model names

    a list of feature dimensions

    a minimum and maximum group size

Users can implement logic to loop through the combinations of models, feature dimensions, and group sizes to send warmup inference requests to the target models.

Refer to `model.py` inside the `warmup_helper` model directory in the model repository under `warmup_with_requests` for implementation details.

Once the `warmup_helper` model is loaded into the Triton server, users only need to send a single request to `warmup_helper` to trigger the warmup for all target models. The script `warmup_models.py` under `warmup_with_requests` folder contains the logic for sending this request to the warmup_helper model.

Limitation:
All models must be loaded and marked as READY in the Triton server before the warmup. The user is responsible for explicitly triggering the warmup logic by sending a request to the `warmup_helper` model.

Example:

Start the Triton server with the model_repository under the `warmup_with_requests` folder. The repository contains four models: `model-v1`, `model-v2`, and `model-v3` are three PyTorch models that are identical except for their different feature dimensions. Each `model.pt` is generated using the `gen_model.py` script.
`warmup_helper` is a Python model that can send BLS inference requests to the other models.

```bash
 $ tritonserver --model-repository ./model_repository --log-verbose 2
```

Run the `warmup_models.py` against the tritonserver:
```bash
 $ python3 warmup_models.py
   ....
   number of warmup request sent: 30
   PASS: Warmup done for models: ['model-v1', 'model-v2', 'model-v3']  
```

## Approach 2

This approach is based on the contents in the `warmup_with_load_api` folder. 

In this approach, we register a `warmup_helper` model in the same model repository as other PyTorch models. It also accepts:

    a list of model names

    a list of feature dimensions

    a minimum and maximum group size

However, in the `model.py` inside the `warmup_helper` model directory, we will use the [Model Loading API](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#model-loading-api) to explicitly load each model. 

Refer to `model.py` inside the `warmup_helper` model directory in the model repository under `warmup_with_load_api` for implementation details. `model.py` This file dynamically inserts warmup configurations with different dimensions into each model's `config.pbtxt`, when the `load_model` API is called, Triton Server will automatically send the warmup requests to the model before marking it as ready.

Note: the model loading API is only supported if the server is running in [explicit model control mode](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html#model-control-mode-explicit). Additionally, the model loading API should only be used after the server has been running, which means that the model that is being loaded in the `model.py` should not be loaded during server startup.

Once the `warmup_helper` model is loaded into the Triton server, users only need to send a single request to `warmup_helper` to trigger loading all target models. The script `warmup_models.py` under `warmup_with_load_api` folder contains the logic for sending this request to the `warmup_helper` model.

Limitation:
The user is responsible for explicitly triggering the warmup logic by sending a request to the `warmup_helper` model.

One advantage of this approach is that the warmup process is initiated before a model is marked as ready or becomes visible. This means the user doesn’t need to manage or maintain the state of each model that is visible in the Triton server.

Example:

Start the Triton server with the model_repository under the `warmup_with_load_api` folder. The repository contains four models: `model-v1`, `model-v2`, and `model-v3` are three PyTorch models that are identical except for their different feature dimensions. Each `model.pt` is generated using the `gen_model.py` script.
`warmup_helper` is a Python model that will explicitly load other models.

```bash
 $ tritonserver --model-repository ./model_repository --log-verbose 2 --model-control-mode=explicit --load-model=warmup_helper
```

Run the `warmup_models.py` against the tritonserver:
```bash
 $ python3 warmup_models.py
   ....
   model loaded: ['model-v1', 'model-v2', 'model-v3']
   PASS: Warmup done for models: ['model-v1', 'model-v2', 'model-v3']   
```

You can also run the `query_model_states.py` agaist the server to check if the models are loaded.
```bash
 $ python3 query_model_states.py
   ....
  [{'name': 'model-v1', 'version': '1', 'state': 'READY'}, {'name': 'model-v2', 'version': '1', 'state': 'READY'}, {'name': 'model-v3', 'version': '1', 'state': 'READY'}, {'name': 'warmup_helper', 'version': '1', 'state': 'READY'}] 
```

You might wonder why we don’t simply add a warmup config to the warmup_helper model to automatically warm up every other model. The reason is that Triton Server need to have at least one model ready to make the server ready to be able to start serving the load model api. 
