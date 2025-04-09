import sys
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

warmup_helper = "warmup_helper"
triton_http_endpoint = "localhost:8000"

# Users can integrate this logic into their application.
# The function will return a list of model names that have been warmed up.
# Users can gate traffic to the models until this method returns,
# and then release the gate for the models returned by this method.
def warm_up_models(group_size_min, group_size_max):
    with httpclient.InferenceServerClient(triton_http_endpoint) as client:
        model_names = ["model-v1", "model-v2", "model-v3"]
        feature_dims = [186, 236, 286]

        # set up the input data for warmup helper
        model_names_data = np.array(model_names, dtype=np.string_)
        feature_dims_data = np.array(feature_dims, dtype=np.int32)
        group_size_min_data = np.array([group_size_min], dtype=np.int32)
        group_size_max_data = np.array([group_size_max], dtype=np.int32)
        inputs = [
            httpclient.InferInput(
                "MODELS", model_names_data.shape, np_to_triton_dtype(model_names_data.dtype)
            ),
            httpclient.InferInput(
                "FEATURE_DIMS", feature_dims_data.shape, np_to_triton_dtype(feature_dims_data.dtype)
            ),
            httpclient.InferInput(
                "GROUP_SIZE_MIN", group_size_min_data.shape, np_to_triton_dtype(group_size_min_data.dtype)
            ),
            httpclient.InferInput(
                "GROUP_SIZE_MAX", group_size_max_data.shape, np_to_triton_dtype(group_size_max_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(model_names_data)
        inputs[1].set_data_from_numpy(feature_dims_data)
        inputs[2].set_data_from_numpy(group_size_min_data)
        inputs[3].set_data_from_numpy(group_size_max_data)

        # setup the output container
        outputs = [httpclient.InferRequestedOutput("LOADED_MODELS")]

        # start the warmup
        response = client.infer(warmup_helper, inputs, request_id=str(1), outputs=outputs)
        # get the warmup result
        output_data = response.as_numpy("LOADED_MODELS")
        model_loaded = [model_name.decode("utf-8") for model_name in output_data.tolist()]

        print(f"model loaded: {model_loaded}")
        return model_loaded


if __name__ == "__main__":
    model_names = warm_up_models(group_size_min=1, group_size_max=10)
    print(f"PASS: Warmup done for models: {model_names}")
    sys.exit(0)
