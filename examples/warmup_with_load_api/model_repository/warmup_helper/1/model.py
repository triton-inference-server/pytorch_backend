import os
import time
import json
from pathlib import Path
import numpy as np
import tritonclient.grpc.model_config_pb2 as mc
from google.protobuf import json_format, text_format
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    This model will take a list of MODELS and FEATURE_DIMS, 
    and loop through them to send warm up request to different
    model and feature dims. 
    GROUP_SIZE_MIN and GROUP_SIZE_MAX will define the second dim 
    of the x and mask. The model will iterate through all group size
    in the range and insert the warmup config to the config.pbtxt 
    """
    model_repository_path = Path(__file__).parent.parent.parent
    model_wait_seconds = 60

    def execute(self, requests):
        request = requests[0]

        # get the list of model we want to warmup
        model_names = pb_utils.get_input_tensor_by_name(request, "MODELS").as_numpy().tolist()
        # get the feature dimensions of each model
        feature_dims = pb_utils.get_input_tensor_by_name(request, "FEATURE_DIMS").as_numpy().tolist()

        # get the range of group size
        group_size_min = pb_utils.get_input_tensor_by_name(request, "GROUP_SIZE_MIN").as_numpy()[0]
        group_size_max = pb_utils.get_input_tensor_by_name(request, "GROUP_SIZE_MAX").as_numpy()[0]

        error = None
        num_model = len(model_names)
        # validate if each model has a feature dimension
        if num_model != len(feature_dims):
            error = pb_utils.TritonError(
                message="model dimension doesn't match feature dimension.",
                code=pb_utils.TritonError.INTERNAL,
            )
        
        # validate the group size
        if group_size_max < group_size_min:
            error = pb_utils.TritonError(
                message=f"group_size_min={group_size_min} is greater than group_size_max={group_size_max}",
                code=pb_utils.TritonError.INTERNAL,
            )

        # return if validation failed
        if error is not None:
            responses = [pb_utils.InferenceResponse(error=error)]
            return responses
    
        loaded_model_names = []
        # loop through model and feature dim and try load the model with its warmup
        for i in range(num_model):
            model_name = model_names[i].decode("utf-8") # covert byte array to string
            feature_dim = feature_dims[i]

            try:
                self.try_load_model_with_warm_up(
                    model_name=model_name, 
                    feature_dim=feature_dim, 
                    group_size_min=group_size_min, 
                    group_size_max=group_size_max
                )
            except Exception as e:
                error = pb_utils.TritonError(
                    message=str(e),
                    code=pb_utils.TritonError.INTERNAL,
                )
                responses = [pb_utils.InferenceResponse(error=error)]
                return responses
            
            loaded_model_names.append(model_name)
        
        # response back which models are loaded
        responses = [
            pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("LOADED_MODELS", np.array(loaded_model_names, dtype=np.string_))
                ]
            )
        ]
        
        return responses

    def try_load_model_with_warm_up(self, model_name, feature_dim, group_size_min, group_size_max):
        
        # check if the model is loaded
        if pb_utils.is_model_ready(model_name=model_name):
            return

        # verify the model and the config.pbtxt exists in the repository
        model_path = os.path.join(str(self.model_repository_path), model_name)
        if not os.path.exists(model_path):
            raise Exception(f"{model_name} doesn't exists in the repository {self.model_repository_path}")

        model_config_path = os.path.join(model_path, "config.pbtxt")
        if not os.path.isfile(model_config_path):
            raise Exception(f"{model_name} doesn't have a config.pbtxt in the repository {self.model_repository_path}")

        json_config = None
        try:
            with open(model_config_path) as config_file:
                # load config.pbtxt into json format
                config = text_format.Parse(config_file.read(), mc.ModelConfig())
                json_config = json.loads(
                    json_format.MessageToJson(config, preserving_proto_field_name=True)
                )
                model_warmups = []

                # insert the warmup config with different group_size into the json config
                for group_size in range(group_size_min, group_size_max + 1):
                    warmup = {}
                    warmup["name"] = f"{model_name}_gs_{group_size}"
                    warmup["batch_size"] = 1
                    warmup["inputs"] = {}
                    warmup["inputs"]["x"] = {
                        "data_type": "TYPE_FP32",
                        "dims": [group_size, feature_dim],
                        "random_data": True
                    }
                    warmup["inputs"]["mask"] = {
                        "data_type": "TYPE_INT64",
                        "dims": [group_size],
                        "random_data": True
                    }
                    model_warmups.append(warmup)
                
                json_config["model_warmup"] = model_warmups
        except Exception as e:
            raise Exception(
                f"Unable config the warmup for model {model_name}. Exception: {str(e)}"
            )

        # convert the json config into json string
        model_config_json_str = json.dumps(json_config)

        # load the model with the newly added warmups
        # the warmups will be executed during loading the model.
        pb_utils.load_model(model_name=model_name, config=model_config_json_str)

        # wait for the model is loaded, config the timeout if this take times.
        timeout = 0
        while timeout < self.model_wait_seconds:
            if pb_utils.is_model_ready(model_name=model_name):
                return

            time.sleep(5)
            timeout += 5
        
        raise Exception(
            f"model {model_name} is not successfully loaded in the timeout {self.model_wait_seconds}s"
        )
