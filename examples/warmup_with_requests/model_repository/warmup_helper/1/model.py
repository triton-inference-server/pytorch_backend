import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    This model will take a list of MODELS and FEATURE_DIMS, 
    and loop through them to send warm up request to different
    model and feature dims. 
    GROUP_SIZE_MIN and GROUP_SIZE_MAX will define the second dim 
    of the x and mask. The model will iterate through all group size
    in the range. 
    """

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

        if error is None:
            request_count = 0
            # loop through model and feature dim
            for i in range(num_model):
                model_name = model_names[i]
                feature_dim = feature_dims[i]

                # loop through group_size
                for group_size in range(group_size_min, group_size_max + 1):
                    input = pb_utils.Tensor(
                        "x", np.random.rand(1, group_size, feature_dim).astype(np.float32))
                    mask = pb_utils.Tensor(
                        "mask", np.random.rand(1, group_size).astype(np.int64))

                    infer_request = pb_utils.InferenceRequest(
                        model_name=model_name,
                        inputs=[input, mask],
                        requested_output_names=["output"],
                    )

                    # send the inference request to the model
                    # users could change this to async execution with async_exec() to use the asyncio to speedup potentially
                    infer_response = infer_request.exec()

                    # check if the warmup failed
                    if infer_response.has_error():
                        error = infer_response.error()
                        print(f"error happened during warming up the {model_name}!")
                        print(f"error: {error}")
                        responses = [pb_utils.InferenceResponse(error=error)]
                        return responses
                    else:
                        request_count += 1

            responses = [
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("REQUEST_COUNT", np.array([request_count], dtype=np.int32))
                    ]
                )
            ]
        else: # return error if failed the validation
            responses = [pb_utils.InferenceResponse(error=error)]
        
        return responses 
