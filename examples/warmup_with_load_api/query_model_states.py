import tritonclient.http as httpclient

triton_http_endpoint = "localhost:8000"

with httpclient.InferenceServerClient(triton_http_endpoint) as client:
    # collect all models loaded
    models = client.get_model_repository_index()
    print(models)
