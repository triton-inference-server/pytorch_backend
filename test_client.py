# test_client.py
import tritonclient.http as httpclient
import numpy as np

# Create client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare input
input_data = np.random.randn(5, 10).astype(np.float32)
inputs = [httpclient.InferInput("INPUT__0", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

# Request outputs by dict key names
outputs = [
    httpclient.InferRequestedOutput("logits"),
    httpclient.InferRequestedOutput("embeddings")
]

# Infer
results = client.infer("dict_model", inputs, outputs=outputs)

# Check output names
print("Output names:", results.get_response())
print("Logits shape:", results.as_numpy("logits").shape)
print("Embeddings shape:", results.as_numpy("embeddings").shape)
