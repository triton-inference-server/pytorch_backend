# test_model.py
import torch
import torch.nn as nn

class DictOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(50, 5)
    
    def forward(self, x):
        features = self.fc1(x)
        logits = self.fc2(features)
        embeddings = self.fc3(features)
        
        # Return dictionary
        return {
            "logits": logits,
            "embeddings": embeddings
        }

# Create and save model
model = DictOutputModel()
model.eval()

# Trace with example input
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input, strict=False)

# Save
torch.jit.save(traced_model, "model.pt")
print("Model saved!")
