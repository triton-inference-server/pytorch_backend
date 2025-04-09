import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        mask = mask.to(dtype=x.dtype).unsqueeze(-1)  # shape (1, 10, 1)
        masked_input = x * mask  # shape (1, 10, 128)
        mask_sum = mask.sum(dim=1)  # shape (1, 1)
        output = masked_input.sum(dim=1) / (mask_sum + 1e-8)  # shape (1, 128)
        return output

if __name__ == "__main__":
    torch.jit.save(torch.jit.script(Model()), "model.pt")
    # model = Model()
    # x = torch.randn(1, 10, 128)
    # mask = torch.randint(0, 2, (1, 10), dtype=torch.int32)

    # output = model(x, mask)
    # print(output)  # torch.Size([1, 128])
