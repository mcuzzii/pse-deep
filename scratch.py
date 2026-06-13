import torch

m = torch.triu(
    torch.ones(4, 5, dtype=bool),
    diagonal=1
)

print(m)