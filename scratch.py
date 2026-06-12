import torch

t = torch.tensor.randn(4,3,5)
m = torch.tensor.rand(4,3) > 0.5

t[m] = 0

print(t)