import torch

t = torch.randn(4,3,5)
m = torch.rand(4,3) > 0.5

t[m] = 0

print(t)