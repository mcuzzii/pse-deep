import torch
import pandas as pd
import matplotlib.pyplot as plt

test_tensor = torch.load('experiments/data/stock_mlp_10m_test.pt', map_location='cpu', weights_only=False)

#test_tensor['X'] = test_tensor['X'].reshape(test_tensor['X'].shape[0] // 30, 30, -1)

x = test_tensor['X'][0, -10:]
matches = torch.isclose(test_tensor['X'][:, -10:], x).all(dim=1)
indices = matches.nonzero(as_tuple=True)[0]
print(indices)

# Compare all S slices against the first one
#first = x[:, 0:1, :]                          # (B, 1, 10)
#identical = (x == first).all(dim=-1).all(dim=0)  # (S,) — True/False per S

#print(identical)          # which S positions match the first
#print(identical.all())    # True if ALL S are identical to each other
