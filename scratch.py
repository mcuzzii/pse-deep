import torch
import pandas as pd
import matplotlib.pyplot as plt

test_tensor = torch.load('experiments/stock_mlp_10/test_outputs.pt', map_location='cpu', weights_only=False)
test_tensor['test_all_targets'] = test_tensor['test_all_targets'].squeeze(1).reshape(30, test_tensor['test_all_targets'].shape[0] // 30).transpose(0, 1)

print(test_tensor['test_all_targets'].shape)