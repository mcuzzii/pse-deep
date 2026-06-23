import torch
import pandas as pd
import matplotlib.pyplot as plt

test_tensor = torch.load('experiments/stock_mlp_10/test_outputs.pt', map_location='cpu', weights_only=False)

print(test_tensor['test_logit_scores'].shape)