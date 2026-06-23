import torch
import pandas as pd
import matplotlib.pyplot as plt

test_tensor = torch.load('experiments/stock_mlp_10/test_outputs.pt', map_location='cpu', weights_only=False)
test_tensor['test_logit_scores'] = test_tensor['test_logit_scores'].squeeze(1).reshape(30, test_tensor['test_logit_scores'].shape[0] // 30, -1).transpose(0, 1)

print(test_tensor['test_logit_scores'].shape)