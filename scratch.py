import torch
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / 'src'))

from experiments import EarlyStopping

model = torch.load('experiments/stock_transformer_10/stock_mlp_10.pt', map_location='cpu', weights_only=False)

print(model['class_weights'])