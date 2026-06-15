import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path.cwd() / 'src'))

from experiments import Experiment, collate_fn

d = torch.load('experiments/stock_transformer_10/checkpoints/stock_transformer_10.pt', map_location='cpu', weights_only=False)

import matplotlib.pyplot as plt

num_checkpoints = len(d['train_losses'])
scaled_x = [(8 * (x + 1)) ** 2 for x in range(num_checkpoints)]

plt.figure(figsize=(8, 5))
plt.plot(scaled_x, d['train_losses'], label='Train loss')
plt.plot(scaled_x, d['val_losses'], label='Val loss', linestyle='--')
plt.xlabel('Validation checkpoint')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('experiments/stock_transformer_10/loss_curve.png')