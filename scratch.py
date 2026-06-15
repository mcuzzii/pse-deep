import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path.cwd() / 'src'))

from experiments import Experiment, collate_fn

d = torch.load('experiments/stock_transformer_10/checkpoints/stock_transformer_10.pt', map_location='cpu', weights_only=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(d['train_losses'], label='Train loss')
plt.plot(d['val_losses'], label='Val loss', linestyle='--')
plt.xlabel('Validation checkpoint')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('experiments/stock_transformer_10/loss_curve.png')