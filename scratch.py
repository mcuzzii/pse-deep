import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path.cwd() / 'src'))

from experiments import Experiment, collate_fn

d = torch.load('experiments/stock_news_transformer_30/checkpoints/stock_news_transformer_30.pt', map_location='cpu', weights_only=False)

print(d['train_losses'])
print(d['val_losses'])
print(d['total_loss'])