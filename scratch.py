import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / 'src'))
from models import AttentionBlock

model = AttentionBlock(
    embedding_dim=512,
    num_heads=8,
    is_causal=True
)

x = torch.randn(3, 2, 4, 512)
x_mask = torch.rand(3, 2, 4) >= 0.5

out = model(x, x, x_mask, x_mask)

print(out)