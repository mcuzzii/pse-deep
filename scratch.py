import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / 'src'))
from models import AttentionBlock

model = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    batch_first=True
)

x = torch.randn(3, 2, 4, 512).flatten(0, 1)
x_mask = (torch.rand(3, 2, 4) >= 0.5).flatten(0, 1)

out = model(x, x, x, key_padding_mask=x_mask, need_weights=True, average_attn_weights=False)[1].detach().numpy()

print(out[5, 7, 2, 0])