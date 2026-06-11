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
y = torch.randn(3, 2, 5, 512)
y_mask = (torch.rand(3, 2, 5) >= 0.5).flatten(0, 1)

out = model(x, y, mask_y=y_mask)[1].detach().numpy()

for i in range(3):
    for j in range(2):
        print(out[i, j, 0])
        print(f'{y_mask[i, j]}\n')
        print(out.shape)
