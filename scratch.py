import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / 'src'))
from models import AttentionBlock

model = AttentionBlock(
    embedding_dim=512,
    num_heads=8
)

x = torch.randn(3, 2, 6, 512)
y = torch.randn(3, 2, 5, 512)

tx = torch.arange(0, 4).unsqueeze(0).unsqueeze(0).expand(3, 2, 6)
ty = torch.arange(0, 5).unsqueeze(0).unsqueeze(0).expand(3, 2, 5)

x_mask = (torch.rand(3, 2, 6) >= 0.75)
y_mask = (torch.rand(3, 2, 5) >= 0.75)

out = model(tx, ty, x, y, mask_y=y_mask)

for i in range(3):
    for j in range(2):
        print(out[i, j, 0])
        print(f'{y_mask[i, j]}\n')
        print(out.shape)
