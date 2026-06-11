import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / 'src'))
from models import AttentionBlock

model = nn.MultiheadAttention(
    embedding_dim=512,
    num_heads=8,
    batch_first=True
)

x = torch.randn(3, 2, 4, 512).flatten(0, 1)
y = torch.randn(3, 2, 5, 512).flatten(0, 1)
y_mask = (torch.rand(3, 2, 5) >= 0.5).flatten(0, 1)

x_sz = x.size(1)
y_sz = y.size(1)
attn_mask = torch.triu(
    torch.ones(x_sz, y_sz, dtype=torch.bool, device=x.device),
    diagonal=1
)

out = model(x, y, y, key_padding_mask=y_mask, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)[1].detach().numpy()

for i in range(3):
    for j in range(2):
        print(out[i, j, 0])
        print(f'{y_mask[i, j]}\n')
        print(out.shape)