import torch

all_masked_y = torch.tensor([0, 1, 0, 0, 1, 0, 1, 1], dtype=bool)             
mask_y = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 0],
    [0, 0, 1]
])         # (b * n,)
safe_mask_y = mask_y.clone()                                           # (b * n, y_seq)
safe_mask_y[all_masked_y, 0] = False

print(safe_mask_y)
print(mask_y)