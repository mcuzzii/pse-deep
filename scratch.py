import torch

model = torch.load('experiments/stock_transformer_10/stock_transformer_10.pt', map_location='cpu', weights_only=False)

print(model['class_weights'])