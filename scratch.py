import torch

d = torch.load('experiments/stock_transformer_30/stock_transformer_30.pt', map_location=torch.device('cpu'), weights_only=False)
print(d['train_losses'])
print(d['val_losses'])
print(d['total_loss'])
print(d)