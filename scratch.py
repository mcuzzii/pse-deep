import torch

d = torch.load('experiments/stock_transformer_30/stock_transformer_30.pt')
print(d['train_losses'])
print(d['val_losses'])
print(d['total_loss'])