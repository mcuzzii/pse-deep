import torch
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / 'src'))

checkpoint = torch.load('experiments/stock_mlp_10/stock_mlp_10.pt', map_location=torch.device('cpu'), weights_only=False)
print(checkpoint.get('best_threshold'))