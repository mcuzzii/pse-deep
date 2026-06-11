import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource, get_stocks
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import gc
import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from models import Time2Vec
from tqdm import tqdm
import signal

from experiments import Experiment

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