import torch
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / 'src'))

from eval import Eval

eval = Eval()
eval.compute_model_drift()