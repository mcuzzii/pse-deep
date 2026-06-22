import torch
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / 'src'))

from experiments import Experiment

test_experiment = Experiment('stock_news_mlp_10', False, False, True, False)
test_experiment.build_dataset(True)