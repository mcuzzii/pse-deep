import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
from river.drift import ADWIN
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource, get_stocks, get_elapsed_time, get_text_window
from collections import Counter
from experiments import Experiment, mcc_curve
from utils import setup_plot_style, COLORS
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
import time
import joblib
import re
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.special import expit
from scipy.stats import wilcoxon
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns
import itertools
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices

news = False
social = False
transformer = True
pred_30 = False
pred_horizon_prefix = 30 if pred_30 else 10

news_prefix = 'news_' if news else ''
social_prefix = 'social_' if social else ''
model_prefix = 'transformer_' if transformer else 'mlp_'
pred_horizon_prefix = 30 if pred_30 else 10

exp_name = f"stock_{news_prefix}{social_prefix}{model_prefix}{pred_horizon_prefix}"

experiment = Experiment(
    experiment_name=exp_name,
    transformer=transformer,
    pred_30=pred_30,
    news=news,
    social=social,
    stock_lookback=60
)
experiment.build_model(
    input_dim=100 if transformer else 110,
    news_input_dim=15,
    social_input_dim=6 if transformer else 15,
    text_input_dim=1024,
    social_embedding_dim=16,
    hidden_dim=384,
    embedding_dim=128,
    num_layers=1 if transformer else 5,
    temporal_embedding_dim=16,
    dropout=0.1,
    K=5,
    num_samples=500,
    sigma=5e-2,
)

best_path = self.experiments_path / exp_name / f'{exp_name}.pt'
best_weights = torch.load(best_path, map_location=device, weights_only=False)['model']

model = experiment.model.to(device)
model.load_state_dict(best_weights)
model.eval()

fin_embed = model.fin_embed
tst = model.time_series_transformer
layer = tst.transformer[0]
attn_blk = layer.attn_blk
