import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

def setup_plot_style():
    sns.set_theme(style='white')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,

        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,

        'savefig.dpi': 300,
        'figure.dpi': 150,
    })

COLORS = {
    'purple':     '#43338a',   # viridis dark end
    'teal':       '#27848a',   # viridis mid
    'green':      '#3a9e5f',   # viridis mid-light
    'yellow':     '#c2df23',   # viridis light end (use sparingly)
    'indigo':     '#30628b',   # viridis blue-purple
    'seafoam':    '#1fa094',   # viridis teal variant
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Force PyTorch operations to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"