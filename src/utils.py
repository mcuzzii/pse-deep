import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

def setup_plot_style():
    font_dir = Path('assets/fonts/EBGaramond')
    for font_file in font_dir.glob('*.ttf'):
        fm.fontManager.addfont(str(font_file))
    
    sns.set_theme(style='white')
    plt.rcParams['font.family'] = 'EB Garamond'

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