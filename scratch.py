import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource, get_unique_instruments
import joblib
import pandas as pd
import pandas_ta as ta
import gc

data = joblib.load('data/processed/emi.joblib')
close = data.df['emi_close']

BB_PERIOD=20

bb = ta.bbands(close, length=BB_PERIOD)
upper = bb.filter(like='BBU').iloc[:, 0]
lower = bb.filter(like='BBL').iloc[:, 0]
pct_b = (close - lower) / (upper - lower)
inf_mask = pct_b.isin([float('inf'), float('-inf')])
print(upper[inf_mask])
print(lower[inf_mask])
print(close[inf_mask])