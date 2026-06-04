import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource, get_unique_instruments
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import gc

stock = 'pgold'

data = joblib.load(f'data/processed/{stock}.joblib')
close = data.df[f'{stock}_close'].astype(float)
high = data.df[f'{stock}_high'].astype(float)
low = data.df[f'{stock}_low'].astype(float)

BB_PERIOD=20
ADX_PERIOD = 14
RSI_PERIOD = 14

bb = ta.bbands(close, length=BB_PERIOD)
upper = bb.filter(like='BBU').iloc[:, 0]
lower = bb.filter(like='BBL').iloc[:, 0]
pct_b = (close - lower) / (upper - lower)
pct_b[(upper == lower) & (lower == close)] = 0.5
mask = pct_b.isin([float('inf'), float('-inf')]) | pct_b.isna()
print(upper[mask])
print(lower[mask])
print(close[mask])

df = pd.DataFrame()
df['upper'] = upper[mask]
df['lower'] = lower[mask]
df['close'] = close[mask]
df['pct_b'] = pct_b[mask]

df.to_csv(f'data/samples/{stock}_bbands_debug.csv')

rsi = ta.rsi(close, length=RSI_PERIOD)
adx  = ta.adx(high, low, close, length=ADX_PERIOD).filter(like='ADX_').iloc[:, 0]

mask = rsi.isin([float('inf'), float('-inf')]) | rsi.isna()
mask = mask.rolling(RSI_PERIOD, min_periods=1).max().astype(bool)

df = pd.DataFrame()
df['rsi'] = rsi[mask]
df['close'] = close[mask]

df.to_csv(f'data/samples/{stock}_rsi_debug.csv')

mask = adx.isin([float('inf'), float('-inf')]) | adx.isna()
mask = mask.rolling(ADX_PERIOD * 2, min_periods=1).max().astype(bool)

df = pd.DataFrame()
df['adx'] = adx[mask]
df['close'] = close[mask]

df.to_csv(f'data/samples/{stock}_adx_debug.csv')