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
close = data.df['emi_close'].astype(float)

BB_PERIOD=20

bb = ta.bbands(close, length=BB_PERIOD)
upper = bb.filter(like='BBU').iloc[:, 0]
lower = bb.filter(like='BBL').iloc[:, 0]
pct_b = (close - lower) / (upper - lower)
mask = pct_b.isin([float('inf'), float('-inf')]) | pct_b.isna()
print(upper[mask])
print(lower[mask])
print(close[mask])

df = pd.DataFrame()
df['upper'] = upper[mask]
df['lower'] = lower[mask]
df['close'] = close[mask]
df['pct_b'] = pct_b[mask]

df.to_csv('data/samples/emi_debug.csv')