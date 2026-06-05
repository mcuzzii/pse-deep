import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource, get_unique_instruments
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import gc

stocks = get_unique_instruments('data/raw/stock')
stocks = list(set(stocks) - {'psei', 'psho', 'psse', 'psmo', 'psfi', 'pspr', 'psin'})

for stock in stocks:
    data = joblib.load(f'data/processed/{stock}.joblib')
    pd.concat([data.df.head(1000), data.df.tail(1000)]).to_csv(f'data/samples/{stock}.csv')