import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource, get_unique_instruments
import joblib
import pandas as pd
import gc

stocks = get_unique_instruments('data/raw/stock')
stocks = list(set(stocks) - {'psei', 'psho', 'psse', 'psmo', 'psfi', 'pspr', 'psin'})

for stock in stocks:
    print(f"Saving {stock}...")
    data = joblib.load(f'data/processed/{stock}.joblib')
    data.df.to_csv(f'data/samples/{stock}.csv')
