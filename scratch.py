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

print("Selecting features...")
features_30 = DataSource()
features_30.create_df(file_name='features_30m', medium='features', target=30, stocks=stocks, ignore_history=True)
features_30.save_selected_features()
del features_30
gc.collect()

features_10 = DataSource()
features_10.create_df(file_name='features_10m', medium='features', target=10, stocks=stocks, ignore_history=True)
features_10.save_selected_features()
del features_10
gc.collect()

print("Finalizing datasets...")
for stock in stocks:
    stock_data = DataSource()
    stock_data.create_df(file_name=stock, medium='final', target=30, ignore_history=True)
    del stock_data
    gc.collect()

for stock in stocks:
    stock_data = DataSource()
    stock_data.create_df(file_name=stock, medium='final', target=10, ignore_history=True)
    del stock_data
    gc.collect()