import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource, get_stocks
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import gc

stocks = get_stocks()
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