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

stocks = ['acen', 'rcr', 'glo', 'tel', 'jfc']
acen = DataSource()
for stock in stocks:
    print(f"Combining instruments for {stock}...")
    stock_data = DataSource()
    stock_data.create_df(file_name=stock, medium='combined', ignore_history=True)
    del stock_data
    gc.collect()