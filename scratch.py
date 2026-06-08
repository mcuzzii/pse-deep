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

lseg_news_data = DataSource()
lseg_news_data.create_df(file_name='news', medium='final_text', ignore_history=True)
del lseg_news_data
gc.collect()

social_media_data = DataSource()
social_media_data.create_df(file_name='social_media', medium='final_text', ignore_history=True)
del social_media_data
gc.collect()