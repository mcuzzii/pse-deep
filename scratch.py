import sys
from pathlib import Path
import joblib
import gc

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource, get_stocks

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