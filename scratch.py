import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
import joblib
from processing import DataSource
import pandas as pd
import numpy as np
import pandas_ta as ta

stock_data = DataSource()
stock_data.create_df(
    processed_path='data/processed',
    raw_path='data/raw/stock',
    medium='stock',
    file_name='ac'
)
print(stock_data.df.head())


def stock_ta():
    unique_dates = stock_data.df.index.normalize().unique().sort_values()
    trading_periods = []
    for date in unique_dates:
        am_start = date + pd.Timedelta(hours = 9, minutes = 31)
        am_end = date + pd.Timedelta(hours = 12, minutes = 0)
        am_period = pd.date_range(start = am_start, end = am_end, freq = '1min')
        pm_start = date + pd.Timedelta(hours = 13, minutes = 1)
        pm_end = date + pd.Timedelta(hours = 15, minutes = 0)
        pm_period = pd.date_range(start = pm_start, end = pm_end, freq = '1min')
        trading_periods.append(am_period)
        trading_periods.append(pm_period)
    datetime_index = pd.DatetimeIndex(np.concatenate(trading_periods)).sort_values()

    stock_data.df = stock_data.df.reindex(datetime_index)

    open_cols = stock_data.df.columns[stock_data.df.columns.str.endswith('open')]
    close_cols = stock_data.df.columns[stock_data.df.columns.str.endswith('close')]
    ohl_cols = stock_data.df.columns[stock_data.df.columns.str.endswith(('open', 'high', 'low'))]
    hlc_cols = stock_data.df.columns[stock_data.df.columns.str.endswith(('high', 'low', 'close'))]

    stock_data.df[close_cols] = stock_data.df[close_cols].ffill()
    for item in ohl_cols:
        stock_data.df[item] = stock_data.df[item].fillna(stock_data.df[item.split('_')[0] + '_close'])
    zero_cols = stock_data.df.columns[stock_data.df.columns.str.endswith(('net', 'perc_chg', 'volume'))]
    stock_data.df[zero_cols] = stock_data.df[zero_cols].fillna(0)
    stock_data.df[open_cols] = stock_data.df[open_cols].bfill()
    for item in hlc_cols:
        stock_data.df[item] = stock_data.df[item].fillna(stock_data.df[item.split('_')[0] + '_open'])

    print(stock_data.df.head())

