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

target = 30

print('Computing common dates and times...')
common_date_times = None
for stock in stocks:
    stock_df = joblib.load(f'data/processed/{stock}.joblib')
    dates = stock_df.df.index
    common_date_times = dates if common_date_times is None else common_date_times.intersection(dates)

print(len(common_date_times.tolist()))

lunch_mask = (
    (common_date_times.time >= pd.Timestamp(f'11:{61 - target}').time()) &
    (common_date_times.time <= pd.Timestamp('12:00').time())
)

print(lunch_mask[:300])

daily_max = common_date_times.to_series().groupby(common_date_times.date).transform('max')
last_mask = common_date_times.to_series() > daily_max - pd.Timedelta(minutes=target)

print(daily_max[:300])
print(last_mask[:300])

filtered_date_times = common_date_times[~lunch_mask & ~last_mask.values]

print(len(filtered_date_time.tolist()))