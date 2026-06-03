import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource, get_unique_instruments
import gc

stocks = get_unique_instruments('data/raw/stock')
stocks = list(set(stocks) - {'psei', 'psho', 'psse', 'psmo', 'psfi', 'pspr', 'psin'})

features_10 = DataSource()
features_10.create_df(file_name='features_10m', medium='features', target=10, stocks=stocks, ignore_history=True)
del features_10
gc.collect()