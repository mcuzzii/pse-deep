import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource, get_unique_instruments
import joblib
import pandas as pd
import gc

print("Selecting features...")
features_30 = DataSource()
features_30.create_df(file_name='features_30m', medium='features', target=30, stocks=stocks)
features_30.save_selected_features()
del features_30
gc.collect()

features_10 = DataSource()
features_10.create_df(file_name='features_10m', medium='features', target=10, stocks=stocks)
features_10.save_selected_features()
del features_10
gc.collect()