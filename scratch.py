import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource, get_unique_instruments
import joblib
import pandas as pd

data = joblib.load('data/processed/features_10m.joblib')
pd.concat([data.df.head(1000), data.df.tail(1000)]).to_csv('data/samples/features_10m.csv')