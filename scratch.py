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

data = joblib.load('data/processed/ac_10m.joblib')

with open('data/samples/dates', 'w', encoding='utf-8') as f:
    json.dump(pd.Series(data.df.index.date).value_counts().to_dict(), f)

pd.concat([data.df.head(1000), data.df.tail(1000)], axis=0).to_csv('data/samples/ac_10m.csv')