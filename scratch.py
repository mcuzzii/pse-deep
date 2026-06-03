import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
import joblib
from processing import DataSource
import pandas as pd
import numpy as np
import pandas_ta as ta
import re

data = joblib.load('data/processed/ac.joblib')


def remove_minutes(group):
    max_time = group['local_time'].max()
    cutoff = max_time - pd.Timedelta(minutes=10)
    return group.loc[group['local_time'] <= cutoff]

print(data.df.groupby(pd.Grouper(key='local_time', freq='D'), group_keys=False).apply(remove_minutes).columns)