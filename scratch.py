import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
import joblib
from processing import DataSource
import pandas as pd
import numpy as np
import pandas_ta as ta

ac = joblib.load('data/processed/ac.joblib')
import pandas as pd
ac.df.to_csv('data/processed/ac.csv')

print(ac.df.columns)

