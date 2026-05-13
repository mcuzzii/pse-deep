import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
import joblib
from processing import DataSource
import pandas as pd
import numpy as np
import pandas_ta as ta

copper = joblib.load('data/processed/acen.joblib')
import pandas as pd
copper.df.to_csv('data/processed/acen.csv')

print(copper.df.columns)

