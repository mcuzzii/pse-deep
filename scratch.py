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

for item in Path('data/processed').glob('*.joblib'):
    print(f'Loading {item.name}...')
    data = joblib.load(item)
    fn = data.file_name
    print(f'Saving {fn}.csv...')
    data.df.head(2000).to_csv(f'data/samples/{fn}.csv')