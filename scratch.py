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
    data = joblib.load(item)
    print(f'Searching {data.file_name}...')
    columns = data.df.columns
    fn = None
    for col in columns:
        if 'no_activity' in col:
            fn = data.file_name
            break
    if fn:
        print(f"Found column 'no_activity' in {fn}.")
        data.df.rename(columns={'no_activity': f'{fn}_no_activity'}, inplace=True)
        if f'{fn}_no_activity' in columns and 'no_activity' not in columns:
            print(f"Successfully replaced column 'no_activity' with '{fn}_no_activity'.")
        print(f'Saving {fn}...')
        joblib.load(f'data/processed/{fn}.joblib')

print(ac.df.columns)

