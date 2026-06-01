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
    columns = data.df.columns
    target = f'{fn}_ad'
    if target in columns:
        print(f"Found column {target} in {fn}.")
        data.df.drop(columns=[target], inplace=True)
        columns = data.df.columns
        if target not in columns:
            print(f"Successfully removed column {target}.")
            print(f'Saving {fn}...')
            joblib.dump(data, f'data/processed/{fn}.joblib')
    else:
        print(f"Found no {target} column; instead found {data.df.columns.tolist()}")