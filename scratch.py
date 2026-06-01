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
    targets = [s for s in columns if re.search(r'.+_ad$', s)]
    if targets:
        print(f"Found columns {targets} in {fn}.")
        data.df.drop(columns=[targets], inplace=True)
        columns = data.df.columns
        if not [s for s in targets if s in columns]:
            print(f"Successfully removed columns {targets}.")
            print(f'Saving {fn}...')
            joblib.dump(data, f'data/processed/{fn}.joblib')
    else:
        print(f"Found no '_ad' columns; instead found {data.df.columns.tolist()}")