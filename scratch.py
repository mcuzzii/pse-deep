import sys
from pathlib import Path
import joblib

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

news = DataSource()
news.create_df('news')

import numpy as np

values, counts = np.unique(news.df.index.date, return_counts=True)
d = dict(zip(values, counts))

print(dict(sorted(d.items(), key=lambda x: x[1], reverse=True)))