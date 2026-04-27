from processing import DataSource
from pathlib import Path
import joblib
import pandas as pd

df = DataSource()
df.load_lseg_news(raw_path=Path('../data/raw/news/all_news.xlsx'))
print(df.df.head(50))