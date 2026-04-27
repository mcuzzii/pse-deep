from processing import DataSource
from pathlib import Path
import joblib

data = joblib.load('../data/processed/social_media.joblib')
print(data.df.head(50)['text'])