import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
import joblib

datasource = joblib.load('data/processed/social_media.joblib')
print(datasource.df.columns)