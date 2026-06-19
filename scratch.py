import sys
from pathlib import Path
import joblib

sys.path.append(str(Path.cwd() / 'src'))

social_final = joblib.load('data/processed/social_media.joblib')
print(social_final.df.index)