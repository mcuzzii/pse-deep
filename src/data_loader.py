import pandas as pd
from pathlib import Path
import json

# Import all social media data from a directory containing all social media
def load_social(folder_path: str) -> pd.DataFrame:

    data_dir = Path(folder_path)

    dfs = []
    for file_path in data_dir.iterdir():
        if file_path.is_file():  # Ensure it's a file, not a subfolder
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            dfs.append(df)
    
    return pd.concat(dfs)
