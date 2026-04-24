import sys
from pathlib import Path

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / "src"))

from processing import create_social_media_df
from dotenv import load_dotenv

load_dotenv()

def main():

    social_df = create_social_media_df('data/raw/social', 'text', 'createdAt')
    print(social_df.head())

if __name__ == "__main__":
    main()
