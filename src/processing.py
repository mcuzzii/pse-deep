import pandas as pd
import cohere
import os
from dotenv import load_dotenv
from pathlib import Path
from data_loader import load_social
import time
from tqdm import tqdm

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Regex patterns
URL_PATTERN = r'(https?://[^\s<>"]+|www\.[^\s<>"]+|[a-zA-Z0-9.-]+\.[a-z]{2,6}/[^\s<>"]*)'
USER_PATTERN = r'@\w+'
HASHTAG_PATTERN = r'#(\w+)'
CASHTAG_PATTERN = r'\$(\w+)'

# Clean text data
def clean_text(text_df, text_col, date_col):
    df = text_df.copy(deep=True)

    # Convert to datetime and sort by date
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(by=date_col, ascending=True, inplace=True)

    # Handle whitespaces
    df[text_col] = df[text_col].str.strip()
    df[text_col] = df[text_col].str.replace(r'\s+', ' ', regex=True)
    df = df[df[text_col] != ""]

    # Deduplicate, keeping only the earliest post
    df.drop_duplicates(subset=[text_col], keep='first', inplace=True)

    # Convert user mentions and links to generic tokens and normalize hashtags and cashtags
    df[text_col] = df[text_col].str.replace(USER_PATTERN, '<USER>', regex=True)
    df[text_col] = df[text_col].str.replace(URL_PATTERN, '<URL>', regex=True)
    df[text_col] = df[text_col].str.replace(HASHTAG_PATTERN, r'\1', regex=True)
    df[text_col] = df[text_col].str.replace(CASHTAG_PATTERN, r'\1', regex=True)

    # Lowercase
    df[text_col] = df[text_col].str.lower()
    
    return df

# Embeddings using cohere embed-v4.0
def cohere_embed(text_df, text_col, batch_size=90, delay=2.8):
    df = text_df.copy(deep=True)
    all_embeddings = []
    texts = list(df[text_col])
    
    # Process text in batches

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        
        response = co.embed(
            model="embed-v4.0",
            texts=batch,
            input_type="classification",
            embedding_types=["float"]
        )

        all_embeddings.extend(response.embeddings)
        
        time.sleep(delay)
    
    df['embeddings'] = pd.Series(all_embeddings)

    return df

# Social media data preprocessing pipeline
def create_social_media_df(raw_path, text_col, date_col):
    raw_path = Path(raw_path)

    processed_path = raw_path.parent.parent / "processed" / "social_media.csv"

    if processed_path.exists():
        df = pd.read_csv(processed_path)

    else:
        df = load_social(raw_path)
        df = clean_text(df, text_col, date_col)
        df = cohere_embed(df, text_col)
        df.to_csv(processed_path, index=False)
    
    return df