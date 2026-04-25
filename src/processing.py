import pandas as pd
import cohere
import os
from dotenv import load_dotenv
from pathlib import Path
from data_loader import load_json_folder
import time
from tqdm import tqdm
from tokenizers import Tokenizer
import requests
from cohere import ClientV2

load_dotenv()

# Initialize Cohere client
co = ClientV2(os.getenv("COHERE_API_KEY"))

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

# Embeddings using cohere embed-v4.0.
def cohere_embed(text_df, text_col, max_batch_size=96, tpm_limit=95000, buffer_duration=10):
    
    # Creating a local tokenizer using the embed-v4.0 tokenizer from Cohere.
    tokenizer = Tokenizer.from_str(requests.get(os.getenv("EMBED_V4_TOKENIZER_URL")).text)
    
    # Initializing data frame for embeddings capture.
    df = text_df.copy(deep=True)
    texts = df[text_col].tolist()
    all_embeddings = []
    
    # Variables for rate limiting.
    minute_tokens = 0
    time_start = time.time()
    
    # Progress bar
    pbar = tqdm(total=len(texts), desc="Embedding (Local Tokenizer)")

    # Batching for embeddings capture
    batch_texts = []

    for i, text in enumerate(texts):

        batch_texts.append(text)
        minute_tokens += len(tokenizer.encode(text, add_special_tokens=False))

        # Rules for terminating the growth of a batch, and subsequently acquiring embeddings.
        if (
            len(batch_texts) == max_batch_size or
            minute_tokens >= tpm_limit or
            i == len(texts) - 1
        ):
            response = co.embed(
                model='embed-v4.0',
                texts=batch_texts,
                embedding_types=['float'],
                input_type='classification',
                output_dimension=1024
            )

            # Extend embeddings as list of lists and update progress bar.
            all_embeddings.extend(response.embeddings.float)
            pbar.update(len(batch_texts))
            batch_texts = []

            # Handling delays between requests.
            if minute_tokens >= tpm_limit:
                time_elapsed = time.time() - time_start

                if time_elapsed < 60:
                    # Buffer period for safety.
                    time.sleep(60 - time_elapsed + buffer_duration)

                minute_tokens = 0
                time_start = time.time()

    # Close the progress bar and add embeddings to the data frame.
    pbar.close()
    df['embeddings'] = pd.Series(all_embeddings)
    return df

# Social media data preprocessing pipeline
def create_social_media_df(raw_path, text_col, date_col):
    raw_path = Path(raw_path)

    processed_path = raw_path.parent.parent / "processed" / "social_media.csv"

    if processed_path.exists():
        df = pd.read_csv(processed_path)

    else:
        df = load_json_folder(raw_path)
        df = clean_text(df, text_col, date_col)
        df = cohere_embed(df, text_col)
        df.to_csv(processed_path, index=False)
    
    return df