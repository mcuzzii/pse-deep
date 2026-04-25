import pandas as pd
import cohere
import os
from dotenv import load_dotenv
from pathlib import Path
from data_loader import load_json_folder
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from cohere import Client, ClientV2

load_dotenv()

# Initialize Cohere client
co_v2 = ClientV2(os.getenv("COHERE_API_KEY"))

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
def cohere_embed(text_df, text_col, tpm_limit=50000):
    # 1. Download/Load the tokenizer locally
    tokenizer = AutoTokenizer.from_pretrained("Xenova/c4ai-command-r-v01-tokenizer")
    
    df = text_df.copy(deep=True).fillna("")
    texts = df[text_col].tolist()
    all_embeddings = []
    
    tokens_in_current_minute = 0
    window_start_time = time.time()
    current_idx = 0
    
    pbar = tqdm(total=len(texts), desc="Embedding (Local Tokenizer)")

    while current_idx < len(texts):
        batch_texts = []
        batch_token_count = 0
        
        # 2. Build the batch locally (Super fast, no API calls here)
        while current_idx < len(texts) and len(batch_texts) < 96:
            text_to_add = texts[current_idx]
            
            # Local token count
            item_tokens = tokenizer.encode(text_to_add, add_special_tokens=False)
            item_token_count = len(item_tokens)
            
            # Check if this single text is too big
            if item_token_count > 512: # Cohere's per-row limit
                 text_to_add = text_to_add[:2000] # Rough truncation
                 item_token_count = len(tokenizer.encode(text_to_add))

            # Check TPM window
            if tokens_in_current_minute + batch_token_count + item_token_count > tpm_limit:
                break
                
            batch_texts.append(text_to_add)
            batch_token_count += item_token_count
            current_idx += 1

        # 3. Manage Timing
        elapsed = time.time() - window_start_time
        if (tokens_in_current_minute + batch_token_count) >= tpm_limit:
            if elapsed < 60:
                sleep_time = 60 - elapsed + 1
                time.sleep(sleep_time)
            window_start_time = time.time()
            tokens_in_current_minute = 0

        # 4. The ONLY API call: Embedding
        if batch_texts:
            response = co.embed(
                model="embed-v4.0",
                texts=batch_texts,
                input_type="classification",
                embedding_types=["float"]
            )
            all_embeddings.extend(response.embeddings.float)
            tokens_in_current_minute += batch_token_count
            pbar.update(len(batch_texts))

    pbar.close()
    df['embeddings'] = all_embeddings
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