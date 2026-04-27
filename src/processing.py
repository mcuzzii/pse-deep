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
import joblib
import functools
import json
import re

load_dotenv()

# Initialize Cohere client.
co = ClientV2(os.getenv("COHERE_API_KEY"))

# Regex patterns.
URL_PATTERN = r'(https?://[^\s<>"]+|www\.[^\s<>"]+|[a-zA-Z0-9.-]+\.[a-z]{2,6}/[^\s<>"]*)'
USER_PATTERN = r'@\w+'
HASHTAG_PATTERN = r'#(\w+)'
CASHTAG_PATTERN = r'\$(\w+)'

# Decorator that records which methods have been called.
def record_history(method):

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._history.append(method.__qualname__)
        return method(self, *args, **kwargs)

    return wrapper

# Helper function for creating snake_case strings.
def snake_case(text_string):
    text_string = re.sub(r'[^A-Za-z0-9]', ' ', text_string)
    text_string = re.sub(r'([a-z])([A-Z])', r'\1 \2', text_string)
    text_string = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text_string)
    text_string = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', text_string)
    text_string = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text_string)
    
    text_string = re.sub(r'\s+', '_', text_string.strip()).lower()

    return text_string

class DataSource:
    """A class for storing and processing a dataset."""

    def __init__(self):
        
        # Flags for whether certain heavy tasks have already been performed.
        self._history = []

    @record_history
    def load_json_folder(self, raw_path, ignore_history=False):
        """Loads JSON data from files in a folder."""

        # If method has already been called, skip the task.
        if 'DataSource.load_json_folder' in self._history and not ignore_history:
            return

        master_json = []
        for file_path in raw_path.iterdir():
            if file_path.is_file():
                with open(file_path, 'r') as f:
                    json_loaded = json.load(f)
                
                master_json.extend(json_loaded)
        
        df = pd.json_normalize(master_json)
        df.columns = [snake_case(col) for col in df.columns]

        self.df = df
    
    @record_history
    def clean_text(self, ignore_history=False):
        """Cleans text data."""

        # If method has already been called, skip the task.
        if 'DataSource.clean_text' in self._history and not ignore_history:
            return self.df

        # Convert to datetime and sort by date.
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df.sort_values(by=self.date_col, ascending=True, inplace=True)

        # Handle whitespaces.
        self.df[self.text_col] = self.df[self.text_col].str.strip()
        self.df[self.text_col] = self.df[self.text_col].str.replace(r'\s+', ' ', regex=True)
        self.df = self.df.loc[self.df[self.text_col] != ""]

        # Deduplicate, keeping only the earliest post.
        self.df.drop_duplicates(subset=[self.text_col], keep='first', inplace=True)

        # Convert user mentions and links to generic tokens and normalize hashtags and cashtags.
        self.df[self.text_col] = self.df[self.text_col].str.replace(USER_PATTERN, '<USER>', regex=True)
        self.df[self.text_col] = self.df[self.text_col].str.replace(URL_PATTERN, '<URL>', regex=True)
        self.df[self.text_col] = self.df[self.text_col].str.replace(HASHTAG_PATTERN, r'\1', regex=True)
        self.df[self.text_col] = self.df[self.text_col].str.replace(CASHTAG_PATTERN, r'\1', regex=True)

        # Lowercase.
        self.df[self.text_col] = self.df[self.text_col].str.lower()

        # Remove rows with NaN text.
        self.df = self.df.loc[self.df[self.text_col].notna()]
    
    # Embeddings using cohere embed-v4.0.
    @record_history
    def cohere_embed(self, max_batch_size=96, tpm_limit=95000, buffer_duration=10, ignore_history=False):
        """ Creates text embeddings using Cohere Embed V4.0"""

        # If method has already been called, skip the task.
        if 'DataSource.cohere_embed' in self._history and not ignore_history:
            return self.df

        # Creating a local tokenizer using the embed-v4.0 tokenizer from Cohere.
        tokenizer = Tokenizer.from_str(requests.get(os.getenv("EMBED_V4_TOKENIZER_URL")).text)
        
        # Initializing list for embeddings capture.
        texts = self.df[self.text_col].tolist()
        all_embeddings = []
        
        # Variables for rate limiting.
        minute_tokens = 0
        time_start = time.time()
        
        # Progress bar.
        pbar = tqdm(total=len(texts), desc="Embedding (Local Tokenizer)")

        # Batching for embeddings capture.
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
        self.df['embeddings'] = pd.Series(all_embeddings)
        
    # Processing pipeline for social media data.
    def create_social_media_df(self, raw_path, processed_path, file_name, text_col, date_col, ignore_history=False):

        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path) / (file_name + '.joblib')
        self.text_col = snake_case(text_col)
        self.date_col = snake_case(date_col)

        if self.processed_path.exists():
            self = joblib.load(self.processed_path)
            init_history = self._history

        else:
            init_history = self._history
            self.load_json_folder(self.raw_path, ignore_history=ignore_history)
        
        self.clean_text(ignore_history=ignore_history)
        self.cohere_embed(ignore_history=ignore_history)

        if self._history != init_history:
            joblib.dump(self, self.processed_path)

