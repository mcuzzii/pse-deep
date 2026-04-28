import pandas as pd
import cohere
import os
from dotenv import load_dotenv
from pathlib import Path
import time
from tqdm import tqdm
from tokenizers import Tokenizer
import requests
from cohere import ClientV2
import joblib
import functools
import json
import ast
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
        method_out = method(self, *args, **kwargs)
        self._history.append(method.__qualname__)
        return method_out

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
    def _load_json_folder(
        self,
        ignore_history: bool = False
    ):
        """ Loads JSON data from files in a folder. """

        # If method has already been called, skip the task.
        if 'DataSource._load_json_folder' in self._history and not ignore_history:
            return

        master_json = []
        for file_path in self.raw_path.iterdir():
            if file_path.is_file():
                with open(file_path, 'r') as f:
                    json_loaded = json.load(f)
                
                master_json.extend(json_loaded)
        
        df = pd.json_normalize(master_json)
        df.columns = [snake_case(col) for col in df.columns]

        self.df = df
    
    @record_history
    def _load_lseg_news(
        self,
        ignore_history: bool = False
    ):
        """ Loads LSEG news data from files in a folder. """

        # If method has already been called, skip the task.
        if 'DataSource._load_lseg_news' in self._history and not ignore_history:
            return
        
        df = pd.read_excel(self.raw_path)

        # Deal with the unique structure of the LSEG news dataset.
        df.iloc[:, 0] = df.iloc[:, 0].ffill()
        df = df.loc[df.iloc[:, 0].notna() & df.iloc[:, 1].notna()]
        df = df.loc[df.iloc[:, 0] != 'Only Important']

        # Create a date column.
        df.iloc[:, 1] = pd.to_datetime(df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str))
        
        # Drop extraneous columns.
        df = df.iloc[:, 1:5]

        # Give column names.
        df.columns = ['date_time', 'source', 'entities', 'text']

        self.df = df
    
    @record_history
    def _clean_text(
        self,
        ignore_history: bool = False
    ):
        """ Cleans text data. """

        # If method has already been called, skip the task.
        if 'DataSource._clean_text' in self._history and not ignore_history:
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
    
    @record_history
    def _cohere_embed(
        self,
        cache_location: Path,
        max_batch_size: int = 96,
        tpm_limit: int = 95000,
        buffer_duration: int = 10,
        ignore_history: bool = False
    ):
        """ Creates text embeddings using Cohere Embed V4.0 """

        # If method has already been called, skip the task.
        if 'DataSource._cohere_embed' in self._history and not ignore_history:
            return

        # Handle cache location.
        cache_location.mkdir(parents=True, exist_ok=True)
        cache_file_path = cache_location / f'{self.file_name}.joblib'
        
        if cache_file_path.exists():
            self.df = joblib.load(cache_file_path)
            if 'embeddings' in self.df.columns:
                self.df['embeddings'] = self.df['embeddings'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
        else:
            self.df['embeddings'] = None

        # Creating a local tokenizer using the embed-v4.0 tokenizer from Cohere.
        tokenizer = Tokenizer.from_str(requests.get(os.getenv("EMBED_V4_TOKENIZER_URL")).text)
        
        # Initializing list for embeddings capture.
        indices_to_embed = self.df.loc[self.df['embeddings'].isna()].index
        texts = self.df.loc[indices_to_embed, self.text_col].tolist()
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
                try:
                    response = co.embed(
                        model='embed-v4.0',
                        texts=batch_texts,
                        embedding_types=['float'],
                        input_type='classification',
                        output_dimension=1024
                    )
                    all_embeddings.extend(response.embeddings.float)
                    pbar.update(len(batch_texts))
                    batch_texts = []
                except Exception as e:
                    print(f"Error at batch {i}: {e}")

                    successful_embeddings_indices = indices_to_embed[:len(all_embeddings)]
                    successful_embeddings = pd.Series(all_embeddings, index=successful_embeddings_indices)
                    self.df.loc[successful_embeddings_indices, 'embeddings'] = successful_embeddings

                    joblib.dump(self.df, cache_file_path)
                    print("Partial progress saved.")

                    raise

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
        self.df.loc[indices_to_embed, 'embeddings'] = pd.Series(all_embeddings, index=indices_to_embed)

        # Delete cache
        if cache_file_path.exists():
            cache_file_path.unlink()
        
    # Processing pipeline for text data.
    def create_text_df(
        self,
        raw_path: str,
        processed_path: str,
        file_name: str,
        medium: ['lseg_news', 'x_posts'],
        text_col: str | None = None,
        date_col: str | None = None,
        ignore_history: bool = False
    ):
        """ Pipeline for preprocessing text-based datasets. """

        self.raw_path = Path(raw_path)
        self.file_name = file_name
        self.processed_path = Path(processed_path)
        data_source_path = self.processed_path / f'{self.file_name}.joblib'

        if data_source_path.exists():
            saved_data_source = joblib.load(data_source_path)
            self.__dict__.update(saved_data_source.__dict__)
            init_history = self._history.copy()

        else:
            init_history = self._history.copy()

            # Processing text-based datasets.
            if medium == 'x_posts':
                self.text_col = snake_case(text_col)
                self.date_col = snake_case(date_col)
                self._load_json_folder(ignore_history=ignore_history)
            
            elif medium == 'lseg_news':
                self._load_lseg_news(ignore_history=ignore_history)
                self.text_col = 'text'
                self.date_col = 'date_time'
        
        self._clean_text(ignore_history=ignore_history)
        self._cohere_embed(
            cache_location=Path(processed_path) / 'cache',
            tpm_limit=95000,
            ignore_history=ignore_history
        )

        if self._history != init_history:
            joblib.dump(self, data_source_path)