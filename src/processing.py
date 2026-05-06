import pandas as pd
import numpy as np
import cohere
import os
from pathlib import Path
import time
from tqdm import tqdm
from tokenizers import Tokenizer
from typing import Literal
from cohere import ClientV2
import requests
import joblib
import functools
import json
import ast
import re
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import langid

# Regex patterns.
URL_PATTERN = r'(https?://[^\s<>"]+|www\.[^\s<>"]+|[a-zA-Z0-9.-]+\.[a-z]{2,6}/[^\s<>"]*)'
USER_PATTERN = r'@\w+'
HASHTAG_PATTERN = r'#(\w+)'
CASHTAG_PATTERN = r'\$(\w+)'

# Decorator that records which methods have been called.
def record_history(method):

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if method.__qualname__ in self._history and not kwargs.get('ignore_history', False):
            return
        
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

# Helper function for getting log softmax probabilities of the language of a text.
def get_lang(text):
    # Get all scores
    ranks = langid.rank(str(text))
    langs = [r[0] for r in ranks]
    scores = np.array([r[1] for r in ranks])
    
    # Standard Softmax: exp(x) / sum(exp(x))
    # We subtract np.max(scores) for numerical stability (prevents overflow)
    shift_scores = scores - np.max(scores)
    exp_scores = np.exp(shift_scores)
    softmax = exp_scores / exp_scores.sum()

    # Get log probabilities.
    log_probabilities = [np.log(s) if s > 0 else -1000 for s in softmax]

    lang_scores = dict(zip(langs, log_probabilities))
    lang = list(lang_scores.keys())[0]
    en_score = lang_scores['en']
    
    # Map back to languages
    return lang, en_score

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
        tpm_limit: int = 30000,
        buffer_duration: int = 10,
        ignore_history: bool = False
    ):
        """ Creates text embeddings using Cohere Embed V4.0 """

        load_dotenv()

        co = ClientV2(api_key=os.getenv('COHERE_API_KEY'))

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
    
    def get_similar_embeddings(
        self,
        index: int,
        n_results: int = 10
    ):
        file_path = self.processed_path / f'{self.file_name}_similar_to_{index}.json'
        if file_path.exists():
            return
        
        # Convert embeddings to numpy array
        df = self.df[['text', 'embeddings']].copy()
        embeddings_matrix = np.stack(df['embeddings'].values)

        # Calculate similarities based on given reference index
        int_loc = df.index.get_loc(index)
        target_vector = embeddings_matrix[int_loc].reshape(1, -1)
        similarities = cosine_similarity(target_vector, embeddings_matrix).flatten()

        # Add similarity scores to the data frame and sort by similarity
        df['similarity_score'] = similarities
        results = df.sort_values(by='similarity_score', ascending=False)

        results.head(n_results).to_json(
            file_path,
            orient='records',
            indent=4
        )
    
    @record_history
    def _translate_headlines(
        self,
        batch_size: int = 16,
        ignore_history: bool = False
    ):

        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        with open('data/raw/languages.json', 'r') as f:
            LANG_MAP = json.load(f)

        # Create a copy to store results
        self.df['cleaned_headline'] = self.df['text']
        
        # Step 1: Fast Language Detection (Row by Row is okay here, langid is fast)
        print("Detecting languages...")
        self.df[['detected_lang', 'en_score']] = self.df['text'].apply(get_lang).apply(pd.Series)
        
        # Step 2: Group by language to batch translate efficiently
        # We ignore 'en' as it doesn't need translation
        non_english_mask = self.df['en_score'] < -30
        langs_to_translate = self.df[non_english_mask]['detected_lang'].unique()
        
        for lang in langs_to_translate:
            nllb_lang = LANG_MAP.get(lang)
            if not nllb_lang:
                continue
                
            # Get all rows for this specific language
            mask = (self.df['detected_lang'] == lang) & non_english_mask
            texts_to_translate = self.df.loc[mask, 'text'].tolist()
            indices = self.df.index[mask].tolist()
            
            print(f"Translating {len(texts_to_translate)} items for language: {lang} ({nllb_lang})")
            
            tokenizer.src_lang = nllb_lang
            translated_results = []
            
            # Batch loop
            for i in tqdm(range(0, len(texts_to_translate), batch_size)):
                batch_texts = texts_to_translate[i : i + batch_size]
                
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                
                with torch.no_grad():
                    translated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                        max_length=100
                    )
                
                batch_outputs = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                translated_results.extend(batch_outputs)
            
            # Update the dataframe with translated text
            self.df.loc[mask, 'cleaned_headline'] = translated_results
    
    def _get_finbert_sentiment(
        self,
        ignore_history: bool = False
    ):
        device = 0 if torch.cuda.is_available() else -1

        finbert = pipeline(
            'text-classification', 
            model='tabularisai/ModernFinBERT', 
            device=device,
            batch_size=16,
            top_k=None  # This ensures we get positive, negative, and neutral scores
        )

        raw_results = finbert(self.df['cleaned_headline'].tolist())

        parsed_results = []
        for row in raw_results:
            parsed_results.append({item['label']: item['score'] for item in row})
        
        sentiment_df = pd.DataFrame(parsed_results)
        
        if 'bullish' in sentiment_df.columns and 'bearish' in sentiment_df.columns:
            sentiment_df['finbert_combined_score'] = sentiment_df['bullish'] - sentiment_df['bearish']
        
        sentiment_df['sentiment'] = sentiment_df[['bullish', 'bearish', 'neutral']].idxmax(axis=1)
        
        self.df = pd.concat([self.df, sentiment_df.set_index(self.df.index)], axis=1)
    
    def _get_multilingual_sentiment(
        self,
        ignore_history: bool = False
    ):
        device = 0 if torch.cuda.is_available() else -1
        multilingual_sentiment = pipeline(
            'text-classification',
            model='tabularisai/multilingual-sentiment-analysis',
            device=device,
            batch_size=16,
            top_k=None
        )

        raw_results = multilingual_sentiment(self.df['text'].tolist())

        parsed_results = []
        for row in raw_results:
            parsed_results.append({item['label']: item['score'] for item in row})
        
        sentiment_df = pd.DataFrame(parsed_results)
        
        probs = sentiment_df[['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']].values
        sentiment_df['sentiment_score'] = np.dot(probs, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))

        sentiment_df['sentiment'] = sentiment_df[list(sentiment_df.columns)].idxmax(axis=1)

        sentiment_df.columns = [snake_case(col) for col in sentiment_df.columns]
        
        self.df = pd.concat([self.df, sentiment_df.set_index(self.df.index)], axis=1)

    def _load_stock(
        self,
        ignore_history: bool = False
    ):
        folder_path = Path("raw")

        master_df = pd.DataFrame(index = pd.Index([], dtype = 'object', name = 'local_time'))
        for item in folder_path.glob('*.xlsx'):
            sheet = pd.read_excel(item)
            header = sheet.index[sheet.isin(["Exchange Date"]).any(axis=1)].tolist()[0]
            cols = sheet.iloc[header, 3:].dropna().str.lower().str.replace(' ', '_').str.replace('%', 'perc_')
            cols.iloc[1:] = item.name.split('.')[0].split('_')[0].lower() + '_' + cols.iloc[1:]
            instrument_df = sheet.iloc[header + 1:, 3:3 + cols.shape[0]].copy()
            instrument_df.columns = cols.tolist()
            instrument_df['local_time'] = pd.to_datetime(instrument_df['local_time'])
            instrument_df = instrument_df.set_index('local_time')
            master_df = master_df.combine_first(instrument_df)
            print(f"Added {item.name}")

        print("Reindexing dates and times")

        unique_dates = master_df.index.normalize().unique().sort_values()
        trading_periods = []
        for date in unique_dates:
            am_start = date + pd.Timedelta(hours = 9, minutes = 31)
            am_end = date + pd.Timedelta(hours = 12, minutes = 0)
            am_period = pd.date_range(start = am_start, end = am_end, freq = '1min')
            pm_start = date + pd.Timedelta(hours = 13, minutes = 1)
            pm_end = date + pd.Timedelta(hours = 15, minutes = 0)
            pm_period = pd.date_range(start = pm_start, end = pm_end, freq = '1min')
            trading_periods.append(am_period)
            trading_periods.append(pm_period)
        datetime_index = pd.DatetimeIndex(np.concatenate(trading_periods)).sort_values()

        master_df = master_df.reindex(datetime_index)

        open_cols = master_df.columns[master_df.columns.str.endswith('open')]
        close_cols = master_df.columns[master_df.columns.str.endswith('close')]
        ohl_cols = master_df.columns[master_df.columns.str.endswith(('open', 'high', 'low'))]
        hlc_cols = master_df.columns[master_df.columns.str.endswith(('high', 'low', 'close'))]

        master_df[close_cols] = master_df[close_cols].ffill()
        for item in ohl_cols:
            master_df[item] = master_df[item].fillna(master_df[item.split('_')[0] + '_close'])
        zero_cols = master_df.columns[master_df.columns.str.endswith(('net', 'perc_chg', 'volume'))]
        master_df[zero_cols] = master_df[zero_cols].fillna(0)
        master_df[open_cols] = master_df[open_cols].bfill()
        for item in hlc_cols:
            master_df[item] = master_df[item].fillna(master_df[item.split('_')[0] + '_open'])
        
    # Processing pipeline for text data.
    def create_text_df(
        self,
        raw_path: str,
        processed_path: str,
        file_name: str,
        medium: Literal['lseg_news', 'x_posts'],
        text_col: str | None = None,
        date_col: str | None = None,
        embedding_dimension: int | None = 1024,
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
            tpm_limit=90000,
            ignore_history=ignore_history
        )
        if medium == 'lseg_news':
            self._translate_headlines(ignore_history=ignore_history)
            self._get_finbert_sentiment(ignore_history=ignore_history)
        
        elif medium == 'x_posts':
            self._get_multilingual_sentiment(ignore_history=ignore_history)

        if self._history != init_history:
            joblib.dump(self, data_source_path)