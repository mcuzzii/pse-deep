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
import pandas_ta as ta

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
def snake_case(text_string: str):
    text_string = re.sub(r'[^A-Za-z0-9]', ' ', text_string)
    text_string = re.sub(r'([a-z])([A-Z])', r'\1 \2', text_string)
    text_string = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text_string)
    text_string = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', text_string)
    text_string = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text_string)
    
    text_string = re.sub(r'\s+', '_', text_string.strip()).lower()

    return text_string

# Helper function for getting log softmax probabilities of the language of a text.
def get_lang(text: str):
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

# Helper function to get unique instrument names from a directory:
def get_unique_instruments(dir_path: str):
    instruments = set()

    for item in Path(dir_path).glob('*.xlsx'):
        instruments.add(item.name.split('.')[0].split('_')[0])
    
    return list(instruments)

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

        df = pd.read_excel(list(self.raw_path.glob('*.xlsx'))[0])

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

        # Set date_time as index
        self.df = self.df.set_index(self.date_col)
    
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
        self.df['cleaned_headline'] = self.df[self.text_col].copy()
        
        # Step 1: Fast Language Detection (Row by Row is okay here, langid is fast)
        print("Detecting languages...")
        self.df[['detected_lang', 'en_score']] = self.df[self.text_col].apply(get_lang).apply(pd.Series)
        
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
            texts_to_translate = self.df.loc[mask, self.text_col].tolist()
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

        self.text_col = 'cleaned_headline'
        
        self.df[self.text_col] = self.df[self.text_col].str.strip().str.lower()
        self.df[self.text_col] = self.df[self.text_col].str.replace(r'\s+', ' ', regex=True)
    
    def get_translated_examples(
        self,
        n: int = 10
    ):
        file_path = self.processed_path / f'{self.file_name}_translated_examples.json'
        if file_path.exists():
            return
        
        sample_df = self.df.loc[self.df['en_score'] < -30].sample(n)
        sample_df = sample_df[['text', 'cleaned_headline']]
        sample_df.to_json(file_path, orient='records', indent=4)
    
    @record_history
    def _get_finbert_sentiment(
        self,
        ignore_history: bool = False
    ):
        device = 0 if torch.cuda.is_available() else -1

        finbert = pipeline(
            'text-classification', 
            model='tabularisai/ModernFinBERT', 
            device=device,
            top_k=None  # This ensures we get positive, negative, and neutral scores
        )

        texts = self.df[self.text_col].tolist()
        raw_results = []
        batch_size = 128

        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
            batch = texts[i : i + batch_size]
            
            # Call the pipeline on the chunk
            batch_results = finbert(
                batch,
                truncation=True,
                max_length=512
            )
            
            raw_results.extend(batch_results)

        parsed_results = []
        for row in raw_results:
            parsed_results.append({item['label']: item['score'] for item in row})
        
        sentiment_df = pd.DataFrame(parsed_results)
        sentiment_df['sentiment'] = sentiment_df[list(sentiment_df.columns)].idxmax(axis=1).str.title()

        if 'bullish' in sentiment_df.columns and 'bearish' in sentiment_df.columns:
            sentiment_df['finbert_combined_score'] = sentiment_df['bullish'] - sentiment_df['bearish']
        
        self.df[sentiment_df.columns] = sentiment_df.values
    
    def get_headline_sentiment_examples(
        self,
        n: int = 15
    ):
        file_path = self.processed_path / f'{self.file_name}_headline_sentiment_examples.json'
        if file_path.exists():
            return
        
        num_positive = num_negative = n // 3
        num_neutral = n - num_positive - num_negative

        positive_examples = self.df.loc[self.df['sentiment'] == 'Bullish'].sample(num_positive)
        negative_examples = self.df.loc[self.df['sentiment'] == 'Bearish'].sample(num_negative)
        neutral_examples = self.df.loc[self.df['sentiment'] == 'Neutral'].sample(num_neutral)

        combined_df = pd.concat([positive_examples, negative_examples, neutral_examples])
        combined_df.reset_index()
        combined_df = combined_df[[self.text_col, 'sentiment', 'finbert_combined_score', 'bullish', 'neutral', 'bearish']]
        combined_df.to_json(file_path, orient='records', indent=4)
    
    @record_history
    def _get_multilingual_sentiment(
        self,
        ignore_history: bool = False
    ):

        device = 0 if torch.cuda.is_available() else -1
        batch_size = 128

        # Initialize pipeline with FP16 for V100 speed
        multilingual_sentiment = pipeline(
            'text-classification',
            model='tabularisai/multilingual-sentiment-analysis',
            device=device,
            top_k=None
        )

        # 1. Clean and prepare the list
        texts = self.df[self.text_col].tolist()
        raw_results = []

        # 2. Manual Batching Loop
        # We iterate in steps of 'batch_size'
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
            batch = texts[i : i + batch_size]
            
            # Call the pipeline on the chunk
            # Note: We don't need batch_size=128 inside the call here 
            # because we are physically handing it a list of 128.
            batch_results = multilingual_sentiment(
                batch, 
                truncation=True, 
                max_length=512
            )
            
            raw_results.extend(batch_results)

        # 3. Parse and Calculate (Same logic as before)
        parsed_results = [{item['label']: item['score'] for item in row} for row in raw_results]
        sentiment_df = pd.DataFrame(parsed_results)
        
        categories = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        probs = sentiment_df[categories].values
        weights = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        sentiment_df['sentiment_score'] = np.dot(probs, weights)
        sentiment_df['sentiment'] = sentiment_df[categories].idxmax(axis=1)

        # 4. Cleanup and Merge
        sentiment_df.columns = [snake_case(col) for col in sentiment_df.columns]
        self.df[sentiment_df.columns] = sentiment_df.values
    
    def get_social_sentiment_examples(
        self,
        n: int = 15
    ):
        file_path = self.processed_path / f'{self.file_name}_social_sentiment_examples.json'
        if file_path.exists():
            return
        
        num_very_positive = num_positive = num_negative = num_very_negative = n // 5
        num_neutral = n - num_very_positive - num_positive - num_negative - num_very_negative

        very_positive_examples = self.df.loc[self.df['sentiment'] == 'Very Positive'].sample(num_very_positive)
        positive_examples = self.df.loc[self.df['sentiment'] == 'Positive'].sample(num_positive)
        neutral_examples = self.df.loc[self.df['sentiment'] == 'Neutral'].sample(num_neutral)
        negative_examples = self.df.loc[self.df['sentiment'] == 'Negative'].sample(num_negative)
        very_negative_examples = self.df.loc[self.df['sentiment'] == 'Very Negative'].sample(num_very_negative)

        sentiment_examples_df = pd.concat(
            [very_positive_examples, positive_examples, neutral_examples, negative_examples, very_negative_examples]
        )
        sentiment_examples_df = sentiment_examples_df[
            [self.text_col, 'sentiment_score', 'sentiment', 'very_negative', 'negative', 'neutral', 'positive', 'very_positive']
        ]
        sentiment_examples_df.to_json(file_path, orient='records', indent=4)
    
    @record_history
    def _load_financial_instrument(
        self,
        ignore_history: bool = False
    ):
        # A single instrument can span multiple files
        self.df = pd.DataFrame()

        for item in self.raw_path.glob('*.xlsx'):
            instrument_name = item.name.split('.')[0].split('_')[0].lower()
            if self.file_name == instrument_name:
                sheet = pd.read_excel(item)

                # Get header row
                header = sheet.index[sheet.isin(["Exchange Date"]).any(axis=1)].tolist()[0]

                # Extract and define columns
                cols = sheet.iloc[header, 3:].dropna().str.lower().str.replace(' ', '_').str.replace('%', 'perc_')
                cols.iloc[1:] = instrument_name + '_' + cols.iloc[1:]

                # Extract data
                partial_df = sheet.iloc[header + 1:, 3:3 + cols.shape[0]].copy()
                partial_df.columns = cols.tolist()
                partial_df['local_time'] = pd.to_datetime(partial_df['local_time'])
                partial_df = partial_df.set_index('local_time')
                
                # Merge with the master dataframe
                self.df = self.df.combine_first(partial_df)
    
    def _col(self, *suffixes):
        return {s: f'{self.file_name}_{s}' for s in suffixes}

    def _process_stock(self):
        c = self._col('open', 'high', 'low', 'close', 'net', 'volume', 'perc_chg')

        self.df[c['close']] = self.df[c['close']].ffill()
        for col in ('open', 'high', 'low'):
            self.df[c[col]] = self.df[c[col]].fillna(self.df[c['close']])

        self.df[c['open']] = self.df[c['open']].bfill()
        for col in ('close', 'high', 'low'):
            self.df[c[col]] = self.df[c[col]].fillna(self.df[c['open']])

        self.df[[c['net'], c['perc_chg'], c['volume']]] = self.df[[c['net'], c['perc_chg'], c['volume']]].fillna(0)

        self._generate_indicators(c)

    def _generate_indicators(self, c: dict):
        fn  = self.file_name
        df  = self.df
        close = df[c['close']].astype(float)
        high = df[c['high']].astype(float)
        low = df[c['low']].astype(float)
        volume = df[c['volume']].astype(float)

        # Periods
        RSI_PERIOD = 14
        STOCH_K = 14
        STOCH_FAST_D = 3
        STOCH_SLOW_D = 3
        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9
        MA_SHORT = 10
        MA_LONG = 50
        EMA_SHORT = 10
        EMA_LONG = 50
        BB_PERIOD = 20
        BB_STD = 2.0
        ATR_PERIOD = 14
        ADX_PERIOD = 14
        PROC_PERIODS = [1, 5, 10, 20] # price momentum look-backs
        RVOL_PERIOD = 20 # rolling window for relative volume
        VROC_PERIOD = 14 # volume rate of change look-back
        PSY_PERIOD = 12 # psychological line window
        RCI_PERIOD = 9 # rank correlation index window

        # RSI
        df[f'{fn}_rsi'] = ta.rsi(close, length=RSI_PERIOD)

        # Stochastic Oscillator (%K, fast %D, slow %D)
        stoch = ta.stoch(high, low, close,
                        k=STOCH_K, d=STOCH_FAST_D, smooth_k=STOCH_SLOW_D)
        df[f'{fn}_stoch_k']      = stoch[f'STOCHk_{STOCH_K}_{STOCH_FAST_D}_{STOCH_SLOW_D}']
        df[f'{fn}_stoch_fast_d'] = stoch[f'STOCHd_{STOCH_K}_{STOCH_FAST_D}_{STOCH_SLOW_D}']
        # Slow %D = SMA of fast %D
        df[f'{fn}_stoch_slow_d'] = df[f'{fn}_stoch_fast_d'].rolling(STOCH_SLOW_D).mean()

        # Log Returns
        df[f'{fn}_log_return'] = np.log(close / close.shift(1))

        # Cumulative Log Return vs. Previous Day's Close
        # At 1-min frequency: group by date, use the last close of the prior date
        prev_day_close = (
            close
            .groupby(close.index.date)
            .transform('first')   # first bar of each day ≈ previous EOD reference
            .shift(1)             # shift one bar so t=0 is the prior close
        )
        df[f'{fn}_cum_log_return'] = np.log(close / prev_day_close)

        # Price Momentum / PROC
        for p in PROC_PERIODS:
            df[f'{fn}_proc_{p}'] = close.diff(p)

        # MACD
        macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        df[f'{fn}_macd'] = macd[f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        df[f'{fn}_macd_signal'] = macd[f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        df[f'{fn}_macd_hist'] = macd[f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']

        # Moving Averages
        df[f'{fn}_ma_short'] = ta.sma(close, length=MA_SHORT)
        df[f'{fn}_ma_long'] = ta.sma(close, length=MA_LONG)
        df[f'{fn}_ema_short'] = ta.ema(close, length=EMA_SHORT)
        df[f'{fn}_ema_long'] = ta.ema(close, length=EMA_LONG)

        # OSCP / SLMA  &  EOSCP / SLEMA
        # OSCP  = MA_short − MA_long          (Son, 2012)
        # EOSCP = EMA_short − EMA_long
        df[f'{fn}_oscp'] = df[f'{fn}_ma_short']  - df[f'{fn}_ma_long']
        df[f'{fn}_eoscp'] = df[f'{fn}_ema_short'] - df[f'{fn}_ema_long']

        # Price Disparity: DISP / EDISP
        # DISP  = (close − MA)  / MA          (Son, 2012)
        # EDISP = (close − EMA) / EMA
        df[f'{fn}_disp'] = (close - df[f'{fn}_ma_long'])  / df[f'{fn}_ma_long']
        df[f'{fn}_edisp'] = (close - df[f'{fn}_ema_long']) / df[f'{fn}_ema_long']

        # Psychological Line
        # Proportion of up-closes in the last N bars (Tanaka-Yamawaki & Tokuoka, 2007)
        up_close = (close.diff(1) > 0).astype(int)
        df[f'{fn}_psy'] = up_close.rolling(PSY_PERIOD).sum() / PSY_PERIOD

        # Rank Correlation Index
        # Spearman ρ between price ranks and time ranks over N bars
        # (Tanaka-Yamawaki & Tokuoka, 2007)
        def _rci(series, n):
            time_ranks = np.arange(1, n + 1)           # fixed time rank
            def _spearman(window):
                price_ranks = pd.Series(window).rank().values
                d_sq = ((time_ranks - price_ranks) ** 2).sum()
                return 1 - (6 * d_sq) / (n * (n ** 2 - 1))
            return series.rolling(n).apply(_spearman, raw=True)

        df[f'{fn}_rci'] = _rci(close, RCI_PERIOD)

        # Accumulation / Distribution Oscillator
        # AD = sum of ((close−low)−(high−close)) / (high−low) × volume
        # (Kong & Azencott, 2020) — pandas-TA exposes this as `ad`
        df[f'{fn}_ad'] = ta.ad(high, low, close, volume)

        # On-Balance Volume
        df[f'{fn}_obv'] = ta.obv(close, volume)

        # Relative Volume
        # Current volume vs. rolling average volume (conviction proxy)
        rolling_avg_vol = volume.rolling(RVOL_PERIOD).mean()
        df[f'{fn}_rvol'] = volume / rolling_avg_vol

        # Volume Rate of Change
        # (Kong & Azencott, 2020)
        df[f'{fn}_vroc'] = volume.pct_change(periods=VROC_PERIOD) * 100

        # Bollinger Band %B
        bb = ta.bbands(close, length=BB_PERIOD, std=BB_STD)
        upper = bb.filter(like='BBU').iloc[:, 0]
        lower = bb.filter(like='BBL').iloc[:, 0]
        df[f'{fn}_bb_pct_b'] = (close - lower) / (upper - lower)

        # Parabolic SAR
        psar = ta.psar(high, low, close)
        # pandas-TA returns the active SAR level in the 'long' or 'short' column;
        # coalesce both into a single series
        psar_long = psar.filter(like='PSARl').iloc[:, 0]
        psar_short = psar.filter(like='PSARs').iloc[:, 0]

        df[f'{fn}_psar'] = psar_long.combine_first(psar_short)

        # Average True Range
        df[f'{fn}_atr'] = ta.atr(high, low, close, length=ATR_PERIOD)

        # ADX (+DI, −DI)
        adx = ta.adx(high, low, close, length=ADX_PERIOD)
        df[f'{fn}_adx']  = adx.filter(like='ADX_').iloc[:, 0]
        df[f'{fn}_adx_di_pos'] = adx.filter(like='DMP_').iloc[:, 0]
        df[f'{fn}_adx_di_neg'] = adx.filter(like='DMN_').iloc[:, 0]

        self.df = df

    def _process_copper(self):
        c = self._col('bid', 'ask')
        self.df = self.df[[c['bid'], c['ask']]]

    def _process_forex(self):
        c = self._col('bid', 'ask', 'bid_net', 'open', 'high', 'low', 'refresh_rate')

    def _process_oil(self):
        c = self._col('bid', 'ask', 'open', 'high', 'low', 'close', 'net', 'volume', 'perc_chg')

    def _process_bond(self, ignore_history: bool = False):
        c = self._col('bid', 'ask', 'askyld', 'bidychg', 'bidyld')

    @record_history
    def _process_high_frequency_instruments(
        self,
        ignore_history: bool = False
    ):

        unique_dates = self.df.index.normalize().unique().sort_values()
        trading_periods = []

        if self._medium == 'stock':
            for date in unique_dates:

                # Define morning stock trading hours
                am_start = date + pd.Timedelta(hours = 9, minutes = 31)
                am_end = date + pd.Timedelta(hours = 12, minutes = 0)
                am_period = pd.date_range(start = am_start, end = am_end, freq = '1min')

                # Define afternoon trading hours
                pm_start = date + pd.Timedelta(hours = 13, minutes = 1)
                pm_end = date + pd.Timedelta(hours = 15, minutes = 0)
                pm_period = pd.date_range(start = pm_start, end = pm_end, freq = '1min')

                trading_periods.append(am_period)
                trading_periods.append(pm_period)

        else:
            for date in unique_dates:

                start = date + pd.Timedelta(hours = 0, minutes = 0)
                end = date + pd.Timedelta(hours = 23, minutes = 59)
                period = pd.date_range(start=start, end=end, freq='1min')

                trading_periods.append(period)
        
        datetime_index = pd.DatetimeIndex(np.concatenate(trading_periods)).sort_values()

        self.df = self.df.reindex(datetime_index)

        self.df['no_activity'] = self.df.isna().all(axis=1).astype(int)
        first_idx = self.df['no_activity'].idxmax()
        last_idx = self.df['no_activity'][::-1].idxmax()
        self.df = self.df.loc[first_idx:last_idx]

        if self._medium == 'stock':
            self._process_stock()
        elif self._medium == 'copper':
            self._process_copper()
        elif self._medium == 'fx':
            self._process_forex()
        elif self._medium == 'oil':
            self._process_oil()
    
    def _text_preprocess(
        self,
        ignore_history: bool = False
    ):
        self._clean_text(ignore_history=ignore_history)
        self._cohere_embed(
            cache_location=Path(self.processed_path) / 'cache',
            tpm_limit=90000,
            ignore_history=ignore_history
        )
        if self._medium == 'lseg_news':
            self._translate_headlines(ignore_history=ignore_history)
            self._get_finbert_sentiment(ignore_history=ignore_history)
        
        elif self._medium == 'x_posts':
            self._get_multilingual_sentiment(ignore_history=ignore_history)
        
    # Processing pipeline for text data.
    def create_df(
        self,
        file_name: str,
        raw_folder_name: str,
        medium: Literal['lseg_news', 'x_posts', 'stock'] | None = None,
        text_col: str | None = None,
        date_col: str | None = None,
        raw_path: str = 'data/raw',
        processed_path: str = 'data/processed',
        embedding_dimension: int | None = 1024,
        ignore_history: bool = False
    ):
        """ Pipeline for preprocessing datasets. """

        self.raw_path = Path(raw_path) / raw_folder_name
        self.file_name = file_name
        self._medium = medium
        self.processed_path = Path(processed_path)
        data_source_path = self.processed_path / f'{self.file_name}.joblib'

        if data_source_path.exists():
            saved_data_source = joblib.load(data_source_path)
            self.__dict__.update(saved_data_source.__dict__)

        init_history = self._history.copy()

        # Processing text-based datasets.
        if self._medium == 'x_posts':
            self.text_col = snake_case(text_col)
            self.date_col = snake_case(date_col)
            self._load_json_folder(ignore_history=ignore_history)
            self._text_preprocess(ignore_history=ignore_history)
        
        elif self._medium == 'lseg_news':
            self._load_lseg_news(ignore_history=ignore_history)
            self.text_col = 'text'
            self.date_col = 'date_time'
            self._text_preprocess(ignore_history=ignore_history)
        
        elif self._medium in ['stock', 'bond', 'copper', 'oil', 'fx']:
            self._load_financial_instrument(ignore_history=ignore_history)
            if self._medium != 'bond':
                self._process_high_frequency_instruments(ignore_history=ignore_history)
            else:
                self._process_bond(ignore_history=ignore_history)

        if self._history != init_history:
            joblib.dump(self, data_source_path)

