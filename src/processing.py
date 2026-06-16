import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from tqdm import tqdm
import requests
import joblib
import functools
import json
import re
from dotenv import load_dotenv

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
    import langid

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

def get_stocks():
    stocks = get_unique_instruments('data/raw/stock')
    stocks = list(set(stocks) - {'psei', 'psho', 'psse', 'psmo', 'psfi', 'pspr', 'psin'})
    return stocks

def get_features(df):

    features = [
        col
        for col in df.columns
        if not col.endswith('m_return') and
        not col.endswith('_no_activity')
    ]

    binary_cols = [col for col in features if df[col].nunique() <= 2]
    continuous_cols = [col for col in features if col not in binary_cols]

    return features, continuous_cols, binary_cols

def get_text_window(timestamp, T, pred_horizon, min_hours=24):

    min_trading_minutes = 270 - 2 * pred_horizon

    cutoff_24h = timestamp - pd.Timedelta(hours=min_hours)
    past_trading_minutes = T[T <= timestamp]

    window_24h = past_trading_minutes[past_trading_minutes > cutoff_24h]
    
    if len(window_24h) >= min_trading_minutes:
        return cutoff_24h, window_24h
    
    else:
        if len(past_trading_minutes) <= min_trading_minutes:
            cutoff_270 = past_trading_minutes.min() - pd.Timedelta(minutes=1)
        else:
            cutoff_270 = past_trading_minutes[-min_trading_minutes - 1]
        
        final_cutoff = min(cutoff_24h, cutoff_270)
        window = past_trading_minutes[past_trading_minutes > final_cutoff]
        return final_cutoff, window

def get_agg_keys(features, agg):
    return [k for k in features.keys() if agg in features[k]]

def get_aggregates(data, features, agg):
    if not get_agg_keys(features, agg):
        return dict()
    if agg != 'follower_weighted_mean':
        str = f"data[get_agg_keys(features)].astype('float32').{agg}().add_suffix('_{agg}')"
        return eval(str, dict(), {'data': data, 'features': features})
    else:
        return data[get_agg_keys(features)].astype('float32').apply(
            lambda col: col * data['author_followers']
        ).sum() / data['author_followers'].sum()

def get_sentiment(data, k, pos=True):
    sentiment = data[k].value_counts()
    fin_sentiment = sentiment.get('bullish' if pos else 'bearish', 0)
    intense_sentiment = sentiment.get('Very Positive' if pos else 'Very Negative', 0)
    mild_sentiment = sentiment.get('Positive' if pos else 'Negative', 0)

    return fin_sentiment + intense_sentiment + 0.5 * mild_sentiment

def get_custom_indicator(data, k, i):

    if i == 'intensity':
        return (get_sentiment(data, k, True) - get_sentiment(data, k, False)) / (data.shape[0] + 1e-4)
    
    elif i == 'strong':
        return (data[k] > 0.7).sum() / (data.shape[0] + 1e-4)
    
    elif i == 'momentum':
        return data[k].iloc[-data.shape[0] // 3:].mean() - data[k].mean()
    
    elif i == 'short_momentum':
        return data[k].iloc[-data.shape[0] // 6:].mean() - data[k].iloc[-data.shape[0] // 2].mean()
    
    elif i == 'pos_neg_ratio':
        return get_sentiment(data, k, True) / (get_sentiment(data, k, False) + 1e-4)
    
    elif i == 'net':
        return (data[k] > 0).sum() / (data.shape[0] + 1e-4)
    
    elif i == 'prop':
        return (
            (sentiment := data[k].value_counts()).get('neutral', 0) + sentiment.get('Neutral', 0)
        ) / (data.shape[0] + 1e-4)
    
    elif i == 'viral_coeff':
        return (data[k] + data['reply_count']) / (data.shape[0] + 1e-4)

def compute_text_stats(text_df, features, cutoffs, trading_minute):
    cutoff = cutoffs[trading_minute]
    
    mask = (text_df.index > cutoff) & (text_df.index <= trading_minute)
    window_data = text_df[mask]

    sums = get_aggregates(window_data, features, 'sum')
    means = get_aggregates(window_data, features, 'mean')
    stds = get_aggregates(window_data, features, 'std')
    maxs = get_aggregates(window_data, features, 'max')
    mins = get_aggregates(window_data, features, 'min')

    custom_indicators = dict()

    for k in features:
        for i in features[k]:
            custom_indicators[f'{k}_{i}'] = get_custom_indicator(window_data, k, i)
    
    custom_indicators = pd.Series(custom_indicators)

    return pd.concat([means, stds, maxs, mins, custom_indicators])

def get_elapsed_time(timestamps):
    reference = pd.Timestamp('2025-03-12 00:00:00')
    num_seconds = (pd.Timestamp('2026-04-17 00:00:00') - reference).total_seconds()
    return (timestamps - reference).total_seconds() / num_seconds

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
        from cohere import ClientV2
        from tokenizers import Tokenizer
        import ast

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
        from sklearn.metrics.pairwise import cosine_similarity

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
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        from transformers import pipeline
        import torch

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
        from transformers import pipeline
        import torch

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
        for col in sentiment_df.columns:
            self.df[col] = sentiment_df[col].values
    
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
        self.df = None

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
                if self.df is None:
                    self.df = partial_df
                else:
                    self.df = self.df.combine_first(partial_df)
    
    def _col(self, *suffixes):
        return {s: f'{self.file_name}_{s}' for s in suffixes}
    
    def _close_indicators(self, c: dict, close_attr='close'):
        import pandas_ta as ta

        fn = self.file_name
        df = self.df
        close = df[c[close_attr]].astype(float)

        LOG_RETURN_PERIODS = [1, 5, 10, 30]
        RSI_PERIOD = 14
        PROC_PERIODS = [5, 10, 20]
        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9
        MA_SHORT = 10
        MA_LONG = 50
        EMA_SHORT = 10
        EMA_LONG = 50
        PSY_PERIOD = 12
        RCI_PERIOD = 9
        BB_PERIOD = 20
        BB_STD = 2.0

        df[f'{fn}_rsi'] = ta.rsi(close, length=RSI_PERIOD)

        for p in LOG_RETURN_PERIODS:
            df[f'{fn}_log_return_{p}'] = np.log(close / close.shift(p))

        # Last close price per day
        last_close_per_day = close.groupby(close.index.date).last()

        # Shift by 1 day to get previous day's last close
        prev_day_last_close = last_close_per_day.shift(1)

        # Map back to every minute row
        prev_day_close = close.index.to_series().apply(
            lambda ts: prev_day_last_close.get(ts.date())
        )

        df[f'{fn}_cum_log_return'] = np.log(close / prev_day_close.values)

        for p in PROC_PERIODS:
            df[f'{fn}_proc_{p}'] = close.pct_change(p)

        macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        df[f'{fn}_macd'] = macd[f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        df[f'{fn}_macd_signal'] = macd[f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        df[f'{fn}_macd_hist'] = macd[f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']

        df[f'{fn}_ma_short'] = ta.sma(close, length=MA_SHORT)
        df[f'{fn}_ma_long'] = ta.sma(close, length=MA_LONG)
        df[f'{fn}_ema_short'] = ta.ema(close, length=EMA_SHORT)
        df[f'{fn}_ema_long'] = ta.ema(close, length=EMA_LONG)

        df[f'{fn}_oscp'] = (df[f'{fn}_ma_short']  - df[f'{fn}_ma_long']) / df[f'{fn}_ma_short']
        df[f'{fn}_eoscp'] = (df[f'{fn}_ema_short'] - df[f'{fn}_ema_long']) / df[f'{fn}_ema_short']

        df[f'{fn}_disp_short'] = (close - df[f'{fn}_ma_short']) / df[f'{fn}_ma_short']
        df[f'{fn}_disp_long'] = (close - df[f'{fn}_ma_long']) / df[f'{fn}_ma_long']
        df[f'{fn}_edisp_short'] = (close - df[f'{fn}_ema_short']) / df[f'{fn}_ema_short']
        df[f'{fn}_edisp_long'] = (close - df[f'{fn}_ema_long']) / df[f'{fn}_ema_long']

        up_close = (close.diff(1) > 0).astype(int)
        df[f'{fn}_psy'] = up_close.rolling(PSY_PERIOD).sum() / PSY_PERIOD

        def _rci(series, n):
            time_ranks = np.arange(1, n + 1)
            def _spearman(window):
                price_ranks = pd.Series(window).rank().values
                d_sq = ((time_ranks - price_ranks) ** 2).sum()
                return 1 - (6 * d_sq) / (n * (n ** 2 - 1))
            return series.rolling(n).apply(_spearman, raw=True)

        df[f'{fn}_rci'] = _rci(close, RCI_PERIOD)

        bb = ta.bbands(close, length=BB_PERIOD, std=BB_STD)
        upper = bb.filter(like='BBU').iloc[:, 0]
        lower = bb.filter(like='BBL').iloc[:, 0]
        df[f'{fn}_bb_pct_b'] = (close - lower) / (upper - lower)
        df.loc[np.isclose(upper, lower) & np.isclose(lower, close), f'{fn}_bb_pct_b'] = 0.5

        self.df = df
    
    def _hlc_indicators(self, c: dict, close_attr='close'):
        import pandas_ta as ta

        fn = self.file_name
        df = self.df
        close = df[c[close_attr]].astype(float)
        high = df[c['high']].astype(float)
        low = df[c['low']].astype(float)

        STOCH_K = 14
        STOCH_FAST_D = 3
        STOCH_SLOW_D = 3
        ATR_PERIOD = 14
        ADX_PERIOD = 14

        stoch = ta.stoch(high, low, close,
                        k=STOCH_K, d=STOCH_FAST_D, smooth_k=1)
        df[f'{fn}_stoch_k']      = stoch[f'STOCHk_{STOCH_K}_{STOCH_FAST_D}_1']
        df[f'{fn}_stoch_fast_d'] = stoch[f'STOCHd_{STOCH_K}_{STOCH_FAST_D}_1']
        df[f'{fn}_stoch_slow_d'] = df[f'{fn}_stoch_fast_d'].rolling(STOCH_SLOW_D).mean()

        psar = ta.psar(high, low, close)
        psar_long = psar.filter(like='PSARl').iloc[:, 0]
        psar_short = psar.filter(like='PSARs').iloc[:, 0]

        psar_value = psar_long.combine_first(psar_short)
        df[f'{fn}_psar_dist'] = (close - psar_value) / close
        df[f'{fn}_psar_dir'] = np.where(psar_long.notna(), 1, -1)
        df[f'{fn}_psar_reversal'] = psar.filter(like='PSARr').iloc[:, 0]

        df[f'{fn}_atr'] = ta.atr(high, low, close, length=ATR_PERIOD)

        adx = ta.adx(high, low, close, length=ADX_PERIOD)
        
        df[f'{fn}_adx']  = adx.filter(like='ADX_').iloc[:, 0]
        df[f'{fn}_adx_di_pos'] = adx.filter(like='DMP_').iloc[:, 0]
        df[f'{fn}_adx_di_neg'] = adx.filter(like='DMN_').iloc[:, 0]

        self.df = df

    def _cv_indicators(self, c: dict):
        import pandas_ta as ta

        fn  = self.file_name
        df  = self.df
        close = df[c['close']].astype(float)
        volume = df[c['volume']].astype(float)

        RVOL_PERIOD = 20

        df[f'{fn}_obv_change'] = ta.obv(close, volume).diff(1)

        rolling_avg_vol = volume.rolling(RVOL_PERIOD).mean()
        df[f'{fn}_rvol'] = volume / rolling_avg_vol
        df.loc[rolling_avg_vol == 0, f'{fn}_rvol'] = 0

        self.df = df
    
    def _bid_ask_indicators(self, c: dict):
        import pandas_ta as ta

        fn = self.file_name
        df = self.df
        ask = df[c['ask']].astype(float)
        bid = df[c['bid']].astype(float)

        SPREAD_MA_PERIODS = [10, 30, 50]

        df[f'{fn}_spread'] = ask - bid
        df[f'{fn}_midprice'] = (ask + bid) / 2
        df[f'{fn}_rel_spread'] = (ask - bid) / df[f'{fn}_midprice']

        c['midprice'] = f'{fn}_midprice'

        for l in SPREAD_MA_PERIODS:
            df[f'{fn}_spread_ma_{l}'] = ta.sma(df[f'{fn}_spread'], length=l)
        
        self.df = df
    
    def _fill(self, c: dict, *attr_names, value=0, kind='value'):
        if attr_names:
            attrs = c[attr_names[0]] if len(attr_names) == 1 else [c[attr_name] for attr_name in attr_names]

            self.df[attrs] = self.df[attrs].ffill() if kind == 'forward' else (
                self.df[attrs].fillna(value) if kind == 'value' else (
                    self.df[attrs].bfill() if kind == 'backward' else self.df[attrs]
                )
            )
    
    def _ohlc_fill(self, c: dict, close_attr='close'):
        self._fill(c, close_attr, kind='forward')
        for col in ('open', 'high', 'low'):
            self._fill(c, col, value=self.df[c[close_attr]])
        
        self._fill(c, 'open', kind='backward')
        for col in (close_attr, 'high', 'low'):
            self._fill(c, col, value=self.df[c['open']])
    
    def _process_stock(self):
        c = self._col('open', 'high', 'low', 'close', 'net', 'volume', 'perc_chg')

        self._ohlc_fill(c)
        self._fill(c, 'net', 'perc_chg', 'volume', value=0)

        self._close_indicators(c)
        self._hlc_indicators(c)
        self._cv_indicators(c)

    def _process_copper(self):
        c = self._col('bid', 'ask', 'no_activity')
        self.df = self.df[[c['bid'], c['ask'], c['no_activity']]]

        self._fill(c, 'bid', 'ask', kind='forward')
        self._fill(c, 'bid', 'ask', kind='backward')
        
        self._bid_ask_indicators(c)
        self._close_indicators(c, close_attr='midprice')

    def _process_forex(self):
        c = self._col('bid', 'ask', 'bidnet', 'open', 'high', 'low', 'refresh_rate')

        self._fill(c, 'bid', 'ask', kind='forward')
        self._fill(c, 'bid', 'ask', kind='backward')

        self._bid_ask_indicators(c)

        self._ohlc_fill(c, close_attr='midprice')
        self._fill(c, 'bidnet', 'refresh_rate', value=0)

        self._close_indicators(c, close_attr='midprice')
        self._hlc_indicators(c, close_attr='midprice')

    def _process_oil(self):
        c = self._col('bid', 'ask', 'open', 'high', 'low', 'close', 'net', 'volume', 'perc_chg')

        self._fill(c, 'bid', 'ask', kind='forward')
        self._fill(c, 'bid', 'ask', kind='backward')

        self._bid_ask_indicators(c)

        self._ohlc_fill(c)
        self._fill(c, 'net', 'perc_chg', 'volume', value=0)

        self._close_indicators(c)
        self._hlc_indicators(c)
        self._cv_indicators(c)
    
    @record_history
    def _process_bond(self, ignore_history: bool = False):
        c = self._col('bid', 'ask', 'askyld', 'bidyld', 'bidychg')

        self._fill(c, 'bid', 'ask', 'askyld', 'bidyld', kind='forward')
        self._fill(c, 'bid', 'ask', 'askyld', 'bidyld', kind='backward')
        self._fill(c, 'bidychg', value=0)
    
    def _within_bond_indicators(self, c: dict):
        fn = self.file_name
        df = self.df
        bid = df[c['bid']].astype(float)
        ask = df[c['ask']].astype(float)
        bidyld = df[c['bidyld']].astype(float)
        askyld = df[c['askyld']].astype(float)
        
        df[f'{fn}_spread'] = ask - bid
        df[f'{fn}_midprice'] = (ask + bid) / 2
        df[f'{fn}_yld_spread'] = askyld - bidyld
        df[f'{fn}_mdyld'] = (askyld + bidyld) / 2

        self.df = df

    @record_history
    def _process_bonds(self, ignore_history: bool = False):
        bond_master = None
        
        bond_dfs = dict()
        for bond in self.processed_path.glob('phgv*.joblib'):
            bond_df = joblib.load(bond)
            bond_dfs[bond_df.file_name] = bond_df

        for fn in bond_dfs:
            c = bond_dfs[fn]._col('bid', 'ask', 'askyld', 'bidyld')
            
            bond_dfs[fn]._within_bond_indicators(c)

            periods = pd.date_range(start = '2025-03-12', end = '2026-04-17', freq = '1min')
            datetime_index = pd.DatetimeIndex(periods).sort_values()

            bond_dfs[fn].df = bond_dfs[fn].df.reindex(datetime_index).ffill().bfill()

            if bond_master is None:
                bond_master = bond_dfs[fn].df
            else:
                bond_master = bond_master.combine_first(bond_dfs[fn].df)
        
        ordered_terms = sorted(
            [int(fn[4:]) for fn in bond_dfs if 'm' not in fn],
            reverse=True
        ) + sorted(
            [fn[4:] for fn in bond_dfs if 'm' in fn],
            key=lambda x: int(x.replace('m', '')),
            reverse=True
        )

        for i, long_term in enumerate(ordered_terms):
            for short_term in ordered_terms[i + 1:]:
                long_term_yld = bond_master[f'phgv{long_term}_mdyld']
                short_term_yld = bond_master[f'phgv{short_term}_mdyld']
                bond_master[f'phgv{long_term}_{short_term}_term_spread'] = long_term_yld - short_term_yld

        bond_master = bond_master.dropna()
        
        self.df = bond_master
    
    @record_history
    def _combine_data(self, ignore_history: bool = False):
        sectors = pd.read_excel('data/raw/info/sectors_and_subsectors.xlsx')
        sectors.columns = [snake_case(col) for col in sectors.columns]

        sectors['sector'] = sectors['sector'].map({
            'Holding Firms': 'psho',
            'Mining and Oil': 'psmo',
            'Services': 'psse',
            'Property': 'pspr',
            'Industrial': 'psin',
            'Financials': 'psfi'
        })
        sectors['stock_symbol'] = sectors['stock_symbol'].str.lower()
        mapping = sectors.set_index('stock_symbol')['sector'].to_dict()

        psei = joblib.load(self.processed_path / 'psei.joblib')
        sector_df = joblib.load(self.processed_path / f'{mapping[self.file_name]}.joblib')

        self.df = self.df.join(psei.df, how='inner')
        self.df = self.df.join(sector_df.df, how='inner')

        for instrument in ['copper', 'lcoc1', 'usd', 'xau']:
            instrument_df = joblib.load(self.processed_path / f'{instrument}.joblib')
            self.df = self.df.join(instrument_df.df, how='inner')

            close = self.df[f'{self.file_name}_close']
            self.df[f'{self.file_name}_10m_return'] = (close.shift(-10) > close).astype(int)
            self.df[f'{self.file_name}_30m_return'] = (close.shift(-30) > close).astype(int)
        
        bond_master = joblib.load(self.processed_path / 'bond_master.joblib')
        self.df = self.df.join(bond_master.df, how='left')
        bond_columns = bond_master.df.columns.tolist()
        self.df[bond_columns] = self.df[bond_columns]

        self.df = self.df.sort_index()

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
        self.df.index.name = 'local_time'

        fn = self.file_name

        self.df[f'{fn}_no_activity'] = self.df.isna().all(axis=1).astype(int)
        first_idx = (self.df[f'{fn}_no_activity'] == 0).idxmax()
        last_idx = (self.df[f'{fn}_no_activity'] == 0)[::-1].idxmax()
        self.df = self.df.loc[first_idx:last_idx]

        if self._medium == 'stock':
            self._process_stock()
        elif self._medium == 'copper':
            self._process_copper()
        elif self._medium == 'fx':
            self._process_forex()
        elif self._medium == 'oil':
            self._process_oil()

        na_counts = self.df.iloc[50:].isna().sum()
        na_cols = na_counts[na_counts > 0]
        if not na_cols.empty:
            print(f"WARNING: NaN values found in columns: {na_cols.to_dict()}")
        
        self.df = self.df.dropna()

        inf_counts = self.df.isin([float('inf'), float('-inf')]).sum()
        inf_cols = inf_counts[inf_counts > 0]
        if not inf_cols.empty:
            print(f"WARNING: inf values found in columns: {inf_cols.to_dict()}")
    
    # Processing pipeline for text data
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
    
    @record_history
    def _create_feature_selection_data(
        self,
        ignore_history: bool = False
    ):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        print('Computing common dates and times...')
        common_date_times = None
        for stock in self._stocks:
            stock_df = joblib.load(self.processed_path / f'{stock}.joblib')
            dates = stock_df.df.index
            common_date_times = dates if common_date_times is None else common_date_times.intersection(dates)
        
        lunch_mask = (
            (common_date_times.time >= pd.Timestamp(f'11:{61 - self._target}').time()) &
            (common_date_times.time <= pd.Timestamp('12:00').time())
        )

        daily_max = common_date_times.to_series().groupby(common_date_times.date).transform('max')
        last_mask = common_date_times.to_series() > daily_max - pd.Timedelta(minutes=self._target)

        self.filtered_date_times = common_date_times[~lunch_mask & ~last_mask.values]
        train_cutoff = self.filtered_date_times[int(0.8 * len(self.filtered_date_times))]

        stacked_df = None
        self.scalers = dict()
        
        for stock in self._stocks:
            print(f"Processing stock: {stock}...")
            stock_df = joblib.load(self.processed_path / f'{stock}.joblib')

            stock_df.df.drop(columns=[f'{stock}_{40 - self._target}m_return'], inplace=True)

            for col in stock_df.df.columns:
                prefix = col.split('_')[0]
                if prefix in ('psfi', 'psin', 'psmo', 'pspr', 'psse', 'psho'):
                    new_col_name = f'sector{col[len(prefix):]}'
                elif prefix == stock:
                    new_col_name = f'stock{col[len(prefix):]}'
                else:
                    new_col_name = col
                stock_df.df.rename(columns={col: new_col_name}, inplace=True)
            
            stock_df.df.index = pd.MultiIndex.from_product([[stock], stock_df.df.index], names=['stock', 'local_time'])

            stock_df.df = stock_df.df.loc[stock_df.df.index.get_level_values('local_time') <= train_cutoff]

            features, continuous_cols, _ = get_features(stock_df.df)

            ct = ColumnTransformer(
                transformers=[
                    ('scaler', StandardScaler(), continuous_cols)
                ],
                remainder='passthrough'
            )

            ct.fit(stock_df.df[features])
            self.scalers[stock] = ct

            stock_df.df = stock_df.df.astype('float32')

            filtered = stock_df.df[stock_df.df.index.get_level_values('local_time').isin(self.filtered_date_times)]
            sample = filtered.sample(43944)

            stacked_df = sample if stacked_df is None else pd.concat([stacked_df, sample], axis=0)

        self.df = stacked_df
        self.train_cutoff = train_cutoff

    @record_history
    def _feature_select(self, ignore_history: bool = False):
        from mrmr import mrmr_classif

        features, continuous_cols, binary_cols = get_features(self.df)

        def standardize(group):

            stock_name = group.index.get_level_values('stock')[0]
            ct = self.scalers[stock_name]

            transformed_data = ct.transform(group[features])
            all_cols = continuous_cols + binary_cols
            new_group_df = pd.DataFrame(transformed_data, columns=all_cols, index=group.index)
            new_group_df[f'stock_{self._target}m_return'] = group[f'stock_{self._target}m_return']
            return(new_group_df)

        self.df = self.df.groupby(pd.Grouper(level='stock'), group_keys=False).apply(standardize)

        selected_features, relevance, redundancy = mrmr_classif(
            X=self.df[features],
            y=self.df[f'stock_{self._target}m_return'],
            K=100,
            return_scores=True
        )

        self.selected_features = selected_features
        self.relevance = relevance
        self.redundancy = redundancy
    
    @record_history
    def save_selected_features(self, ignore_history: bool = False):
        features = self.relevance[self.selected_features].to_dict()
        for key in features:
            features[key] = {
                'relevance': features[key],
                'redundancy': self.redundancy[key].to_dict()
            }
        with open(self.processed_path / f'{self.file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=4)
    
    def _add_elapsed_time(self):
        self.df['elapsed_time'] = get_elapsed_time(self.df.index).astype('float32')
        self.time_vec_input = 'elapsed_time'
    
    @record_history
    def _finalized_stock(
        self,
        ignore_history: bool = False
    ):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        features_df = joblib.load(self.processed_path / f'features_{self._target}m.joblib')
        sector = next(c.split('_')[0] for c in self.df.columns if c.startswith('ps') and c.split('_')[0] != 'psei')

        self.features = [
            f'{self.file_name}{feature[5:]}'
            if 'stock' in feature else (
                f'{sector}{feature[6:]}'
                if 'sector' in feature else feature
            )
            for feature in features_df.selected_features
        ]
        self.target = f'{self.file_name}_{self._target}m_return'
        self.no_activity_col = f'{self.file_name}_no_activity'

        selected_cols = self.features + [self.no_activity_col, self.target]

        self.df = self.df.loc[features_df.filtered_date_times, selected_cols]

        self.df["date_day_of_month_sin"] = np.sin(2 * np.pi * self.df.index.day / self.df.index.days_in_month)
        self.df["date_day_of_month_cos"] = np.cos(2 * np.pi * self.df.index.day / self.df.index.days_in_month)

        days_in_year = np.where(self.df.index.is_leap_year, 366, 365)
        self.df["date_day_of_year_sin"] = np.sin(2 * np.pi * self.df.index.dayofyear / days_in_year)
        self.df["date_day_of_year_cos"] = np.cos(2 * np.pi * self.df.index.dayofyear / days_in_year)

        self.df["date_day_of_week"] = self.df.index.dayofweek.astype('float32')

        self.df["date_quarter"] = self.df.index.quarter.astype('float32')

        self.df["date_minute_sin"] = np.sin(
            2 * np.pi * (self.df.index.hour * 60 + self.df.index.minute) / 1440
        )
        self.df["date_minute_cos"] = np.cos(
            2 * np.pi * (self.df.index.hour * 60 + self.df.index.minute) / 1440
        )

        t = self.df.index.time
        am_end = pd.Timestamp("12:00").time()

        self.df["session_id"] = (t > am_end).astype('float32')

        # Minutes to end of each session
        am_end_minutes = 12 * 60
        pm_end_minutes = 15 * 60

        current_minutes = self.df.index.hour * 60 + self.df.index.minute

        self.df["minutes_to_session_end"] = np.where(
            self.df["session_id"] == 0,
            am_end_minutes - current_minutes,
            pm_end_minutes - current_minutes
        ).astype('float32')

        self.benchmark_time_features = [
            'date_day_of_month_sin',
            'date_day_of_month_cos',
            'date_day_of_year_sin',
            'date_day_of_year_cos',
            'date_day_of_week',
            'date_quarter',
            'date_minute_sin',
            'date_minute_cos',
            'session_id',
            'minutes_to_session_end',
        ]
        
        features, continuous_cols, binary_cols = get_features(self.df)

        ct = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), continuous_cols)
            ],
            remainder='passthrough'
        )

        self.train_cutoff = features_df.train_cutoff
        train_mask = self.df.index.get_level_values('local_time') <= self.train_cutoff

        all_cols = continuous_cols + binary_cols
        self.df.loc[train_mask, all_cols] = ct.fit_transform(self.df.loc[train_mask, features]).astype('float32')
        self.df.loc[~train_mask, all_cols] = ct.transform(self.df.loc[~train_mask, features]).astype('float32')

        self.scaler = ct

        self._add_elapsed_time()

        print(f'Final dataframe for {self.file_name}; shape: {self.df.shape}.')
        
        self.file_name = f'{self.file_name}_{self._target}m'
        self.filtered_date_times = features_df.filtered_date_times

        self.data_source_path = self.processed_path / f'{self.file_name}.joblib'
    
    @record_history
    def _text_indicators(
        self,
        ignore_history: bool = False
    ):
        from sklearn.preprocessing import StandardScaler
        from mrmr import mrmr_classif

        reference_df = joblib.load(self.processed_path / f'ac_{self._target}m.joblib')
        self.train_cutoff = reference_df.train_cutoff
        self.filtered_date_times = reference_df.filtered_date_times

        text_df = joblib.load(self.processed_path / f'{self.file_name}.joblib')

        cutoffs = pd.Series(
            [get_text_window(ts, self.filtered_date_times, self._target)[0] for ts in self.filtered_date_times],
            index=self.filtered_date_times
        )

        indicator_instructions = {
            'bearish': ('mean', 'std', 'momentum', 'strong', 'net'),
            'bullish': ('mean', 'std', 'momentum', 'strong', 'net'),
            'neutral': ('mean', 'prop'),
            'finbert_combined_score': ('mean', 'std', 'momentum', 'short_momentum', 'max', 'min'),
            'sentiment': ('intensity', 'pos_neg_ratio')
        } if self._medium == 'news_sentiment' else {
            'very_negative': ('follower_weighted_mean', 'std', 'momentum', 'strong', 'net'),
            'negative': ('follower_weighted_mean', 'std', 'momentum', 'strong', 'net'),
            'neutral': ('follower_weighted_mean', 'prop'),
            'positive': ('follower_weighted_mean', 'std', 'momentum', 'strong', 'net'),
            'very_positive': ('follower_weighted_mean', 'std', 'momentum', 'strong', 'net'),
            'sentiment_score': ('follower_weighted_mean', 'std', 'momentum', 'short_momentum', 'max', 'min'),
            'sentiment': ('intensity', 'pos_neg_ratio'),
            'retweet_count': ('sum', 'viral_coeff'),
            'reply_count': ('sum',),
            'like_count': ('sum',),
            'quote_count': ('sum',),
            'view_count': ('mean',),
            'bookmark_count': ('sum',),
            'author_is_blue_verified': ('sum',),
            'author_followers': ('mean',),
            'author_following': ('mean',),
            'author_favourites_count': ('mean',),
            'author_media_count': ('mean',),
            'author_statuses_count': ('mean',),
        }

        self.df = pd.DataFrame(
            [compute_text_stats(text_df.df, indicator_instructions, cutoffs, ts) for ts in self.filtered_date_times],
            index=self.filtered_date_times
        )

        self.text_indicators = [
            f'{k}_{i}'
            for k in indicator_instructions
            for i in indicator_instructions[k]
        ]

        self.df = self.df[self.text_indicators].astype('float32')
        self.df.index.name = 'local_time'

        self.scaler = StandardScaler()

        train_mask = self.df.index.get_level_values('local_time') <= self.train_cutoff

        self.df.loc[train_mask] = self.scaler.fit_transform(self.df.loc[train_mask]).astype('float32')
        self.df.loc[~train_mask] = self.scaler.transform(self.df.loc[~train_mask]).astype('float32')

        stocks = get_stocks()

        mrmr_df = []
        for stock in stocks:
            df = self.df.copy(deep=True)
            stock_df = joblib.load(f'{stock}_{self._target}m.joblib')
            df[f'stock_{self._target}m_return'] = stock_df.df[f'{stock}_{self._target}m_return'].astype('float32')
            mrmr_df.append(df)

        mrmr_df = pd.concat(mrmr_df)

        self.selected_features, self.relevance, self.redundancy = mrmr_classif(
            X=mrmr_df[self.text_indicators],
            y=mrmr_df[f'stock_{self._target}m_return'],
            K=15,
            return_scores=True
        )

        self.df = self.df[self.selected_features]
        print(f'Selected features: {self.selected_features}...')

        self.file_name = f'{self.file_name}_{self._target}m'
        self.data_source_path = self.processed_path / f'{self.file_name}.joblib'
    
    @record_history
    def _finalized_text(
        self,
        ignore_history: bool = False
    ):
        self.df.index = pd.to_datetime(self.df.index)
        self.df.index.name = 'local_time'
        self._add_elapsed_time()
        
    # Processing pipeline for all data.
    def create_df(
        self,
        file_name: str,
        raw_folder_name: str | None = None,
        medium: str | None = None,
        text_col: str | None = None,
        date_col: str | None = None,
        raw_path: str = 'data/raw',
        processed_path: str = 'data/processed',
        embedding_dimension: int | None = 1024,
        stocks: list | None = None,
        target: int | None = None,
        ignore_history: bool = False
    ):
        """ Pipeline for preprocessing datasets. """

        self.processed_path = Path(processed_path)
        data_source_path = self.processed_path / f'{file_name}.joblib'

        if data_source_path.exists():
            saved_data_source = joblib.load(data_source_path)
            self.__dict__.update(saved_data_source.__dict__)
        
        self.file_name = file_name
        self.data_source_path = data_source_path
        self.raw_path = Path(raw_path) / raw_folder_name if raw_folder_name else None
        self._medium = medium
        self._stocks = stocks
        self._target = target

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
        
        elif self._medium == 'bonds':
            self._process_bonds(ignore_history=ignore_history)
        
        elif self._medium == 'combined':
            self._combine_data(ignore_history=ignore_history)

        elif self._medium == 'features':
            self._create_feature_selection_data(ignore_history=ignore_history)
            self._feature_select(ignore_history=ignore_history)
        
        elif self._medium == 'final':
            self._finalized_stock(ignore_history=ignore_history)
        
        elif self._medium == 'final_text':
            self._finalized_text(ignore_history=ignore_history)

        elif self._medium == 'news_sentiment' or self._medium == 'social_indicators':
            self._text_indicators(ignore_history=ignore_history)

        if self._history != init_history:
            joblib.dump(self, self.data_source_path)