import refinitiv.data as rd
from refinitiv.data.content import news
import base64
from urllib.parse import urlparse, parse_qs, unquote
import re
from htmldate import find_date
import requests
import json
import dateparser
import pytz
from bs4 import BeautifulSoup
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import traceback
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dotenv import load_dotenv
import os
import asyncio
import aiohttp
from aiolimiter import AsyncLimiter
from rapidfuzz import fuzz
import random
import numpy as np

load_dotenv()

custom_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

def extract_news_url(url):

    # Step 1: Parse the URL to get the 'u' query parameter
    try:
        parsed_url = urlparse(url)
    except Exception as _:
        return None
    
    query_params = parse_qs(parsed_url.query)
    encoded_payload = query_params.get('u', [None])[0]

    if encoded_payload:
        # Step 2: Fix padding for Base64 standard if required
        # Base64 strings must be multiples of 4 bytes. We pad with '='
        missing_padding = len(encoded_payload) % 4
        if missing_padding:
            encoded_payload += '=' * (4 - missing_padding)
            
        # Step 3: Decode the URL-safe Base64 payload
        decoded_bytes = base64.urlsafe_b64decode(encoded_payload)
        decoded_url = decoded_bytes.decode('utf-8')
        
        print("--- Decoded Real Destination ---")
        print(decoded_url)
        match = re.search(r'storyId=(.*)&type=([A-Za-z]*)', decoded_url)
        story_id = match.group(1)
        type = match.group(2)
        if type == "WebUrl":
            try:
                response = news.story.Definition(story_id).get_data()

                if response is not None and response.data is not None:
                    url = response.data.raw.get('webURL', None)

                    rate_limit = response.data._owner._http_headers['ratelimit-remaining']
                    volume_limit = response.data._owner._http_headers['volumelimit-remaining']
                    queue_limit = response.data._owner._http_headers['queuelimit-remaining']

                    print(f'Rate limit: {rate_limit}; Volume limit: {volume_limit}; Queue limit: {queue_limit}')
                    return url
                else:
                    print("Error: Received an empty data response. Your desktop session might be unresponsive.")
                    return None

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise
        else:
            print(f'Type is not a URL; type = {type}')
            return None
    else:
        print("No encoded parameter found in the URL.")
        return None

with open('data/raw/news/domains.json', 'r', encoding='utf-8') as f:
    DOMAIN_TIMEZONE_MAP = json.load(f)

AMBIGUOUS_TZ_ABBREVS = {
    'PST',  # Philippine ST (UTC+8) vs Pacific ST (UTC-8)
    'IST',  # Indian ST (UTC+5:30) vs Irish ST (UTC+1) vs Israel ST (UTC+2)
    'CST',  # China ST (UTC+8) vs Central ST (UTC-6) vs Cuba ST (UTC-5)
    'BST',  # Bangladesh ST (UTC+6) vs British Summer Time (UTC+1)
    'SST',  # Singapore ST (UTC+8) vs Samoa ST (UTC-11)
    'MST',  # Malaysia ST (UTC+8) vs Mountain ST (UTC-7)
    'AST',  # Arabia ST (UTC+3) vs Atlantic ST (UTC-4)
    'NST',  # Nepal ST (UTC+5:45) vs Newfoundland ST (UTC-3:30)
    'GST',  # Gulf ST (UTC+4) vs South Georgia ST (UTC-2)
}

PRESERVE = {'AM', 'PM', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'UTC', 'GMT'}

def preprocess_date_string(date_str):
    stripped = False
    def replacer(m):
        nonlocal stripped
        token = m.group()
        if token in PRESERVE:
            return token
        if token in AMBIGUOUS_TZ_ABBREVS:
            stripped = True
            return ''
        return token
    result = re.sub(r'\b[A-Z]{2,5}\b', replacer, date_str).strip()
    return result, stripped

def get_fallback_timezone(url):
    domain = urlparse(url).netloc.lower().replace("www.", "")

    if domain in DOMAIN_TIMEZONE_MAP:
        return pytz.timezone(DOMAIN_TIMEZONE_MAP[domain])

    if ".com.ph" in domain or ".gov.ph" in domain:
        return pytz.timezone("Asia/Manila")
    elif ".co.uk" in domain or ".org.uk" in domain:
        return pytz.timezone("Europe/London")
    elif ".com.au" in domain or ".net.au" in domain:
        return pytz.timezone("Australia/Sydney")
    elif ".co.za" in domain:
        return pytz.timezone("Africa/Johannesburg")
    elif ".com.sg" in domain:
        return pytz.timezone("Asia/Singapore")
    elif ".com.my" in domain:
        return pytz.timezone("Asia/Kuala_Lumpur")
    elif ".co.nz" in domain:
        return pytz.timezone("Pacific/Auckland")
    elif ".ng" in domain:
        return pytz.timezone("Africa/Lagos")
    elif ".ca" in domain:
        return pytz.timezone("America/Toronto")
    elif ".de" in domain:
        return pytz.timezone("Europe/Berlin")

    return pytz.utc


def ensure_utc(parsed_dt, url):
    if parsed_dt is None:
        return None

    if parsed_dt.tzinfo is not None:
        return parsed_dt.astimezone(pytz.utc)

    fallback_tz = get_fallback_timezone(url)

    try:
        localized = fallback_tz.localize(parsed_dt)
    except Exception:
        localized = parsed_dt.replace(tzinfo=fallback_tz)

    return localized.astimezone(pytz.utc)

DATE_FIELDS_PRIORITY = ["datePublished", "dateCreated", "uploadDate", "date", "dateModified"]

def find_date_fields(obj):
    """
    Recursively search JSON-LD for date strings, yielding in priority order.
    datePublished is preferred over dateModified over generic date.
    """
    found = {}  # key -> first value found for each date field

    def _recurse(o):
        if isinstance(o, dict):
            for key, value in o.items():
                if key in DATE_FIELDS_PRIORITY and isinstance(value, str) and key not in found:
                    found[key] = value
                _recurse(value)
        elif isinstance(o, list):
            for item in o:
                _recurse(item)

    _recurse(obj)

    for key in DATE_FIELDS_PRIORITY:
        if key in found:
            yield found[key]

def extract_exact_timestamp(url):

    custom_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    try:
        res = requests.get(
            url,
            headers=custom_headers,
            timeout=10
        )

        res.raise_for_status()

    except Exception as e:
        print(f"Network error: {e}")
        return None, None

    soup = BeautifulSoup(res.text, "html.parser")

    # ============================================================
    # LAYER 1: JSON-LD
    # ============================================================

    for script in soup.find_all(
        "script",
        type="application/ld+json"
    ):
        try:
            raw = script.string

            if not raw:
                continue

            data = json.loads(raw)

            for candidate in find_date_fields(data):

                candidate, was_stripped = preprocess_date_string(candidate)
                parsed = dateparser.parse(candidate, settings={
                    'RETURN_AS_TIMEZONE_AWARE': not was_stripped
                })

                if parsed:
                    timestamp = ensure_utc(parsed, url)
                    print(f'Found ld+json timestamp: {timestamp}')
                    return timestamp, 'application/ld+json'

        except Exception:
            continue

    # ============================================================
    # LAYER 2: Metadata tags
    # ============================================================

    meta_selectors = [
        {"property": "article:published_time"},
        {"itemprop": "datePublished"},
        {"property": "og:published_time"},
        {"name": "parsely-pub-date"},
        {"name": "publish-date"},
        {"name": "publication_date"},
        {"name": "pubdate"},
        {"name": "date"},
        {"property": "article:modified_time"},
        {"itemprop": "dateModified"},
    ]

    for selector in meta_selectors:

        meta_tag = soup.find("meta", attrs=selector)

        if not meta_tag:
            continue

        content = meta_tag.get("content")

        if not content:
            continue

        content, was_stripped = preprocess_date_string(content)
        parsed = dateparser.parse(content, settings={
            'RETURN_AS_TIMEZONE_AWARE': not was_stripped
        })

        if parsed:
            timestamp = ensure_utc(parsed, url)
            print(f'Found metadata timestamp: {timestamp}')
            return timestamp, 'metadata'

    # ============================================================
    # LAYER 3: Semantic time tag
    # ============================================================

    for time_tag in soup.find_all("time"):
        time_str = time_tag.get("datetime") or time_tag.get_text()
        time_str, was_stripped = preprocess_date_string(time_str)
        parsed = dateparser.parse(time_str, settings={
            'RETURN_AS_TIMEZONE_AWARE': not was_stripped
        })
        if parsed:
            timestamp = ensure_utc(parsed, url)
            print(f'Found time tag: {timestamp}')
            return timestamp, 'time_tag'
    
    # -------------------------------------------------------------
    # LAYER 4: Regex Catch-All (Almost always missing timezone)
    # -------------------------------------------------------------
    page_text = soup.get_text(separator=" ")
    fuzzy_date_pattern = r"(\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})[^\w\n]{1,5}(\d{1,2}:\d{2}(?:\s*[ap]m)?)"
    
    matches = re.findall(fuzzy_date_pattern, page_text, re.IGNORECASE)
    for match in matches:
        combined_text = f"{match[0]} {match[1]}"
        combined_text, was_stripped = preprocess_date_string(combined_text)
        parsed = dateparser.parse(combined_text, settings={
            'RETURN_AS_TIMEZONE_AWARE': not was_stripped
        })
        if parsed:
            timestamp = ensure_utc(parsed, url)
            print(f'Found in article text: {timestamp}')
            return timestamp, 'article_text'

    return None, None

def load_news(file_name='news', filter=True, get_raw=False):
    print("Loading news...")

    raw_path = Path('data/raw/news')
    processed_path = Path('data/processed/news')
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    if (processed_path / f'{file_name}.csv').exists() and not get_raw:
        news_df = pd.read_csv(processed_path / f'{file_name}.csv', index_col=0)
    else:
        news_df = pd.read_csv(raw_path / 'news.csv', index_col=0)
    
    news_df['text'] = news_df['text'].str.strip()
    news_df['text'] = news_df['text'].str.replace(r'\s+', ' ', regex=True)
    news_df = news_df.loc[news_df['text'] != ""]
    news_df.drop_duplicates(subset=['text'], keep='first', inplace=True)

    if filter:
        news_df = news_df.loc[
            (news_df['source'] != "RTRS") &
            (news_df['source'] != "GLORDP") &
            (news_df['source'] != "GLFILE")
        ]
    
    return news_df, processed_path

def get_lseg_news_urls(news_df, processed_path):
    finished = False
    while not finished:
        news_urls = []
        indices = news_df.index.tolist()

        if 'news_urls' not in news_df.columns.tolist() or not news_df['news_urls'].notna().sum():
            news_df['news_urls'] = None
            to_extract = indices
        else:
            to_extract = indices[indices.index(news_df['news_urls'].last_valid_index()) + 1:]
        
        try:
            rd.open_session()
            finished = True
            for url in tqdm(news_df.loc[to_extract, 'url']):
                news_url = extract_news_url(url)
                news_urls.append(news_url)

            print("Success!")

        except KeyboardInterrupt:
            print("\nStopped by user.")

        except Exception as e:
            traceback.print_exc()
            print(f'Error: {e}')
            finished = False
        
        finally:
            new_indices = to_extract[:len(news_urls)]
            news_df.loc[new_indices, 'news_urls'] = pd.Series(news_urls, index=new_indices)

            news_df.to_csv(processed_path / 'news_lseg.csv')
            try:
                rd.close_session()
            except Exception:
                pass
    
    return news_df

def get_news_distribution(news_df, processed_path):
    urls = news_df.loc[news_df['news_urls'].notna(), 'news_urls']
    website_dist = urls.str.extractall(r'https?://(?:www\.)*(.*?)/').value_counts().to_dict()
    website_dist = {k[0]: v for k, v in website_dist.items()}

    with open(processed_path / 'news_dist.json', 'w', encoding='utf-8') as f:
        json.dump(website_dist, f, indent=4)

    return website_dist

# per-domain rate limiting
domain_locks = defaultdict(threading.Lock)
domain_last_called = defaultdict(float)
RATE_LIMIT_SECONDS = 1.0  # min seconds between requests to the same domain

def rate_limited_extract(idx, url):
    domain = urlparse(url).netloc
    lock = domain_locks[domain]
    
    with lock:  # hold the lock for the entire request
        elapsed = time.time() - domain_last_called[domain]
        wait = RATE_LIMIT_SECONDS - elapsed
        if wait > 0:
            time.sleep(wait)
        domain_last_called[domain] = time.time()
        return idx, extract_exact_timestamp(url)  # inside the lock


def get_news_timestamps(news_df, processed_path):
    indices = news_df.index[news_df['news_urls'].notna()].tolist()
    
    if 'published_at' not in news_df.columns.tolist() or not news_df['published_at'].notna().sum():
        news_df['published_at'] = None
        news_df['timestamp_from'] = None
        to_extract = indices
    else:
        if 'timestamp_from' not in news_df.columns.tolist():
            news_df['timestamp_from'] = None
        to_extract = indices[indices.index(news_df['published_at'].last_valid_index()) + 1:]

    results = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(rate_limited_extract, idx, url): idx
            for idx, url in news_df.loc[to_extract, 'news_urls'].items()
        }
        try:
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    idx, (dt, kind) = future.result()
                    results[idx] = (dt, kind)

                except KeyboardInterrupt:
                    raise

                except Exception as e:
                    traceback.print_exc()
        
        except KeyboardInterrupt:
            print('\nStopped by user.')
            executor.shutdown(wait=False, cancel_futures=True)
        
        finally:
            filled_indices = [i for i in to_extract if i in results]
            news_df.loc[filled_indices, 'published_at'] = pd.Series({k: v[0] for k, v in results.items()})
            news_df.loc[filled_indices, 'timestamp_from'] = pd.Series({k: v[1] for k, v in results.items()})
            news_df.to_csv(processed_path / 'news_urls.csv')
        
    return news_df

def filter_news(news_df, processed_path):
    news_df = news_df.loc[news_df['published_at'].notna()]

    news_df['published_at'] = pd.to_datetime(
        news_df['published_at'], format='mixed', utc=True
    ).dt.tz_convert('Asia/Manila').dt.tz_localize(None)

    news_df['date_time'] = pd.to_datetime(news_df['date_time'])

    dates_df = news_df[['date_time', 'published_at']]

    news_df['date_time'] = news_df['published_at']
    news_df = news_df.iloc[:, :4]

    dates_df.to_csv(processed_path / 'dates.csv')
    news_df.to_csv(processed_path / 'news_cleaned.csv')

    return news_df

SIMILARITY_THRESHOLD = 0
MAX_RETRIES = 5
CONCURRENCY = 20
REQUESTS_PER_SECOND = 40
SAVE_EVERY = 100

rate_limiter = AsyncLimiter(REQUESTS_PER_SECOND, 1)
save_lock = asyncio.Lock()
processed = 0

async def query_serper(session, headline, api_key, exact=True, news_endpoint=True):

    payload = {
        "q": f'"{headline}"' if exact else headline
    }

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    try:

        async with session.post(
            f"https://google.serper.dev/{'news' if news_endpoint else 'search'}",
            json=payload,
            headers=headers
        ) as response:
            
            if response.status == 400:
                print(await response.text())
                return "error", None

            if response.status == 401:
                return "error", None

            if response.status == 403:
                return "error", None

            if response.status == 429:
                return "retry", None

            if response.status >= 500:
                return "retry", None

            if response.status != 200:
                print(response.status, headline)
                print(await response.text())
                return "no_match", None

            data = await response.json()

    except (
        aiohttp.ClientError,
        asyncio.TimeoutError
    ):
        return "retry", None

    except Exception:
        traceback.print_exc()
        return "retry", None
    
    news = data.get("news")

    if not news:
        news = data.get("organic")

    if not news:
        return "no_match", None

    best = None
    best_score = -1

    for article in news:

        score = fuzz.partial_ratio(
            headline.lower(),
            article["title"].lower()
        )

        if score > best_score:
            best = article
            best_score = score

    if best is None or best_score < SIMILARITY_THRESHOLD:
        return "no_match", None

    return "success", {
        "matched_title": best["title"],
        "similarity": best_score,
        "source": best.get("source"),
        "url": best.get("link")
    }

def clean_headline(headline):
    headline = re.sub(r"\(\d+\)$", "", headline)
    headline = re.sub(r"\bBRIEF\b", "", headline, flags=re.I)
    return " ".join(headline.split())

async def worker(
    idx,
    df,
    session,
    semaphore,
    rate_limiter,
    api_key,
    processed_path
):

    global processed

    headline = df.at[idx, "text"]

    if not isinstance(headline, str):
        return

    headline = headline.strip()
    headline = clean_headline(headline)

    if not headline:
        return

    async with semaphore:

        for attempt in range(MAX_RETRIES):

            async with rate_limiter:

                status, result = await query_serper(
                    session,
                    headline,
                    api_key,
                    exact=True,
                    news_endpoint=True
                )

            if status == "success":

                df.at[idx, "matched_title"] = result["matched_title"]
                df.at[idx, "similarity"] = result["similarity"]
                df.at[idx, "source"] = result["source"]
                df.at[idx, "news_urls"] = result["url"]

                print(f"{idx}: ✓")

                break

            elif status == "no_match":

                print(f"{idx}: no match")

                break

            elif status == "retry":

                wait = 2 ** attempt

                print(
                    f"{idx}: retry "
                    f"{attempt + 1}/{MAX_RETRIES} "
                    f"in {wait}s"
                )

                await asyncio.sleep(wait)
            
            elif status == "error":
                raise RuntimeError(f"Error encountered for idx {idx}")

        async with save_lock:

            processed += 1

            if processed % SAVE_EVERY == 0:

                print(f"Saving checkpoint ({processed})")

                df.to_csv(
                    processed_path / "news.csv"
                )

async def get_news_urls_async(
    df,
    processed_path,
    headline_column="text",
    concurrency=CONCURRENCY
):

    global processed

    api_key = os.getenv("SERPER_API_TOKEN")

    if api_key is None:
        raise RuntimeError(
            "SERPER_API_TOKEN not found."
        )

    for col in (
        "news_urls",
        "matched_title",
        "similarity",
        "source"
    ):

        if col not in df.columns:
            df[col] = None

    # Resume logic
    to_extract = df.index[
        df["news_urls"].isna()
    ].tolist()

    print(
        f"{len(to_extract)} headlines remaining."
    )

    rng = random.Random(42)
    
    rng.shuffle(to_extract)

    semaphore = asyncio.Semaphore(concurrency)

    rate_limiter = AsyncLimiter(
        REQUESTS_PER_SECOND,
        1
    )

    connector = aiohttp.TCPConnector(
        limit=concurrency
    )

    timeout = aiohttp.ClientTimeout(
        total=30
    )

    try:

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as session:

            tasks = [

                worker(
                    idx,
                    df,
                    session,
                    semaphore,
                    rate_limiter,
                    api_key,
                    processed_path
                )

                for idx in to_extract

            ]

            results = await asyncio.gather(
                *tasks,
                return_exceptions=True
            )

            for r in results:

                if isinstance(r, Exception):

                    traceback.print_exception(r)

    finally:

        print("Saving final checkpoint...")

        df.to_csv(
            processed_path / "news.csv"
        )

    print("Done.")

    return df

def combine_dfs(serper_df, lseg_df):
    serper_df.loc[lseg_df['news_urls'].notna(), 'news_urls'] = lseg_df.loc[lseg_df['news_urls'].notna(), 'news_urls']
    serper_df = serper_df.loc[
        (90 <= serper_df['similarity']) |
        lseg_df['news_urls'].notna(),
        [col for col in serper_df.columns if col not in ('matched_title', 'similarity')]
    ]

    serper_df.to_csv('data/processed/news/news_urls.csv')

def append_dfs(cleaned_df, raw_df):
    raw_df = raw_df.loc[raw_df['source'] == 'RTRS'].iloc[:, :4]
    raw_df['source'] = 'Reuters'
    combined_df = pd.concat([cleaned_df, raw_df])

    combined_df["date_time"] = pd.to_datetime(combined_df["date_time"], format='mixed')
    combined_df = combined_df.reset_index()
    combined_df = combined_df.sort_values("index")
    combined_df["date_ordinal"] = combined_df["date_time"].map(pd.Timestamp.toordinal)

    x = np.arange(len(combined_df))
    y = combined_df["date_ordinal"].values

    # Fit line: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)
    combined_df["fitted"] = slope * x + intercept
    combined_df["residual"] = combined_df["date_ordinal"] - combined_df["fitted"]

    # Remove high residuals using IQR
    Q1 = combined_df["residual"].quantile(0.25)
    Q3 = combined_df["residual"].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    is_outlier = (combined_df["residual"] < lower) | (combined_df["residual"] > upper)
    clean_df = combined_df[~is_outlier].drop(columns=["date_ordinal", "fitted", "residual"])
    clean_df = clean_df.set_index('index')
    outliers_df = combined_df[is_outlier]

    print(f"Removed {len(outliers_df)} rows")
    print(outliers_df[["index", "date_time", "residual", "text"]])
    
    clean_df.to_csv('data/processed/news/news_final.csv')

if __name__ == '__main__':
    
    news_df, processed_path = load_news()
    news_df = asyncio.run(
        get_news_urls_async(
            news_df,
            processed_path,
            concurrency=20
        )
    )
    news_df_lseg, _ = load_news('news_lseg')
    news_df_lseg = get_lseg_news_urls(news_df_lseg, processed_path)
    combine_dfs(news_df, news_df_lseg)
    
    news_urls_df, processed_path = load_news('news_urls')
    website_dist = get_news_distribution(news_urls_df, processed_path)
    news_urls_df = get_news_timestamps(news_urls_df, processed_path)
    news_urls_df = filter_news(news_urls_df, processed_path)

    cleaned_df, _ = load_news('news_cleaned')
    raw_df, _ = load_news(filter=False, get_raw=True)
    append_dfs(cleaned_df, raw_df)