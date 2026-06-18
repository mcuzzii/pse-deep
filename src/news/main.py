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

custom_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

def extract_news_url(url):

    # Step 1: Parse the URL to get the 'u' query parameter
    parsed_url = urlparse(url)
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
        match = re.search(r'storyId=(.*)&type=([A-Z][a-z]*)', decoded_url)
        story_id = match.group(1)
        type = match.group(2)
        if type == "WebUrl":
            try:
                response = news.story.Definition(story_id).get_data()

                if response is not None and response.data is not None:
                    return response.data.raw.get('webURL', None)
                else:
                    print("Error: Received an empty data response. Your desktop session might be unresponsive.")
                    return None

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            finally:
                return None
        else:
            print('Type is not a URL')
            return None
    else:
        print("No encoded parameter found in the URL.")
        return None

def get_fallback_timezone(url, response_headers):
    """
    Determines the most likely timezone based on URL markers or server headers.
    """
    url_lower = url.lower()
    if ".com.ph" in url_lower or "/philippines/" in url_lower:
        return pytz.timezone("Asia/Manila")
    elif ".co.uk" in url_lower or "/uk/" in url_lower:
        return pytz.timezone("Europe/London")
    elif ".ca" in url_lower or "ca.news" in url_lower:
        return pytz.timezone("America/Toronto")
    
    # Fallback to UTC if no regional indicators are found
    return pytz.utc

def ensure_utc(parsed_dt, url, response_headers):
    """
    Takes a parsed datetime object and ensures it is safely mapped to UTC.
    If it lacks a timezone (naive), it applies a contextual fallback first.
    """
    if parsed_dt is None:
        return None
        
    # If the string already had a timezone (e.g. "GMT+8"), convert it straight to UTC
    if parsed_dt.tzinfo is not None:
        return parsed_dt.astimezone(pytz.utc)
        
    # If it's a naive datetime (Layers 3 & 4 text), localize it using context
    fallback_tz = get_fallback_timezone(url, response_headers)
    localized_dt = fallback_tz.localize(parsed_dt)
    return localized_dt.astimezone(pytz.utc)

def extract_exact_timestamp(url):
    custom_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."
    }
    try:
        res = requests.get(url, headers=custom_headers, timeout=10)
        res.raise_for_status()
        html_content = res.text
    except Exception as e:
        print(f"Network error: {e}")
        return None

    soup = BeautifulSoup(html_content, "html.parser")
    
    # -------------------------------------------------------------
    # LAYER 1: JSON-LD (Usually contains explicit offset data)
    # -------------------------------------------------------------
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            if "@graph" in data:
                items.extend(data["@graph"])
            for item in items:
                for key in ["datePublished", "dateModified"]:
                    if key in item and isinstance(item[key], str):
                        parsed = dateparser.parse(item[key])
                        if parsed:
                            # Run through UTC gate
                            return ensure_utc(parsed, url, res.headers)
        except Exception:
            continue

    # -------------------------------------------------------------
    # LAYER 2: Metadata Tags (Usually contains explicit offset data)
    # -------------------------------------------------------------
    meta_selectors = [{"property": "article:published_time"}, {"name": "parsely-pub-date"}, {"itemprop": "datePublished"}]
    for selector in meta_selectors:
        meta_tag = soup.find("meta", attrs=selector)
        if meta_tag and meta_tag.get("content"):
            parsed = dateparser.parse(meta_tag["content"])
            if parsed:
                return ensure_utc(parsed, url, res.headers)

    # -------------------------------------------------------------
    # LAYER 3: Semantic <time> Tags (Frequently missing timezone)
    # -------------------------------------------------------------
    time_tag = soup.find("time")
    if time_tag:
        time_str = time_tag.get("datetime") or time_tag.get_text()
        parsed = dateparser.parse(time_str)
        if parsed:
            return ensure_utc(parsed, url, res.headers)

    # -------------------------------------------------------------
    # LAYER 4: Regex Catch-All (Almost always missing timezone)
    # -------------------------------------------------------------
    page_text = soup.get_text(separator=" ")
    fuzzy_date_pattern = r"(\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})[^\w\n]{1,5}(\d{1,2}:\d{2}(?:\s*[ap]m)?)"
    
    matches = re.findall(fuzzy_date_pattern, page_text, re.IGNORECASE)
    for match in matches:
        combined_text = f"{match[0]} {match[1]}"
        parsed = dateparser.parse(combined_text)
        if parsed:
            # Matches found deep inside random layout text are seamlessly 
            # safe-guarded against timezone drift here:
            return ensure_utc(parsed, url, res.headers)

    return None

if __name__ == '__main__':
    try:
        rd.open_session()
        print("Loading news...")

        raw_path = Path('data/raw/news')
        processed_path = Path('data/processed/news')

        raw_path.mkdir(parents=True, exist_ok=True)
        processed_path.mkdir(parents=True, exist_ok=True)

        news = pd.read_csv(raw_path / 'news.csv')

        tqdm.pandas()
        news['news_urls'] = news['url'].progress_apply(extract_news_url)

        news.to_csv(processed_path / 'news.csv')

        print("Success!")
    except BaseException as e:
        
    finally:
        rd.close_session()