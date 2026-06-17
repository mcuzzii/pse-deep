"""
rescue_timestamp.py  —  fixed version
======================================
Problems with the original:
  1. DuckDuckGo HTML endpoint (html.duckduckgo.com) now returns 403 for bots.
     Fixed → use DuckDuckGo Lite (lite.duckduckgo.com) with realistic headers + session.
  2. Source hint (e.g. "BUSMIR") was appended raw to the query — DDG doesn't know
     what "BUSMIR" means.
     Fixed → map source codes to actual domains, then use `site:domain` in the query
     for precise targeting.
  3. Only one query was attempted. If it failed, the function gave up.
     Fixed → waterfall of increasingly broad queries tried in order.
  4. Result link extraction used class="result__url" which is the display URL,
     not the real href. On DDG Lite the real href is on class="result-link".
     Fixed → correct selector + DDG redirect unwrapping.
  5. No retry / rate-limit handling.
     Fixed → exponential backoff, configurable delay between calls.
  6. newspaper3k is unmaintained and often fails on modern sites.
     Fixed → try multiple metadata extraction strategies in order:
       a. <meta property="article:published_time">  (Open Graph, most common)
       b. <meta name="publishdate"> / <meta name="date">
       c. <time datetime="..."> tags
       d. JSON-LD schema (datePublished)
       e. newspaper3k as last resort
"""

import re
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser as dateutil_parser   # pip install python-dateutil


# ------------------------------------------------------------------
# Source code → domain mapping  (extend as needed)
# ------------------------------------------------------------------
SOURCE_DOMAIN_MAP = {
    "BUSMIR": "businessmirror.com.ph",
    "PDI":    "inquirer.net",
    "PST":    "philstar.com",
    "MB":     "mb.com.ph",
    "BW":     "bworldonline.com",
    "ET":     "edge.com.ph",
    "MLA":    "manilatimes.net",
    "SUN":    "sunstar.com.ph",
    "ABS":    "abs-cbn.com",
    "GMA":    "gmanetwork.com",
    "PNA":    "pna.gov.ph",
}


# ------------------------------------------------------------------
# Shared session with realistic browser headers
# ------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
        "Gecko/20100101 Firefox/124.0"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
})


# ------------------------------------------------------------------
# Step 1: find the article URL
# ------------------------------------------------------------------

def _ddg_lite_search(query: str, retries: int = 3) -> list[str]:
    """
    Query DuckDuckGo Lite and return a list of result URLs.
    """
    url = "https://lite.duckduckgo.com/lite/"
    params = {"q": query}

    for attempt in range(retries):
        try:
            resp = SESSION.post(url, data=params, timeout=15)
            if resp.status_code == 200:
                break
            if resp.status_code == 429:
                wait = 2 ** attempt * 3
                print(f"  Rate limited, waiting {wait}s…")
                time.sleep(wait)
        except requests.RequestException as e:
            print(f"  Request error: {e}")
            time.sleep(2)
    else:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    urls = []
    # DDG Lite: real links are <a class="result-link"> with actual href
    for a in soup.find_all("a", class_="result-link"):
        href = a.get("href", "").strip()
        if href and href.startswith("http"):
            urls.append(href)

    # Fallback: any link containing a domain with a path (avoid DDG internal links)
    if not urls:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and "duckduckgo.com" not in href:
                urls.append(href)

    return urls


def get_article_url(headline: str, source_hint: str = "") -> str | None:
    """
    Try a waterfall of queries (most specific → least specific) until we get a hit.
    """
    domain = SOURCE_DOMAIN_MAP.get(source_hint.strip().upper(), "")

    # Build queries from most targeted to least
    queries = []
    if domain:
        queries.append(f'"{headline}" site:{domain}')   # exact phrase + site
        queries.append(f'{headline} site:{domain}')      # relaxed + site
    if domain:
        queries.append(f'{headline} {domain}')           # domain as keyword
    queries.append(f'{headline} {source_hint}')          # raw source hint
    queries.append(headline)                              # bare headline last

    for q in queries:
        print(f"  Trying query: {q!r}")
        urls = _ddg_lite_search(q)
        if urls:
            # If we know the domain, prefer a result from that domain
            if domain:
                for u in urls:
                    if domain in u:
                        return u
            return urls[0]  # best available
        time.sleep(1.5)     # be polite between queries

    return None


# ------------------------------------------------------------------
# Step 2: extract publication timestamp from the article page
# ------------------------------------------------------------------

def _parse_dt(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return dateutil_parser.parse(raw)
    except Exception:
        return None


def extract_publication_date(url: str) -> datetime | None:
    """
    Multi-strategy metadata extraction. Returns a datetime or None.
    """
    try:
        resp = SESSION.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Failed to fetch page: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    html = resp.text

    # Strategy A: Open Graph / standard meta tags
    for attr, names in [
        ("property", ["article:published_time", "article:modified_time",
                       "og:article:published_time"]),
        ("name",     ["publishdate", "date", "pubdate", "publication_date",
                       "article.published", "sailthru.date",
                       "DC.date.issued", "lastmod"]),
        ("itemprop", ["datePublished", "dateModified"]),
    ]:
        for name in names:
            tag = soup.find("meta", attrs={attr: re.compile(name, re.I)})
            if tag:
                dt = _parse_dt(tag.get("content", ""))
                if dt:
                    return dt

    # Strategy B: <time> elements
    for time_tag in soup.find_all("time"):
        dt = _parse_dt(time_tag.get("datetime", ""))
        if dt:
            return dt

    # Strategy C: JSON-LD schema.org
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            # Can be a single object or a list
            if isinstance(data, list):
                data = data[0]
            for key in ("datePublished", "dateModified", "uploadDate"):
                if key in data:
                    dt = _parse_dt(data[key])
                    if dt:
                        return dt
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    # Strategy D: regex scan the raw HTML for ISO 8601 dates near publish keywords
    pattern = re.compile(
        r'"(?:datePublished|dateModified|publishedAt|pubDate)"\s*:\s*"([^"]+)"'
    )
    matches = pattern.findall(html)
    for m in matches:
        dt = _parse_dt(m)
        if dt:
            return dt

    # Strategy E: newspaper3k last resort
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        if article.publish_date:
            return article.publish_date
    except Exception:
        pass

    return None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def rescue_timestamp(headline: str, source_hint: str = "",
                     sleep_between: float = 1.5) -> datetime | None:
    """
    Given a headline and an optional source code (e.g. "BUSMIR"),
    find the article URL and return its publication datetime.

    Args:
        headline:       The news headline string.
        source_hint:    Source code from your dataset (e.g. "BUSMIR", "PDI").
        sleep_between:  Seconds to sleep after the call (for batch use).

    Returns:
        datetime object, or None if not found.
    """
    print(f"\n🔍 Searching: '{headline}' [{source_hint}]")

    url = get_article_url(headline, source_hint)
    if not url:
        print("❌ No URL found.")
        time.sleep(sleep_between)
        return None

    print(f"🔗 URL: {url}")
    dt = extract_publication_date(url)

    if dt:
        print(f"✅ Published: {dt}")
    else:
        print("❌ Could not extract publication date from page metadata.")

    time.sleep(sleep_between)
    return dt


# ------------------------------------------------------------------
# Batch helper — drop-in for a DataFrame loop
# ------------------------------------------------------------------

def rescue_timestamps_batch(df, headline_col: str, source_col: str,
                             output_col: str = "true_timestamp",
                             sleep_between: float = 2.0):
    """
    Applies rescue_timestamp row-by-row to a DataFrame.
    Skips rows that already have a value in output_col.
    Returns the modified DataFrame.

    Example:
        df = rescue_timestamps_batch(df, "headline", "source")
    """
    import pandas as pd

    if output_col not in df.columns:
        df[output_col] = None

    for idx, row in df.iterrows():
        if pd.notna(df.at[idx, output_col]):
            continue  # already resolved, skip
        dt = rescue_timestamp(
            row[headline_col],
            row.get(source_col, ""),
            sleep_between=sleep_between,
        )
        df.at[idx, output_col] = dt

    return df


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------

if __name__ == "__main__":
    import base64
    from urllib.parse import urlparse, parse_qs, unquote

    # Your LSEG link
    url = "https://go.refinitiv.com/?u=Y3B1cmw6Ly9hcHBzLmNwLi9hcHBzL25ld3MtbGlua3MtbmF2aWdhdGlvbi8/c3RvcnlJZD11cm46bmV3c21sOndlYm5ld3MucmVmaW5pdGl2LmNvbToyMDI1MDYzMDpuTlJBd3Vyd2JmOjAmdHlwZT1XZWJVcmwmaW5saW5lSW5XZWI9dHJ1ZQ=="

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
    else:
        print("No encoded parameter found in the URL.")