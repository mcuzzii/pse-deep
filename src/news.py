import time
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from newspaper import Article

def get_article_url_from_headline(headline, source_hint=""):
    """
    Uses DuckDuckGo HTML search to find the top URL for a headline.
    """
    # Combine headline and source abbreviation to make search highly accurate
    query = f"{headline} {source_hint}".strip()
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        # DuckDuckGo HTML version stores search result links in 'a.result__url'
        links = soup.find_all("a", class_="result__url")
        
        if links:
            # Return the first search result link
            actual_url = links[0]['href'].strip()
            # Clean DuckDuckGo redirect wrappers if present
            if "uddg=" in actual_url:
                actual_url = requests.utils.unquote(actual_url.split("uddg=")[1].split("&")[0])
            return actual_url
    except Exception as e:
        print(f"Search failed for '{headline}': {e}")
    return None

def extract_true_publication_date(url):
    """
    Downloads the webpage and extracts the underlying publication datetime metadata.
    """
    try:
        # Newspaper3k handles complex tasks like finding <meta property="article:published_time">
        article = Article(url)
        article.download()
        article.parse()
        
        if article.publish_date:
            return article.publish_date # Returns a datetime object (often with timezone)
    except Exception as e:
        print(f"Metadata extraction failed for {url}: {e}")
    return None

def rescue_timestamp(headline, source_hint=""):
    print(f"🔍 Searching for: '{headline}'...")
    url = get_article_url_from_headline(headline, source_hint)
    
    if not url:
        print("❌ Could not find a valid URL via search.")
        return None
    
    print(f"🔗 Found URL: {url}")
    print("⏳ Extracting true publication time...")
    true_time = extract_true_publication_date(url)
    
    if true_time:
        print(f"✅ Success! True Pub Date: {true_time}")
        return true_time
    else:
        print("❌ Could not find publication timestamp metadata inside the HTML.")
        return None

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Simulate a row in your LSEG dataset
    sample_headline = "Apple Launches M4 Mac Mini with Smaller Design"
    sample_source = "Reuters" 
    
    true_timestamp = rescue_timestamp(sample_headline, sample_source)
    
    # Anti-banning safety break if you loop through a dataframe
    time.sleep(1)