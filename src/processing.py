import pandas as pd

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

