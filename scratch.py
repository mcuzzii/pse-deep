import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd

def _get_multilingual_sentiment(
    df,
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

    raw_results = multilingual_sentiment(df['text'].tolist())

    parsed_results = []
    for row in raw_results:
        parsed_results.append({item['label']: item['score'] for item in row})
    
    sentiment_df = pd.DataFrame(parsed_results)
    
    sentiment_df['sentiment'] = sentiment_df[list(sentiment_df.columns)].idxmax(axis=1)
    
    df = pd.concat([df, sentiment_df.set_index(df.index)], axis=1)
    
    return df

df = pd.DataFrame({'text': ["This is extremely frustrating", "It's not good", "The laptop is on the table", "This look alright", "That's extremely good"]})
print(_get_multilingual_sentiment(df))