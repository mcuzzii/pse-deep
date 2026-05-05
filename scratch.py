import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json
import langid

df = pd.read_csv('data/processed/lseg_news_translated.csv')

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

with open('data/raw/languages.json', 'r') as f:
    LANG_MAP = json.load(f)

# Create a copy to store results
df['cleaned_headline'] = df['text']

# Step 1: Fast Language Detection (Row by Row is okay here, langid is fast)
print("Detecting languages...")
def get_en_relative_position(text):
    # rank() returns a list of tuples: [('en', -245.2), ('fr', -500.1), ...]
    ranks = langid.rank(str(text))
    
    scores = [score for lang, score in ranks]
    highest = max(scores)
    lowest = min(scores)
    
    # Find the specific score for 'en'
    en_score = next(score for lang, score in ranks if lang == 'en')
    
    # Prevent DivisionByZero if all scores are identical (rare)
    if highest == lowest:
        return 0.0
    
    # Your formula: (highest - en) / (highest - lowest)
    # 0.0 = English is the top result
    # 1.0 = English is the bottom result
    return (highest - en_score) / (highest - lowest)


def get_softmax(text):
    # Get all scores
    ranks = langid.rank(str(text))
    langs = [r[0] for r in ranks]
    scores = np.array([r[1] for r in ranks])
    
    # Standard Softmax: exp(x) / sum(exp(x))
    # We subtract np.max(scores) for numerical stability (prevents overflow)
    shift_scores = scores - np.max(scores)
    exp_scores = np.exp(shift_scores)
    probabilities = exp_scores / exp_scores.sum()
    
    # Map back to languages
    return dict(zip(langs, probabilities))

# Apply to your dataframe
df['softmax'] = df['text'].apply(get_softmax)

df.loc[df['detected_lang'] != 'en', ['text', 'softmax']].to_json('data/processed/lseg_news_untranslated.json', orient='records', indent=4)
