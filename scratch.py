import pandas as pd
from transformers import pipeline

# 1. Initialize the FinBERT sentiment pipeline
# 'ProsusAI/finbert' is the most popular pre-trained version
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# 2. Sample Dataframe
# Assuming your headlines are in a column named 'headline'
df = pd.DataFrame({
    'headline': [
        "Stocks rally as inflation data shows cooling prices.",
        "Company XYZ reports massive losses in Q3 earnings call.",
        "Market remains steady despite global uncertainty."
    ]
})

# 3. Apply the model to your headlines
# We use .tolist() because pipelines are faster when processing lists
results = sentiment_analyzer(df['headline'].tolist())

# 4. Extract results back into the dataframe
df['sentiment'] = [res['label'] for res in results]
df['score'] = [res['score'] for res in results]

print(df)