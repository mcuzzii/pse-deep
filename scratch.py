import pandas as pd

lseg_news_translated = pd.read_csv('data/processed/lseg_news_translated.csv')
lseg_news_translated.loc[lseg_news_translated['detected_lang'] != 'en', ['text', 'cleaned_headline']].to_json('data/processed/lseg_news_untranslated.json', orient='records', indent=4)