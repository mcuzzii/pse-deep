from tokenizers import Tokenizer
import requests

response = requests.get(os.getenv("EMBED_V4_TOKENIZER_URL"))
tokenizer = Tokenizer.from_str(response.text)

result = tokenizer.encode(sequence="nagpapababa ng kalidad ng edukasyon ang mga influencer", add_special_tokens=False)
print(result.tokens)