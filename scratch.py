import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load the model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. Prepare your financial text
text = "The company's revenue grew by 20%, but net profit declined due to rising costs."

# 3. Tokenize and get model output
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    # The raw scores are in 'logits'
    logits = outputs.logits

# 4. Apply Softmax to get probabilities
# dim=-1 ensures we calculate softmax across the classes for each input
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# 5. Map probabilities to labels
# For ProsusAI/finbert, the order is [Positive, Negative, Neutral]
labels = ["Positive", "Negative", "Neutral"]
probs_list = probabilities[0].tolist()

results = {label: round(prob, 4) for label, prob in zip(labels, probs_list)}

print(f"Text: {text}")
print(f"Probabilities: {results}")