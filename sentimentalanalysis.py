from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

text = "I love using transformers, it's amazing!"
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)
print(outputs.logits)