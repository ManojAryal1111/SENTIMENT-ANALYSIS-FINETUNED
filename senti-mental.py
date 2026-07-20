from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import TrainingArguments,Trainer
dataset=load_dataset("imdb")
tokenizer=AutoTokenizer.from_pretrained("bert-based-uncased")
model=AutoModelForSequenceClassification.from_pretrained("bert-based-uncased",num_labels=2)