# Install required libraries first:
# pip install transformers datasets torch scikit-learn

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset (IMDb reviews)
dataset = load_dataset("imdb")

# 2. Load tokenizer and model (BERT for binary classification)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 3. Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# 4. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # small batch size for GPU memory
    num_train_epochs=1,              # demo run, increase later
    weight_decay=0.01,
)

# 6. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(2000)),  # demo subset
    eval_dataset=tokenized_dataset["test"].shuffle(seed=42).select(range(500)),
    compute_metrics=compute_metrics
)

# 7. Train the model
trainer.train()

# 8. Evaluate
print(trainer.evaluate())

# 9. Save both model and tokenizer
trainer.save_model("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")
