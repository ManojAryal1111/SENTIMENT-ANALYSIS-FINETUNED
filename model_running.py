from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Load the fine-tuned model and tokenizer
model_path = "./sentiment_model"   # folder where you saved both model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2. Helper function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class  # 0 = negative, 1 = positive

# 3. Helper function to respond conversationally
def respond_to_user(text):
    sentiment = predict_sentiment(text)
    if sentiment == 1:
        return "Glad you enjoyed it! Sounds like the movie really worked for you."
    else:
        return "Sorry to hear that. Maybe the plot or pacing wasn’t what you expected."

# 4. Interactive loop
print("🎬 Movie Review Bot 🎬")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("How was the movie? ")
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye! 👋")
        break
    response = respond_to_user(user_input)
    print("Bot:", response)
