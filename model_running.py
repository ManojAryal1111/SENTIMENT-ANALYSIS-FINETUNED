from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
model_path = "./sentiment_model"   
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class 
def respond_to_user(text):
    sentiment = predict_sentiment(text)
    if sentiment == 1:
        return "Glad you enjoyed it! Sounds like the movie really worked for you."
    else:
        return "Sorry to hear that. Maybe the plot or pacing wasn’t what you expected."
print("Movie Review")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("How was the movie? ")
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break
    response = respond_to_user(user_input)
    print("Bot:", response)
