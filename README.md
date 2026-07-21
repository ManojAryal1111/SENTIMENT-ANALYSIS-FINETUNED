# Sentiment Analysis Fine-Tuning

Fine-tunes a transformer model (BERT/DistilBERT) for sentiment classification using Hugging Face `Trainer`.

## What it does
- Loads a pretrained model and tokenizes a sentiment dataset
- Fine-tunes using `TrainingArguments` (2e-5 learning rate, 1 epoch, batch size 8)
- Evaluates on a held-out test split
- Saves the fine-tuned model and tokenizer locally

## Files
- `senti-ment.py` — training script (model setup, training args, Trainer)
- `model_running.py` — script to load and run the saved model for inference

## Requirements
```bash
pip install transformers datasets torch
```

## Usage
```bash
python senti-ment.py
python model_running.py
```
## Notes
Trained model weights are not included in this repo due to size. Re-run `senti-ment.py` to regenerate them.
