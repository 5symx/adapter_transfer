import os

import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up argument parsing
parser = argparse.ArgumentParser(description="Load different transformer models for classification.")
parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name (e.g., roberta-large)")

# Parse arguments
args = parser.parse_args()

# Load model and tokenizer dynamically
# model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Load pre-trained BERT model from Hugging Face Hub
# The `BertAdapterModel` class is specifically designed for working with adapters
# It can be used with different prediction heads
model = AutoAdapterModel.from_pretrained(args.model_name)
adapter_name = model.load_adapter("sentiment/sst-2@ukp", config='houlsby')
model.set_active_adapters(adapter_name)

print("test on benchmark")
#from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
# Load the SST-2 dataset
dataset = load_dataset("glue", "sst2")

test_texts = dataset["validation"]["sentence"]
test_labels = dataset["validation"]["label"]
# Tokenize dataset
inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

# Run inference
with torch.no_grad():
    logits = model(**inputs).logits

# Get predictions
predictions = torch.argmax(logits, dim=-1).numpy()

# Compute accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Validation Accuracy: {accuracy:.4f}")
from sklearn.metrics import classification_report

print(classification_report(test_labels, predictions, target_names=["negative", "positive"]))

