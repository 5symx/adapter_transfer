import os

import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up argument parsing
parser = argparse.ArgumentParser(description="Load different transformer models for classification.")
parser.add_argument("--model_name", type=str, help="Pretrained model name (e.g., roberta-large)",default="roberta-base")
parser.add_argument("--new_adapter", action="store_true", help="Enable scratch adapter model",default=True)
# Parse arguments
args = parser.parse_args()

# Load model and tokenizer dynamically
# model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Load pre-trained BERT model from Hugging Face Hub
# The `BertAdapterModel` class is specifically designed for working with adapters
# It can be used with different prediction heads

model = AutoAdapterModel.from_pretrained(args.model_name)
if args.new_adapter:
    adapter_name = model.load_adapter("./result/final_adapter", config='houlsby')
else:
    adapter_name = model.load_adapter("/home/ymx/adapters/adapter_transfer/example/result/my_large2base_after", config='houlsby')
model.set_active_adapters(adapter_name)

print("test on benchmark")
#from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
# Load the SST-2 dataset
dataset = load_dataset("glue", "sst2")

# print(set(dataset["validation"]["label"])) # 0 , 1


test_texts = dataset["validation"]["sentence"]
test_labels = dataset["validation"]["label"]
# Tokenize dataset
inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

# Run inference
with torch.no_grad():
    logits = model(**inputs).logits

# Get predictions
predictions = torch.argmax(logits, dim=-1).numpy()

# print(predictions[:5])  # Inspect first few predictions -- test for 
# print(predictions.shape)  # Check output dimensions

# Compute accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Validation Accuracy: {accuracy:.4f}")
from sklearn.metrics import classification_report

print(classification_report(test_labels, predictions, target_names=["negative", "positive"]))

