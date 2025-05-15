import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModel
# Set up argument parsing
parser = argparse.ArgumentParser(description="Load different transformer models for classification.")
parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name (e.g., roberta-large, textattack/roberta-base-SST-2)")

# Parse arguments
args = parser.parse_args()

# Load model and tokenizer dynamically
model = AutoModel.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

print(f"Loaded model: {args.model_name}")

# Load dataset
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


print(predictions[:5])  # Inspect first few predictions
print(predictions.shape)  # Check output dimensions

# # Compute accuracy
# accuracy = accuracy_score(test_labels, predictions)
# print(f"Validation Accuracy: {accuracy:.4f}")

# from sklearn.metrics import classification_report

# print(classification_report(test_labels, predictions, target_names=["negative", "positive"]))

