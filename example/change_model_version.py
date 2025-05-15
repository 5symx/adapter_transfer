from transformers import BertConfig, BertModel
import torch
from adapters import AutoAdapterModel, RobertaAdapterModel
from transformers import AutoModelForSequenceClassification
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Load different transformer models for classification.")
parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name (e.g., roberta-large)")
parser.add_argument("--adapter", action="store_true", help="Enable adapter model")

# Parse arguments
args = parser.parse_args()
# Ensure the directory exists


# Load the base model
if args.adapter:
    print("set with pre-trained adapter")
    model = AutoAdapterModel.from_pretrained(args.model_name)
else:
    print("set with pretrained model only")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)


for name, param in model.named_parameters():
    if str(model.config.hidden_size) in str(param.shape):
        print(f"Layer using hidden_size: {name} -> Shape: {param.shape}")
        
if args.model_name == "roberta-base":
    print("set to large ")
        
    model.config.num_hidden_layers = 24
    model.config.num_attention_heads = 16
    model.config.intermediate_size = 4096
    model.config.prediction_heads["default"]["embedding_size"] = 1024

elif  args.model_name == "roberta-large":
    print("set to base ")
    model.config.hidden_size = 768
    model.config.num_hidden_layers = 12
    model.config.num_attention_heads = 12
    model.config.intermediate_size = 3072
    model.config.prediction_heads["default"]["embedding_size"] = 768

else:
    print("invalid model name ")


print("write to file the updated conf")
file_2 = f"result/{args.model_name}_adaarch_{args.adapter}.txt"
with open(file_2, "w") as f:
    f.write(str(model))




model_update = RobertaAdapterModel(model.config)
print(model_update.config)
print("successfully set to same model layer and arch")



