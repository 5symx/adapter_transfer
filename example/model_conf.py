from adapters import AutoAdapterModel

import os
os.makedirs("result", exist_ok=True)

import argparse
# Set up argument parsing
parser = argparse.ArgumentParser(description="Load different transformer models for classification.")
parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name (e.g., roberta-large)")

# Parse arguments
args = parser.parse_args()
# Ensure the directory exists


# Load the base model
model = AutoAdapterModel.from_pretrained(args.model_name)

# Save model structure to file_1 before adapter loading
file_1 = f"result/{args.model_name}_before.txt"
with open(file_1, "w") as f:
    f.write(str(model))

# Load adapter and activate it
adapter_name = model.load_adapter("sentiment/sst-2@ukp", config="houlsby")
model.set_active_adapters(adapter_name)

# Save updated model structure to file_2 after adapter activation
file_2 = f"result/{args.model_name}_after.txt"
with open(file_2, "w") as f:
    f.write(str(model))

print("Model structure saved to name_before.txt (before) and name_after.txt (after adapter activation).")