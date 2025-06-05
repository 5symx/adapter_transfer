from transformers import BertConfig, BertModel
from transformers import AutoModel
import torch
from adapters import AutoAdapterModel, RobertaAdapterModel
from transformers import AutoModelForSequenceClassification
import argparse
import torch.nn as nn
from transformers import AutoTokenizer

# Set up argument parsing
parser = argparse.ArgumentParser(description="Load different transformer models for classification.")
parser.add_argument("--model_name", type=str,  help="Pretrained model name (e.g., roberta-large)", default="roberta-large") #required=True,
parser.add_argument("--adapter", action="store_true", help="Enable adapter model",default=True)

# Parse arguments
args = parser.parse_args()
# Ensure the directory exists


# Load the base model
if args.adapter:
    print("set with pre-trained adapter")
    model = AutoAdapterModel.from_pretrained(args.model_name) # large 768

    import torch.nn.functional as F

    model_state = model.state_dict()

    # new_size = 768
    # old_size = 1024

    size_mapping = {
        1024: 768,   # Example: Change 768 → 1024
        4096: 3072,    # Example: Change 512 → 640
        24: 12,    # Example: Change 256 → 320
        16: 12,
    }
    # def update_sizes(a, b, mapping):
    #     return (mapping.get(a, a), mapping.get(b, b))
    def resize_param(param, size_mapping):
        shape = param.shape
        
        if len(shape) == 1 and shape[0] in size_mapping:
            new_shape = size_mapping[shape[0]]
            return F.interpolate(param.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(1, new_shape), mode="nearest").squeeze(0).squeeze(0).squeeze(0)

        elif len(shape) == 2:
            new_shape = tuple(size_mapping.get(dim, dim) for dim in shape)
            if new_shape != shape:
                return F.interpolate(param.unsqueeze(0).unsqueeze(0), size=new_shape, mode="nearest").squeeze(0).squeeze(0)

        return None  # Return the original if no resizing is needed
    
    # old_size = 0
    # new_size = 0
    # tuple_list =[]
    # Iterate through state_dict and update matching tensors
    for name, param in model.state_dict().items():
        resized_param = resize_param(param, size_mapping)
        if resized_param != None:
            model_state[name] = resized_param.clone()
            print(f"Updated {name} from {param.shape} → {resized_param.shape}")
        # tuple_list.clear()
        # Determine if 768 appears in shape[0] or shape[1]
        # shape = param.shape
        # if len(shape) == 1:
        #     if shape[0] in size_mapping:  # 1D tensor
        #     # print(param.shape)
        #     # new_shape = tuple(new_size if dim == 1024 else dim for dim in param.shape)
        #         new_shape = size_mapping[shape[0]]
        #         resized_param = F.interpolate(param.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(1, new_shape), mode="nearest").squeeze(0).squeeze(0).squeeze(0)
        # elif len(shape) == 2:
        #     new_shape = update_sizes(shape[0], shape[1], size_mapping)
        #     if new_shape != shape:
        #         resized_param = F.interpolate(param.unsqueeze(0).unsqueeze(0), size=new_shape, mode="nearest").squeeze(0).squeeze(0)
        # else:
        #     continue

        # for dim in shape:
        #     if dim in size_mapping:
        #         print(f"Matched dimension: {dim}")  # Print matched dimensions
        #         old_size = dim
        #         new_size = size_mapping[dim]
        #         print(old_size, new_size)
        #         # break
        #     # else:
        #         # print("not found")
        # if new_size == 0:
        #     continue

        # # if old_size in shape:  # Check if 768 appears in tensor shape
        # if len(shape) == 1:  # 1D tensor
        #     # print(param.shape)
        #     # new_shape = tuple(new_size if dim == 1024 else dim for dim in param.shape)
        #     resized_param = F.interpolate(param.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(1, new_size), mode="nearest").squeeze(0).squeeze(0).squeeze(0)
        #     new_shape = new_size
        # elif len(shape) == 2: 
        #     # print(param.shape)
        #     for (old_size,new_size) in tuple_list:
        #         if param.shape[0] == old_size and param.shape[1] == old_size:
        #             new_shape = (new_size, new_size)  # Resize first dimension
        #         elif param.shape[0] == old_size:
        #             new_shape = (new_size, param.shape[1])
        #         elif param.shape[1] == old_size:
        #             new_shape = (param.shape[0], new_size)  # Resize second dimension
        #         else:
        #             new_shape = param.shape  # No resizing needed
        #     # new_shape = tuple(new_size if dim == 768 else dim for dim in param.shape)
        #     # print(new_shape)
        #     # Interpolating for resizing (Only applicable for certain layers)
        #     resized_param = F.interpolate(param.unsqueeze(0).unsqueeze(0), size=new_shape, mode="nearest").squeeze(0).squeeze(0)

            # Assign resized parameter back to model
            # model.state_dict()[name].copy_(resized_param)
        # model_state[name] = resized_param.clone()

        # print(f"Updated {name} from {param.shape} → {new_shape}")
            # break

    print("✅ All matching parameters resized successfully!")
    
else:
    print("set with pretrained model only")
    # model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    
    # print("classifier")
    # print(model.classifier)

    # print(model.classifier.dense.weight.mean())  # Check mean of initialized weights
    # print(model.classifier.dense.bias)  # Bias values
model = AutoAdapterModel.from_pretrained("roberta-base")
model.load_state_dict(model_state, strict=False) #  ignore unexpected keys


adapter_name = model.load_adapter("AdapterHub/roberta-base-sst_houlsby")
model.set_active_adapters(adapter_name)


print("test on benchmark")
#from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
# Load the SST-2 dataset
dataset = load_dataset("glue", "sst2")

# print(set(dataset["validation"]["label"])) # 0 , 1
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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



# # check name and dimension
# for name, param in model.state_dict().items():
#     print(f"{name}: {param.shape}")

# # # Load adapter trained on BERT-Base
# adapter_name = model.load_adapter("AdapterHub/roberta-base-sst_houlsby")
# print(adapter_name)

# # Activate the adapter
# model.set_active_adapters(adapter_name)

# print(f"Loaded adapter '{adapter_name}' into ROBERTA-Large!")

# # model.add_adapter(adapter_name, overwrite_ok=True)


