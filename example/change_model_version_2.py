from transformers import BertConfig, BertModel
from transformers import AutoModel
import torch
from adapters import AutoAdapterModel, RobertaAdapterModel
from transformers import AutoModelForSequenceClassification
import argparse
import torch.nn as nn

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
    model = AutoAdapterModel.from_pretrained(args.model_name) # large 768
    adapter_name = model.load_adapter("AdapterHub/roberta-large-sst_houlsby")
    print(adapter_name)

    import torch.nn.functional as F

    new_size = 768
    old_size = 1024
    model_state = model.state_dict()

    # Iterate through state_dict and update matching tensors
    for name, param in model.state_dict().items():
        # Determine if 768 appears in shape[0] or shape[1]
        shape = param.shape
        if old_size in shape:  # Check if 768 appears in tensor shape
            if len(shape) == 1:  # 1D tensor
                print(param.shape)
                # new_shape = tuple(new_size if dim == 1024 else dim for dim in param.shape)
                resized_param = F.interpolate(param.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(1, new_size), mode="nearest").squeeze(0).squeeze(0).squeeze(0)
            elif len(shape) == 2: 
                print(param.shape)
                if param.shape[0] == old_size:
                    new_shape = (new_size, param.shape[1])  # Resize first dimension
                elif param.shape[1] == old_size:
                    new_shape = (param.shape[0], new_size)  # Resize second dimension
                else:
                    new_shape = param.shape  # No resizing needed
                # new_shape = tuple(new_size if dim == 768 else dim for dim in param.shape)
                print(new_shape)
                # Interpolating for resizing (Only applicable for certain layers)
                resized_param = F.interpolate(param.unsqueeze(0).unsqueeze(0), size=new_shape, mode="nearest").squeeze(0).squeeze(0)

            # Assign resized parameter back to model
            # model.state_dict()[name].copy_(resized_param)
            model_state[name] = resized_param.clone()

            print(f"Updated {name} from {param.shape} → {new_shape}")
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
# model = AutoAdapterModel.from_pretrained("roberta-base")
# model.load_state_dict(model_state)
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


