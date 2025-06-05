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
parser.add_argument("--base_model_name", type=str,  help="Pretrained model name (e.g., roberta-large)", default="roberta-base") #required=True,
# Parse arguments
args = parser.parse_args()
# Ensure the directory exists


# Load the base model
if args.adapter:
    print("set with pre-trained adapter")
    model = AutoAdapterModel.from_pretrained(args.model_name) # large 768

    import torch.nn.functional as F

    model_state = model.state_dict()

    adapter_name = model.load_adapter("AdapterHub/roberta-large-sst_houlsby")
    model.set_active_adapters(adapter_name)
    new_model_state = model.state_dict()

    assert len(model_state) != len(new_model_state) # numbers of keys


    # new_size = 768
    # old_size = 1024

    size_mapping = {
        1024: 768,   # Example: Change 768 → 1024
        4096: 3072,    # Example: Change 512 → 640
        24: 12,    # Example: Change 256 → 320
        16: 12,
        64: 48,
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
    
    
    for name, param in new_model_state.items():
        resized_param = resize_param(param, size_mapping)
        if resized_param != None:
            new_model_state[name] = resized_param.clone()
            # print(f"Updated {name} from {param.shape} → {resized_param.shape}")
       

    print("✅ All matching parameters resized successfully!")
    
else:
    print("set with pretrained model only")
    # model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    

def test():
    print("test on benchmark")
    
    #from transformers import Trainer, TrainingArguments
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score
    # Load the SST-2 dataset
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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



model = AutoAdapterModel.from_pretrained(args.base_model_name)
print("before add adapter")
print(len(model.state_dict()))


adapter_name = model.load_adapter("AdapterHub/roberta-base-sst_houlsby")
model.set_active_adapters(adapter_name)

print("after add adapter")
print(len(model.state_dict()))

test()

model.load_state_dict(new_model_state, strict=False) #  ignore unexpected keys
print("save custom adapter")
model.save_adapter(save_directory="/home/ymx/adapters/adapter_transfer/example/result/my_large2base", adapter_name=adapter_name)


model = AutoAdapterModel.from_pretrained(args.base_model_name)
print("before add adapter")
print(len(model.state_dict()))

adapter_name = model.load_adapter("/home/ymx/adapters/adapter_transfer/example/result/my_large2base")
model.set_active_adapters(adapter_name)


print("after add adapter")
print(len(model.state_dict()))

test()
