globals()["TC"] = True  # Define dynamically STD TC

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import BertConfig, BertModel
from transformers import AutoModel
import torch
from adapters import AutoAdapterModel, RobertaAdapterModel, AdapterTrainer
from transformers import AutoModelForSequenceClassification
import argparse
from datasets import load_dataset
import torch.nn as nn
from transformers import AutoTokenizer

import time
import sys
# Generate timestamp
timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


# Define log file
log_filename = f"./result/log/output_{timestamp}.log"

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

sys.stdout = Tee(log_filename)

# Set up argument parsing
parser = argparse.ArgumentParser(description="Load different transformer models for classification.")
parser.add_argument("--model_name", type=str,  help="Pretrained model name (e.g., roberta-large)", default="roberta-large") #required=True,
parser.add_argument("--new_adapter", action="store_true", help="Enable scratch adapter model",default=True)
parser.add_argument("--base_model_name", type=str,  help="Pretrained model name (e.g., roberta-large)", default="roberta-base") #required=True,
parser.add_argument("--dataset", type=str,  help="dataset for training (e.g., sst2)", default="sst2") #required=True,
# Parse arguments
args = parser.parse_args()
print("Parsed Parameters:")
print(vars(args))

model = AutoAdapterModel.from_pretrained(args.base_model_name)

if args.new_adapter:
    adapter_name = model.add_adapter("my-adapter-s",config="houlsby")
    model.add_classification_head(
        "my-adapter-s",
        num_labels=2,
        id2label={ 0: "👎", 1: "👍"}
    )
    model.train_adapter("my-adapter-s")
else:
    adapter_name = model.load_adapter("/home/ymx/adapters/adapter_transfer/example/result/my_large2base")
    model.set_active_adapters(adapter_name)
    model.train_adapter(adapter_name)

# model_name = "textattack/roberta-base-SST-2"  # Using a fine-tuned model on SST-2
tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

# dataset = load_dataset("imdb")
dataset = load_dataset("glue", args.dataset)
# train_texts, train_labels = dataset["train"]["sentence"], dataset["train"]["label"]
# val_texts, val_labels = dataset["test"]["sentence"], dataset["test"]["label"]

# train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
# val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt")

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["sentence"], max_length=80, truncation=True, padding="max_length")
# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# training_args = TrainingArguments(
#     output_dir="./model_output",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,

# )

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
)

if "ESC" in globals():
    from transformers import EarlyStoppingCallback

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stops after 2 epochs without improvement
    )

    trainer.train()

if "STD" in globals():

    # small_train_dataset = train_dataset.select(range(int(len(train_dataset) * 0.2)))

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        
        # tokenizer=tokenizer,
        # data_collator=data_collator,
    )

    trainer.train()

if "TC" in globals():
    import time
    from transformers import TrainerCallback, EarlyStoppingCallback
    class TimeCallback(TrainerCallback):
        def __init__(self):
            self.epoch_times = []

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch_time = time.time()
            self.epoch_times.append(epoch_time)
            print(f"Epoch {state.epoch}: Time taken - {epoch_time:.2f} seconds")

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[TimeCallback(), EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
trainer.evaluate()

if args.new_adapter:
    model.save_adapter("./result/final_adapter", "my-adapter-s")
else:
    model.save_adapter("./result/my_large2base_after", adapter_name)
# model.save_pretrained("fine_tuned_model")
# Reset stdout to default
sys.stdout.file.close()
sys.stdout = sys.stdout.stdout