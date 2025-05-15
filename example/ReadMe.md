# Simple test to start for adapter transfer between models

## quick start
`python test_*.py --model_name roberta_large`

## model / adapter selection

model :  roberta-base / roberta-large
adapter: sst-2 from adapter hub

```
# original
model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

# adapter
model = AutoAdapterModel.from_pretrained(args.model_name)
adapter_name = model.load_adapter("sentiment/sst-2@ukp", config='houlsby')
model.set_active_adapters(adapter_name)
```

# Model Evaluation Results

## setup
dataset "sst-2" from "glue" to evaluate

## 1. Roberta-base With/Without Pre-Trained Adapter

### Adapter Validation Accuracy: 0.9427

| Class        | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Negative    | 0.94      | 0.94   | 0.94     | 428     |
| Positive    | 0.94      | 0.95   | 0.94     | 444     |
| Accuracy   |    -      |   -    | 0.94     | 872     |
| Macro Avg  | 0.94      | 0.94   | 0.94     | 872     |
| Weighted Avg | 0.94      | 0.94   | 0.94     | 872     |

---

### Original Validation Accuracy: 0.4908

| Class        | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Negative    | 0.49      | 1.00   | 0.66     | 428     |
| Positive    | 0.00      | 0.00   | 0.00     | 444     |
| Accuracy   |    -      |   -    | 0.49     | 872     |
| Macro Avg  | 0.25      | 0.50   | 0.33     | 872     |
| Weighted Avg | 0.24      | 0.49   | 0.32     | 872     |

---

## 2. Roberta-large With/Without Pre-Trained Adapter

## Adapter Validation Accuracy: 0.9610

| Class        | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Negative    | 0.96      | 0.96   | 0.96     | 428     |
| Positive    | 0.96      | 0.97   | 0.96     | 444     |
| Accuracy   |    -      |   -    | 0.96     | 872     |
| Macro Avg  | 0.96      | 0.96   | 0.96     | 872     |
| Weighted Avg | 0.96      | 0.96   | 0.96     | 872     |

---

## Original Validation Accuracy: 0.5092

| Class        | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Negative    | 0.00      | 0.00   | 0.00     | 428     |
| Positive    | 0.51      | 1.00   | 0.67     | 444     |
| Accuracy   |    -      |   -    | 0.51     | 872     |
| Macro Avg  | 0.25      | 0.50   | 0.34     | 872     |
| Weighted Avg | 0.26      | 0.51   | 0.34     | 872     |

