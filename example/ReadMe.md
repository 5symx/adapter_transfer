# experiment

`python test_*.py --model_name roberta_large`
# Model Evaluation Results

## 1. With Pre-Trained Adapter

**Validation Accuracy:** 0.9610  

| Class     | Precision | Recall | F1-Score | Support |
|-----------|------------|------------|------------|------------|
| Negative  | 0.96 | 0.96 | 0.96 | 428 |
| Positive  | 0.96 | 0.97 | 0.96 | 444 |
| Accuracy    | | | 0.96 | 872 |
| Macro Avg   | 0.96 | 0.96 | 0.96 | 872 |
| Weighted Avg | 0.96 | 0.96 | 0.96 | 872 |

---

## 2. Only Pre-Trained Model

**Validation Accuracy:** 0.5092  

| Class     | Precision | Recall | F1-Score | Support |
|-----------|------------|------------|------------|------------|
| Negative  | 0.00 | 0.00 | 0.00 | 428 |
| Positive  | 0.51 | 1.00 | 0.67 | 444 |
| Accuracy    |  |  | 0.51 | 872 |
| Macro Avg   | 0.25 | 0.50 | 0.34 | 872 |
| Weighted Avg | 0.26 | 0.51 | 0.34 | 872 |

