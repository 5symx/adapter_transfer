# experiment

`python test_*.py --model_name roberta_large`
# Model Evaluation Results

## 1. With Pre-Trained Adapter

**Validation Accuracy:** 0.9610  

| Class     | Precision | Recall | F1-Score | Support |
|-----------|------------|------------|------------|------------|
| Negative  | 0.96 | 0.96 | 0.96 | 428 |
| Positive  | 0.96 | 0.97 | 0.96 | 444 |

**Overall Metrics:**
| Metric      | Score |
|-------------|------------|
| Accuracy    | 0.96 |
| Macro Avg   | 0.96 |
| Weighted Avg | 0.96 |

---

## 2. Only Pre-Trained Model

**Validation Accuracy:** 0.5092  

| Class     | Precision | Recall | F1-Score | Support |
|-----------|------------|------------|------------|------------|
| Negative  | 0.00 | 0.00 | 0.00 | 428 |
| Positive  | 0.51 | 1.00 | 0.67 | 444 |

**Overall Metrics:**
| Metric      | Score |
|-------------|------------|
| Accuracy    | 0.51 |
| Macro Avg   | 0.25 |
| Weighted

