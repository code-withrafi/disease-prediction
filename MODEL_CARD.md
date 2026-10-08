# Model Card — Disease Prediction

## Model

**Model:** Random Forest
**Task:** Binary heart disease classification
**Dataset:** UCI Heart Disease
**Final threshold:** 0.29
**Purpose:** Educational machine learning project

The model predicts whether a clinical record belongs to the positive or negative heart disease class.

## Intended Use

The model demonstrates classification, model comparison, and threshold optimization when false negatives are considered more costly than false positives.

It must not be used for diagnosis, screening, treatment, triage, or healthcare decisions.

## Data

* 920 records
* 411 negative cases
* 509 positive cases
* Original target `num` converted to binary
* `id` and `dataset` columns excluded

## Model Comparison

| Model               |     Recall |    ROC-AUC |
| ------------------- | ---------: | ---------: |
| Logistic Regression |     88.24% |     90.35% |
| **Random Forest**   | **90.20%** | **92.37%** |
| XGBoost             |     89.22% |     89.92% |

Random Forest achieved the strongest baseline performance.

## Threshold Decision

Because false negatives were considered more costly, the default `0.50` threshold was evaluated against lower thresholds.

The final threshold was selected using **5-fold cross-validation on training data only**.

**Selected threshold: `0.29`**

| Metric          |   0.50 |       0.29 |
| --------------- | -----: | ---------: |
| Recall          | 91.18% | **98.04%** |
| Specificity     | 81.71% |     62.20% |
| False Negatives |      9 |      **2** |
| False Positives |     15 |         31 |

The threshold increased recall and reduced false negatives, at the cost of more false positives.

## Limitations

* Small dataset
* Multiple clinical sources
* No external validation
* No prospective clinical testing
* No clinical deployment evaluation

Performance may differ on other patient populations.

## Responsible Use

This model is for **educational purposes only**.

It must not be used to diagnose patients, recommend treatment, perform medical triage, or make healthcare decisions.

The `0.29` threshold reflects this project's error-cost assumption and is **not a clinically validated threshold**.

## Reproducibility

The repository contains the code, dataset, evaluation results, and tests.

Current test result:

```text
11 passed
```

