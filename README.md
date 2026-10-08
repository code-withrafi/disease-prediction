# Disease Prediction from Medical Data

Machine learning project for predicting heart disease from clinical data, developed as a part of my AgenticX Machine Learning Internship. It compares three classifiers and applies threshold optimization to prioritize reducing false negatives.

---

## Project Overview

The project builds a complete, leakage-safe binary classification pipeline using the UCI Heart Disease dataset.

### Main Objectives

* Prepare structured medical data for machine learning
* Handle missing values safely
* Compare multiple classification algorithms
* Evaluate models using clinically relevant metrics
* Analyze classification thresholds
* Prioritize recall when false negatives are more costly
* Document model limitations and responsible use

---

## Models

Three classifiers are compared:

1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**

Random Forest produced the strongest baseline performance.

---

## Dataset

The project uses the **UCI Heart Disease dataset**.

* Total records: **920**
* Original features: **16 columns**
* Positive cases: **509**
* Negative cases: **411**

The original `num` target contains values from 0–4 and is converted into binary classification:

```text
0 → No heart disease
1 → Heart disease
```

The `id` column is removed because it is an identifier.

The `dataset` column is also removed to avoid using source-specific information as a predictive feature.

---

## Machine Learning Pipeline

```text
UCI Heart Disease Dataset
          ↓
Remove ID / Dataset Source
          ↓
Binary Target Conversion
          ↓
Stratified Train/Test Split
          ↓
Missing Value Imputation
          ↓
Categorical Encoding
          ↓
Numerical Scaling
          ↓
Model Training
          ↓
Model Evaluation
          ↓
Threshold Analysis
```

All preprocessing is implemented inside scikit-learn pipelines to reduce the risk of data leakage.

### Numerical Features

* Median imputation
* StandardScaler

### Categorical Features

* Most-frequent imputation
* OneHotEncoder
* `handle_unknown="ignore"`

---

## Train/Test Split

The dataset uses an 80/20 stratified split.

```text
Training samples: 736
Testing samples: 184
```

A fixed `random_state=42` is used for reproducibility.

---

## Baseline Results

The initial evaluation uses the standard classification threshold of **0.50**.

| Model               |   Accuracy |     Recall | Specificity |         F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ----------: | ---------: | ---------: |
| Logistic Regression |     84.24% |     88.24% |      79.27% |     86.12% |     90.35% |
| **Random Forest**   | **86.41%** | **90.20%** |  **81.71%** | **88.04%** | **92.37%** |
| XGBoost             |     84.78% |     89.22% |      79.27% |     86.67% |     89.92% |

Random Forest achieved the strongest baseline results and was selected for threshold analysis.

---

## Threshold Analysis

Because false negatives are considered more costly in this project, the default threshold of 0.50 was not treated as automatically optimal.

Thresholds were evaluated using **5-fold stratified cross-validation on the training data only**.

The selected threshold was:

```text
0.29
```

The final threshold was then evaluated on the untouched test set.

| Metric          | Threshold 0.50 | Threshold 0.29 |
| --------------- | -------------: | -------------: |
| Accuracy        |         86.96% |         82.07% |
| Precision       |         86.11% |         76.34% |
| Recall          |         91.18% |     **98.04%** |
| Specificity     |         81.71% |         62.20% |
| F1              |         88.57% |         85.84% |
| False Negatives |              9 |          **2** |
| False Positives |             15 |             31 |

The threshold of 0.29 increases recall from **91.18% to 98.04%** and reduces false negatives from **9 to 2**.

The trade-off is lower specificity and more false positives.

This demonstrates why threshold selection should reflect the cost of different classification errors.

---

## Project Structure

```text
disease-prediction/
│
├── data/
│   └── heart.csv
│
├── reports/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── threshold_analysis.csv
│   ├── threshold_analysis.png
│   └── threshold_comparison.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── threshold_analysis.py
│   └── visualize.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_training.py
│
├── inspect_data.py
├── MODEL_CARD.md
├── README.md
├── requirements.txt
├── pytest.ini
└── .gitignore
```

---

## Installation

Clone the repository and create a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Run the Project

### Evaluate Models

```powershell
python -m src.evaluate
```

This trains the three models and saves:

```text
reports/model_comparison.csv
```

### Generate Visualization

```powershell
python -m src.visualize
```

This generates:

```text
reports/model_comparison.png
```

### Run Threshold Analysis

```powershell
python -m src.threshold_analysis
```

This generates the threshold evaluation reports.

---

## Testing

The project uses pytest for automated testing.

Run:

```powershell
python -m pytest
```

Current result:

```text
11 passed
```

The tests cover:

* Dataset loading
* Feature and target preparation
* Binary target validation
* Removed columns
* Preprocessor creation
* Preprocessor transformation
* Model configuration
* Probability prediction support
* End-to-end model training and prediction

---

## Key Decisions

### Why Random Forest?

Random Forest achieved the strongest baseline performance among the three evaluated models, including the highest ROC-AUC and recall.

### Why Change the Threshold?

A 0.50 threshold is only a default decision rule. Since false negatives were considered more costly, a lower threshold was investigated to increase recall.

### Why 0.29?

The threshold was selected using training data with cross-validation while keeping the final test set untouched.

---

## Limitations

This is an educational machine learning project.

The dataset is relatively small and comes from multiple clinical sources. No external validation, prospective clinical testing, calibration study, or clinical deployment evaluation was performed.

The reported performance therefore does **not** establish clinical effectiveness or generalization to other patient populations.

---

## Responsible Use

This project must not be used to:

* Diagnose patients
* Screen patients clinically
* Recommend treatment
* Perform medical triage
* Make healthcare decisions

The model is intended only to demonstrate machine learning concepts, model comparison, evaluation metrics, and threshold selection.

For detailed methodology, results, limitations, and safety considerations, see [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Technologies

* Python 3.12
* pandas
* NumPy
* scikit-learn
* XGBoost
* Matplotlib
* pytest

---

## Reproducibility

The project uses fixed random seeds and a defined train/test split.

The repository includes the dataset, source code, automated tests, evaluation results, and generated visualizations required to reproduce the main experiment.

---

## Author And License

Copyright (c) 2026 Iftekhar Ibne Masud Rafi
