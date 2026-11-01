# Customer Churn Prediction

Predicts which telecom customers will leave, and compares Logistic Regression, Random Forest and XGBoost, including hyperparameter tuning.

## Results

Scores are for the churn class on a test set of 1,407 customers, of whom 380 churned.

| Model | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.575 | 0.663 | 0.508 | 0.797 |
| XGBoost (tuned) | 0.556 | 0.685 | 0.468 | 0.798 |
| XGBoost | 0.521 | 0.592 | 0.466 | 0.769 |
| Random Forest | 0.500 | 0.633 | 0.413 | 0.777 |

Logistic Regression gave the best F1. Tuning raised XGBoost F1 from 0.521 to 0.556. For Random Forest, the search kept 100 trees, so the tuned result matches the default result.

All models find about half of the customers who churn or fewer, so recall is the main thing to improve.

## Data

The dataset is `data/Telco-Customer-Churn.csv`, which has 7,043 customers. After removing the 11 rows with no `TotalCharges`, 7,032 remain, and about 27% of them churned.

## Method

- Checked data types, missing values and class distribution.
- Converted `TotalCharges` to a number.
- Removed `customerID`, because a unique ID has no predictive value. One-hot encoding it had added about 7,000 useless columns.
- One-hot encoded categorical features.
- Used an 80/20 train and test split, and scaled features for Logistic Regression.
- Tuned Random Forest and XGBoost with `GridSearchCV`, using 3-fold cross-validation and F1 scoring.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

## Project structure

```text
├── data/            Dataset
├── main.py          Full analysis and training
└── requirements.txt
```

## Author

Murad Shikhmuradov
