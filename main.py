import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv('data/Telco-Customer-Churn.csv')

print("First 5 row: ")
print(df.head())

print("\nData info: ")
print(df.info())

print("\nStatistics: ")
print(df.describe())

print("\nMissing Values: ")
print(df.isnull().sum())

print("\nClass distribution: ")
print(df["Churn"].value_counts())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dropna(inplace=True)

df =pd.get_dummies(df, drop_first=True)

X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1918)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lr = LogisticRegression(max_iter=1000, random_state=1918)
model_lr.fit(X_train_scaled, y_train)
model_lr_prediction = model_lr.predict(X_test_scaled)

accuracy = accuracy_score(y_test, model_lr_prediction)
print("Accuracy LR: ", accuracy)
print("Confusion Matrix LR: ")
print(confusion_matrix(y_test, model_lr_prediction))
print("Classification Report LR: ")
print(classification_report(y_test, model_lr_prediction))

model_rf = RandomForestClassifier(n_estimators=100, random_state=1918)
model_rf.fit(X_train, y_train)
model_rf_prediction = model_rf.predict(X_test)

accuracy = accuracy_score(y_test, model_rf_prediction)
print("Accuracy RF: ", accuracy)
print("Confusion Matrix RF: ")
print(confusion_matrix(y_test, model_rf_prediction))
print("Classification Report RF: ")
print(classification_report(y_test, model_rf_prediction))

param_grid_rf = {
    "n_estimators" : [100, 200]
}

grid_rf = GridSearchCV(
    estimator = RandomForestClassifier(random_state=1918),
    param_grid = param_grid_rf,
    cv = 3,
    n_jobs = -1,
    scoring = "f1"
)

grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
grid_rf_prediction = grid_rf.predict(X_test)
accuracy = accuracy_score(y_test, grid_rf_prediction)
print("Accuracy RF: ", accuracy)
print("Confusion Matrix RF: ")
print(confusion_matrix(y_test, grid_rf_prediction))
print("Classification Report RF: ")
print(classification_report(y_test, grid_rf_prediction))
print(grid_rf.best_params_)

model_xgb = XGBClassifier(
    random_state=1918,
    eval_metric='logloss'
)

model_xgb.fit(X_train, y_train)
model_xgb_prediction = model_xgb.predict(X_test)

accuracy = accuracy_score(y_test, model_xgb_prediction)
print("Accuracy XGBoost: ", accuracy)
print("Confusion Matrix XGBoost: ")
print(confusion_matrix(y_test, model_xgb_prediction))
print("Classification Report XGBoost: ")
print(classification_report(y_test, model_xgb_prediction))

param_grid_xgb = {
    "learning_rate" : [0.05, 0.1],
    "max_depth" : [5, 10],
    "n_estimators" : [100, 200]
}

grid_xgb = GridSearchCV(
    estimator = XGBClassifier(random_state=1918, eval_metric='logloss'),
    param_grid = param_grid_xgb,
    cv = 3,
    n_jobs = -1,
    scoring = "f1"
)

grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
grid_xgb_prediction = grid_xgb.predict(X_test)

accuracy = accuracy_score(y_test, grid_xgb_prediction)
print("Accuracy XGBoost: ", accuracy)
print("Confusion Matrix XGBoost: ")
print(confusion_matrix(y_test, grid_xgb_prediction))
print("Classification Report XGBoost: ")
print(classification_report(y_test, grid_xgb_prediction))
print(grid_xgb.best_params_)









