import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

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

model_lr = LogisticRegression(max_iter=1000)
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






