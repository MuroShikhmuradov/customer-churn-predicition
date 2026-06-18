import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
