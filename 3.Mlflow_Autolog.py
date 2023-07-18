import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
import datetime

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Set experiment name
mlflow.set_experiment("CustomerChurn")

df = pd.read_csv("telecom_churn.csv")

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Manual Logging
with mlflow.start_run(run_name=f'Regression in Unbalanced data {timestamp}'):
    mlflow.set_tag("description", "Churn Classification in unbalanced dataset")

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    joblib.dump(model, 'model/prediction1.joblib')

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(report)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_artifact("artifacts1.txt")

# Auto Logging
with mlflow.start_run(run_name=f'Regression in balanced data {timestamp}'):
    mlflow.autolog()

    oversampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    model1 = LogisticRegression()
    model1.fit(X_train_resampled, y_train_resampled)
    y_pred = model1.predict(X_test)
    joblib.dump(model1, 'model/prediction2.joblib')

    report = classification_report(y_test, y_pred)
    print(report)
