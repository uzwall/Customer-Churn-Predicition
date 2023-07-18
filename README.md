# General Overview
Project made with model tracking using mlflow(Machine Learing) and Docker implementation

# Customer Churn Prediction using MLflow

This code repository contains a Python script for predicting customer churn using MLflow, a machine learning lifecycle management platform. The code performs the following steps:

1. **Data Loading**: The script loads the customer churn data from the "telecom_churn.csv" file.

2. **Data Preprocessing**: The loaded data is preprocessed by scaling the features using StandardScaler to ensure they are on a similar scale.

3. **Data Split**: The preprocessed data is split into training and testing sets using a 80:20 ratio.

4. **Model Training and Evaluation (Unbalanced Data)**: A logistic regression model is trained on the unbalanced data using the training set. The performance of the model is evaluated by making predictions on the test set and generating a classification report.

5. **Model Saving and Manual Logging**: The trained model is saved using joblib. MLflow is used for manual logging, where metrics such as accuracy are logged, a description tag is set, and an artifact file ("artifacts1.txt") is logged.

6. **Data Oversampling**: To address class imbalance, the script applies oversampling to the training data using RandomOverSampler from the imbalanced-learn library.

7. **Model Training and Evaluation (Balanced Data)**: Another logistic regression model is trained on the balanced data obtained from oversampling. Similar to the previous step, the model's performance is evaluated by generating a classification report.

8. **Auto-Logging with MLflow**: MLflow's auto-logging feature is enabled in this step, which automatically tracks metrics, parameters, and artifacts without explicit logging. The trained model is saved using joblib, and MLflow records relevant information about the run.

9. **Experiment Tracking**: The code utilizes MLflow's experiment tracking capabilities to manage and track the different runs, allowing for easy comparison and reproducibility of experiments.

This code serves as an example of using MLflow for customer churn prediction, showcasing its features for experiment tracking, model training, evaluation, and logging.
