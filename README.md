# Customer Churn Predictor

A machine learning API that predicts whether a customer is likely to churn based on their account information.

## Overview
This project uses a logistic regression model trained on the Telco Customer Churn dataset. The model is served via a Flask REST API that accepts customer data and returns a churn prediction and probability score.

- Python
- Scikit-learn (Pipeline, LogisticRegression, ColumnTransformer)
- Flask
- Pandas
- Joblib

## Setup
pip install -r requirements.txt
python app.py

Example request:
```json
{
    "tenure": 12,
    "MonthlyCharges": 65.50,
    "TotalCharges": 786.00,
    "SeniorCitizen": 0,
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
}
```

Example response:
```json
{
    "churned": true,
    "churn_probability": 0.823
}
```

## Dataset
Trained on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle — 7032 customers with 21 features.
