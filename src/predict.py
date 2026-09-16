import pandas as pd 
import numpy as np
import joblib as jb

model = jb.load("./models/churn_pipeline.pkl")

customer = pd.DataFrame([{
    "gender": "...",
    "SeniorCitizen": 0,
    "Partner": "...",
    "Dependents": "...",
    "tenure": 0,
    "PhoneService": "...",
    "MultipleLines": "...",
    "InternetService": "...",
    "OnlineSecurity": "...",
    "OnlineBackup": "...",
    "DeviceProtection": "...",
    "TechSupport": "...",
    "StreamingTV": "...",
    "StreamingMovies": "...",
    "Contract": "...",
    "PaperlessBilling": "...",
    "PaymentMethod": "...",
    "MonthlyCharges": 0.0,
    "TotalCharges": 0.0
}])
THRESHOLD = 0.62

prob = model.predict_proba(customer)[:,1]
prediction = "Yes" if prob[0] >= THRESHOLD else "No"

print(prediction)
print("Churn Probability:", prob[0])