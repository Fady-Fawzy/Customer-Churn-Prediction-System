import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression



df = pd.read_csv("./data/telco_churn_training.csv")
x = df.drop(columns=["Churn","customerID"])
y = df["Churn"]

num = [
    "tenure",
    "MonthlyCharges"
]

categ = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]
TotalCharges_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", MinMaxScaler())
])
preprocessor = ColumnTransformer([
    ("total", TotalCharges_pipeline, ["TotalCharges"]),
    ("cate", OneHotEncoder(handle_unknown="ignore"), categ),
    ("num", MinMaxScaler(), num),
    ("senior", "passthrough", ["SeniorCitizen"])
])


final_model = Pipeline([
    ("Preprocessor", preprocessor),
    ("model",LogisticRegression(class_weight="balanced",C=0.1,max_iter=100,solver="liblinear"))

])
final_model.fit(x, y)
joblib.dump(final_model, "./models/churn_pipeline.pkl")