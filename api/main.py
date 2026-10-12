from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from io import StringIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(PROJECT_ROOT / "models" / "churn_pipeline.pkl")
THRESHOLD = 0.62

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {"model_type": type(model.named_steps["model"]).__name__, "threshold": THRESHOLD}


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict(customer: CustomerData):
    df = pd.DataFrame([customer.model_dump()])
    prob = model.predict_proba(df)[:, 1]
    probability = prob[0]
    prediction = "Churn" if probability >= THRESHOLD else "Not Churn"
    risk_level = None
    if probability < 1 / 3:
        risk_level = "Low"
    elif probability < 2 / 3:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {"prediction": prediction, "probability": probability, "risk_level": risk_level}


@app.post("/predict-batch")
def predict_batch(file: UploadFile = File(...)):

    try:
        df = pd.read_csv(file.file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Upload a valid, non-empty UTF-8 CSV.") from exc

    missing = sorted(set(CustomerData.model_fields) - set(df.columns))
    if missing:
        raise HTTPException(status_code=400, detail={"missing_columns": missing})
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV must contain at least one customer.")

    try:
        probabilities = model.predict_proba(df)[:, 1]
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="CSV contains invalid feature values.") from exc

    predictions = ["Yes" if probability >= THRESHOLD else "No" for probability in probabilities]

    risk_levels = [
        "Low" if probability < 1 / 3 else "Medium" if probability < 2 / 3 else "High"
        for probability in probabilities
    ]

    df["prediction"] = predictions
    df["probability"] = probabilities
    df["risk_level"] = risk_levels

    output = StringIO()

    df.to_csv(output, index=False)

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
