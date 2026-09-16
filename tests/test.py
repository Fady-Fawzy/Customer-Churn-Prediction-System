from fastapi.testclient import TestClient
from api.main import app
import io
import pandas as pd

client = TestClient(app)


def test_predict_batch_endpoint_exists():
    response = client.post("/predict-batch")

    assert response.status_code != 404


def test_predict_batch_returns_csv_with_predictions():

    csv_content = """gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
Female,0,No,No,6,Yes,No,Fiber optic,No,No,No,No,Yes,No,Month-to-month,Yes,Electronic check,85.5,513.0
"""

    files = {
        "file": ("customers.csv", csv_content, "text/csv")
    }

    response = client.post(
        "/predict-batch",
        files=files
    )

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]

    result_df = pd.read_csv(
        io.StringIO(response.text)
    )

    assert "prediction" in result_df.columns
    assert "probability" in result_df.columns
    assert "risk_level" in result_df.columns