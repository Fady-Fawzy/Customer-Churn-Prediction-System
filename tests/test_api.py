import io
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from api import main

client = TestClient(main.app)


@pytest.fixture
def customer():
    row = pd.read_csv(main.PROJECT_ROOT / "data/telco_churn_training.csv").dropna().iloc[0]
    return {
        key: row[key].item() if hasattr(row[key], "item") else row[key]
        for key in main.CustomerData.model_fields
    }


def test_metadata_and_frontend():
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client.get("/model-info").json() == {
        "model_type": "LogisticRegression",
        "threshold": 0.62,
    }
    assert client.get("/").status_code == 200


def test_real_single_inference(customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    result = response.json()
    expected = float(main.model.predict_proba(pd.DataFrame([customer]))[0, 1])
    assert result["probability"] == pytest.approx(expected)
    assert result["prediction"] == ("Churn" if expected >= 0.62 else "Not Churn")


@pytest.mark.parametrize(
    "probability,prediction,risk",
    [
        (0.0, "Not Churn", "Low"),
        (1 / 3, "Not Churn", "Medium"),
        (0.619, "Not Churn", "Medium"),
        (0.62, "Churn", "Medium"),
        (2 / 3, "Churn", "High"),
        (1.0, "Churn", "High"),
    ],
)
def test_threshold_boundaries(customer, monkeypatch, probability, prediction, risk):
    monkeypatch.setattr(
        main.model, "predict_proba", lambda _: np.array([[1 - probability, probability]])
    )
    result = client.post("/predict", json=customer).json()
    assert result["prediction"] == prediction
    assert result["risk_level"] == risk


def test_request_validation():
    assert client.post("/predict", json={"tenure": "invalid"}).status_code == 422


def test_real_batch_preserves_columns_and_matches_single(customer):
    source = pd.DataFrame([{**customer, "customerID": "test-customer"}])
    response = client.post(
        "/predict-batch", files={"file": ("input.csv", source.to_csv(index=False), "text/csv")}
    )
    assert response.status_code == 200
    output = pd.read_csv(io.StringIO(response.text))
    assert output.customerID.iloc[0] == "test-customer"
    assert set(source.columns).issubset(output.columns)
    single = client.post("/predict", json=customer).json()
    assert output.probability.iloc[0] == pytest.approx(single["probability"])
    assert output.prediction.iloc[0] == ("Yes" if single["prediction"] == "Churn" else "No")


@pytest.mark.parametrize("csv", ["", "a,b\n1,2", 'a,b\n"unterminated', "a,b\n"])
def test_invalid_batch_is_client_error(csv):
    assert (
        client.post("/predict-batch", files={"file": ("bad.csv", csv, "text/csv")}).status_code
        == 400
    )


def test_header_only_batch(customer):
    csv = pd.DataFrame(columns=customer.keys()).to_csv(index=False)
    assert (
        client.post("/predict-batch", files={"file": ("empty.csv", csv, "text/csv")}).status_code
        == 400
    )


def test_invalid_numeric_batch(customer):
    customer["tenure"] = "invalid"
    csv = pd.DataFrame([customer]).to_csv(index=False)
    assert (
        client.post("/predict-batch", files={"file": ("bad.csv", csv, "text/csv")}).status_code
        == 400
    )


def test_import_from_another_working_directory(tmp_path):
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(main.PROJECT_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", "from api.main import app; assert app"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
