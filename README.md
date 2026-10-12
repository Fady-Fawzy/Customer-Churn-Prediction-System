# Customer Churn Prediction

An end-to-end tabular machine learning application with a scikit-learn pipeline, FastAPI inference endpoints, and a browser interface for single-customer and CSV batch predictions.

## Problem and implementation

The application estimates churn probability from customer contract, service, tenure, and billing features. A shared serialized pipeline applies preprocessing and logistic regression during inference, reducing the risk of training/serving preprocessing drift.

The pipeline imputes missing `TotalCharges` with zero, scales numeric features with MinMaxScaler, encodes categorical features with OneHotEncoder (`handle_unknown="ignore"`), and passes through `SeniorCitizen`. Logistic regression uses balanced class weights, `C=0.1`, `solver="liblinear"`, and `max_iter=100`. The application uses a decision threshold of **0.62**. Risk labels are separate probability bands, not calibrated business-loss estimates.

## Architecture and files

The browser calls FastAPI; FastAPI validates single requests or reads a CSV, constructs a Pandas DataFrame, and calls the saved pipeline. `src/train.py` fits that pipeline on all bundled training rows; it does not calculate a new held-out score.

| Path | Purpose |
| --- | --- |
| `api/main.py` | API routes, input schema, batch export, static frontend |
| `src/train.py` | Training and artifact export |
| `src/predict.py` | Manual inference example |
| `frontend/` | HTML, CSS, JavaScript UI |
| `data/telco_churn_training.csv` | Bundled 5,000-row, 21-column dataset |
| `models/churn_pipeline.pkl` | Trusted repository pipeline artifact |
| `notebooks/01_eda.ipynb`, `notebooks/02_modeling.ipynb` | Exploration and saved modeling results |
| `tests/test_api.py` | Inference, batch validation, threshold, and serving checks |

## Setup and usage

Use Python 3.12. The pinned scikit-learn version matches the artifact's recorded version (1.9.1). Only load pickle/joblib artifacts from trusted sources.

```bash
git clone https://github.com/Fady-Fawzy/Customer-Churn-Prediction-System.git
cd Customer-Churn-Prediction-System
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/ or http://127.0.0.1:8000/docs. No environment variables or external provider credentials are needed. Run `python src/train.py` from the repository root only when you intentionally want to replace the local artifact.

| Route | Contract |
| --- | --- |
| `GET /health` | Reports whether the imported model is loaded |
| `GET /model-info` | Model type and threshold |
| `POST /predict` | JSON with the 19 fields documented by `/docs`; returns prediction, probability, risk level |
| `POST /predict-batch` | Multipart `file` CSV; returns CSV with `prediction`, `probability`, `risk_level` |

```bash
curl http://127.0.0.1:8000/model-info
curl -F "file=@data/telco_churn_training.csv" http://127.0.0.1:8000/predict-batch -o predictions.csv
```

Single predictions use `Churn`/`Not Churn`; batch output retains the original `Yes`/`No` convention. Original CSV columns, including customer IDs, are preserved. Missing columns, empty or malformed CSVs, and invalid feature values receive HTTP 400.

## Evaluation: evidence and limits

The bundled dataset has 188 missing `TotalCharges` values, zero duplicate rows, and a target distribution of 62.44% `No` / 37.56% `Yes` (checked directly).

The modeling notebook uses stratified 90/5/5 train/validation/test splits with `random_state=1`. Its saved classification reports show **77% validation accuracy** after threshold selection and **70% held-out test accuracy** on 250 rows. These are historical notebook outputs, not a fresh training run. The validation ROC-AUC is approximately 0.80 in the saved outputs. See [evaluation notes](docs/evaluation.md) for precise provenance and caveats.

The production artifact is fitted on the full dataset. Do not interpret its training-set predictions as independent performance evidence. Probability calibration, deployment behavior, and business impact have not been demonstrated.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check api src tests
python -m ruff format --check api src tests
python -m pytest
```

CI runs those checks and tests the bundled artifact, including a request made after changing the working directory.

## Limitations and next steps

- The single-request schema checks types but does not constrain every category or range.
- CSV batches are read into memory; there is no upload size limit, authentication, or rate limiting.
- An artifact loading failure prevents application startup.
- Threshold choice is validation-driven; a reproducible evaluation script and calibration analysis would strengthen the project.
- The `CLASSIFICATION - Neural Network/` directory is a separate experiment and is outside the serving path.
