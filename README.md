# 📊 Customer Churn Prediction System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-Frontend-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-Styling-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-Frontend-F7DF1E?logo=javascript&logoColor=black)

</p>

---

## 📑 Table of Contents

- [1-EDA Findings](#1-eda-findings)
- [2-ML Pipeline & Architecture](#2-ml-pipeline--architecture)
- [3-Model Comparison & Evaluation](#3-model-comparison--evaluation)
- [4-Why Logistic Regression Was Selected](#4-why-logistic-regression-was-selected)
- [5-Production Training Pipeline](#5-production-training-pipeline-trainpy)
- [6-Model Verification](#6-model-verification-predictpy)
- [FastAPI Backend & Endpoints](#-fastapi-backend--endpoints-mainpy)
- [How to Run](#️-how-to-run)

---

## 1-EDA Findings

### **Data Cleaning & Quality**

- The dataset contains **no duplicate rows**.
- Identified **188 missing values** in `TotalCharges` (handled during the preprocessing phase).

### **Target Distribution (`Churn`)**

- **No (Retained):** 62.44%
- **Yes (Churned):** 37.56%

➜ There is a moderate class imbalance, which is why **Accuracy alone was not used** to evaluate model performance.

### **Key Business Insights**

- **Contract Type:** Strong correlation with churn. Customers with **Month-to-month** contracts have the highest churn rate.
- **Tenure:** As customer tenure increases, churn drops significantly (long-term customers are much more loyal).
- **Monthly Charges:** Higher `MonthlyCharges` correspond to higher churn rates (higher bills lead to customer dissatisfaction).

---

## 2-ML Pipeline & Architecture

The entire data preparation and model inference flow is encapsulated within a single unified scikit-learn `Pipeline` to prevent data leakage and ensure seamless production serving via FastAPI.

```text
Customer Churn Prediction Pipeline
==================================

                            ┌───────────────────┐
                            │     RAW DATA      │
                            └─────────┬─────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
       TotalCharges         tenure, MonthlyCharges    Categorical Features
            │                         │                         │
            ▼                         │                         │
  ┌───────────────────┐               │                         │
  │   SimpleImputer   │               │                         │
  │ (fill_value = 0)  │               │                         │
  └─────────┬─────────┘               │                         │
            │                         │                         │
            ▼                         ▼                         ▼
  ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
  │   MinMaxScaler    │     │   MinMaxScaler    │     │   OneHotEncoder   │
  │                   │     │                   │     │ (handle_unknown)  │
  └─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                        + SeniorCitizen (passthrough)
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │       PREPROCESSOR        │
                        │     ColumnTransformer     │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │     INITIAL PIPELINE      │
                        │    Logistic Regression    │
                        │ (class_weight='balanced') │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │       GridSearchCV        │
                        │      model__C = 0.1       │
                        │ model__solver='liblinear' │
                        │   model__max_iter = 100   │
                        └─────────────┬─────────────┘
                                      │
                                Best Estimator
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │        FINAL MODEL        │
                        │   (Decision Threshold)    │
                        │      threshold = 0.62     │
                        └───────────────────────────┘
```

---

## 3-Model Comparison & Evaluation

Three baseline models were evaluated on the validation set using a stratified split:

| Model | Train Accuracy | Val Accuracy | Recall (Churn) | F1-Score (Churn) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Default @ 0.50)** | 75.22% | 71.60% | **71.28%** | 0.65 | **~0.80** |
| **Logistic Regression (Tuned @ 0.62)** | 75.22% | **77.00%** | 63.83% | **0.67** | **~0.80** |
| **Decision Tree** | 82.51% | 69.60% | 55.32% | 0.58 | 0.72 |
| **Random Forest** | **83.47%** | 74.40% | 60.64% | 0.64 | 0.77 |

---

## 4-Why Logistic Regression Was Selected

**Logistic Regression (Balanced)** was chosen as the final model because it achieved the optimal balance across all primary evaluation metrics:

- **Balanced Performance:** Delivers the best combined score across **Accuracy, Recall, F1-Score, and ROC-AUC (~0.80)**.
- **No Overfitting:** Unlike Decision Tree and Random Forest (which showed a noticeable gap between training and validation accuracy), Logistic Regression generalized consistently with nearly zero overfitting.
- **Effective Churn Identification:** Captures at-risk churners effectively while maintaining high overall precision and probability calibration for business risk scoring.

---

## 5-Production Training Pipeline (`train.py`)

- **Data Ingestion:** Loaded the raw dataset (`telco_churn_training.csv`) into a pandas DataFrame.
- **Feature & Target Separation:** Split the dataset into features ($X$) and target ($y$), dropping `customerID` and `Churn`, to utilize 100% of the available data for maximum learning capacity.
- **Unified Pipeline Assembly:** Built the final end-to-end `Pipeline` (`ColumnTransformer` + `LogisticRegression`) incorporating the exact preprocessing and hyperparameter choices validated during the EDA and modeling phases.
- **Model Training & Artifact Export:** Trained the pipeline on all available data and serialized it into `churn_pipeline.pkl` using `joblib` for instant loading in the API without retraining.

---

## 6-Model Verification (`predict.py`)

- **Purpose:** Manually tests the saved model (`churn_pipeline.pkl`) to verify error-free inference on raw data[cite: 5].
- **Execution:** Loads the pipeline via `joblib`, feeds a single-row sample DataFrame, evaluates churn probability against the `0.62` threshold, and prints the result[cite: 5].

---

## 🚀 FastAPI Backend & Endpoints (`main.py`)

- **Model Initialization & Thresholding:** Loads `churn_pipeline.pkl` globally once upon startup and enforces the optimized decision threshold of `0.62`[cite: 1, 6].
- **`GET /health`:** Health check endpoint to verify server readiness and confirm the pipeline artifact is loaded in memory[cite: 1, 6].
- **`GET /model-info`:** Returns active metadata including the underlying model name (`LogisticRegression`) and classification threshold[cite: 1, 6].
- **Input Validation (`BaseModel`):** Uses a Pydantic schema (`CustomerData`) to validate data types and structure for all required raw features before inference[cite: 1, 6].
- **`POST /predict` (Single Prediction):** Receives customer data from the web form, predicts churn probability, and returns the verdict (`Churn` / `Not Churn`), probability score, and risk level (`Low`, `Medium`, `High`)[cite: 1, 6].
- **`POST /predict-batch` (Batch Prediction):** Accepts a full CSV file, generates predictions for all records without altering the original columns (preserving `customerID`), appends `prediction`, `probability`, and `risk_level`, and streams the updated file back as an instant CSV download[cite: 1, 6].
- **Frontend Integration:** Mounts the static UI files (`index.html`, `style.css`, `script.js`) at the root URL (`/`) for direct access via the browser[cite: 1, 6].

---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone <https://github.com/Fady-Fawzy/Customer-Churn-Prediction-System>
cd Customer-Churn-Prediction
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the Required Packages

```bash
pip install -r requirements.txt
```

### 4. Train the Model

If `models/churn_pipeline.pkl` is not already available:

```bash
python src/train.py
```

This will train the final pipeline and save the trained model artifact.

### 5. Start the FastAPI Server

From the project root directory:

```bash
uvicorn api.main:app --reload
```

### 6. Open the Application

Open your browser and navigate to:

```text
http://127.0.0.1:8000/
```

The web interface supports:

- **Single Customer Prediction**
- **Whole CSV Batch Prediction**

### 7. API Documentation

FastAPI automatically provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

Available endpoints include:

```text
GET  /health
GET  /model-info
POST /predict
POST /predict-batch
```

---