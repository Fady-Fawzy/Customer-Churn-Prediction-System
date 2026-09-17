## 1-EDA Findings

* **Data Cleaning & Quality:**
  * The dataset contains **no duplicate rows**.
  * Identified **188 missing values** in `TotalCharges` (handled during the preprocessing phase).

* **Target Distribution (`Churn`):**
  * **No (Retained):** 62.44%
  * **Yes (Churned):** 37.56%
  * ➜ There is a moderate class imbalance, which is why **Accuracy alone was not used** to evaluate model performance.

* **Key Business Insights:**
  * **Contract Type:** Strong correlation with churn. Customers with **Month-to-month** contracts have the highest churn rate.
  * **Tenure:** As customer tenure increases, churn drops significantly (long-term customers are much more loyal).
  * **Monthly Charges:** Higher `MonthlyCharges` correspond to higher churn rates (higher bills lead to customer dissatisfaction).

## 2-ML Pipeline & Architecture

The entire data preparation and model inference flow is encapsulated within a single unified scikit-learn `Pipeline` to prevent data leakage and ensure seamless production serving via FastAPI.


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
                        │     threshold = 0.62      │
                        └───────────────────────────┘
```

</details>


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

* **Balanced Performance:** Delivers the best combined score across **Accuracy, Recall, F1-Score, and ROC-AUC (~0.80)**.
* **No Overfitting:** Unlike Decision Tree and Random Forest (which showed a noticeable gap between training and validation accuracy), Logistic Regression generalized consistently with nearly zero overfitting.
* **Effective Churn Identification:** Captures at-risk churners effectively while maintaining high overall precision and probability calibration for business risk scoring.


## 5-Production Training Pipeline (`train.py`)

* **Data Ingestion:** Loaded the raw dataset (`telco_churn_training.csv`) into a pandas DataFrame.
* **Feature & Target Separation:** Split the dataset into features ($X$) and target ($y$), dropping `customerID` and `Churn`, to utilize 100% of the available data for maximum learning capacity.
* **Unified Pipeline Assembly:** Built the final end-to-end `Pipeline` (`ColumnTransformer` + `LogisticRegression`) incorporating the exact preprocessing and hyperparameter choices validated during the EDA and modeling phases.
* **Model Training & Artifact Export:** Trained the pipeline on all available data and serialized it into `churn_pipeline.pkl` using `joblib` for instant loading in the API without retraining.


## 6-Model Verification (`predict.py`)

* **Purpose:** Manually tests the saved model (`churn_pipeline.pkl`) to verify error-free inference on raw data[cite: 5].
* **Execution:** Loads the pipeline via `joblib`, feeds a single-row sample DataFrame, evaluates churn probability against the `0.62` threshold, and prints the result[cite: 5].



