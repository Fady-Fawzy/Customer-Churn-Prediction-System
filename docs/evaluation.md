# Evaluation provenance

Source: `notebooks/02_modeling.ipynb` as checked out for the portfolio review.

The notebook splits the supplied data into 90% training, 5% validation, and 5% test, using stratification and `random_state=1`. Threshold 0.62 is selected using validation probabilities from `best_logistic`. Saved reports show validation accuracy 0.77 and held-out test accuracy 0.70; both evaluation sets contain 250 rows (156 No, 94 Yes). The test report's churn recall is 0.54 and F1 is 0.58 after rounding.

A saved validation ROC-AUC near 0.80 supports a validation-specific statement only. Neither a score for the current full-data artifact nor absence of overfitting or probability calibration is established by those outputs. Saved outputs may be stale; the portfolio review did not rerun hyperparameter searches or claim a new benchmark.

`src/train.py` fits all 5,000 rows and saves a serving artifact. Its purpose is inference preparation; it does not perform a held-out evaluation. A future evaluator should preserve the test split, serialize its split configuration and selected hyperparameters, and output a machine-readable metrics file before fitting a separate full-data serving artifact.
