# ML Practice 1 – Bank Deposit Subscription Prediction

Machine Learning coursework (2025-2026) at Universidad Carlos III de Madrid.

## Problem

Binary classification task: predict whether a bank client will subscribe
to a term deposit, based on demographic, financial, and interaction data.
Target variable: `deposit` (yes/no).

## Dataset
- `bank_xx.pkl` — training data (EDA, HPO, model selection, evaluation)
- `bank_competition_xx.pkl` — unlabelled competition data for final predictions

## Pipeline

1. **EDA** — variable types, missing values, class balance, `pdays` treatment
2. **Evaluation strategy** — Holdout (2/3 train / 1/3 test) + CV=3 inner
3. **Basic methods** — KNN and Decision Trees
   - Scaling comparison: MinMax, Standard, Robust
   - Imputation: mean vs. median
   - Default hyperparameters + HPO
   - Hyperparameter effect plots
4. **Advanced methods** — Logistic Regression (L1), SVM
   - Default + HPO
   - Feature importance analysis
5. **Model selection** — best model chosen via inner evaluation
6. **Final model** — trained on full data, saved as `modelo_final.joblib`
7. **Streamlit deployment** — web app for real-time predictions on new clients

## Deliverables
- `notebook_main.ipynb` — EDA, HPO, model selection
- `notebook_predictions.ipynb` — load final model, generate competition predictions
- `mystreamlit.py` — Streamlit web application
- `modelo_final.joblib` — serialized final pipeline
- `predicciones.csv` — predictions for competition dataset

## Stack
Python · scikit-learn · pandas · numpy · matplotlib · Streamlit · Jupyter
