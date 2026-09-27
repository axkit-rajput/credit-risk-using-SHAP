# Credit Ledger — Explainable Credit Risk Assessment using SHAP

An end-to-end machine learning project that predicts **loan default risk** and explains every prediction using **SHAP (SHapley Additive exPlanations)**. The trained model is served through a **FastAPI** backend with a polished browser-based underwriting interface.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [License](#license)

---

## Overview

Banks and lending institutions need to decide quickly whether a loan applicant is likely to default. This project builds a **binary classifier** on consumer-loan data, tunes it rigorously, and then goes a step further — using **SHAP** to make every prediction **transparent and explainable**.

The final model is wrapped in a lightweight **FastAPI** service with a static frontend ("Credit Ledger") so anyone can fill in applicant details and receive a real-time risk assessment with a visual gauge and verdict stamp.

---

## Demo

| Input Form | Risk Verdict |
|:---:|:---:|
| Fill in applicant & loan details across three sections | An animated gauge and stamp show the default probability and decision |

> The app is deployable on [Render](https://render.com) (config included) or any platform that supports Python web services.

---

## Features

- **Full ML notebook** — from EDA through model comparison, threshold optimization, probability calibration, to SHAP interpretation
- **XGBoost + Logistic Regression** comparison with stratified cross-validation
- **Class imbalance handling** via scale weights
- **Hyperparameter tuning** with `RandomizedSearchCV`
- **Optimal classification threshold** selection (not just 0.5)
- **Probability calibration** to ensure predicted probabilities are well-calibrated
- **SHAP explainability** — global feature importance + local per-applicant breakdowns
- **False positive / false negative analysis**
- **FastAPI REST API** with Pydantic validation
- **Responsive dark-themed frontend** with animated risk gauge and verdict stamp
- **One-click deploy** to Render via `render.yaml`

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML / Data Science | Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP |
| Backend | FastAPI, Uvicorn, Pydantic, Joblib |
| Frontend | HTML5, CSS3 (custom design tokens), Vanilla JS |
| Deployment | Render (render.yaml) |
| Runtime | Python 3.11.9 |

---

## Project Structure

```
Credit-Risk-Assesment-using-SHAP-main/
│
├── Credit_Risk.ipynb          # Full ML pipeline notebook (EDA → SHAP)
├── credit_risk_dataset.csv    # Raw dataset
│
├── credit_risk_model.pkl      # Trained XGBoost pipeline (serialized)
├── best_threshold.pkl         # Optimal classification threshold
│
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python version for deployment
├── render.yaml                # Render deployment configuration
│
├── static/
│   ├── index.html             # Credit Ledger UI
│   ├── style.css              # Dark-themed responsive styles
│   └── script.js              # Form handling, gauge animation, API calls
│
└── .gitignore
```

---

## ML Pipeline

The Jupyter notebook (`Credit_Risk.ipynb`) walks through **16 clearly documented steps**:

| # | Step | Description |
|---|---|---|
| 1 | **Load Data** | Import the credit risk dataset |
| 2 | **EDA** | Distributions, target balance, missing values, outliers, correlations |
| 3 | **Data Validation** | Remove duplicates, impossible ages (>100), excessive employment length (>60) |
| 4 | **Train-Test Split** | Stratified split preserving class ratios |
| 5 | **Class Imbalance** | Compute scale weights for the minority class |
| 6 | **Preprocessing Pipelines** | Separate `ColumnTransformer` pipelines for Logistic Regression (with scaling) and XGBoost (without scaling) |
| 7 | **Evaluation Helper** | Unified function for accuracy, precision, recall, F1, confusion matrix |
| 8 | **Cross-Validation** | 5-fold stratified CV on both models |
| 9 | **Baseline Model** | Logistic Regression as benchmark |
| 10 | **Hyperparameter Tuning** | `RandomizedSearchCV` on XGBoost |
| 11 | **Model Comparison** | Side-by-side metric comparison |
| 12 | **Threshold Optimization** | Sweep thresholds to balance FP/FN for the business case |
| 13 | **Probability Calibration** | Calibration curves + `CalibratedClassifierCV` |
| 14 | **SHAP Interpretation** | Global summary plot + local waterfall for individual applicants |
| 15 | **Error Analysis** | Inspect false positives and false negatives |
| 16 | **Save Model** | Export model pipeline + threshold as `.pkl` files |

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- `pip` package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Credit-Risk-Assesment-using-SHAP.git
cd Credit-Risk-Assesment-using-SHAP

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The Credit Ledger UI will load automatically.

---

## API Reference

### `POST /predict`

Predict the default risk for a loan application.

**Request body** (JSON):

```json
{
  "person_age": 30,
  "person_income": 600000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 100000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

**Response** (JSON):

```json
{
  "default_probability": 0.042,
  "default_prediction": 0,
  "threshold": 0.35,
  "Result": "Low Risk"
}
```

| Field | Type | Description |
|---|---|---|
| `person_age` | `int` | Applicant's age (18–100) |
| `person_income` | `float` | Annual income |
| `person_home_ownership` | `str` | `RENT`, `MORTGAGE`, `OWN`, or `OTHER` |
| `person_emp_length` | `float` | Employment length in years |
| `loan_intent` | `str` | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| `loan_grade` | `str` | Loan grade (`A` through `G`) |
| `loan_amnt` | `float` | Requested loan amount |
| `loan_int_rate` | `float` | Interest rate (%) |
| `loan_percent_income` | `float` | Loan amount as a ratio of annual income (0–1) |
| `cb_person_default_on_file` | `str` | Prior default on credit bureau file (`Y` / `N`) |
| `cb_person_cred_hist_length` | `int` | Credit history length in years |

---

## Deployment

The project includes a [`render.yaml`](render.yaml) for one-click deployment to **Render**:

```yaml
services:
  - type: web
    name: credit-ledger
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    autoDeploy: true
```

1. Push the repo to GitHub
2. Connect your Render account to the repo
3. Render will auto-detect the `render.yaml` and deploy

---

## License

This project is open-source. Feel free to use, modify, and distribute.
