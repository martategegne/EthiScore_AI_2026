# FastAPI API Documentation
API Overview

The EthiScore AI service exposes a REST API built with FastAPI that enables lenders and financial institutions to submit loan applicant information and receive an AI-powered credit assessment.

## The API performs the following steps:

Loads the trained credit risk model.
Preprocesses applicant information.
Predicts repayment probability.
Calculates the EthiScore.
Determines the applicant's risk level.
Generates explainable AI insights.
Returns financial recommendations.
Installation

Create and activate a virtual environment.

# Windows

python -m venv .venv
.venv\Scripts\activate

Install project dependencies.

pip install -r requirement.txt
Running the API


# Start the FastAPI development server.

## on ai_services folder
uvicorn app.main:app --reload

## or 
from root directory

uvicorn ai_services.app.main:app --reload
The API will be available at:

http://127.0.0.1:8000
Interactive API Documentation

FastAPI automatically generates API documentation.

Swagger UI:

http://127.0.0.1:8000/docs

ReDoc:

http://127.0.0.1:8000/redoc
Prediction Endpoint
POST /predict

Predicts the repayment probability and generates an EthiScore.

Request Example
{
  "annual_income": 60000,
  "credit_score": 720,
  "debt_to_income_ratio": 0.25,
  "loan_amount": 15000,
  "interest_rate": 12.5,
  "employment_status": "Employed",
  "loan_purpose": "Business"
}
Response Example
{
  "repayment_probability": 0.94,
  "ethiscore": 88,
  "risk_level": "Low",
  "strengths": [
    "Strong credit score",
    "Stable income"
  ],
  "areas_to_improve": [
    "Reduce debt-to-income ratio"
  ],
  "recommendation": "Loan application presents low repayment risk."
}
Project Structure
ai_services/
│
├── app/
│   ├── main.py
│   ├── predict.py
│   ├── credit_engine.py
│   ├── preprocess.py
│   ├── explain.py
│   ├── xai_rules.py
│   ├── schemas.py
│   └── __init__.py
│
├── model/
│   └── credit_model.pkl
│
├── data/
├── training.py
├── README.md
└── requirement.txt
Technologies Used
Python
FastAPI
Uvicorn
Pydantic
Scikit-learn
SHAP
Pandas
Joblib
Features
AI-powered credit risk prediction
EthiScore calculation
Risk level classification
Explainable AI (SHAP)
Human-readable financial recommendations
Interactive API documentation
REST API for easy integration


# pictures