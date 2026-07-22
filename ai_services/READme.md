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

## Technologies Used

Python

FastAPI

Uvicorn

Pydantic

Scikit-learn

SHAP

Pandas

Joblib

# Features

AI-powered credit risk prediction

EthiScore calculation

Risk level classification

Explainable AI (SHAP)

Human-readable financial recommendations

Interactive API documentation

REST API for easy integration



# pictures

<img width="1021" height="680" alt="p1" src="https://github.com/user-attachments/assets/c25d5f04-2195-404d-b7a0-8a3691747643" />                


<img width="1024" height="685" alt="p2" src="https://github.com/user-attachments/assets/da2b0658-7331-496e-ac1c-d7f4b9a98756" />                    



<img width="1020" height="689" alt="p3" src="https://github.com/user-attachments/assets/381d92de-fd2e-4a96-be45-b244e0813f3b" />


<img width="963" height="352" alt="p4" src="https://github.com/user-attachments/assets/e0178e36-ac79-4f41-8c4c-dd5aac2e39c3" />



<img width="972" height="414" alt="p5" src="https://github.com/user-attachments/assets/ac172914-4f4c-427b-bd5f-f4694e09e8c9" />



<img width="982" height="369" alt="p6" src="https://github.com/user-attachments/assets/122301dc-f8f8-4e41-806b-779c8588d1df" />



<img width="1024" height="657" alt="p7" src="https://github.com/user-attachments/assets/e849eb93-4009-402c-aab7-af22e50c0f57" />



<img width="1023" height="692" alt="p8" src="https://github.com/user-attachments/assets/eb02ce53-7e24-4b46-a5f8-ae8c7823bcc6" />


<img width="1019" height="690" alt="p9" src="https://github.com/user-attachments/assets/f0dd26bc-ce07-46d0-bd30-d5087c35d2bf" />


