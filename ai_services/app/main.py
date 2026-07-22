from fastapi import FastAPI

from .schemas import ApplicantRequest, PredictionResponse
from .predict import predict_credit

app = FastAPI(
    title="EthiScore AI API",
    version="1.0.0",
    description="Explainable AI Credit Intelligence Engine for SMEs"
)


@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {
        "message": "Welcome to EthiScore AI API"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantRequest):
    """
    Generate an EthiScore prediction for an applicant.
    """
    result = predict_credit(applicant.model_dump())
    return result