from pydantic import BaseModel


class ApplicantRequest(BaseModel):
    annual_income: float
    credit_score: int
    debt_to_income_ratio: float
    loan_amount: float
    interest_rate: float
    employment_status: str
    loan_purpose: str


class PredictionResponse(BaseModel):
    ethi_score: int
    score_category: str
    repayment_probability: float
    confidence: float
    risk_level: str
    creditworthiness: str
    financial_health: str
    loan_suitability: str
    strengths: list[str]
    areas_to_improve: list[str]
    recommendations: list[str]
    summary_for_lender: str
    summary_for_sme: str
    key_positive_factors: list[str]
    key_risk_factors: list[str]
    top_features: list[dict]
    xai_advice: list[str]