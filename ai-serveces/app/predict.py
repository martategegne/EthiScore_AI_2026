import sys
from pathlib import Path
import pandas as pd
import joblib

from explain import explain_prediction
from xai_rules import generate_xai_advice


# Configure project path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.preprocess import preprocess_data
from app.credit_engine import calculate_ethiscore


# Load trained credit prediction model
MODEL_PATH = BASE_DIR / "model" / "credit_model.pkl"
model = joblib.load(MODEL_PATH)


def predict_credit(applicant_data):
    """
    Predict EthiScore for one applicant.
    """

    # Convert applicant data into dataframe
    df = pd.DataFrame([applicant_data])

    # Apply preprocessing pipeline
    processed_data, _ = preprocess_data(df)

    # Display processed features for validation
    print(processed_data.columns.tolist())

    # Generate EthiScore assessment
    result = calculate_ethiscore(processed_data)

    # Generate Explainable AI insights
    explanation = explain_prediction(
        model,
        processed_data
    )

    # Add explanation results to EthiScore report
    result.update(explanation)

    # Generate financial improvement recommendations
    result["xai_advice"] = generate_xai_advice(
        explanation["key_risk_factors"]
    )

    return result


if __name__ == "__main__":

    applicant = {
        "annual_income": 50000,
        "credit_score": 520,
        "debt_to_income_ratio": 0.65,
        "loan_amount": 120000,
        "interest_rate": 18,
        "employment_status": "Unemployed",
        "loan_purpose": "Debt consolidation"
    }

    result = predict_credit(applicant)

    print("\nEthiScore Result:")
    print(result)