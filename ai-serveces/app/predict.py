import sys
from pathlib import Path
import pandas as pd

# Add ai-serveces to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.preprocess import preprocess_data
from app.credit_engine import calculate_ethiscore


def predict_credit(applicant_data):
    """
    Predict EthiScore for one applicant.

    Parameters
    ----------
    applicant_data : dict
        Applicant financial information.

    Returns
    -------
    dict
        EthiScore report.
    """

    # Convert input dictionary to dataframe
    df = pd.DataFrame([applicant_data])

    # Apply preprocessing
    processed_data, _ = preprocess_data(df)

    # Debug: check processed features
    print(processed_data.columns.tolist())

    # Send processed data to EthiScore engine
    result = calculate_ethiscore(processed_data)

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