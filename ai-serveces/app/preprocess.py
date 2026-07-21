from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_PATH = BASE_DIR / "model" / "feature_names.pkl"
def preprocess_data(df):

    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # Feature Engineering

    df["monthly_income"] = df["annual_income"] / 12

    df["estimated_debt"] = (
        df["annual_income"] *
        df["debt_to_income_ratio"]
    )

    df["loan_to_income_ratio"] = (
        df["loan_amount"] /
        (df["annual_income"] + 1)
    )

    df["estimated_interest_cost"] = (
        df["loan_amount"] *
        df["interest_rate"] / 100
    )

    df["disposable_income_estimate"] = (
        df["annual_income"] -
        df["estimated_debt"]
    )

    df["affordability_index"] = (
        df["disposable_income_estimate"] /
        (df["loan_amount"] + 1)
    )

    df["income_after_debt_ratio"] = (
        df["disposable_income_estimate"] /
        (df["annual_income"] + 1)
    )

    df["credit_debt_score"] = (
        df["credit_score"] /
        (df["debt_to_income_ratio"] + 0.01)
    )

    df["loan_pressure"] = (
        df["loan_amount"] /
        (df["monthly_income"] + 1)
    )

    df["income_loan_ratio"] = (
        df["annual_income"] /
        (df["loan_amount"] + 1)
    )

    features = [
        "annual_income",
        "monthly_income",
        "credit_score",
        "debt_to_income_ratio",
        "loan_amount",
        "interest_rate",
        "estimated_debt",
        "loan_to_income_ratio",
        "estimated_interest_cost",
        "disposable_income_estimate",
        "affordability_index",
        "income_after_debt_ratio",
        "credit_debt_score",
        "loan_pressure",
        "income_loan_ratio",
        "employment_status",
        "loan_purpose"
    ]

    if "loan_paid_back" in df.columns:
        features.append("loan_paid_back")

    df = df[features]

    # One-hot encode
    df = pd.get_dummies(
        df,
        columns=[
            "employment_status",
            "loan_purpose"
        ],
        drop_first=False
    )
    # TRAINING MODE
    if "loan_paid_back" in df.columns:

        X = df.drop(columns=["loan_paid_back"])

        y = df["loan_paid_back"]

        return X, y

    # PREDICTION MODE

    expected_columns = joblib.load(FEATURE_PATH)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    return df, None