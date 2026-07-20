import pandas as pd
def preprocess_data(df):
    """
    Preprocess the dataset for EthiScore AI.
    Returns:
        X -> Features
        y -> Target (None during prediction)
    """
    # Remove ID column
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
        df["annual_income"]
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
        df["loan_amount"]
    )
    # Select Features
    selected_columns = [
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
        "employment_status",
        "loan_purpose"
    ]

    # Include target only if available (training mode)
    if "loan_paid_back" in df.columns:
        selected_columns.append("loan_paid_back")

    df = df[selected_columns]
    # Encode categorical variables
    df = pd.get_dummies(
        df,
        columns=[
            "employment_status",
            "loan_purpose"
        ],
        drop_first=True
    )
    # Training Mode
    if "loan_paid_back" in df.columns:

        X = df.drop(columns=["loan_paid_back"])
        y = df["loan_paid_back"]

        return X, y

    # Prediction Mode

    return df, None
