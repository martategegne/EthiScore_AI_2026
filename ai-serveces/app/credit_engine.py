{
    "ethi_score": 87,

    "score_category": "Very Strong",

    "repayment_probability": 91.3,

    "confidence": 94.1,

    "risk_level": "Low",

    "creditworthiness": "Strong",

    "financial_health": "Healthy",

    "loan_suitability": "Suitable for Standard Financing",

    "decision_support": "Recommend Standard Review",

    "strengths": [],

    "areas_to_improve": [],

    "recommendations": [],

    "key_risk_factors": [],

    "positive_factors": [],

    "summary_for_sme": "",

    "summary_for_lender": "",

    "next_best_actions": [],

    "explanation": ""
}
from pathlib import Path
import joblib
# Load trained model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "credit_model.pkl"

model = joblib.load(MODEL_PATH)
# EthiScore Table
ETHISCORE_TABLE = [
    (95, 100, "Exceptional", "Very Low"),
    (90, 94, "Excellent", "Very Low"),
    (85, 89, "Very Strong", "Low"),
    (80, 84, "Strong", "Low"),
    (75, 79, "Good", "Moderate-Low"),
    (70, 74, "Fairly Good", "Moderate"),
    (65, 69, "Fair", "Moderate"),
    (60, 64, "Watch List", "Moderate-High"),
    (55, 59, "Needs Improvement", "High"),
    (50, 54, "Weak", "High"),
    (45, 49, "Poor", "Very High"),
    (40, 44, "Very Poor", "Very High"),
    (35, 39, "Critical Risk", "Severe"),
    (30, 34, "Extremely High Risk", "Severe"),
    (25, 29, "Financial Distress", "Critical"),
    (20, 24, "Unsustainable", "Critical"),
    (15, 19, "Near Default Risk", "Extreme"),
    (10, 14, "Default Warning", "Extreme"),
    (5, 9, "Imminent Default Risk", "Extreme"),
    (0, 4, "Financial Emergency", "Extreme"),
]

# Convert score to category
def get_score_category(score: int):
    """
    Returns:
        category
        risk_level
    """
    for low, high, category, risk in ETHISCORE_TABLE:
        if low <= score <= high:
            return category, risk

    return "Unknown", "Unknown"

def get_risk_level(score: int) -> str:
    """
    Classify applicant risk level based on EthiScore.

    Parameters
    ----------
    score : int
        EthiScore (0–100)

    Returns
    -------
    str
        Low, Moderate, or High
    """

    if score >= 85:
        return "Low"
    elif score >= 70:
        return "Moderate"
    else:
        return "High"
def identify_strengths(preprocessed_data):
    """
    Identify the applicant's financial strengths.

    Parameters
    ----------
    preprocessed_data : pandas.DataFrame
        Preprocessed applicant features (one row).

    Returns
    -------
    list
        List of positive financial strengths.
    """

    strengths = []

    applicant = preprocessed_data.iloc[0]

    # High credit score
    if applicant.get("credit_score", 0) > 700:
        strengths.append("High credit score")

    # Healthy debt ratio
    if applicant.get("debt_to_income_ratio", 1) < 0.30:
        strengths.append("Healthy debt ratio")

    # Stable employment
    if applicant.get("employment_status_Employed", 0) == 1:
        strengths.append("Stable employment")

    # Strong repayment capacity
    if applicant.get("affordability_index", 0) > 3:
        strengths.append("Strong repayment capacity")

    return strengths   
def identify_areas_to_improve(preprocessed_data):
    """
    Identify financial areas that could be improved.

    Parameters
    ----------
    preprocessed_data : pandas.DataFrame
        Preprocessed applicant features (one row).

    Returns
    -------
    list
        List of improvement areas.
    """

    areas = []

    applicant = preprocessed_data.iloc[0]

    # Credit score
    if applicant.get("credit_score", 0) < 620:
        areas.append("Credit score needs improvement.")

    # Loan-to-income ratio
    if applicant.get("loan_to_income_ratio", 0) > 0.50:
        areas.append("Requested loan is high relative to income.")

    # Debt-to-income ratio
    if applicant.get("debt_to_income_ratio", 0) > 0.45:
        areas.append("Debt burden is relatively high.")

    return areas
def generate_recommendations(preprocessed_data):
    """
    Generate personalized financial recommendations.

    Parameters
    ----------
    preprocessed_data : pandas.DataFrame
        Preprocessed applicant features (one row).

    Returns
    -------
    list
        List of actionable recommendations.
    """

    recommendations = []

    applicant = preprocessed_data.iloc[0]

    if applicant.get("debt_to_income_ratio", 0) > 0.45:
        recommendations.append(
            "Reduce existing debt before applying for larger financing."
        )

    if applicant.get("affordability_index", 10) < 2:
        recommendations.append(
            "Increase monthly savings to improve repayment capacity."
        )

    if applicant.get("credit_score", 1000) < 620:
        recommendations.append(
            "Maintain timely repayment of existing loans to improve your credit score."
        )

    if applicant.get("loan_to_income_ratio", 0) > 0.50:
        recommendations.append(
            "Request a smaller loan amount initially."
        )

    if applicant.get("employment_status_Employed", 0) == 0:
        recommendations.append(
        "Provide additional business income documentation."
    )

    return recommendations
def generate_lender_summary(
    ethi_score,
    risk_level,
    strengths,
    areas_to_improve,
):
    """
    Generate a concise decision-support summary
    for the financial institution.

    Parameters
    ----------
    ethi_score : int
    risk_level : str
    strengths : list
    areas_to_improve : list

    Returns
    -------
    str
    """

    if ethi_score >= 85:
        return (
            "Applicant demonstrates strong repayment potential "
            "with positive financial indicators. "
            "Recommend standard review procedures."
        )

    if ethi_score >= 70:
        return (
            "Applicant has moderate repayment potential. "
            "Consider requesting additional financial documentation "
            "before making a lending decision."
        )

    return (
        "Applicant shows elevated financial risk. "
        "Additional assessment is recommended before "
        "offering higher-value financing."
    )

def assess_creditworthiness(score):

    if score >= 90:
        return "Excellent"

    elif score >= 75:
        return "Good"

    elif score >= 60:
        return "Average"

    elif score >= 45:
        return "Limited"

    return "Poor" 

def assess_financial_health(data):

    credit_score = float(data["credit_score"].iloc[0])
    dti = float(data["debt_to_income_ratio"].iloc[0])

    if credit_score >= 750 and dti <= 0.30:
        return "Healthy"

    elif credit_score >= 650 and dti <= 0.45:
        return "Stable"

    elif credit_score >= 550:
        return "Needs Improvement"

    return "Financially Vulnerable"

def assess_loan_suitability(score):

    if score >= 85:
        return "Highly Suitable"

    elif score >= 70:
        return "Suitable"

    elif score >= 55:
        return "Suitable with Conditions"

    return "Currently Not Suitable"

def generate_sme_summary(score, strengths, improvements):

    summary = (
        f"Your EthiScore is {score}. "
    )

    if score >= 85:
        summary += (
            "Your financial profile is excellent. "
        )

    elif score >= 70:
        summary += (
            "Your financial profile is strong. "
        )

    elif score >= 55:
        summary += (
            "Your profile is acceptable but has room for improvement. "
        )

    else:
        summary += (
            "Your financial profile needs significant improvement. "
        )

    if strengths:

        summary += (
            "Your strengths include: "
            + ", ".join(strengths)
            + ". "
        )

    if improvements:

        summary += (
            "Focus on improving: "
            + ", ".join(improvements)
            + "."
        )

    return summary

# Main EthiScore Calculation
def calculate_ethiscore(preprocessed_data):
    """
    Calculate complete EthiScore Credit Intelligence Report.

    Parameters
    ----------
    preprocessed_data
        Preprocessed applicant features.

    Returns
    -------
    dict
        Complete EthiScore report.
    """

    # Predict repayment probability
    probability = float(
        model.predict_proba(preprocessed_data)[0][1]
    )

    # Calculate EthiScore
    ethi_score = int(
        round(probability * 100)
    )

    # Determine score category and risk level
    score_category, _ = get_score_category(
        ethi_score
    )

    risk_level = get_risk_level(
        ethi_score
    )

    # Calculate prediction confidence
    confidence = round(
        max(probability, 1 - probability) * 100,
        2
    )

    # Assess applicant profile
    creditworthiness = assess_creditworthiness(
        ethi_score
    )

    financial_health = assess_financial_health(
        preprocessed_data
    )

    loan_suitability = assess_loan_suitability(
        ethi_score
    )

    # Generate explainability
    strengths = identify_strengths(
        preprocessed_data
    )

    areas_to_improve = identify_areas_to_improve(
        preprocessed_data
    )

    recommendations = generate_recommendations(
        preprocessed_data
    )

    # Generate summaries
    summary_for_lender = generate_lender_summary(
        ethi_score,
        risk_level,
        strengths,
        areas_to_improve,
    )

    summary_for_sme = generate_sme_summary(
        ethi_score,
        strengths,
        areas_to_improve,
    )

    # Return complete report
    return {

        "ethi_score": ethi_score,

        "score_category": score_category,

        "repayment_probability": round(
            probability * 100,
            2
        ),

        "confidence": confidence,

        "risk_level": risk_level,

        "creditworthiness": creditworthiness,

        "financial_health": financial_health,

        "loan_suitability": loan_suitability,

        "strengths": strengths,

        "areas_to_improve": areas_to_improve,

        "recommendations": recommendations,

        "summary_for_lender": summary_for_lender,

        "summary_for_sme": summary_for_sme,
    }