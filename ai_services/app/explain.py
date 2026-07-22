import shap
import pandas as pd
import numpy as np


def explain_prediction(model, input_data):

    """
    Explain model prediction using SHAP
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_data)


    # Random Forest classifier returns list
    if isinstance(shap_values, list):

        values = shap_values[1][0]

    else:

        values = shap_values[0]


    explanation = pd.DataFrame(
        {
            "feature": input_data.columns,
            "impact": values
        }
    )


    explanation["absolute_impact"] = (
        explanation["impact"]
        .abs()
    )


    explanation = explanation.sort_values(
        by="absolute_impact",
        ascending=False
    )


    positive = []
    negative = []


    for _, row in explanation.head(5).iterrows():

        feature = (
            row["feature"]
            .replace("_"," ")
        )


        if row["impact"] > 0:

            positive.append(
                f"{feature} improves repayment probability"
            )

        else:

            negative.append(
                f"{feature} increases financial risk"
            )


    return {

        "key_positive_factors": positive,

        "key_risk_factors": negative,

        "top_features": 
            explanation.head(5)
            [["feature","impact"]]
            .to_dict("records")
    }