def generate_xai_advice(risk_factors):

    advice = []


    for factor in risk_factors:


        if "credit score" in factor.lower():

            advice.append(
                "Improve credit history by making loan repayments on time."
            )


        elif "debt" in factor.lower():

            advice.append(
                "Reduce existing debt obligations before applying for additional financing."
            )


        elif "loan amount" in factor.lower():

            advice.append(
                "Consider requesting a loan amount that matches your income capacity."
            )


        elif "income" in factor.lower():

            advice.append(
                "Increase stable income sources and maintain proper financial records."
            )


    return advice