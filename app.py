
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION AND ASSET LOADING ---
st.set_page_config(page_title="Lender Recommendation", layout="wide")
st.title("💸 Loan Recommendation Engine")

try:
    # Note: In a real environment, you must ensure model.pkl, etc., are created first!
    # Since we are simulating, we must skip the file loading which will fail.
    # In a proper Streamlit deployment, this would load the assets.

    # Placeholder variables for running in Colab without local files:
    TRAINING_COLUMNS = ['FICO_score', 'Debt_To_Income', 'Reason_Home Improvement', 'Reason_Other', 'Employment_Status_Self-Employed', 'Employment_Status_Unemployed', 'Employment_Sector_Finance', 'Employment_Sector_Healthcare', 'Employment_Sector_Tech', 'Employment_Sector_Unknown', 'Lender_B', 'Lender_C']
    BOUNTY_MAP = {'A': 0, 'B': 100, 'C': 250}

    # Simulating a basic prediction function for demonstration in a limited environment
    # In a real app, this would use the loaded model and scaler.
    def model_predict(X_final):
        # Simplistic probability based on FICO and DTI for demonstration
        fico = X_final['FICO_score'].iloc[0]
        dti = X_final['Debt_To_Income'].iloc[0]
        prob = np.clip((fico / 850) - (dti * 0.1) + 0.1, 0.1, 0.9)
        return np.array([[1 - prob, prob]])

    # Reassigning the actual model predict function placeholder for the demo
    model = type('MockModel', (object,), {'predict_proba': model_predict})()

    # Mock Scaler transform
    def mock_transform(data):
        return data

    class MockScaler:
        def transform(self, data):
            # In a real scenario, this would standardize FICO and DTI
            return data

    scaler = MockScaler()

except Exception as e:
    st.error(f"Error loading model assets (Running in mock mode): {e}")


# --- 2. PREDICTION FUNCTION (Simplified for Colab) ---
def predict_expected_payout(input_df):

    results = {}
    for lender in BOUNTY_MAP:

        customer_features = input_df.copy()

        # 1. Handle Categorical Encoding
        customer_features = pd.get_dummies(customer_features, drop_first=True)

        # 2. Add Lender Dummies
        customer_features['Lender_B'] = 1 if lender == 'B' else 0
        customer_features['Lender_C'] = 1 if lender == 'C' else 0

        # 3. Reindex and Scale (using mock scaler/model)
        X_final = customer_features.reindex(columns=TRAINING_COLUMNS, fill_value=0)
        X_final[['FICO_score', 'Debt_To_Income']] = scaler.transform(X_final[['FICO_score', 'Debt_To_Income']])

        # 4. Predict P(Approved) and Calculate Payout
        prob_approved = model.predict_proba(X_final)[:, 1][0]
        payout = prob_approved * BOUNTY_MAP[lender]
        payout_results[lender] = payout

    return payout_results


# --- 3. STREAMLIT UI ---
st.sidebar.header("Customer Profile")
fico_score = st.sidebar.slider("FICO Score", 600, 850, 720)
income = st.sidebar.number_input("Monthly Gross Income ($)", 1000, 10000, 5000)
housing = st.sidebar.number_input("Monthly Housing Payment ($)", 0, 3000, 1500)
reason = st.sidebar.selectbox("Loan Reason", ['Debt Consolidation', 'Home Improvement', 'Other'])
employment = st.sidebar.selectbox("Employment Status", ['Employed', 'Self-Employed', 'Unemployed'])
sector = st.sidebar.selectbox("Employment Sector", ['Tech', 'Finance', 'Healthcare', 'Unknown'])

if st.sidebar.button("Recommend Lender"):

    dti = housing / income if income > 0 else 0

    input_data = {
        'FICO_score': [fico_score],
        'Debt_To_Income': [dti],
        'Reason': [reason],
        'Employment_Status': [employment],
        'Employment_Sector': [sector]
    }
    input_df = pd.DataFrame(input_data)

    payout_results = predict_expected_payout(input_df)

    max_payout = max(payout_results.values())
    recommended_lender = max(payout_results, key=payout_results.get)

    st.success(f"## Recommended Lender: {recommended_lender}")
    st.metric("Maximum Expected Payout", f"${max_payout:,.2f}")

    payout_df = pd.DataFrame(payout_results.items(), columns=['Lender', 'Expected Payout'])
    payout_df['Bounty'] = payout_df['Lender'].map(BOUNTY_MAP)
    st.dataframe(payout_df.style.format({'Expected Payout': '${:,.2f}', 'Bounty': '${:,.0f}'}))
