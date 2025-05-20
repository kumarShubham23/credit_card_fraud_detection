import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("model/xgboost_fraud_model.pkl")

# Features expected by model
all_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                'V28', 'Amount']

# Mapping user-friendly names to feature codes
feature_name_map = {
    "Transaction Velocity (V14)": "V14",
    "Amount Deviation (V12)": "V12",
    "Transaction Frequency (V10)": "V10",
    "Account Age (V17)": "V17",
    "Transaction Amount (₹)": "Amount"
}

# Dropdown mappings for V14, V10, V17
velocity_map = {
    "Low": -5.0,
    "Medium": 0.0,
    "High": 5.0
}

amount_deviation_map = {
    "Low": -10.0,
    "Medium": 0.0,
    "High": 10.0
}

frequency_map = {
    "Rare": -5.0,
    "Normal": 0.0,
    "Frequent": 5.0
}

account_age_map = {
    "New": -5.0,
    "Established": 0.0,
    "Old": 5.0
}

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered", page_icon="💳")

st.title("💳 Credit Card Fraud Detection System")
st.markdown("""
Predict if a transaction is fraudulent using a trained **XGBoost** model.
Use the dropdowns and sliders on the sidebar to input transaction features.
""")

# Sidebar inputs with friendly names
st.sidebar.header("🔧 Input Transaction Features")

def get_user_input():
    inputs = {}
    for label, feature_code in feature_name_map.items():
        if feature_code == "Amount":
            inputs[feature_code] = st.sidebar.number_input(label, min_value=0.0, max_value=100000.0, value=100.0, step=10.0)
        elif feature_code == "V14":  # Transaction Velocity dropdown
            selected = st.sidebar.selectbox(label, options=list(velocity_map.keys()))
            inputs[feature_code] = velocity_map[selected]
        elif feature_code == "V10":  # Transaction Frequency dropdown
            selected = st.sidebar.selectbox(label, options=list(frequency_map.keys()))
            inputs[feature_code] = frequency_map[selected]
        elif feature_code == "V17":  # Account Age dropdown
            selected = st.sidebar.selectbox(label, options=list(account_age_map.keys()))
            inputs[feature_code] = account_age_map[selected]
        elif feature_code == "V12":  # Amount Deviation dropdown
            selected = st.sidebar.selectbox(label, options=list(amount_deviation_map.keys()))
            inputs[feature_code] = amount_deviation_map[selected]
        else:
            inputs[feature_code] = st.sidebar.slider(label, -20.0, 20.0, 0.0, step=0.1)
    
    return inputs  # ✅ Make sure this is at the end


user_input = get_user_input()

# Create DataFrame with all zeros
input_data = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)

# Fill user input values in DataFrame
for feature, value in user_input.items():
    input_data.at[0, feature] = value

with st.expander("🔍 View Input Transaction Data"):
    st.write(input_data)

if st.button("🚀 Predict Transaction"):
    prediction = model.predict(input_data)
    
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

st.markdown("""
---
<p style='text-align:center; color:gray; font-size:14px;'>
Made with ❤️ by Shubham | Model: XGBoost
</p>
""", unsafe_allow_html=True)
