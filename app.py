import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set your chosen threshold
THRESHOLD = 0.35

st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Fraud Transaction Detector")
st.markdown("Enter transaction details to check for potential fraud.")

# Input form
with st.form("fraud_form"):
    customer_id = st.number_input("Customer ID", min_value=0)
    terminal_id = st.number_input("Terminal ID", min_value=0)
    tx_amount = st.number_input("Transaction Amount", min_value=0.0)
    tx_time_seconds = st.number_input("Transaction Time (Seconds)", min_value=0)
    tx_time_days = st.number_input("Transaction Time (Days)", min_value=0)
    tx_hour = st.slider("Transaction Hour", 0, 23)
    tx_day = st.slider("Transaction Day", 1, 31)
    tx_weekday = st.slider("Transaction Weekday", 0, 6)
    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    input_data = pd.DataFrame([[
        customer_id, terminal_id, tx_amount, tx_time_seconds, tx_time_days,
        tx_hour, tx_day, tx_weekday
    ]], columns=[
        "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_TIME_SECONDS",
        "TX_TIME_DAYS", "TX_HOUR", "TX_DAY", "TX_WEEKDAY"
    ])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Get predicted probability
    fraud_proba = model.predict_proba(input_scaled)[0][1]
    is_fraud = fraud_proba >= THRESHOLD

    # Show result
    st.markdown(f"**Fraud Probability:** `{fraud_proba:.2f}`")
    if is_fraud:
        st.error("⚠️ This transaction is predicted to be **FRAUDULENT**.")
    else:
        st.success("✅ This transaction is predicted to be **LEGITIMATE**.")
