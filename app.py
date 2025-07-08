import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
THRESHOLD = 0.35

# Page config
st.set_page_config(page_title="Fraud Detection App", layout="centered", page_icon="💳")

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💳 Fraud Transaction Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>A smart machine learning model to detect suspicious transactions in real-time.</p>", unsafe_allow_html=True)
st.markdown("---")

# Form layout
st.subheader("📋 Transaction Details")

with st.form("fraud_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        customer_id = st.number_input("👤 Customer ID", min_value=0)
        tx_amount = st.number_input("💵 Transaction Amount", min_value=0.0)
        tx_time_seconds = st.number_input("⏱️ TX Time Seconds", min_value=0)
        tx_hour = st.slider("🕐 Transaction Hour", 0, 23)
    
    with col2:
        terminal_id = st.number_input("🏪 Terminal ID", min_value=0)
        tx_time_days = st.number_input("📅 TX Time Days", min_value=0)
        tx_day = st.slider("📆 Day of Month", 1, 31)
        tx_weekday = st.slider("📅 Weekday (0=Mon)", 0, 6)

    submitted = st.form_submit_button("🔍 Predict")

# Prediction
if submitted:
    input_data = pd.DataFrame([[
        customer_id, terminal_id, tx_amount, tx_time_seconds, tx_time_days,
        tx_hour, tx_day, tx_weekday
    ]], columns=[
        "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_TIME_SECONDS",
        "TX_TIME_DAYS", "TX_HOUR", "TX_DAY", "TX_WEEKDAY"
    ])

    input_scaled = scaler.transform(input_data)
    fraud_proba = model.predict_proba(input_scaled)[0][1]
    is_fraud = fraud_proba >= THRESHOLD

    st.markdown("---")
    st.subheader("🧠 Prediction Result")

    st.markdown(f"**Fraud Probability:** `{fraud_proba:.2f}`")

    if is_fraud:
        st.error("⚠️ This transaction is predicted to be **FRAUDULENT**.")
    else:
        st.success("✅ This transaction is predicted to be **LEGITIMATE**.")
