import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
THRESHOLD = 0.35

# Streamlit config
st.set_page_config(page_title="Fraud Detector", layout="centered", page_icon="💳")

# Sidebar
with st.sidebar:
    st.markdown("### 👤 Made by [M ABHISHEK](https://github.com/Abhishek071104)")
    st.markdown("🔗[GitHub](https://github.com/Abhishek071104)")
    st.markdown("🔗[LinkedIn](https://www.linkedin.com/in/-mabhishek/)")
    st.markdown("📧 [manipatruniabhishek07@gmail.com](mailto:manipatruniabhishek07@gmail.com)")
    st.markdown("---")
    st.markdown("### 🔎 About This App")
    st.write("This Streamlit application uses a trained **XGBoost model** to detect potentially fraudulent transactions based on historical behavioral patterns and transaction features.")

# Title
st.title("💳 Fraud Transaction Detector")
st.markdown("Enter transaction details below to check whether it's **fraudulent or legitimate**.")

# Session state for prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Tabs: Predict & History
tab1, tab2 = st.tabs(["🔮 Predict", "📜 History"])

with tab1:
    st.subheader("🧾 Transaction Information")

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

    if st.button("🔍 Predict Fraud"):
        # Prepare input
        input_data = pd.DataFrame([[
            customer_id, terminal_id, tx_amount, tx_time_seconds, tx_time_days,
            tx_hour, tx_day, tx_weekday
        ]], columns=[
            "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_TIME_SECONDS",
            "TX_TIME_DAYS", "TX_HOUR", "TX_DAY", "TX_WEEKDAY"
        ])

        # Loading bar
        progress_text = "Running fraud prediction..."
        progress = st.progress(0, text=progress_text)
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i + 1, text=progress_text)

        # Predict
        scaled_input = scaler.transform(input_data)
        fraud_proba = model.predict_proba(scaled_input)[0][1]
        is_fraud = fraud_proba >= THRESHOLD

        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.markdown(f"**Fraud Probability:** `{fraud_proba:.2f}`")

        if is_fraud:
            st.error("⚠️ This transaction is likely **FRAUDULENT**.")
        else:
            st.success("✅ This transaction is likely **LEGITIMATE**.")

        # Save to session history
        st.session_state.history.append({
            "Customer ID": customer_id,
            "Terminal ID": terminal_id,
            "Amount ($)": tx_amount,
            "Hour": tx_hour,
            "Weekday": tx_weekday,
            "Fraud Probability": round(fraud_proba, 2),
            "Prediction": "FRAUD" if is_fraud else "LEGIT"
        })

with tab2:
    st.subheader("📜 Prediction History")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history).iloc[::-1]
        st.dataframe(df_history, use_container_width=True)

        csv = df_history.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download as CSV", csv, "fraud_history.csv", "text/csv")

        if st.button("🧹 Clear History"):
            st.session_state.history = []
            st.success("Prediction history cleared.")
    else:
        st.info("No predictions yet.")

# Optional footer banner
st.image("fraud_banner.png", use_container_width=True)
