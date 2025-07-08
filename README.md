# 💳 Fraud Transaction Detection ML App
End-to-end Machine Learning solution to detect fraudulent credit card transactions using a real-time, deployable Streamlit app. The pipeline includes data preprocessing, handling class imbalance with SMOTE, training an XGBoost model, applying threshold tuning for optimized fraud recall, and building an interactive UI for live prediction.

## 🚀 Features
- Trained on a large synthetic transaction dataset with simulated fraud scenarios
- Handles extreme class imbalance using **SMOTE**
- Uses **XGBoost** with **threshold tuning** for optimal fraud recall
- Clean and intuitive **Streamlit app interface**
- Ready for deployment and real-time testing

---

## 📁 Project Structure
fraud-detection-ml-project/
├── app.py # Streamlit UI code
├── fraud_model.pkl # Trained XGBoost model
├── scaler.pkl # StandardScaler used for input normalization
├── notebook.ipynb # Google Colab end-to-end training notebook
├── requirements.txt # Python package requirements
├── sample_inputs.csv # Example inputs for testing in Streamlit app
└── README.md # Project documentation


---

## 🧠 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, Imbalanced-learn
- Matplotlib, Seaborn (for EDA)
- Streamlit (for deployment)

---

