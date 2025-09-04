import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Credit Card Fraud Detection App")

st.write("Enter transaction details below:")

# --- Only 2 inputs for user ---
time = st.number_input("Transaction Time (seconds)", step=1, value=0)
amount = st.number_input("Transaction Amount ($)", step=0.01, value=0.0)

# --- Fill V1–V28 with zeros automatically ---
v_features = [0.0] * 28
features = np.array([[time, amount] + v_features])

# --- Predict Button ---
if st.button("Check Fraud"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1] * 100  # fraud probability
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Fraud Probability: {prob:.2f}%)")
    else:
        st.success(f"✅ Legit Transaction (Fraud Probability: {prob:.2f}%)")

# --- Sample Transaction Button ---
if st.button("Use Sample Transaction"):
    sample = [2, 0.02] + [0.0]*28  # Example sample
    features = np.array([sample])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1] * 100
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Fraud Probability: {prob:.2f}%)")
    else:
        st.success(f"✅ Legit Transaction (Fraud Probability: {prob:.2f}%)")




