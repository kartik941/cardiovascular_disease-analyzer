import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ----------------------------
# App UI
# ----------------------------
st.set_page_config(page_title="Cardiovascular Disease Predictor", layout="centered")
st.title("❤️ Cardiovascular Disease Prediction App")

st.write("Enter patient details to predict cardiovascular disease risk.")

# ----------------------------
# Input fields
# ----------------------------
gender = st.selectbox("Gender", [1, 2])  # 1: Female, 2: Male
age_years = st.number_input("Age (years)", min_value=1, max_value=120, value=50)

height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=250, value=120)
ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=150, value=80)

cholesterol = st.selectbox("Cholesterol", [1, 2, 3])  
gluc = st.selectbox("Glucose", [1, 2, 3])

smoke = st.selectbox("Smoking", [0, 1])
alco = st.selectbox("Alcohol intake", [0, 1])
active = st.selectbox("Physically Active", [0, 1])

# BMI calculation
bmi = weight / ((height / 100) ** 2)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    input_data = pd.DataFrame([[
        gender, height, weight, ap_hi, ap_lo,
        cholesterol, gluc, smoke, alco, active,
        age_years, bmi
    ]], columns=[
        "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active",
        "age_years", "bmi"
    ])

    # Probability prediction
    prob = model.predict_proba(input_data)[:, 1][0]

    # Threshold (change if you tuned it)
    threshold = 0.5
    prediction = int(prob >= threshold)

    st.subheader("🧾 Result")
    st.write(f"**Risk Probability:** `{prob:.2f}`")

    if prediction == 1:
        st.error("⚠️ High risk of cardiovascular disease")
    else:
        st.success("✅ Low risk of cardiovascular disease")
