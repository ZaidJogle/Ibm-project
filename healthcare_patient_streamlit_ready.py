import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Title Section
# -----------------------------
st.title("🩺 Diabetes Prediction Using Machine Learning")

st.write(
"""
This web application predicts whether a patient is **Diabetic or Non-Diabetic**
based on health parameters.

Developed using **Machine Learning and Streamlit**.
"""
)

st.write("---")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Patient Health Details")

age = st.sidebar.slider("Age", 18, 100, 30)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)

bp = st.sidebar.number_input("Blood Pressure", 40, 200, 80)

glucose = st.sidebar.number_input("Glucose Level", 50, 300, 120)

insulin = st.sidebar.number_input("Insulin Level", 0, 500, 80)

activity = st.sidebar.selectbox(
    "Physical Activity Level",
    ["Low", "Medium", "High", "Very High"]
)

family = st.sidebar.selectbox(
    "Family History of Diabetes",
    ["No", "Yes"]
)

smoking = st.sidebar.selectbox(
    "Smoking",
    ["No", "Yes"]
)

# -----------------------------
# Convert Inputs to Numeric
# -----------------------------
gender = 1 if gender == "Male" else 0
family = 1 if family == "Yes" else 0
smoking = 1 if smoking == "Yes" else 0

activity_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

activity = activity_map[activity]

# -----------------------------
# Feature Engineering
# -----------------------------
if bmi < 18.5:
    bmi_cat = 0
elif bmi < 25:
    bmi_cat = 1
elif bmi < 30:
    bmi_cat = 2
else:
    bmi_cat = 3

if age < 30:
    age_group = 0
elif age < 50:
    age_group = 1
else:
    age_group = 2

# -----------------------------
# Prediction Button
# -----------------------------
st.subheader("Prediction")

if st.button("Predict Diabetes Risk"):

    input_data = np.array([[age, gender, bmi, bp, glucose, insulin,
                            activity, family, smoking, bmi_cat, age_group]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.write("---")

    if prediction[0] == 1:
        st.error("⚠ The patient is **likely Diabetic**")
    else:
        st.success("✅ The patient is **likely Non-Diabetic**")

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.markdown(
"""
**Project:** Diabetes Prediction Using Machine Learning  
**Student:** Zaid Jogle  
**Course:** B.Sc Data Science  
"""
)
