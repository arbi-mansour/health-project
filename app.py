import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("my_healthcare_model.joblib")

st.title("Health Diagnosis Prediction")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
blood_type = st.selectbox("Blood Type", ["A", "B", "AB", "O"])
medical_condition = st.text_input("Medical Condition")
admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])

if st.button("Predict Billing Amount"):
    input_data = {
        "Age": [age],
        "Gender": [gender],
        "Blood Type": [blood_type],
        "Medical Condition": [medical_condition],
        "Admission Type": [admission_type],
    }
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)
    st.write(f"Predicted Billing Amount: ${prediction[0]:.2f}")
