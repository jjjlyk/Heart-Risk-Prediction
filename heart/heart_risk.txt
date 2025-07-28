import streamlit as st
import numpy as np

#for models
import joblib

# Load models and scaler
svm_linear = joblib.load('models/svm_heart_risk_linear_model.joblib')
svm_rbf = joblib.load('models/svm_heart_risk_rbf_model.joblib')
svm_poly = joblib.load('models/svm_heart_risk_poly_model.joblib')
scaler = joblib.load('models/scaler.joblib')  # You’ll need to save and load the scaler too!

# App title
st.title("❤️ Heart Attack Risk Prediction")
st.write("Enter the patient's health indicators below:")

# Kernel selection
kernel = st.sidebar.selectbox("Choose SVM Kernel", ["Linear", "RBF", "Polynomial"])

# Input form
age = st.number_input("Age", 1, 120, 50)
chol = st.number_input("Cholesterol", 100, 400, 200)
sbp = st.number_input("Systolic Blood Pressure", 80, 200, 120)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
exercise = st.number_input("Exercise Hours Per Week", 0.0, 20.0, 3.0)
smoking = st.selectbox("Smoking Status", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])

# Encode categorical inputs
smoking = 1 if smoking == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0

# Prepare and scale input
input_data = np.array([[age, chol, sbp, bmi, exercise, smoking, diabetes]])
input_scaled = scaler.transform(input_data)

# Make prediction
if st.button("Predict Heart Attack Risk"):
    model = {
        "Linear": svm_linear,
        "RBF": svm_rbf,
        "Polynomial": svm_poly
    }[kernel]

    prediction = model.predict(input_scaled)
    risk_level = "High Risk 🚨" if prediction[0] == 1 else "Low Risk ✅"

    st.subheader("Prediction Result:")
    st.success(f"The patient is predicted to be at **{risk_level}** of a heart attack.")
