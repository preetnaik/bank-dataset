import streamlit as st
import pickle
import numpy as np
import joblib

# Load the model
model = joblib.load("bank.pkl")

def encode_input(data):
    return data  

# Streamlit app
st.title("Bank Loan Prediction App")
st.write("Enter the customer information below:")

# Input fields (example — customize based on your actual features)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ["admin.", "technician", "services", "management", "retired", "blue-collar"])
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary"])
default = st.selectbox("Default?", ["no", "yes"])
balance = st.number_input("Balance", value=1000)
housing = st.selectbox("Housing Loan?", ["no", "yes"])
loan = st.selectbox("Personal Loan?", ["no", "yes"])
contact = st.selectbox("Contact Type", ["cellular", "telephone"])
day = st.number_input("Day", min_value=1, max_value=31, value=15)
month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
duration = st.number_input("Duration", value=100)
campaign = st.number_input("Campaign", value=1)

# Collect all inputs
input_data = [age, job, marital, education, default, balance, housing,
              loan, contact, day, month, duration, campaign]

# Encode if needed
processed_data = encode_input(input_data)

# Prediction
if st.button("Predict"):
    try:
        # Reshape for prediction
        features = np.array(processed_data).reshape(1, -1)
        prediction = model.predict(features)[0]
        st.success(f"Prediction: {'Subscribed' if prediction == 1 else 'Not Subscribed'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
