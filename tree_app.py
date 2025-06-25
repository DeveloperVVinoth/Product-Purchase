import streamlit as st
import joblib
import pandas as pd

# Load the trained decision tree model
model = joblib.load("model/tree_model.pkl")

# UI
st.title("🧠 Product Purchase Predictor (Decision Tree)")
st.write("Enter Age and Income (in ₹1000s) to check if the person is likely to buy.")

# Inputs
age = st.number_input("Enter Age:", min_value=10, max_value=100, step=1)
income = st.number_input("Enter Income (in ₹1000s):", min_value=10.0, max_value=150.0, step=1.0)

# Predict
if st.button("Predict"):
    input_data = pd.DataFrame([[age, income]], columns=["Age", "Income"])
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: Will Buy? → **{prediction}**")
