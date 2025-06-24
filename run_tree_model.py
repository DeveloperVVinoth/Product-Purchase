import joblib
import pandas as pd

# Load the model
model = joblib.load("model/tree_model.pkl")

# Input from user
age = float(input("Enter age: "))
income = float(input("Enter income (in ₹1000s): "))

# Prepare input
input_data = pd.DataFrame([[age, income]], columns=["Age", "Income"])
prediction = model.predict(input_data)[0]

print(f"Prediction: Will the person buy? → {prediction}")