import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Sample data: [Age, Income (in ₹1000s)]
data = {
    "Age": [25, 30, 35, 40, 45, 50, 55],
    "Income": [30, 40, 50, 60, 70, 80, 90],
    "Buy": ["No", "No", "Yes", "Yes", "Yes", "No", "No"]
}

df = pd.DataFrame(data)

# Features and label
X = df[["Age", "Income"]]
y = df["Buy"]

# Train decision tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Save model
joblib.dump(model, "model/tree_model.pkl")

print("✅ Decision tree model trained and saved.")
