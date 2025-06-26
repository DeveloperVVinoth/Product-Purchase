import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

# Sample data
data = {
    "Age": [25, 30, 35, 40, 45, 50, 55, 28, 33, 48],
    "Income": [30, 40, 50, 60, 70, 80, 90, 35, 45, 85],
    "Buy": ["No", "No", "Yes", "Yes", "Yes", "No", "No", "No", "Yes", "No"]
}
df = pd.DataFrame(data)

# Features and labels
X = df[["Age", "Income"]]
y = df["Buy"]

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="Yes"))
print("Recall:", recall_score(y_test, y_pred, pos_label="Yes"))
print("F1 Score:", f1_score(y_test, y_pred, pos_label="Yes"))

# Show confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Yes", "No"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Yes", "No"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()