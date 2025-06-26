import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    "Age": [25, 30, 35, 40, 45, 50, 55, 28, 33, 48],
    "Income": [30, 40, 50, 60, 70, 80, 90, 35, 45, 85],
    "Buy": ["No", "No", "Yes", "Yes", "Yes", "No", "No", "No", "Yes", "No"]
}
df = pd.DataFrame(data)

X = df[["Age", "Income"]]
y = df["Buy"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "Logistic Regression": LogisticRegression()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.2f}")
    
    cm = confusion_matrix(y_test, y_pred, labels=["Yes", "No"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Yes", "No"])
    disp.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")

plt.tight_layout()
plt.show()