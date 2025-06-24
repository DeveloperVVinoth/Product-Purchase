from sklearn.tree import export_graphviz
import joblib
import graphviz

# Load model
model = joblib.load("model/tree_model.pkl")

# Export tree structure to DOT format
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=["Age", "Income"],
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    special_characters=True
)

# Render the tree
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=False)  # Saves as decision_tree.png
graph.view()  # Opens the image