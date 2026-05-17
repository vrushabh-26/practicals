
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import subprocess
subprocess.run(["pip", "install", "lime"], capture_output=True)
from lime import lime_tabular

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train a supervised classification model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# 3. Create a LIME explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# 4. Select a sample from the test set
index = 0
sample = X_test[index]

# Explain the prediction
explanation = explainer.explain_instance(
    data_row=sample,
    predict_fn=model.predict_proba,
    num_features=4
)

# Print explanation as text
print("\nLIME Explanation (Feature Contributions):")
for feature, weight in explanation.as_list():
    print(f"{feature}: {weight:.4f}")

# 5. Visualize the explanation
fig = explanation.as_pyplot_figure()
plt.title("LIME Explanation for Iris Sample")
plt.show()

# 6. Display the predicted class and probabilities
predicted_class = model.predict([sample])[0]
probabilities = model.predict_proba([sample])[0]

print("\nPredicted Class:", class_names[predicted_class])
print("Prediction Probabilities:", probabilities)