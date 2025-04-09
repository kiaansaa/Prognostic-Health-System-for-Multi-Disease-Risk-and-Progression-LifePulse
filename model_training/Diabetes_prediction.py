# Diabetes_prediction.py

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Dynamically resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the cleaned dataset
data_path = os.path.join(current_dir, "..", "Clean_Dataset", "Cleaned_Diabetes_Dataset.csv")

# Path to save the trained model
model_path = os.path.join(current_dir, "..", "models", "diabetes.pkl")

# Load data
df = pd.read_csv(data_path)

# Split into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy on test set: {acc:.4f}")
print(classification_report(y_test, y_pred))

# 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"✅ 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Save trained model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Trained model saved at: {model_path}")
