import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ✅ Set your path to the cleaned dataset
data_path = os.path.join('..', 'Clean_Dataset', 'clean_heart.csv')  
data = pd.read_csv(data_path)

print("✅ Loaded data shape:", data.shape)

# ✅ Features and target
X = data.drop('target', axis=1)
y = data['target']

# ✅ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Prediction and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy on test set: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# ✅ Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"✅ 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ✅ Save the model
model_output_path = os.path.join('..', 'models', 'heart.pkl')
with open(model_output_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Model saved to: {model_output_path}")
