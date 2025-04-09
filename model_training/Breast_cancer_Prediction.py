import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os


# Load the cleaned dataset
file_path = r'C:\Users\user\Desktop\Production Project\LifePulse\Clean_Dataset\Cleaned_Cancer_Dataset.csv'
df = pd.read_csv(file_path)
print("✅ Loaded data shape:", df.shape)

# Split into features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ✅ Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ✅ Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# ✅ Grid Search with cross-validation
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train_resampled, y_train_resampled)

# ✅ Best Parameters
print("✅ Best Parameters:", grid.best_params_)

# ✅ Evaluation
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy on test set: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# ✅ Cross-validation score
cv_scores = cross_val_score(grid.best_estimator_, X, y, cv=5)
print(f"✅ 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Save model
model_path = r'C:\Users\user\Desktop\Production Project\LifePulse\models\breast_cancer.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(grid.best_estimator_, model_path)
print(f"✅ Trained model saved at: {model_path}")
