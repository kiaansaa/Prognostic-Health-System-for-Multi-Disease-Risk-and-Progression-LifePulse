import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ✅ Load the cleaned liver dataset
file_path = r'C:/Users/user/Desktop/Production Project/LifePulse/Clean_Dataset/cleaned_liver_dataset.csv'
df = pd.read_csv(file_path)
print(f"✅ Loaded data shape: {df.shape}")

print("Available columns:", df.columns.tolist())


# ✅ Features & Labels
X = df.drop('Result', axis=1)
y = df['Result'].replace({1: 0, 2: 1})



# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ✅ Define XGBoost model and parameters
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
params = {
    'n_estimators': [100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid = GridSearchCV(xgb, params, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_resampled, y_train_resampled)

# ✅ Best parameters and model
print("✅ Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# ✅ Test set evaluation
y_pred = best_model.predict(X_test)
print("✅ Accuracy on test set:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

# ✅ Cross-validation score
cv_scores = cross_val_score(best_model, X, y, cv=5)
print("✅ 5-Fold CV Accuracy:", round(np.mean(cv_scores), 4), "±", round(np.std(cv_scores), 4))

# ✅ Save model
model_path = '../models/liver.pkl'
joblib.dump(best_model, model_path)
print(f"✅ Trained model saved at: {model_path}")
