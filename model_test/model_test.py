import os
import pickle
import numpy as np
import joblib

# List of models and their paths
model_paths = {
    'diabetes': r'C:\Users\user\Desktop\Production Project\LifePulse\models\diabetes.pkl',
    'heart': r'C:\Users\user\Desktop\Production Project\LifePulse\models\heart.pkl',
    'kidney': r'C:\Users\user\Desktop\Production Project\LifePulse\models\kidney.pkl',
    'liver': r'C:\Users\user\Desktop\Production Project\LifePulse\models\liver.pkl',
    'breast_cancer': r'C:\Users\user\Desktop\Production Project\LifePulse\models\breast_cancer.pkl'
}


# Dummy inputs per model (adjust according to your model features)
dummy_inputs = {
    'diabetes': np.random.rand(1, 8),
    'heart': np.random.rand(1, 13),
    'kidney': np.random.rand(1, 10),
    'liver': np.random.rand(1, 10),
    'breast_cancer': np.random.rand(1, 10),
}

for name, path in model_paths.items():
    print(f"🔍 Testing model: {name}")
    if os.path.exists(path):
        try:
            model = joblib.load(path) if path.endswith(".pkl") else pickle.load(open(path, "rb"))
            print("✅ Model loaded:", type(model))

            # Run dummy prediction
            dummy_input = dummy_inputs.get(name)
            if hasattr(model, "predict"):
                result = model.predict(dummy_input)
                print("✅ Prediction successful:", result)
            else:
                print("❌ Model does not have a 'predict' method. It's likely invalid.")
        except Exception as e:
            print(f"❌ Error with model {name}: {e}")
    else:
        print(f"❌ Model file not found: {path}")

    print("-" * 50)

