from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import os
from PIL import Image
from lime.lime_tabular import LimeTabularExplainer
from werkzeug.utils import secure_filename
import warnings

import sys
import sklearn.ensemble._forest

from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import uuid


# Suppress warnings
warnings.filterwarnings("ignore")
sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble._forest

# Custom scoring imports
from scoring import (
    diabetes_health_score,
    heart_health_score,
    kidney_health_score,
    liver_health_score,
    breast_cancer_health_score
)

app = Flask(__name__)

# ✅ Load Models
models = {
    'diabetes': 'models/diabetes.pkl',
    'breast_cancer': 'models/breast_cancer.pkl',
    'heart': 'models/heart.pkl',
    'kidney': 'models/kidney.pkl',
    'liver': 'models/liver.pkl',
    'malaria': 'models/malaria.h5',
    'pneumonia': 'models/pneumonia.h5'
}

for key, path in models.items():
    if os.path.exists(path):
        try:
            models[key] = load_model(path) if path.endswith(".h5") else joblib.load(path)
            print(f"✅ Loaded model: {key}")
        except Exception as e:
            print(f"❌ Error loading {key}: {e}")

# ✅ Route setup
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/diabetes')
def diabetesPage():
    return render_template('diabetes.html')

@app.route('/cancer')
def cancerPage():
    return render_template('breast_cancer.html')

@app.route('/heart')
def heartPage():
    return render_template('heart.html')

@app.route('/kidney')
def kidneyPage():
    return render_template('kidney.html')

@app.route('/liver')
def liverPage():
    return render_template('liver.html')

@app.route('/malaria')
def malariaPage():
    return render_template('malaria.html')

@app.route('/pneumonia')
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route('/predict', methods=['POST'])
def predictPage():
    try:
        to_predict_dict = request.form.to_dict()
        disease = to_predict_dict.get('disease')
        print(f"\n➡️ Received data for: {disease}")
        print("📥 Form values received:", to_predict_dict)

        if not disease or disease not in models:
            return render_template("home.html", message="Invalid disease type.")

        model = models[disease]

        # ✅ For malaria/pneumonia: Handle image
        if disease in ['malaria', 'pneumonia']:
            file = request.files.get('image')
            if not file or file.filename == '':
                return render_template("home.html", message="Please upload an image.")

            try:
                target_size = (64, 64) if disease == 'malaria' else (150, 150)
                img = Image.open(file).convert('RGB').resize(target_size)
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                prediction = int(prediction[0][0] > 0.5)

                result_template = 'malaria_predict.html' if disease == 'malaria' else 'pneumonia_predict.html'
                return render_template(result_template, pred=prediction, lime_path=None)

            except Exception as e:
                print(f"❌ Image processing error: {e}")
                return render_template("home.html", message="Error processing image.")

        # ✅ For tabular models
        to_predict_dict.pop('disease', None)
        raw_input = list(to_predict_dict.values())
        print("🔎 Raw Inputs:", raw_input)

        health_score = None
        fields = []
        processed_inputs = []

        if disease == 'kidney':
            fields = [
                'BMI', 'SystolicBP', 'FastingBloodSugar', 'HbA1c',
                'SerumCreatinine', 'BUNLevels', 'GFR',
                'ProteinInUrine', 'MuscleCramps', 'Itching'
            ]
            for i, f in enumerate(fields):
                val = raw_input[i].strip().lower()
                if f in ['ProteinInUrine', 'MuscleCramps', 'Itching']:
                    processed_inputs.append(1 if val in ['yes', '1', 'true'] else 0)
                else:
                    processed_inputs.append(float(val))
            to_predict_np = np.array(processed_inputs).reshape(1, -1)
            row = dict(zip(fields, processed_inputs))
            health_score = kidney_health_score(row)

        else:
            processed_inputs = list(map(float, raw_input))
            to_predict_np = np.array(processed_inputs).reshape(1, -1)

            if disease == 'diabetes':
                fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                row = dict(zip(fields, processed_inputs))
                health_score = diabetes_health_score(row)

            elif disease == 'heart':
                fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                row = dict(zip(fields, processed_inputs))
                health_score = heart_health_score(row)

            elif disease == 'liver':
                fields = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkphos', 'Sgpt', 'Sgot', 'Total_Proteins', 'Albumin', 'A/G_Ratio']
                row = dict(zip(fields, processed_inputs))
                health_score = liver_health_score(row)

            elif disease == 'breast_cancer':
                fields = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                          'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
                          'fractal_dimension_mean']
                row = dict(zip(fields, processed_inputs))
                health_score = breast_cancer_health_score(row)

        prediction = model.predict(to_predict_np)
        prediction = prediction[0]
        print(f"🧠 Prediction Raw Output: {prediction}")

        if isinstance(prediction, (np.ndarray, list)):
            prediction = prediction[0]

        if isinstance(prediction, str) and prediction.upper() in ['B', 'BENIGN']:
            prediction = 0
        elif isinstance(prediction, str) and prediction.upper() in ['M', 'MALIGNANT']:
            prediction = 1
        else:
            prediction = int(prediction)

        if fields:
            dummy_data = np.array([to_predict_np[0]] * 100)
            explainer = LimeTabularExplainer(
                training_data=dummy_data,
                feature_names=fields,
                class_names=['Not at Risk', 'At Risk'],
                mode='classification'
            )
            explanation = explainer.explain_instance(to_predict_np[0], model.predict_proba)
            explanation.save_to_file('static/lime_explanation.html')

        return render_template('predict.html', pred=prediction, disease=disease, health_score=health_score)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return render_template("home.html", message=f"Unexpected error: {e}")

@app.route('/predict_malaria', methods=['POST'])
def malariapredictPage():
    try:
        img = request.files['image']
        if img.filename == '':
            return render_template('malaria.html', message="Please upload an image.")

        img_path = os.path.join('static/uploads', secure_filename(img.filename))
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path)

        img_loaded = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img_loaded)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        model = models['malaria']
        prediction = model.predict(img_array)[0][0]
        pred = 1 if prediction > 0.5 else 0

        return render_template('malaria_predict.html', pred=pred, lime_path=None)

    except Exception as e:
        print("❌ Error during malaria prediction:", e)
        return render_template('malaria.html', message=f"Error: {e}")



    

@app.route('/predict_pneumonia', methods=['POST'])
def pneumoniapredictPage():
    try:
        img = request.files['image']
        if img.filename == '':
            return render_template('pneumonia.html', message="Please upload an image.")

        # Save image to static/uploads
        filename = secure_filename(img.filename)
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        img_path = os.path.join(upload_folder, filename)
        img.save(img_path)

        # Create web-accessible path for display
        image_path = f"/{img_path.replace(os.sep, '/')}"

        # Preprocess image for model
        img_loaded = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img_loaded)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        model = models['pneumonia']
        prediction = model.predict(img_array)[0][0]
        pred = 1 if prediction > 0.5 else 0  # 1 = Pneumonia, 0 = Normal

        return render_template('pneumonia_predict.html', pred=pred, image_path=image_path)

    except Exception as e:
        print("❌ Error during pneumonia prediction:", e)
        return render_template('pneumonia.html', message=f"Error: {e}")




    
if __name__ == '__main__':
    app.run(debug=True)
