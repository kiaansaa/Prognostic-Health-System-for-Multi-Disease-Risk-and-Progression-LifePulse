from flask import Flask, render_template, request
from flask import redirect, url_for
from datetime import datetime, timedelta
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import os
from PIL import Image
from lime.lime_tabular import LimeTabularExplainer
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from flask import request

import warnings

import sys
import sklearn.ensemble._forest

from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import uuid

from flask_pymongo import PyMongo
from mongo_setup import log_disease_prediction
from urllib.parse import quote_plus


from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

from mongo_setup import User, get_db
from mongo_setup import get_db

from bson.objectid import ObjectId

from mongo_setup import get_users_collection

import base64
from io import BytesIO

from flask import jsonify

from markupsafe import Markup

import glob
import uuid
import os
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory

from flask import abort

from flask_login import login_required, current_user
from functools import wraps

from flask_mail import Mail, Message

app = Flask(__name__)





# app = Flask(
#     __name__,
#     static_folder='static',    
#     template_folder='templates'
# )

# Mail config
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "nishah8535@gmail.com"
app.config["MAIL_PASSWORD"] = "oynl bygs wmbq mhcl"  # Not Gmail password. Use App Password.
app.config["MAIL_DEFAULT_SENDER"] = "LifePulse AI <nishah8535@gmail.com>"

mail = Mail(app) 
app.extensions['mail'] = mail


app.secret_key = "supersecretkey123"

bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)

username = "Nishant8535"
password = quote_plus("Gop65792")  # safely encoded
host = "cluster0.dwdu3pu.mongodb.net"

# Connect to your MongoDB Atlas cluster
app.config["MONGO_URI"] = f"mongodb+srv://{username}:{password}@{host}/lifepulse_db?retryWrites=true&w=majority"
mongo = PyMongo(app)



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


@app.route("/about")
def about():
    return render_template("about.html")


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

        # ✅ Image-based prediction (no LIME)
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

                # Only log if user is authenticated
                if current_user.is_authenticated:
                    log_disease_prediction(
                        username=current_user.username,
                        disease=disease,
                        input_data="image",
                        prediction=prediction,
                        health_score=None
                    )

                return render_template(result_template, pred=prediction, lime_path=None)

            except Exception as e:
                print(f"❌ Image processing error: {e}")
                return render_template("home.html", message="Error processing image.")

        # ✅ Tabular input handling
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

        # ✅ Log prediction only if user is logged in
        if current_user.is_authenticated:
            log_disease_prediction(
                username=current_user.username,
                disease=disease,
                input_data=processed_inputs,
                prediction=prediction,
                health_score=health_score
            )

        # ✅ Generate new LIME HTML file for everyone (guest or logged in)
        try:
            if fields and to_predict_np is not None:
                dummy_data = np.tile(to_predict_np, (100, 1)) + np.random.normal(0, 0.1, size=(100, to_predict_np.shape[1]))
                explainer = LimeTabularExplainer(
                    training_data=dummy_data,
                    feature_names=fields,
                    class_names=['Not at Risk', 'At Risk'],
                    mode='classification'
                )
                explanation = explainer.explain_instance(to_predict_np[0], model.predict_proba)

                lime_id = str(uuid.uuid4())
                lime_path = f'lime_cache/{lime_id}.html'
                os.makedirs('lime_cache', exist_ok=True)
                explanation.save_to_file(lime_path)

                session['lime_filename'] = lime_id + ".html"
        except Exception as e:
            print(f"❌ LIME generation failed: {e}")
            session['lime_filename'] = None

        session['last_disease'] = disease

        return render_template(
            'predict.html',
            pred=prediction,
            disease=disease,
            health_score=health_score
        )

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return render_template("home.html", message=f"Unexpected error: {e}")


    


@app.route('/predict_malaria', methods=['POST'])
def malariapredictPage():
    try:
        img = request.files['image']
        if img.filename == '':
            return render_template('malaria.html', message="Please upload an image.")

        # Save and preprocess image
        img_path = os.path.join('static/uploads', secure_filename(img.filename))
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path)

        img_loaded = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img_loaded)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        model = models['malaria']
        raw_output = model.predict(img_array)[0][0]

        print(f"📊 Raw malaria model output: {raw_output}")

        # 🛠 Flip logic: 0 = infected, 1 = healthy
        pred = 1 if raw_output < 0.5 else 0

        # ✅ Log to MongoDB if user is logged in
        if current_user.is_authenticated:
            log_disease_prediction(
                username=current_user.username,
                disease='malaria',
                input_data="image",
                prediction=pred,
                health_score=None
            )

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


        if current_user.is_authenticated:
            log_disease_prediction(
            username=current_user.username,
            disease='pneumonia',
            input_data="image",
            prediction=pred,
            health_score=None
        )


        return render_template('pneumonia_predict.html', pred=pred, image_path=image_path)

    except Exception as e:
        print("❌ Error during pneumonia prediction:", e)
        return render_template('pneumonia.html', message=f"Error: {e}")


# @app.route('/dashboard/<username>')
# @login_required
# def userDashboard(username):
#     from mongo_setup import get_db
#     db = get_db()
#     logs = list(db["disease_logs"].find({"username": username}).sort("timestamp", -1))
#     return render_template('dashboard.html', username=username, logs=logs)

@app.route("/dashboard/<username>")
@login_required
def userDashboard(username):
    disease_filter = request.args.get("disease")
    date_filter = request.args.get("date")

    db = get_db()
    query = {"username": username}

    if disease_filter:
        query["disease"] = disease_filter.lower()
    if date_filter:
        try:
            selected_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
            query["timestamp"] = {
                "$gte": datetime.combine(selected_date, datetime.min.time()),
                "$lt": datetime.combine(selected_date, datetime.max.time())
            }
        except Exception as e:
            print("Date filter parsing error:", e)

    logs = list(db["disease_logs"].find(query).sort("timestamp", -1))
    return render_template("dashboard.html", username=username, logs=logs)




@app.route('/signup', methods=['GET', 'POST'])
def signup():
    users_collection = get_users_collection()

    from mongo_setup import get_db  # Ensure it's imported

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        db = get_db()
        users_collection = db["users"]

        if users_collection.find_one({"username": username}):
            return render_template('signup.html', message="Username already exists")

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed_pw,
            "created_at": datetime.now()
        })

        return redirect(url_for('login'))

    return render_template('signup.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    users_collection = get_users_collection()

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user_data = users_collection.find_one({"username": username})
        if user_data and bcrypt.check_password_hash(user_data['password'], password):
            user_obj = User(
            user_data['username'],
            user_data['email'],
            user_data['_id'],
            role=user_data.get("role", "user")
                            )
            login_user(user_obj)
            return redirect(url_for('userDashboard', username=user_obj.username))  # ✅ Redirect to dashboard
        else:
            return render_template('login.html', message="Invalid credentials")
    return render_template('login.html')



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))  # Redirects to login page after logout


@app.route('/progression/<disease>')
@login_required
def disease_progression(disease):
    return render_template('progression.html', disease=disease)



@app.route('/api/progression/<disease>')
@login_required
def progression_api(disease):
    from datetime import datetime, timedelta
    db = get_db()

    # Get days filter from query param (default to 7)
    days = int(request.args.get("days", 7))
    from_time = datetime.now() - timedelta(days=days)

    logs = list(
        db["disease_logs"]
        .find({
            "username": current_user.username,
            "disease": disease,
            "health_score": {"$ne": None},
            "timestamp": {"$gte": from_time}
        }).sort("timestamp", 1)
    )




    # Return as JSON
    data = {
        "timestamps": [log["timestamp"].strftime("%Y-%m-%d %H:%M") for log in logs],
        "scores": [log["health_score"] for log in logs]
    }
    return data



@app.route('/lime/<filename>')
def serve_lime_file(filename):
    try:
        return send_from_directory('lime_cache', filename)
    except Exception as e:
        return f"<h3>Error loading LIME explanation: {e}</h3>"


@app.route('/view_lime')
def view_lime_page():
    lime_filename = session.get('lime_filename')
    disease = session.get('last_disease')

    if lime_filename and os.path.exists(os.path.join('lime_cache', lime_filename)):
        return render_template("view_lime.html", disease=disease)

    return render_template("error.html", message="No LIME explanation found.")



@app.route("/notifications")
@login_required
def get_notifications():
    db = get_db()
    notifications = list(db["notifications"].find({
        "username": current_user.username,
        "seen": False
    }).sort("timestamp", -1))

    # Mark as seen after fetching
    db["notifications"].update_many(
        {"username": current_user.username, "seen": False},
        {"$set": {"seen": True}}
    )

    return jsonify([{
        "message": n["message"],
        "timestamp": n["timestamp"].strftime("%Y-%m-%d %H:%M")
    } for n in notifications])





@app.route("/edit_log/<log_id>", methods=["GET", "POST"])
@login_required
def edit_log(log_id):
    db = get_db()
    log = db["disease_logs"].find_one({"_id": ObjectId(log_id)})

    if not log or log["username"] != current_user.username:
            return redirect(url_for("edit_log", log_id=log_id, updated="true"))

    if request.method == "POST":
        updated_inputs = request.form.getlist("inputs")
        db["disease_logs"].update_one({"_id": ObjectId(log_id)}, {"$set": {"input_data": updated_inputs}})

        return redirect(url_for("edit_log", log_id=log_id, updated="true"))

    disease = log["disease"]
    field_map = {
        "diabetes": ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'],
        "heart": ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting Blood Sugar', 'Rest ECG', 'Max HR', 'Exercise Induced Angina', 'Oldpeak', 'Slope', 'CA', 'Thal'],
        "liver": ['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin', 'Alk Phos', 'SGPT', 'SGOT', 'Total Proteins', 'Albumin', 'A/G Ratio'],
        "kidney": ['BMI', 'Systolic BP', 'Fasting Blood Sugar', 'HbA1c', 'Creatinine', 'BUN', 'GFR', 'Protein in Urine', 'Muscle Cramps', 'Itching'],
        "breast_cancer": ['Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean']
    }
    fields = field_map.get(disease, [])

    return render_template("edit_log.html", log=log, disease=disease, fields=fields, zip=zip)  




@app.context_processor
def utility_processor():
    import uuid
    return dict(uuid=lambda: str(uuid.uuid4()))

def clean_lime_cache():
    for f in glob.glob("lime_cache/*.html"):
        try:
            os.remove(f)
        except Exception as e:
            print(f"❌ Failed to delete {f}: {e}")






from collections import Counter
from statistics import mean

# Decorator to restrict to admins
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/admin")
@admin_required
def admin_dashboard():
    db = get_db()
    logs = list(db["disease_logs"].find().sort("timestamp", -1))
    users = list(db["users"].find({}, {"username": 1}))

    # === Quick Stats ===
    total_users = len(set(user["username"] for user in users))
    total_predictions = len(logs)
    health_scores = [log["health_score"] for log in logs if log.get("health_score") is not None]
    avg_health_score = round(mean(health_scores), 2) if health_scores else "N/A"
    disease_names = [log["disease"] for log in logs]
    most_common_disease = Counter(disease_names).most_common(1)[0][0] if disease_names else "N/A"

    return render_template(
        "admin_dashboard.html",
        logs=logs,
        users=users,
        total_users=total_users,
        total_predictions=total_predictions,
        avg_health_score=avg_health_score,
        most_common_disease=most_common_disease
    )

# Call this when app starts
clean_lime_cache()

if __name__ == '__main__':
    app.run(debug=True)
    # port = int(os.environ.get("PORT", 5000))  
    # app.run(host='0.0.0.0', port=port)
