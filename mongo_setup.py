from pymongo import MongoClient
from datetime import datetime
from flask import current_app
from flask_login import UserMixin
from bson.objectid import ObjectId
from flask_mail import Message

# === DATABASE CONNECTION ===
def get_db():
    uri = current_app.config["MONGO_URI"]
    client = MongoClient(uri)
    return client["lifepulse_db"]

# === USER CREATION ===
def create_user(username, email, password_hash, role="user"):
    db = get_db()
    users_collection = db["users"]
    users_collection.insert_one({
        "username": username,
        "email": email,
        "password": password_hash,
        "role": role,
        "created_at": datetime.now()
    })

# === EMAIL ON AT-RISK PREDICTION ===
def send_risk_email(username, disease, health_score):
    db = get_db()
    user = db["users"].find_one({"username": username})
    if not user or not user.get("email"):
        return

    suggestion_map = {
        "heart": "Reduce salt intake. Consult a cardiologist.",
        "diabetes": "Lower sugar intake and exercise.",
        "kidney": "Stay hydrated. Visit a nephrologist.",
        "liver": "Avoid alcohol. See a hepatologist.",
        "breast_cancer": "Schedule a screening with an oncologist.",
        "malaria": "Seek urgent care and hydration.",
        "pneumonia": "Start antibiotics immediately and rest."
    }

    hospital_map = {
        "heart": "Narayana Health, Dr. Ramesh Gupta",
        "diabetes": "Medanta, Dr. Sneha Rao",
        "kidney": "Apollo Kidney Center, Dr. Rajesh",
        "liver": "Fortis Liver Clinic, Dr. Ramesh ",
        "breast_cancer": "AIIMS Oncology, Dr. Reena Sen",
        "malaria": "Manipal Emergency, Dr. Iqbal",
        "pneumonia": "Max Chest Center, Dr. Ramesh"
    }

    disease_key = disease.lower()
    suggestion = suggestion_map.get(disease_key, "Please consult a specialist.")
    hospital_info = hospital_map.get(disease_key, "Visit your nearest hospital.")

    msg = Message(
        subject=f"⚠️ Health Alert: At Risk for {disease.capitalize()}",
        recipients=[user["email"]],
        body=f"""
Hi {username},

Based on your recent submission on LifePulse, our model predicts you may be at risk for {disease.capitalize()}.

🩺 Health Suggestion:
{suggestion}

🏥 Recommended Doctor/Hospital:
{hospital_info}

📉 Your Health Score: {health_score}

Please consult a doctor immediately if symptoms persist.

Stay safe,
Team LifePulse AI
"""
    )

    try:
        mail = current_app.extensions["mail"]
        with current_app.app_context():
            mail.send(msg)
    except Exception as e:
        print("❌ Email sending failed:", e)

# === DISEASE PREDICTION LOGGER ===
def log_disease_prediction(username, disease, input_data, prediction, health_score):
    db = get_db()
    logs_collection = db["disease_logs"]
    logs_collection.insert_one({
        "username": username,
        "disease": disease,
        "input_data": input_data,
        "prediction": prediction,
        "health_score": health_score,
        "timestamp": datetime.now()
    })

    # Create a notification
    message = f"You are {'at risk' if prediction == 1 else 'not at risk'} for {disease.capitalize()}"
    db["notifications"].insert_one({
        "username": username,
        "message": message,
        "seen": False,
        "timestamp": datetime.now()
    })

    # 📧 Trigger email if at risk
    if prediction == 1:
        send_risk_email(username, disease, health_score)

# === USER CLASS FOR LOGIN ===
class User(UserMixin):
    def __init__(self, username, email, user_id, role="user"):
        self.username = username
        self.email = email
        self.id = str(user_id)
        self.role = role

    @staticmethod
    def get_by_id(user_id):
        db = get_db()
        users_collection = db["users"]
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})
        if user_data:
            return User(
                username=user_data["username"],
                email=user_data["email"],
                user_id=user_data["_id"],
                role=user_data.get("role", "user")
            )
        return None

# === UTILITY FUNCTIONS ===
def get_users_collection():
    return get_db()["users"]

def get_logs_collection():
    return get_db()["disease_logs"]
