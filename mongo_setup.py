from pymongo import MongoClient
from datetime import datetime
from flask import current_app
from flask_login import UserMixin
from bson.objectid import ObjectId

# Get DB instance from URI
def get_db():
    uri = current_app.config["MONGO_URI"]
    client = MongoClient(uri)
    return client["lifepulse_db"]

# Create a new user document
def create_user(username, email, password_hash):
    db = get_db()
    users_collection = db["users"]
    users_collection.insert_one({
        "username": username,
        "email": email,
        "password": password_hash,
        "created_at": datetime.now()
    })

# ✅ Log a prediction and also store a real-time notification
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

    # ✅ Insert notification into 'notifications' collection
    message = f"You are {'at risk' if prediction == 1 else 'not at risk'} for {disease.capitalize()}"
    db["notifications"].insert_one({
        "username": username,
        "message": message,
        "seen": False,
        "timestamp": datetime.now()
    })

# ✅ User class for login system
class User(UserMixin):
    def __init__(self, username, email, _id=None):
        self.id = str(_id)
        self.username = username
        self.email = email

    @staticmethod
    def get_by_username(username):
        db = get_db()
        users_collection = db["users"]
        user_data = users_collection.find_one({"username": username})
        if user_data:
            return User(user_data["username"], user_data["email"], user_data["_id"])
        return None

    @staticmethod
    def get_by_id(user_id):
        db = get_db()
        users_collection = db["users"]
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})
        if user_data:
            return User(user_data["username"], user_data["email"], user_data["_id"])
        return None

def get_users_collection():
    db = get_db()
    return db["users"]

def get_logs_collection():
    db = get_db()
    return db["disease_logs"]
