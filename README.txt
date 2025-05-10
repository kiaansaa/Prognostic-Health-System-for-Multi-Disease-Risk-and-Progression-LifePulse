==========================================
 LifePulse: README
 A Smart Multi-Disease Risk Prediction System
==========================================

PROJECT OVERVIEW:
-----------------
LifePulse is a web-based health intelligence platform that allows users to predict multiple disease risks using trained machine learning models. 
It also provides personalized health recommendations, risk progression charts, and email alerts, supporting both admin and user roles.

Developed using: 
- Python (Flask Framework)
- MongoDB Atlas (Cloud DB)
- HTML/CSS/JS + Bootstrap (Frontend)
- Scikit-learn, TensorFlow, Keras, Lime (AI/ML libraries)

------------------------------------------
FOLDER & FILE STRUCTURE:
------------------------------------------

1. app.py
   - Main Flask application entry point

2. hash_pw.py
   - Utility for hashing and verifying user passwords

3. mongo_setup.py
   - MongoDB Atlas connection initialization

4. scoring.py
   - Logic for generating health scores and recommendations

5. requirements.txt
   - List of Python libraries needed to run the app

6. runtime.txt / Procfile
   - Used for deployment on cloud platforms (Render, Heroku)

------------------------------------------

FOLDERS:
--------

1. models/
   - Contains trained ML model files (.pkl for tabular models, .h5 for CNNs)

2. model_training/
   - Contains Jupyter notebooks or Python scripts for training models
   - Also includes dataset_sources.txt to list external dataset links

3. model_test/
   - Test scripts to validate trained models' performance

4. static/
   - CSS, JavaScript, images used by the frontend templates

5. templates/
   - HTML templates for rendering pages (login, dashboard, prediction results)

6. lime_cache/
   - Temporarily stores LIME explanation images

------------------------------------------

HOW TO RUN LOCALLY:
--------------------
1. Create a virtual environment (recommended)
2. Install dependencies using:
   pip install -r requirements.txt

3. Ensure MongoDB Atlas is set up and URI is configured in mongo_setup.py
4. Run the app:
   python app.py

5. Access the app on:
   http://localhost:5000/

------------------------------------------

NOTES:
------
- Admin can access logs, manage users, and view system usage.
- Users can log in, submit data or images for prediction, view risk charts, and receive alerts.
- All health logs are stored in MongoDB Atlas under `lifepulse_db`.

------------------------------------------

CONTACT:
--------
Developer: Nishant Shah
Supervisor: Dr. Hari Prashad Joshi
