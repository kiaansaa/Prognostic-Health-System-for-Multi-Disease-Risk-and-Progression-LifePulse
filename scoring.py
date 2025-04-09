# scoring.py

def diabetes_health_score(row):
    score = 100
    if row['Glucose'] > 180: score -= 30
    elif row['Glucose'] > 140: score -= 20
    elif row['Glucose'] > 120: score -= 10

    if row['BMI'] > 35: score -= 25
    elif row['BMI'] > 30: score -= 15
    elif row['BMI'] > 25: score -= 5

    if row['BloodPressure'] > 90: score -= 10
    elif row['BloodPressure'] > 80: score -= 5

    if row['Insulin'] > 200: score -= 10
    elif row['Insulin'] > 150: score -= 5

    if row['Age'] > 60: score -= 10
    elif row['Age'] > 50: score -= 5

    if row['SkinThickness'] > 40: score -= 5
    if row['Pregnancies'] > 6: score -= 5
    return max(0, score)


def heart_health_score(row):
    score = 100
    if row['age'] > 60: score -= 10
    if row['trestbps'] > 140: score -= 10
    if row['chol'] > 240: score -= 15
    if row['thalach'] < 100: score -= 15
    if row['oldpeak'] > 2.0: score -= 10
    if row['cp'] == 4: score -= 10  # chest pain type: typical angina
    if row['exang'] == 1: score -= 10  # exercise-induced angina
    return max(0, score)


def kidney_health_score(data):
    score = 100
    if data['GFR'] < 60:
        score -= 20
    if data['SerumCreatinine'] > 1.5:
        score -= 20
    if data['ProteinInUrine'] == 1:
        score -= 15
    if data['Itching'] == 1:
        score -= 5
    return max(score, 0)



def liver_health_score(row):
    score = 100
    if row['Age'] > 60: score -= 10
    if row['Total_Bilirubin'] > 2: score -= 10
    if row['Direct_Bilirubin'] > 1: score -= 10
    if row['Alkphos'] > 120: score -= 10
    if row['Sgpt'] > 45: score -= 10
    if row['Sgot'] > 50: score -= 10
    if row['Total_Proteins'] < 6: score -= 10
    return max(0, score)


def breast_cancer_health_score(row):
    score = 100
    if row['radius_mean'] > 17: score -= 15
    if row['texture_mean'] > 22: score -= 10
    if row['perimeter_mean'] > 115: score -= 10
    if row['area_mean'] > 1000: score -= 10
    if row['concavity_mean'] > 0.3: score -= 15
    if row['fractal_dimension_mean'] > 0.08: score -= 10
    return max(0, score)
