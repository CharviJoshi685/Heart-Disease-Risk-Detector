# predict_heart_risk.py

import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

print("\n💓 HEART DISEASE RISK PREDICTOR 💓")
print("Please enter the following details:\n")

# Get user input
age = int(input("Age: "))
sex = int(input("Sex (1 = Male, 0 = Female): "))
cp = int(input("Chest Pain Type (0 = Typical, 1 = Atypical, 2 = Non-anginal, 3 = Asymptomatic): "))
trestbps = int(input("Resting Blood Pressure (mm Hg): "))
chol = int(input("Cholesterol (mg/dL): "))
fbs = int(input("Fasting Blood Sugar > 120 mg/dL? (1 = Yes, 0 = No): "))
restecg = int(input("Resting ECG (0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy): "))
thalach = int(input("Maximum Heart Rate Achieved: "))
exang = int(input("Exercise Induced Angina? (1 = Yes, 0 = No): "))
oldpeak = float(input("ST Depression (Oldpeak): "))
slope = int(input("Slope of Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping): "))
ca = int(input("Number of Major Vessels Colored by Fluoroscopy (0-3): "))
thal = int(input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect): "))

# Create input vector
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# Scale the input
features_scaled = scaler.transform(features)

# Make prediction
prediction = model.predict(features_scaled)[0]

# Generate report
print("\n📄 PREDICTION REPORT")
print("---------------------------")
print(f"Age: {age} | Sex: {'Male' if sex == 1 else 'Female'}")
print(f"Prediction: {'🔴 At Risk of Heart Disease' if prediction == 1 else '🟢 Not at Risk'}")
print("---------------------------")

if prediction == 1:
    print("⚠️ Advice: Please consult a cardiologist for further diagnosis.")
else:
    print("✅ Advice: Keep maintaining a healthy lifestyle!")

