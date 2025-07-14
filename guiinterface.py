# heart_gui.py

import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# Predict function
def predict_risk():
    try:
        # Collect data
        data = [
            int(age_entry.get()),
            int(sex_var.get()),
            int(cp_entry.get()),
            int(trestbps_entry.get()),
            int(chol_entry.get()),
            int(fbs_var.get()),
            int(restecg_entry.get()),
            int(thalach_entry.get()),
            int(exang_var.get()),
            float(oldpeak_entry.get()),
            int(slope_entry.get()),
            int(ca_entry.get()),
            int(thal_entry.get())
        ]

        # Scale and predict
        data = scaler.transform([data])
        prediction = model.predict(data)[0]

        # Show result
        if prediction == 1:
            messagebox.showerror("Prediction Result", "🔴 High Risk of Heart Disease! Consult a doctor.")
        else:
            messagebox.showinfo("Prediction Result", "🟢 Low Risk of Heart Disease. Keep living healthy!")

    except Exception as e:
        messagebox.showwarning("Input Error", f"Please enter valid inputs.\n\nDetails: {e}")

# Create window
root = tk.Tk()
root.title("Heart Disease Risk Detector")
root.geometry("400x600")

# Labels and entries
tk.Label(root, text="Heart Disease Predictor", font=("Arial", 16, "bold")).pack(pady=10)

fields = [
    ("Age", "age_entry"),
    ("Sex (1=Male, 0=Female)", "sex_var"),
    ("Chest Pain Type (0-3)", "cp_entry"),
    ("Resting BP", "trestbps_entry"),
    ("Cholesterol", "chol_entry"),
    ("Fasting Blood Sugar >120? (1/0)", "fbs_var"),
    ("Rest ECG (0-2)", "restecg_entry"),
    ("Max Heart Rate", "thalach_entry"),
    ("Exercise Angina (1=Yes, 0=No)", "exang_var"),
    ("Oldpeak", "oldpeak_entry"),
    ("Slope (0-2)", "slope_entry"),
    ("Major Vessels (0-3)", "ca_entry"),
    ("Thalassemia (1-3)", "thal_entry")
]

entry_vars = {}

for label, var_name in fields:
    tk.Label(root, text=label).pack()
    if "var" in var_name:
        entry_vars[var_name] = tk.StringVar()
        tk.Entry(root, textvariable=entry_vars[var_name]).pack()
    else:
        entry_vars[var_name] = tk.Entry(root)
        entry_vars[var_name].pack()

# Map entry widgets to variables
age_entry = entry_vars["age_entry"]
sex_var = entry_vars["sex_var"]
cp_entry = entry_vars["cp_entry"]
trestbps_entry = entry_vars["trestbps_entry"]
chol_entry = entry_vars["chol_entry"]
fbs_var = entry_vars["fbs_var"]
restecg_entry = entry_vars["restecg_entry"]
thalach_entry = entry_vars["thalach_entry"]
exang_var = entry_vars["exang_var"]
oldpeak_entry = entry_vars["oldpeak_entry"]
slope_entry = entry_vars["slope_entry"]
ca_entry = entry_vars["ca_entry"]
thal_entry = entry_vars["thal_entry"]

# Predict Button
tk.Button(root, text="Predict Risk", command=predict_risk, bg="blue", fg="white").pack(pady=20)

root.mainloop()
