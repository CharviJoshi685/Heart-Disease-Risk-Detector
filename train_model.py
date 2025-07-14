# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load dataset
df = pd.read_csv("heart.csv")  # Make sure the CSV file is in the same directory

# Step 2: Understand your data (optional but good practice)
print("Columns:", df.columns.tolist())
print("Sample Data:\n", df.head())

# Step 3: Features and target
X = df.drop('target', axis=1)  # Independent variables
y = df['target']              # Dependent variable (0 = No Disease, 1 = Disease)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test_scaled)
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save the model and scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved successfully.")
