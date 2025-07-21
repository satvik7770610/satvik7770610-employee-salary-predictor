# retrain_model.py
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load sample dataset
df = pd.read_csv("model/sample_dataset.csv")

# Re-encode categorical columns
categorical = ['company_type', 'job_role', 'location', 'remote_work', 'job_level', 'gender']
encoders = {col: LabelEncoder() for col in categorical}
for col in categorical:
    df[col] = encoders[col].fit_transform(df[col])

# Train model
X = df.drop(columns=['estimated_salary'])
y = df['estimated_salary']
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model + encoders
joblib.dump(model, "model/model.pkl")
joblib.dump(encoders, "model/encoders.pkl")
print("✅ Model retrained and saved.")
