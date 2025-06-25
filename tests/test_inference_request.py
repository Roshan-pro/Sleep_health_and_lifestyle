import requests
import pandas as pd
import json

# 🧪 Step 1: Create sample data
data = {
    "Person ID": [1],
    "Gender": ["Male"],
    "Age": [30],
    "Occupation": ["Software Engineer"],
    "Sleep Duration": [6.5],
    "Quality of Sleep": [7],
    "Physical Activity Level": [40],
    "Stress Level": [5],
    "BMI Category": ["Normal"],
    "Blood Pressure": ["120/80"],
    "Heart Rate": [70],
    "Daily Steps": [6000]
}

df = pd.DataFrame(data)

# 🧪 Step 2: Format the data in MLflow's expected input format
payload = {
    "columns": list(df.columns),
    "data": df.values.tolist()
}

# 🧪 Step 3: Send POST request to MLflow prediction endpoint
response = requests.post("http://127.0.0.1:1234/invocations", json=payload)

# 🧪 Step 4: Print response
print("Prediction:", response.json())
