from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# load trained modek and scaler 
model = joblib.load("model.pkl")     
scaler = joblib.load("scaler.pkl")   

app = FastAPI(title="Customer Purchase Prediction API")

# input schema
class CustomerData(BaseModel):
    gender: str      # "male" or "female"
    age: int
    salary: float


# prediction endpoint
@app.post("/predict")
def predict_purchase(data: CustomerData):


    # encode the Gender
    gender_encoded = 1 if data.gender.lower() == "male" else 0

    
    input_array = np.array([[gender_encoded, data.age, data.salary]])

    # apply Scaling 
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Convert prediction to readable output
    if prediction == 1:
        result = "The customer will purchase the product"
    else:
        result = "The customer will not purchase the product"

    return {
        "prediction": result
    }


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}


