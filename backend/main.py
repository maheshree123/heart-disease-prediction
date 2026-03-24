from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("../model/model.pkl")

# -------------------------------
# DEFINE INPUT FORMAT
# -------------------------------
class InputData(BaseModel):
    data: list

# -------------------------------
# HOME
# -------------------------------
@app.get("/")
def home():
    return {"message": "API is running"}

# -------------------------------
# PREDICT
# -------------------------------
@app.post("/predict")
def predict(input_data: InputData):
    try:
        data = np.array(input_data.data).reshape(1, -1)
        prediction = model.predict(data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}