import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load the trained model when the app starts
model = joblib.load('model.joblib')

class InputData(BaseModel):
    # Define your input data structure based on the model's features
    feature_1: float
    feature_2: float
    feature_3: float
    # Add all the features used for training

@app.post("/predict")
async def predict(data: InputData):
    # Convert input data to a numpy array
    input_data = np.array([[data.feature_1, data.feature_2, data.feature_3]])  # Add more features as needed
    
    # Get the prediction
    prediction = model.predict(input_data)
    
    return {"prediction": prediction[0]}
