from fastapi import FastAPI
import pickle
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the model at startup (local model)
# with open('pipeline_v1.bin', 'rb') as f:
#     model = pickle.load(f)

# Load the model at startup (using pipeline_v2.bin from base image)
with open('pipeline_v2.bin', 'rb') as f:
    model = pickle.load(f)

# Define the input data structure
class LeadData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Define the prediction endpoint
@app.post("/predict")
def predict(data: LeadData):
    # Convert to dictionary format for DictVectorizer
    record = data.dict()
    
    # Make prediction (DictVectorizer expects a list of dicts)
    prediction = model.predict([record])
    probability = model.predict_proba([record])
    
    return {
        "prediction": int(prediction[0]),
        "probability": {
            "not_convert": float(probability[0][0]),
            "convert": float(probability[0][1])
        },
        "lead_will_convert": bool(prediction[0]) 
    }

# Optional: health check endpoint
@app.get("/")
def read_root():
    return {"message": "Lead scoring API is running"}