import os
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from pathlib import Path

app = FastAPI(title="Sentiment Analysis API")

model_path = Path("models/best_model.pkl")
vectorizer_path = Path("models/vectorizer.pkl")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "IMDB Sentiment Analysis API is running!"}

@app.post("/predict")
def predict(input_text: InputText):
    try:
        # Transform input text
        text_vectorized = vectorizer.transform([input_text.text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)

        # Convert to human-readable label
        sentiment = "positive" if prediction[0] == 1 else "negative"
        
        return {"text": input_text.text, "sentiment": sentiment}
    except Exception as e:
        return {"error": str(e)}
    
# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)