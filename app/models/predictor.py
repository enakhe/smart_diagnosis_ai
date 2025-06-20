import joblib
from app.schemas.diagnose import DiagnosisRequest, DiagnosisResponse
from app.config import settings

# Load artifacts
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

def predict_disease(data: DiagnosisRequest) -> DiagnosisResponse:
    # Vectorize input
    input_vector = vectorizer.transform([data.symptoms])
    
    # Predict
    prediction = model.predict(input_vector)[0]
    proba = model.predict_proba(input_vector).max()

    # Return response
    return DiagnosisResponse(
        predictions=[
            {
                "condition": label_encoder.inverse_transform([prediction])[0], 
                "confidence": round(proba, 2)
            }
        ],
        advice="This is an AI-generated suggestion. Please consult a doctor for confirmation."
    )
