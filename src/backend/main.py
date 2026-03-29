import os
import joblib
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from .llm_service import LLMService

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")

app = FastAPI(title="MedGuard-AI API")

# Setup CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For simplicity in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize paths and dependencies
ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
PROC_DIR   = ROOT / "data" / "processed"

model_path = MODELS_DIR / "diagnosis_classifier.pkl"
le_path    = MODELS_DIR / "label_encoder.pkl"
symptom_list_path = PROC_DIR / "symptom_list.json"

if not model_path.exists() or not le_path.exists():
    raise RuntimeError("Model files not found. Run training script first.")

model = joblib.load(model_path)
le = joblib.load(le_path)
with open(symptom_list_path) as f:
    VOCABULARY = json.load(f)

llm_service = LLMService()

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    symptoms_extracted: list[str]
    diagnosis: str
    confidence: float
    explanation: str

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_symptoms(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # 1. Parse symptoms using LLM
    extracted = llm_service.parse_symptoms(req.text)
    if not extracted:
        # Graceful fallback or rejection
        return AnalyzeResponse(
            symptoms_extracted=[],
            diagnosis="Unknown",
            confidence=0.0,
            explanation="I could not identify any recognizable medical symptoms from your input. Please try again with more specific medical terms."
        )

    # 2. Build feature vector
    feature_vector = [1 if sym in extracted else 0 for sym in VOCABULARY]
    
    # 3. Predict Diagnosis
    import numpy as np
    X = np.array(feature_vector).reshape(1, -1)
    proba = model.predict_proba(X)[0]
    
    top_idx = int(np.argmax(proba))
    diagnosis = le.classes_[top_idx]
    confidence = float(proba[top_idx])

    # 4. Generate Clinical Reasoning
    explanation = llm_service.explain_diagnosis(extracted, diagnosis)

    return AnalyzeResponse(
        symptoms_extracted=extracted,
        diagnosis=diagnosis,
        confidence=confidence,
        explanation=explanation
    )
    
class FeedbackRequest(BaseModel):
    diagnosis: str
    is_correct: bool
    notes: str = ""

@app.post("/api/feedback")
async def receive_feedback(req: FeedbackRequest):
    # Log feedback. In a production app, this would be written to a database
    print(f"[FEEDBACK] Diagnosis: {req.diagnosis} | Correct: {req.is_correct} | Notes: {req.notes}")
    return {"status": "recorded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000, reload=True)
