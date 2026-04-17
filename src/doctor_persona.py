"""
MedGuard-AI — Doctor Persona Module
Generates a conversational, clinical explanation of the pipeline's findings using the Gemini API.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Try importing generativeai
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    # Load .env file
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    genai = None

class DoctorPersona:
    """
    Consumes the AgentResult/Verification output and explains it
    empathetically as a local clinical doctor would.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if self.api_key and genai is not None:
            genai.configure(api_key=self.api_key)
            # Use gemini-2.5-flash as it is fast and excellent for conversational tasks
            self.model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            self.model = None

    def generate_response(self, prompt: str, pipeline_result: dict) -> str:
        """
        Generates the doctor's response.
        Returns a fallback string if the API key is not configured or fails.
        """
        if not self.model:
            return (
                "⚠️ [System Notice] Gemini API Key not found. "
                "Unable to generate clinical doctor persona. "
                "Please add GEMINI_API_KEY to your .env file."
            )

        diagnosis = pipeline_result.get("diagnosis", "Unknown")
        treatment = pipeline_result.get("treatment", "Unknown")
        safety_score = pipeline_result.get("safety", {}).get("score", 0)
        risk_label = pipeline_result.get("safety", {}).get("risk_label", "UNKNOWN")
        penalties = pipeline_result.get("safety", {}).get("penalties", {})
        clarifying_qs = pipeline_result.get("clarifying_questions", [])
        qs_text = f"\n- Clarifying Questions Needed: {clarifying_qs}" if clarifying_qs else ""
        
        # Build prompt for LLM
        llm_prompt = f"""
        You are an empathetic, highly efficient clinical AI doctor. 
        A patient has reported: "{prompt}"
        
        Your MedGuard-AI reasoning engine analyzed this globally:
        - Diagnosis: {diagnosis}
        - Verified Dataset Treatments/Precautions: {treatment}
        - Safety Score: {safety_score}/100 (Risk: {risk_label}){qs_text}
        
        Speak directly to the patient. State the diagnosis and any precautions. 
        If there are clarifying questions, ASK THEM directly to the patient so they can follow up with a real doctor.
        Keep your entire response STRICTLY under 4 sentences. Be direct. Do NOT hallucinate dosages.
        """

        try:
            response = self.model.generate_content(llm_prompt)
            return response.text
        except Exception as e:
            return f"❌ [Error generating Doctor Response]: {str(e)}"
