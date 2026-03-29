import os
import json
from pathlib import Path
from google import genai
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent
SYMPTOM_LIST_PATH = ROOT / "data" / "processed" / "symptom_list.json"

# Load vocabulary
with open(SYMPTOM_LIST_PATH) as f:
    VOCABULARY = json.load(f)


class ExtractedSymptoms(BaseModel):
    model_config = ConfigDict(extra='forbid')
    symptoms: list[str]

class LLMService:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash"

    def parse_symptoms(self, text: str) -> list[str]:
        """
        Extracts formal symptoms from natural language explicitly matching the 
        canonical vocabulary.
        """
        prompt = f"""
        You are an expert clinical coding assistant.
        Your task is to extract symptoms from the patient's natural language input 
        and map them strictly to the following canonical vocabulary list.

        Canonical Vocabulary: {VOCABULARY}

        Patient Input: "{text}"

        Please return a strictly formatted JSON object with a single key 'symptoms' 
        containing a list of matching strings from the vocabulary.
        If a symptom is mentioned that is not in the list, try to map it to the closest 
        synonymous term in the list. If it cannot be mapped, omit it.
        Return ONLY valid JSON. 
        Example: {{"symptoms": ["headache", "chest_pain"]}}
        """
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        try:
            json_str = response.text.replace('```json', '').replace('```', '').strip()
            data = json.loads(json_str)
            return data.get("symptoms", [])
        except Exception as e:
            print(f"Failed to parse LLM structured output: {e}\nRaw output: {response.text}")
            return []

    def explain_diagnosis(self, symptoms: list[str], diagnosis: str) -> str:
        """
        Generates clinical reasoning for the user based on ML prediction.
        """
        prompt = f"""
        You are a supportive, knowledgeable AI clinical assistant.
        A probabilistic machine learning model has evaluated a patient presenting with the following symptoms:
        {symptoms}

        The model predicts the most likely diagnosis is: {diagnosis}
        
        Write a brief, patient-friendly explanation (3-5 sentences) summarizing why these specific symptoms 
        might lead to a diagnosis of {diagnosis}. Keep the tone professional but empathetic. 
        Add a standard medical disclaimer at the end that this is an AI tool and they should consult a real doctor.
        """
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return response.text.strip()
