# MedGuard-AI 🩺

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikitlearn&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4?logo=google&logoColor=white)
![NetworkX](https://img.shields.io/badge/Graph-NetworkX-green)
![PubMedBERT](https://img.shields.io/badge/Embeddings-PubMedBERT-yellow)
![License](https://img.shields.io/badge/License-MIT-purple)

**MedGuard-AI** is a high-fidelity, agentic Clinical Decision-Support System (CDSS) that combines **Generative AI reasoning** with a rigid **Deterministic Safety Verification Engine**. Designed as a research-grade clinical copilot, it mimics the human hospital workflow: a reasoning agent acts as the attending physician (diagnosing and forming a treatment plan), while a causal knowledge graph acts as the clinical pharmacist (auditing the plan against the patient's exact medical profile before approval).

> ⚠️ **Disclaimer**: MedGuard-AI is a research prototype. It is not a licensed medical device and must not be used for actual clinical decisions or self-diagnosis.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Hybrid Reasoning** | Primary diagnosis via **Gemini 2.5 Flash** LLM with graceful offline fallback to an ensemble ML classifier |
| 🧠 **ReAct Agent Loop** | Multi-step `Thought → Action → Observe` reasoning loop with a full reasoning trace output |
| 🎯 **Ensemble Classifier** | Soft-Voting Classifier (Random Forest + Gradient Boosting + SVM) mapping 131 symptoms → 41 diseases |
| 🔬 **Semantic Retrieval** | Local **PubMedBERT** 768-dim embeddings retrieve the most relevant clinical guidelines per diagnosis |
| 🛡️ **3-Layer Safety Verification** | Deterministic checks for **Allergies**, **Drug-Drug Interactions**, and **Medical History Contraindications** |
| 🕸️ **Knowledge Graph** | `NetworkX` directed graph encoding all safety rules for rigorous, explainable penalty calculation |
| 🩻 **Doctor Persona** | Gemini-powered empathetic persona that translates the technical output into patient-facing clinical language |
| 💻 **Rich CLI** | Beautiful terminal UI with interactive patient intake and batch JSON processing |

---

## 🏗️ Architecture

The pipeline is a 4-layer sequential system:

```
Patient Input (CLI / JSON)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Layer 0: Evidence Extractor                │
│  Validates & structures patient data        │
│  (age, symptoms, medications, history)      │
└────────────────────┬────────────────────────┘
                     │ EvidenceBundle
                     ▼
┌─────────────────────────────────────────────┐
│  Layer 1: ReAct Reasoning Agent             │
│                                             │
│  Step 1 → Fuzzy Symptom Matcher             │
│  Step 2 → Gemini LLM / Ensemble Classifier  │
│  Step 3 → PubMedBERT Guideline Retrieval    │
│  Step 4 → Treatment Plan Generation        │
└────────────────────┬────────────────────────┘
                     │ AgentResult
                     ▼
┌─────────────────────────────────────────────┐
│  Layer 2: Causal Verification Engine        │
│                                             │
│  ├─ Allergy Cross-Reactivity Check          │
│  ├─ Drug-Drug Interaction Check             │
│  ├─ Medical History Contraindications       │
│  └─ Risk Score Calculation (0–100)          │
└────────────────────┬────────────────────────┘
                     │ VerificationResult
                     ▼
┌─────────────────────────────────────────────┐
│  Layer 3: Decision Module                   │
│                                             │
│  Safety ≥ 85 & Confidence ≥ 60% → APPROVED │
│  Interaction detected            → ABSTAIN  │
└────────────────────┬────────────────────────┘
                     │ FinalDecision
                     ▼
        Doctor Persona (Gemini 2.5 Flash)
        Converts output to patient language
```

---

## 📁 Project Structure

```
MedGuard-AI/
│
├── cli/
│   └── main.py                  # Rich CLI with interactive + batch modes
│
├── data/
│   ├── preprocessor.py          # Cleans & engineers features from raw CSV
│   ├── processed/               # Engineered feature matrix, symptom list, encoders
│   └── raw/                     # Raw Kaggle dataset (gitignored)
│
├── knowledge_base/
│   ├── allergy_map.json          # Drug allergy cross-reactivity rules
│   ├── drug_interactions.json    # Drug-drug interaction pairs
│   ├── history_contraindications.json  # Condition-based drug contraindications
│   ├── treatment_guidelines.json # Evidence-based treatment protocols
│   └── build_knowledge_graph.py  # Builds the NetworkX safety graph
│
├── models/
│   ├── train_classifier.py       # Trains the ensemble ML classifier
│   ├── embed_guidelines.py       # Generates PubMedBERT guideline embeddings
│   ├── diagnosis_classifier.pkl  # Trained model artifact (gitignored)
│   └── label_encoder.pkl         # Label encoder artifact (gitignored)
│
├── schema/
│   ├── patient_schema.json       # JSON Schema for patient input validation
│   └── example_patients.json     # Sample patient records for batch mode
│
├── scripts/
│   └── download_dataset.py       # Downloads the Kaggle dataset
│
├── src/
│   ├── pipeline.py               # End-to-end pipeline orchestrator
│   ├── decision_module.py        # Layer 3: Final approve/abstain decision logic
│   ├── doctor_persona.py         # Gemini-powered empathetic Doctor output
│   │
│   ├── agent/
│   │   ├── evidence_extractor.py # Layer 0: Structures raw patient data
│   │   ├── reasoning_agent.py    # Layer 1: Full ReAct reasoning loop
│   │   ├── symptom_matcher.py    # Fuzzy + semantic symptom canonicalization
│   │   ├── guideline_retriever.py # PubMedBERT semantic guideline search
│   │   └── evidence_extractor.py
│   │
│   └── verifier/
│       ├── causal_verifier.py    # Layer 2: Orchestrates all safety checks
│       ├── contraindication_checker.py # History-based rule engine
│       ├── interaction_checker.py # Drug-drug interaction engine
│       ├── risk_scorer.py        # Penalty aggregation → safety score
│       └── knowledge_graph.py    # NetworkX graph loader
│
├── .env.example                  # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- A [Kaggle account](https://www.kaggle.com) (for dataset download)
- A [Google Gemini API Key](https://aistudio.google.com/app/apikey) *(optional, enables LLM reasoning & Doctor persona)*

### 1. Clone the repository

```bash
git clone https://github.com/Krrish258/MedGuard-AI.git
cd MedGuard-AI
```

### 2. Set up a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required for LLM diagnosis + Doctor Persona (optional but recommended)
GEMINI_API_KEY=your_gemini_api_key_here

# Required for dataset download
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# Optional: speeds up PubMedBERT download
HF_TOKEN=hf_your_token_here
```

> 🔒 The `.env` file is listed in `.gitignore` and will **never** be committed to version control.

### 4. Build the pipeline (one-time setup)

Run these scripts in order to prepare all models and knowledge artifacts:

```bash
python scripts/download_dataset.py     # Download clinical dataset from Kaggle
python data/preprocessor.py            # Feature engineering
python models/train_classifier.py      # Train the ensemble ML classifier
python knowledge_base/build_knowledge_graph.py  # Build the safety knowledge graph
python models/embed_guidelines.py      # Generate PubMedBERT guideline embeddings
```

---

## 💻 Usage

### Interactive Mode *(Recommended)*

Enter patient demographics and symptoms conversationally in the terminal:

```bash
python cli/main.py --interactive
```

You will be prompted for:
- Patient ID, Age, Sex
- Presenting symptoms (free-text, comma-separated)
- Current medications and allergies
- Relevant medical history

### Batch / JSON Mode

Process pre-configured patient records from a JSON file:

```bash
python cli/main.py --patient schema/example_patients.json --id PT-001
```

Example patient schema:
```json
{
  "PT-001": {
    "age": 45,
    "sex": "M",
    "symptoms": ["chest pain", "shortness of breath", "fatigue"],
    "medications": [{"name": "warfarin", "dose": "5mg"}],
    "allergies": ["penicillin"],
    "medical_history": ["hypertension", "type 2 diabetes"]
  }
}
```

---

## 📊 Output Format

Each pipeline run produces a full structured output including:

```
✅ DECISION:  APPROVED
🔬 Diagnosis: Pneumonia  (Confidence: 87.3%)
⚠️  Risk Label: LOW RISK  (Safety Score: 91/100)

Differential Diagnoses:
  1. Pneumonia       — 87.3%
  2. Bronchitis      — 6.1%
  3. COPD            — 3.4%

Treatment:  rest, consult doctor, medication, steam therapy

Safety Verification:
  Allergy Issues:      None
  Interaction Issues:  None
  History Issues:      None

Doctor's Assessment (Gemini):
  "Based on your symptoms of chest pain, shortness of breath, and
   fatigue, the most likely diagnosis is pneumonia..."

Reasoning Trace:
  Step 1: ANALYSE_SYMPTOMS  → 3 symptoms matched
  Step 2: PREDICT_DIAGNOSIS → Pneumonia (87.3%)
  Step 3: RETRIEVE_GUIDELINES → Guideline similarity: 0.923
  Step 4: GENERATE_TREATMENT → Dataset precaution protocol
```

---

## 🛠️ Technical Details

### Ensemble ML Classifier
- **Algorithm**: Soft-Voting Classifier (Random Forest + Gradient Boosting + SVM)
- **Input**: 131-dimensional binary symptom feature vector
- **Output**: 41 disease classes with calibrated probability scores
- **Fallback role**: Activated automatically when Gemini API is unavailable

### Semantic Guideline Retrieval
- **Model**: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (PubMedBERT)
- **Method**: Cosine similarity search over pre-computed 768-dim embeddings
- **Index**: `models/guideline_index.json` — 80+ clinical guidelines

### Causal Verification Engine
- **Graph**: `NetworkX` directed graph with typed safety edges
- **Rule sources**: `allergy_map.json`, `drug_interactions.json`, `history_contraindications.json`
- **Scoring**: Penalty-based safety score from 0–100 (threshold: 85)
- **Decision threshold**: Safety ≥ 85 AND Confidence ≥ 60% → `APPROVED`

---

## 📦 Dependencies

| Category | Packages |
|---|---|
| **Core ML** | `scikit-learn`, `pandas`, `numpy`, `joblib` |
| **Knowledge Graph** | `networkx` |
| **NLP / Embeddings** | `sentence-transformers`, `torch` |
| **LLM Integration** | `google-generativeai`, `python-dotenv` |
| **Fuzzy Matching** | `thefuzz`, `python-Levenshtein` |
| **CLI / Visualization** | `rich`, `matplotlib`, `seaborn` |
| **Dataset** | `kaggle` |

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request on the `final_version` branch.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Built with ❤️ for clinical safety research.*
