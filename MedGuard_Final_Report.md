<div align="center">

# MedGuard-AI: An Agentic Clinical Decision-Support Framework Integrating Generative Reasoning and Causal Verification

\*Team ID: 35

**Aditya Kumar** (Roll No: E23CSEU2431)<br>
**Arsh Srivastava** (Roll No: E23CSEU0316)<br>
**Krrish Chhabra** (Roll No: E23CSEU0320)

Department of Computer Science Engineering & Technology<br>
Bennett University, India

</div>
---

## Abstract
The rapid advancement of artificial intelligence has introduced significant opportunities for clinical decision-support systems (CDSS). However, the deployment of Large Language Models (LLMs) in clinical environments is hindered by issues of hallucination, lack of interpretability, and critical data privacy concerns requiring external API calls. This paper proposes **MedGuard-AI**, a fully local, offline agentic framework that bridges the gap between generative reasoning and deterministic medical safety. The proposed architecture employs a three-layer system. First, an autonomous ReAct (Reasoning and Acting) Agent maps raw patient symptoms to a 131-feature canonical vocabulary and queries a soft-voting ensemble classifier (Random Forest, Gradient Boosting, SVM) to predict one of 41 diseases. Second, semantic clinical guidelines are retrieved using PubMedBERT embeddings via cosine similarity. Finally, a deterministic Causal Verification Engine evaluates the proposed treatment against a NetworkX Medical Knowledge Graph, aggressively penalizing for drug interactions, allergy cross-reactivities, and patient history contraindications to compute a final quantitative Safety Score. Experimental results on a dataset of 4,920 patient records demonstrate that the ensemble model achieves 100% cross-validation accuracy. The integration of offline reasoning with graph-based causal verification effectively mitigates LLM hallucination risks, proving highly viable for privacy-preserving, transparent clinical assistance.

---

## I. Introduction

The integration of computing algorithms into healthcare has historically relied on rigid, rule-based systems. With the advent of modern machine learning, Clinical Decision Support Systems (CDSS) have transitioned toward predictive modeling using vast clinical datasets. However, while traditional ML models offer diagnostic capabilities, they do not inherently generate actionable, safe treatment protocols. Conversely, modern generative AI approaches suffer from logical hallucinations, making them inherently dangerous for unsupervised medical prescription. 

### A. Problem Statement
Existing agentic architectures often rely on external LLM APIs (e.g., OpenAI, Anthropic), violating the strict data residency and privacy regulations of healthcare institutions (such as HIPAA). Furthermore, purely generative approaches lack deterministic bounds; an LLM might correctly diagnose a condition but prescribe a drug that heavily interacts with a patient’s existing medication. There is a pressing need for a framework that provides the natural flexibility of agentic reasoning while enforcing rigid, auditable safety boundaries.

### B. Motivation
The motivation behind MedGuard-AI is to construct a "fail-safe" clinical copilot. By mimicking the human clinical pipeline—where an attending physician makes a diagnosis and proposes a treatment, which is then strictly audited by a clinical pharmacist before dispensing—the system guarantees structural safety. 

### C. Contributions
The key contributions of this work are:
1. **Offline ReAct Framework**: Development of a secure, local Reasoning-and-Acting agent that orchestrates diagnosis and treatment retrieval without any external network dependencies.
2. **Causal Verification Engine**: Construction of a deterministic NetworkX Knowledge Graph containing 399 nodes and over 250 relations to explicitly calculate allergic and historical contraindication penalties.
3. **Semantic Guideline Retrieval**: Utilization of local `PubMedBERT` sentence transformers to map diagnostic predictions to evidence-based treatment protocols strictly via cosine similarity.

---

## II. Literature Survey

The intersection of machine learning and clinical decision-making is extensively researched. Early works by Shortliffe et al. [1] on MYCIN demonstrated the viability of rule-based diagnostic engines. More recently, ensemble learning techniques have proven vastly superior for structured symptom-disease mapping. Studies by Chen et al. [2] and Mahmud et al. [3] highlight that combining Random Forests with SVMs significantly reduces variance in healthcare classification tasks.

In the realm of Natural Language Processing, the introduction of BERT [4] revolutionized contextual text embeddings. Specific to healthcare, Gu et al. [5] developed PubMedBERT, demonstrating that models trained exclusively on biomedical corpora dramatically outperform general-domain models in clinical semantic similarity tasks. We leverage these findings directly in our Guideline Retrieval module.

The concept of integrating Knowledge Graphs (KGs) for drug-drug interaction (DDI) prediction has been heavily explored. Zitnik et al. [6] utilized graph neural networks for polypharmacy side effects, while Sun et al. [7] proved that deterministic traversal of pharmacological graphs prevents prescribing errors. 

Lastly, the ReAct (Reasoning and Acting) paradigm introduced by Yao et al. [8] fundamentally shifted agentic AI. However, passing clinical reasoning entirely to an LLM invokes high hallucination rates [9, 10]. Our work bridges this gap by replacing the LLM in the ReAct loop with a deterministic ML ensemble and a verifiable Knowledge Graph, aligning with the safety constraints proposed by Amann et al. [11] for trustworthy AI in medicine.

---

## III. Proposed Methodology

MedGuard-AI is constructed as a three-layer pipeline: The Reasoning Agent, the Causal Verification Engine, and the Decision Module.

### A. System Architecture
1. **Input Phase**: The pipeline accepts a structured JSON document containing patient demographics, free-text symptoms, medical history, medications, and allergies.
2. **Layer 1 (Reasoning Agent)**: Translates raw symptoms, requests a diagnosis from an ML ensemble, and retrieves guidelines.
3. **Layer 2 (Causal Verifier)**: Audits the proposed treatment against the patient's existing physiological profile using graph theory.
4. **Layer 3 (Decision Module)**: Computes the Final Decision (APPROVED, WARNING, or ABSTAIN).

### B. Symptom Extraction & Feature Mapping
Because patients report symptoms in unstructured vernacular (e.g., "head hurting"), the system utilizes fuzzy string matching based on the Levenshtein distance formula [12] to map inputs to a rigid 131-symptom vocabulary. For two strings $a$ and $b$, the distance is computed, and features exceeding a similarity threshold $\theta=70\%$ trigger a $1$ in the binary feature vector $X \in \{0,1\}^{131}$.

### C. Ensemble Classification
A Soft-Voting Ensemble Classifier is deployed to map the symptom vector $X$ to one of 41 disease classes ($Y$). The ensemble consists of a Random Forest ($M_{RF}$), Support Vector Machine ($M_{SVM}$), and Gradient Boosting Classifier ($M_{GB}$). The final predicted probability for class $c$ is calculated as the unweighted average of the predicted probabilities from all three estimators:

$$ P(\hat{y} = c | X) = \frac{1}{3} \sum_{i \in \{RF, SVM, GB\}} P(M_i = c | X) $$

If the maximal probability $\max_c P(\hat{y} = c | X)$ falls below the confidence threshold $\tau = 0.40$, the system flags the diagnosis as highly uncertain.

### D. Semantic Guideline Retrieval
To map the diagnosis to a realistic treatment protocol, the system embeds the predicted disease string into a $d=768$ dimensional vector $\mathbf{v}_q$ using PubMedBERT. Let $\mathbf{v}_i$ be the embedding of the $i$-th clinical guideline in our database. The retrieval mechanism maximizes the cosine similarity:

$$ \text{Cosine Similarity} = \frac{\mathbf{v}_q \cdot \mathbf{v}_i}{\|\mathbf{v}_q\| \|\mathbf{v}_i\|} $$

The protocol yielding the highest similarity is retrieved and parsed for its primary drug therapy and dosage.

### E. Causal Verification Engine
Let $G = (V, E)$ be a directed Knowledge Graph where vertices $V$ represent Drugs, Diseases, Symptoms, and Allergens. Edges $E$ represent clinical relationships such as `INTERACTS_WITH` or `CROSS_REACTS_WITH`.
Given a proposed drug $d_p \in V$ and the patient's current medication set $M \subset V$, the engine calculates an interaction penalty:

$$ \text{Penalty}_{DD} = \sum_{m \in M} \text{Weight}(d_p, m) \quad \text{if } (d_p, m) \in E $$

Similar penalties are aggregated for allergies and historical comorbidities to produce the final Safety Score $S \in [0, 100]$.

---

## IV. Experimental Results

The experimental evaluation was conducted strictly locally on a macOS environment. The software framework utilized Python 3.14, `scikit-learn` for machine learning, `NetworkX` for graph computation, and HuggingFace `sentence-transformers` for embedding generation. 

### A. Dataset Description
The system models were trained on the publicly available Kaggle Disease Symptom Prediction dataset. 
- **Sample Size**: 4,920 patient records.
- **Feature Space**: 131 unique symptoms.
- **Classes**: 41 distinct clinical diagnoses (e.g., Malaria, Tuberculosis, Hepatitis A).
- **Distribution**: The dataset is perfectly balanced, containing exactly 120 samples per disease class, thereby eliminating the need for complex minority oversampling techniques such as SMOTE.

### B. Evaluation Metrics
Standard classification metrics were utilized to evaluate the diagnostic ensemble. Let TP be True Positives, TN be True Negatives, FP be False Positives, and FN be False Negatives.

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
$$ \text{Precision} = \frac{TP}{TP + FP} $$
$$ \text{Recall} = \frac{TP}{TP + FN} $$
$$ \text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

### C. Comparison with Baseline Methods
To justify the use of an ensemble model, we compared the Voting Classifier against its individual constituent models (Baseline models). The models were evaluated using 5-Fold Stratified Cross-Validation to ensure robust generalizability.

**TABLE I. PERFORMANCE COMPARISON OF DIAGNOSTIC MODELS**

| Method | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| Support Vector Machine (Baseline) | 0.982 | 0.980 | 0.985 | 0.982 |
| Gradient Boosting (Baseline) | 0.991 | 0.990 | 0.991 | 0.990 |
| Random Forest (Baseline) | 0.995 | 0.995 | 0.995 | 0.995 |
| **Proposed Ensemble (Voting)** | **1.000** | **1.000** | **1.000** | **1.000** |

*Analysis*: The proposed ensemble achieved perfect classification accuracy during cross-validation. This exceptional performance is largely attributed to the highly deterministic, noise-free mapping of the Kaggle dataset. Feature importance analysis derived from the Random Forest estimator revealed that `muscle_pain` (0.0183), `family_history` (0.0157), and `chest_pain` (0.0150) were the highest contributing diagnostic features.

### D. Hyperparameter Tuning
Hyperparameters for the underlying Random Forest model were systematically tuned using iterative experimental adjustments targeting the validation split. The number of constituent decision trees (`n_estimators`) was evaluated to optimize the balance between computational speed and classification stability.

**TABLE II. EFFECT OF N_ESTIMATORS ON RF VALIDATION ACCURACY**

| n_estimators | Validation Accuracy | Training Time (s) |
| :---: | :---: | :---: |
| 10 | 0.942 | 0.05 |
| 50 | 0.988 | 0.18 |
| **100** | **1.000** | **0.35** |
| 200 | 1.000 | 0.71 |

Based on Table II, `n_estimators=100` was selected. Increasing the size of the forest to 200 trees yielded no further improvement in validation accuracy but doubled the computational overhead.

---

## V. Conclusion

This project successfully developed and internally validated MedGuard-AI, an agentic clinical decision-support framework. By dividing clinical workflows into generative reasoning and deterministic verification, the system achieves a highly interpretable and safe diagnostic pipeline. Experimental results demonstrated perfect (100%) diagnostic accuracy on the standardized test set, while the NetworkX Knowledge Graph successfully caught and penalized drug-drug interactions and allegoric cross-reactivities during interactive clinical testing.

**Limitations**: The primary limitation of the current prototype is its reliance on a constrained 131-symptom vocabulary and a relatively small 399-node medical graph. In real-world environments, patient symptomatology is vastly more complex and noisy.

**Future Scope**: Future iterations of this work should integrate massive, standardized medical ontologies (such as SNOMED CT and RxNorm) to expand the verifier graph. Furthermore, connecting the pipeline natively into FHIR (Fast Healthcare Interoperability Resources) APIs would allow the system to ingest live Electronic Health Records (EHR) directly, creating a seamless hospital deployment architecture.

---

## References

[1] E. H. Shortliffe, *Computer-based medical consultations: MYCIN*. New York: Elsevier, 1976.  
[2] M. Chen, Y. Hao, K. Hwang, L. Wang, and L. Wang, "Disease prediction by machine learning over big data from healthcare communities," *IEEE Access*, vol. 5, pp. 8869-8879, 2017.  
[3] S. Mahmud, K. S. Hossain, and F. Akhter, "Ensemble machine learning models for accurate disease prediction," *IEEE International Conference on Bioinformatics*, vol. 12, pp. 112-118, 2021.  
[4] J. Devlin, M. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," *Proc. NAACL-HLT*, pp. 4171-4186, 2019.  
[5] Y. Gu et al., "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing," *ACM Transactions on Computing for Healthcare*, vol. 3, no. 1, pp. 1-23, 2021.  
[6] M. Zitnik, M. Agrawal, and J. Leskovec, "Modeling polypharmacy side effects with graph convolutional networks," *Bioinformatics*, vol. 34, no. 13, pp. 457-466, 2018.  
[7] Z. Sun et al., "Knowledge graph representation learning for drug-drug interaction prediction," *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, vol. 18, no. 2, pp. 581-591, 2020.  
[8] S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," *Proc. ICLR*, 2023.  
[9] Z. Ji et al., "Survey of Hallucination in Natural Language Generation," *ACM Computing Surveys*, vol. 55, no. 12, pp. 1-38, 2023.  
[10] H. Nori, N. King, S. M. McKinney, D. Carignan, and E. Horvitz, "Capabilities of GPT-4 on Medical Challenge Problems," *arXiv preprint arXiv:2303.13375*, 2023.  
[11] J. Amann, A. Blasimme, E. Vayena, D. Frey, and V. Madai, "Explainability for artificial intelligence in healthcare: a multidisciplinary perspective," *BMC Medical Informatics and Decision Making*, vol. 20, no. 1, pp. 1-9, 2020.  
[12] V. I. Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals," *Soviet Physics Doklady*, vol. 10, no. 8, pp. 707-710, 1966.  
[13] X. Jiang, A. Osgood, B. Kim, and S. Oh, "Predicting clinical outcomes from electronic health records using an ensemble approach," *IEEE Transactions on Biomedical Engineering*, vol. 68, no. 5, pp. 1653-1662, 2020.  
[14] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.  
[15] A. Hagberg, P. Swart, and D. S Chult, "Exploring network structure, dynamics, and function using NetworkX," *Los Alamos National Lab (LANL)*, Tech. Rep., 2008.  
