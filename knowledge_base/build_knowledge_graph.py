#!/usr/bin/env python3
"""
MedGuard-AI — Medical Knowledge Graph Builder
Constructs a NetworkX directed multigraph from all knowledge base files.

Node types: Disease, Symptom, Drug, Condition
Edge types:
  Disease  --HAS_SYMPTOM-->          Symptom
  Disease  --TREATED_BY-->           Drug
  Drug     --INTERACTS_WITH-->        Drug
  Drug     --CONTRAINDICATED_IN-->   Condition
  Allergen --CROSS_REACTS_WITH-->    Drug

Outputs:
  knowledge_base/medical_graph.pkl   → serialised NetworkX graph

Usage:
    python knowledge_base/build_knowledge_graph.py
"""

import json
import sys
from pathlib import Path

import joblib
import networkx as nx

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
KB   = ROOT / "knowledge_base"
DATA = ROOT / "data" / "processed"

TREATMENT_FILE      = KB / "treatment_guidelines.json"
INTERACTIONS_FILE   = KB / "drug_interactions.json"
ALLERGY_FILE        = KB / "allergy_map.json"
CONTRAINDIC_FILE    = KB / "history_contraindications.json"
SYMPTOM_LIST_FILE   = DATA / "symptom_list.json"
GRAPH_OUTPUT        = KB / "medical_graph.pkl"

# ── Severity → numeric weight mapping ─────────────────────────────────────────
SEVERITY_WEIGHT = {"CRITICAL": 1.0, "MAJOR": 0.75, "MODERATE": 0.5, "MINOR": 0.25}


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[WARN] File not found: {path} — skipping.")
        return {}
    with open(path) as f:
        return json.load(f)


def add_disease_symptom_nodes(G: nx.DiGraph, treatment_data: dict) -> None:
    """Add Disease nodes and TREATED_BY edges to Drug nodes."""
    for disease, data in treatment_data.items():
        if disease.startswith("_"):
            continue
        G.add_node(disease, node_type="Disease",
                   icd10=data.get("icd10", ""),
                   evidence_level=data.get("evidence_level", ""))

        drug = data.get("first_line_drug", "")
        if drug:
            G.add_node(drug, node_type="Drug",
                       drug_class=data.get("drug_class", ""))
            G.add_edge(disease, drug,
                       relation="TREATED_BY",
                       dose=data.get("dose", ""),
                       reasoning=data.get("reasoning", ""),
                       evidence_level=data.get("evidence_level", ""))

        # Add alternative drugs
        for alt in data.get("alternative", "").split(";"):
            alt = alt.strip()
            if alt:
                G.add_node(alt, node_type="Drug", drug_class="alternative")
                G.add_edge(disease, alt,
                           relation="ALTERNATIVE_TREATMENT",
                           evidence_level=data.get("evidence_level", ""))

    print(f"[OK] Added {sum(1 for n,d in G.nodes(data=True) if d.get('node_type')=='Disease')} disease nodes")
    print(f"[OK] Added {sum(1 for n,d in G.nodes(data=True) if d.get('node_type')=='Drug')} drug nodes")


def add_symptom_nodes(G: nx.DiGraph, symptom_list: list[str]) -> None:
    """Add canonical Symptom nodes (vocabulary from training data)."""
    for sym in symptom_list:
        if not G.has_node(sym):
            G.add_node(sym, node_type="Symptom")
    print(f"[OK] Added {len(symptom_list)} symptom nodes")


def add_drug_interactions(G: nx.DiGraph, interactions_data: dict) -> None:
    """Add INTERACTS_WITH bidirectional edges between drug pairs."""
    interactions = interactions_data.get("interactions", [])
    for item in interactions:
        drug_a    = item.get("drug_a", "")
        drug_b    = item.get("drug_b", "")
        severity  = item.get("severity", "MINOR")
        weight    = SEVERITY_WEIGHT.get(severity, 0.25)

        for d in (drug_a, drug_b):
            if not G.has_node(d):
                G.add_node(d, node_type="Drug")

        attrs = dict(
            relation="INTERACTS_WITH",
            severity=severity,
            weight=weight,
            effect=item.get("effect", ""),
            mechanism=item.get("mechanism", ""),
            management=item.get("management", ""),
        )
        G.add_edge(drug_a, drug_b, **attrs)
        G.add_edge(drug_b, drug_a, **attrs)  # bidirectional

    print(f"[OK] Added {len(interactions)} drug-drug interaction edges")


def add_allergy_contraindications(G: nx.DiGraph, allergy_data: dict) -> None:
    """Add CROSS_REACTS_WITH edges from allergen nodes to unsafe drugs."""
    count = 0
    for allergen, data in allergy_data.items():
        if allergen.startswith("_"):
            continue
        allergen_node = f"ALLERGEN:{allergen}"
        G.add_node(allergen_node, node_type="Allergen",
                   severity=data.get("severity", "MAJOR"))

        for drug in data.get("avoid_drugs", []):
            if not G.has_node(drug):
                G.add_node(drug, node_type="Drug")
            G.add_edge(allergen_node, drug,
                       relation="CROSS_REACTS_WITH",
                       severity=data.get("severity", "MAJOR"),
                       weight=SEVERITY_WEIGHT.get(data.get("severity", "MAJOR"), 0.5))
            count += 1

    print(f"[OK] Added {count} allergy cross-reactivity edges")


def add_history_contraindications(G: nx.DiGraph, contraindic_data: dict) -> None:
    """Add CONTRAINDICATED_IN edges from conditions to drugs."""
    count = 0
    for condition, data in contraindic_data.items():
        if condition.startswith("_"):
            continue
        cond_node = f"CONDITION:{condition}"
        G.add_node(cond_node, node_type="Condition",
                   severity=data.get("severity", "MAJOR"))

        for drug in data.get("contraindicated_drugs", []):
            if not G.has_node(drug):
                G.add_node(drug, node_type="Drug")
            G.add_edge(drug, cond_node,
                       relation="CONTRAINDICATED_IN",
                       severity=data.get("severity", "MAJOR"),
                       weight=SEVERITY_WEIGHT.get(data.get("severity", "MAJOR"), 0.5),
                       reason=data.get("reason", ""))
            count += 1

    print(f"[OK] Added {count} history contraindication edges")


def print_graph_summary(G: nx.DiGraph) -> None:
    """Print a summary of the constructed graph."""
    type_counts: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        t = data.get("node_type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    rel_counts: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "Unknown")
        rel_counts[r] = rel_counts.get(r, 0) + 1

    print("\n── Knowledge Graph Summary ──────────────────────────────")
    print(f"  Total nodes : {G.number_of_nodes()}")
    print(f"  Total edges : {G.number_of_edges()}")
    print("\n  Node types:")
    for ntype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ntype:<20} {count:>4}")
    print("\n  Edge (relation) types:")
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"    {rel:<35} {count:>4}")


def main() -> None:
    print("=" * 60)
    print("  MedGuard-AI — Knowledge Graph Builder")
    print("=" * 60)

    # Load all knowledge base files
    treatment_data  = load_json(TREATMENT_FILE)
    interactions    = load_json(INTERACTIONS_FILE)
    allergy_data    = load_json(ALLERGY_FILE)
    contraindic     = load_json(CONTRAINDIC_FILE)

    # Load symptom vocabulary (requires preprocessor to have run)
    symptom_list = []
    if SYMPTOM_LIST_FILE.exists():
        with open(SYMPTOM_LIST_FILE) as f:
            symptom_list = json.load(f)
    else:
        print("[WARN] symptom_list.json not found. Run: python data/preprocessor.py")

    # Build directed multigraph
    G = nx.DiGraph()
    G.graph["name"]    = "MedGuard-AI Medical Knowledge Graph"
    G.graph["version"] = "1.0"

    print("\n── Building graph nodes and edges ────────────────────────")
    add_disease_symptom_nodes(G, treatment_data)
    if symptom_list:
        add_symptom_nodes(G, symptom_list)
    add_drug_interactions(G, interactions)
    add_allergy_contraindications(G, allergy_data)
    add_history_contraindications(G, contraindic)

    # Summary
    print_graph_summary(G)

    # Save
    joblib.dump(G, GRAPH_OUTPUT, compress=3)
    print(f"\n[OK] Knowledge graph saved → {GRAPH_OUTPUT}")
    print("\n[DONE] Next step:")
    print("  python models/embed_guidelines.py")


if __name__ == "__main__":
    main()
