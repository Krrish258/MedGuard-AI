"""
MedGuard-AI — Knowledge Graph Query Interface
Provides clean query methods over the NetworkX medical knowledge graph.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import joblib
import networkx as nx

ROOT        = Path(__file__).resolve().parent.parent.parent
GRAPH_PATH  = ROOT / "knowledge_base" / "medical_graph.pkl"


class KnowledgeGraph:
    """Thread-safe singleton wrapper around the NetworkX knowledge graph."""

    _instance: Optional["KnowledgeGraph"] = None
    _graph: Optional[nx.DiGraph]          = None

    def __new__(cls) -> "KnowledgeGraph":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._graph is not None:
            return
        if not GRAPH_PATH.exists():
            raise FileNotFoundError(
                f"Knowledge graph not found at {GRAPH_PATH}. "
                "Run: python knowledge_base/build_knowledge_graph.py"
            )
        self._graph = joblib.load(GRAPH_PATH)

    @property
    def G(self) -> nx.DiGraph:
        self._ensure_loaded()
        return self._graph

    # ── Disease queries ───────────────────────────────────────────────────────

    def get_treatments_for(self, disease: str) -> list[dict]:
        """Return all drugs linked to a disease via TREATED_BY edges."""
        results = []
        disease_lower = disease.lower()
        for node in self.G.nodes:
            if str(node).lower() == disease_lower:
                for _, drug, data in self.G.out_edges(node, data=True):
                    if data.get("relation") in ("TREATED_BY", "ALTERNATIVE_TREATMENT"):
                        results.append({
                            "drug"           : drug,
                            "relation"       : data.get("relation"),
                            "dose"           : data.get("dose", ""),
                            "evidence_level" : data.get("evidence_level", ""),
                        })
        return results

    # ── Drug queries ──────────────────────────────────────────────────────────

    def get_interactions_for(self, drug: str) -> list[dict]:
        """Return all known drug-drug interactions for a given drug."""
        drug_lower = drug.lower()
        results    = []
        for node in self.G.nodes:
            if str(node).lower() == drug_lower:
                for _, target, data in self.G.out_edges(node, data=True):
                    if data.get("relation") == "INTERACTS_WITH":
                        results.append({
                            "interacts_with": target,
                            "severity"      : data.get("severity", "MINOR"),
                            "weight"        : data.get("weight", 0.25),
                            "effect"        : data.get("effect", ""),
                            "management"    : data.get("management", ""),
                        })
        return results

    def get_contraindications_for(self, drug: str) -> list[dict]:
        """Return all conditions for which the drug is contraindicated."""
        drug_lower = drug.lower()
        results    = []
        for node in self.G.nodes:
            if str(node).lower() == drug_lower:
                for _, target, data in self.G.out_edges(node, data=True):
                    if data.get("relation") == "CONTRAINDICATED_IN":
                        condition = str(target).replace("CONDITION:", "")
                        results.append({
                            "condition": condition,
                            "severity" : data.get("severity", "MAJOR"),
                            "weight"   : data.get("weight", 0.5),
                            "reason"   : data.get("reason", ""),
                        })
        return results

    def get_allergen_avoid_list(self, allergen: str) -> list[str]:
        """Return list of drugs unsafe for a given allergen."""
        allergen_node = f"ALLERGEN:{allergen}"
        if not self.G.has_node(allergen_node):
            return []
        return [
            target
            for _, target, data in self.G.out_edges(allergen_node, data=True)
            if data.get("relation") == "CROSS_REACTS_WITH"
        ]

    def node_exists(self, name: str) -> bool:
        return any(str(n).lower() == name.lower() for n in self.G.nodes)
