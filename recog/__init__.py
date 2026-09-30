"""ReCoG: a compact, methodology-focused reference implementation."""

from .graph import build_graph
from .matching import run_league
from .pipeline import recognize
from .retrieval import retrieve_top3
from .schemas import Candidate, Concept, ContrastGraph, PairEvidence

__all__ = [
    "Candidate", "Concept", "ContrastGraph", "PairEvidence",
    "build_graph", "recognize", "retrieve_top3", "run_league",
]
