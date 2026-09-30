"""The complete online data flow, with model and storage dependencies explicit."""

from collections.abc import Sequence

from .matching import PairJudge, run_league
from .retrieval import retrieve_top3
from .schemas import ContrastGraph, Prediction


def recognize(
    query_image: str,
    query_embedding: Sequence[float],
    graph: ContrastGraph,
    judge: PairJudge,
) -> Prediction:
    """Retrieve up to three graph nodes, compare every pair, and select a winner."""
    candidates = retrieve_top3(query_embedding, list(graph.nodes.values()))
    return run_league(query_image, candidates, graph, judge)
