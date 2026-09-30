"""Online pairwise comparisons and round-robin win aggregation."""

from collections.abc import Sequence
from itertools import combinations
from typing import Protocol

from .prompts import comparison_prompt
from .schemas import Candidate, Concept, ContrastGraph, Match, Prediction


class PairJudge(Protocol):
    """Frozen VLM adapter returning the exact winning concept identifier.

    Resolve and send images in query/A/B order, then the provided prompt.
    Parse the model's JSON Answer field; do not infer a winner from a substring.
    """

    def __call__(self, query_image: str, a: Concept, b: Concept, prompt: str) -> str: ...


def run_league(
    query_image: str,
    candidates: Sequence[Candidate],
    graph: ContrastGraph,
    judge: PairJudge,
) -> Prediction:
    """Each pair awards one win; final ties use CLIP similarity, then list order."""
    if not candidates:
        raise ValueError("Recognition requires at least one retrieved concept.")
    names = [candidate.concept.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("League candidates must be unique.")
    for candidate in candidates:
        if graph.nodes.get(candidate.concept.name) != candidate.concept:
            raise ValueError("Every candidate must match its graph node.")

    pairs = list(combinations(candidates, 2))
    # Validate graph coverage before making any model calls.
    evidence = [graph.evidence(a.concept.name, b.concept.name) for a, b in pairs]
    wins = dict.fromkeys(names, 0)
    matches = []
    for (a, b), cues in zip(pairs, evidence):
        winner = judge(query_image, a.concept, b.concept, comparison_prompt(cues))
        if winner not in (a.concept.name, b.concept.name):
            raise ValueError("Pair judge must return one of the two exact concept identifiers.")
        wins[winner] += 1
        matches.append(Match(a.concept.name, b.concept.name, winner))

    selected = max(candidates, key=lambda c: (wins[c.concept.name], c.similarity))
    return Prediction(selected.concept.name, tuple(candidates), wins, tuple(matches))
