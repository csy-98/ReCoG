"""Offline construction of a concept graph with directed contrastive cues."""

from collections.abc import Sequence
from itertools import combinations
from typing import Protocol

from .prompts import extraction_prompt
from .schemas import Concept, ContrastGraph, PairEvidence


class EdgeExtractor(Protocol):
    """Frozen VLM adapter: inspect A/B support images and parse two cue lists.

    The adapter resolves support_image handles, sends images in A/B order with
    the prompt, and returns validated PairEvidence. Model loading and decoding
    belong to the caller. This repository ships no model or generated cue DB.
    """

    def __call__(self, a: Concept, b: Concept, prompt: str) -> PairEvidence: ...


def build_graph(concepts: Sequence[Concept], extract: EdgeExtractor) -> ContrastGraph:
    """One extraction call per unordered pair, storing both directed cue sets."""
    nodes = {concept.name: concept for concept in concepts}
    if len(nodes) != len(concepts) or any(not name for name in nodes):
        raise ValueError("Concept identifiers must be nonempty and unique.")
    graph = ContrastGraph(nodes)
    for a, b in combinations(concepts, 2):
        evidence = extract(a, b, extraction_prompt(a, b))
        graph.add(evidence.oriented(a.name, b.name))
    return graph
