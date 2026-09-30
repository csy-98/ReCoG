"""CLIP cosine-similarity retrieval over one support embedding per concept."""

from math import hypot, isfinite
from collections.abc import Sequence

from .schemas import Candidate, Concept


def normalize(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or not all(isfinite(value) for value in values):
        raise ValueError("An embedding must contain finite values and be nonempty.")
    norm = hypot(*values)
    if norm == 0 or not isfinite(norm):
        raise ValueError("An embedding must have a finite, nonzero norm.")
    return tuple(value / norm for value in values)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = normalize(left), normalize(right)
    if len(a) != len(b):
        raise ValueError("Query and support embedding dimensions must match.")
    return sum(x * y for x, y in zip(a, b))


def retrieve_top3(
    query_embedding: Sequence[float], concepts: Sequence[Concept]
) -> tuple[Candidate, ...]:
    """Rank concepts by cosine similarity and retain at most three.

    Embeddings must come from the same frozen CLIP image encoder. The original
    encoder configuration is openai/clip-vit-large-patch14-336. For normalized
    vectors, this scoring is the inner product used by a flat FAISS index.
    No question, target label, or concept-name hint is an input to retrieval.
    """
    names = [concept.name for concept in concepts]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Concept identifiers must be nonempty and unique.")
    query = normalize(query_embedding)
    ranked = [Candidate(concept, cosine(query, concept.embedding)) for concept in concepts]
    # Python's stable sort preserves input order for exactly equal scores.
    ranked.sort(key=lambda candidate: candidate.similarity, reverse=True)
    return tuple(ranked[:3])
