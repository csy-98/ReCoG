"""Explicit contracts for support concepts, directional cues, and predictions."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Concept:
    """One concept, one support image, and its frozen CLIP image embedding.

    support_image is an opaque identifier resolved by the caller's VLM adapter.
    No image loading, database access, or model download happens here.
    """

    name: str
    support_image: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class Candidate:
    concept: Concept
    similarity: float


@dataclass(frozen=True)
class PairEvidence:
    """Two directed descriptions from one unordered-pair extraction call.

    a_to_b describes A relative to B; b_to_a describes B relative to A.
    These are textual discriminative cues, not numerical edge weights.
    """

    a: str
    b: str
    a_to_b: tuple[str, ...]
    b_to_a: tuple[str, ...]
    shared: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.a or not self.b or self.a == self.b:
            raise ValueError("An edge must connect two distinct, named concepts.")
        for cues in (self.a_to_b, self.b_to_a):
            if len(cues) != 3 or any(not isinstance(cue, str) or not cue.strip() for cue in cues):
                raise ValueError("Each direction requires three nonempty distinctive cues.")

    def oriented(self, a: str, b: str) -> "PairEvidence":
        if (a, b) == (self.a, self.b):
            return self
        if (a, b) == (self.b, self.a):
            return PairEvidence(a, b, self.b_to_a, self.a_to_b, self.shared)
        raise ValueError("Requested concepts do not match this edge.")


@dataclass
class ContrastGraph:
    nodes: dict[str, Concept]
    pairs: dict[tuple[str, str], PairEvidence] = field(default_factory=dict)

    def add(self, evidence: PairEvidence) -> None:
        if evidence.a not in self.nodes or evidence.b not in self.nodes:
            raise ValueError("Both edge endpoints must exist in the graph.")
        self.pairs[tuple(sorted((evidence.a, evidence.b)))] = evidence

    def evidence(self, a: str, b: str) -> PairEvidence:
        return self.pairs[tuple(sorted((a, b)))].oriented(a, b)


@dataclass(frozen=True)
class Match:
    a: str
    b: str
    winner: str


@dataclass(frozen=True)
class Prediction:
    winner: str
    candidates: tuple[Candidate, ...]
    wins: dict[str, int]
    matches: tuple[Match, ...]
