"""Readable prompt templates; image content is supplied by a caller-owned adapter."""

import json

from .schemas import Concept, PairEvidence


def extraction_prompt(a: Concept, b: Concept) -> str:
    schema = {
        f"{a.name}_distinct_features": ["cue 1", "cue 2", "cue 3"],
        f"{b.name}_distinct_features": ["cue 1", "cue 2", "cue 3"],
    }
    return f"""Compare the two support concepts.
Image 1: {a.name}. Image 2: {b.name}.
For each concept, describe exactly three fine-grained physical features that
distinguish it from the other concept. Exclude background, environmental context,
and generic features common to both. The descriptions must be pair-specific.
Return only valid JSON in this form:
{json.dumps(schema, indent=2)}"""


def comparison_prompt(evidence: PairEvidence) -> str:
    return f"""Identify which support concept the query image matches.
Image 1: query. Image 2: {evidence.a}. Image 3: {evidence.b}.
Features of {evidence.a} relative to {evidence.b}:
{json.dumps(evidence.a_to_b)}
Features of {evidence.b} relative to {evidence.a}:
{json.dumps(evidence.b_to_a)}
Shared features to ignore as a basis for the decision:
{json.dumps(evidence.shared)}
Compare the query to both support images using the distinctive cues.
Ignore superficial changes in pose, background, and lighting.
Return only JSON with "Answer" (exactly one of {json.dumps([evidence.a, evidence.b])})
and "Reasoning" (a concise explanation grounded in the visible cues)."""
