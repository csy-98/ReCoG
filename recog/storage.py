"""Private local inputs and generated databases; never a bundled concept list."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .schemas import Concept, ContrastGraph, PairEvidence
from .retrieval import normalize


@dataclass(frozen=True)
class SupportRecord:
    name: str
    image: str
    info: dict


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    """Refuse to silently replace existing private artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a rerun from overwriting an existing DB/result.
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def resolve_image(value, base):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Support/query image does not exist: {path}")
    return str(path)


def load_support(path, *, r2p=False, image_root=None):
    """Accept a private manifest or R2P concept_dict DB; select one support/image."""
    path = Path(path).expanduser().resolve()
    data = read_json(path)
    base = Path(image_root).expanduser().resolve() if image_root else path.parent
    records = []
    if r2p:
        concepts = data.get("concept_dict", {})
        if not isinstance(concepts, dict) or not concepts:
            raise ValueError("The R2P DB must contain a nonempty concept_dict.")
        for name, item in concepts.items():
            image = item.get("image")
            if not image:
                image = next((p for p, c in data.get("path_to_concept", {}).items()
                              if c == name), None)
            if isinstance(image, list):
                image = image[0] if image else None
            if not image:
                raise ValueError(f"Missing support image for {name}.")
            info = item.get("info", {})
            if isinstance(info, str):
                info = {"general": info}
            records.append(SupportRecord(name.strip("<>"), resolve_image(image, base), info))
    else:
        rows = data.get("concepts", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("The manifest must contain a nonempty concepts list.")
        for row in rows:
            info = row.get("info", {})
            records.append(SupportRecord(row["name"], resolve_image(row["image"], base), info))
    names = [record.name for record in records]
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise ValueError("Support concept names must be nonempty and unique.")
    if any(not isinstance(record.info, dict) for record in records):
        raise ValueError("Each concept's info must be an object.")
    return records


def save_database(path, graph, descriptions, clip_model):
    write_json(path, {
        "schema_version": 1,
        "clip_model": clip_model,
        "concepts": [asdict(concept) for concept in graph.nodes.values()],
        "descriptions": descriptions,
        "pairs": [asdict(evidence) for evidence in graph.pairs.values()],
    })


def load_database(path):
    data = read_json(path)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported database schema_version.")
    concepts = [Concept(row["name"], row["support_image"], tuple(row["embedding"]))
                for row in data["concepts"]]
    nodes = {concept.name: concept for concept in concepts}
    if not nodes or len(nodes) != len(concepts) or any(not name for name in nodes):
        raise ValueError("Database must contain unique, nonempty concept names.")
    dimensions = set()
    for concept in concepts:
        dimensions.add(len(normalize(concept.embedding)))
    if len(dimensions) != 1:
        raise ValueError("Support embeddings must all have the same dimension.")
    graph = ContrastGraph(nodes)
    for row in data.get("pairs", []):
        if any(not isinstance(row.get(key, []), list) for key in ("a_to_b", "b_to_a", "shared")):
            raise ValueError("Stored directional/shared cues must be JSON arrays.")
        evidence = PairEvidence(row["a"], row["b"], tuple(row["a_to_b"]),
                                tuple(row["b_to_a"]), tuple(row.get("shared", [])))
        key = tuple(sorted((evidence.a, evidence.b)))
        if key in graph.pairs:
            raise ValueError("Database contains duplicate concept pairs.")
        graph.add(evidence)
    return graph, data.get("descriptions", {}), data["clip_model"]
