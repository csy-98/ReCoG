"""Local database preparation and R2P/ReCoG inference entry points."""
import argparse
from dataclasses import asdict
import gc
from itertools import combinations
import json
from pathlib import Path

from .graph import build_graph
from .matching import run_league
from .retrieval import retrieve_top3
from .schemas import Concept, ContrastGraph, PairEvidence
from .storage import load_database, load_support, resolve_image, save_database, write_json


def make_backend(args):
    from .runtime import MiniCPMBackend
    return MiniCPMBackend(args.vlm_model, args.device, args.dtype, args.max_new_tokens, args.seed)


def release_encoder(encoder):
    # Free GPU memory before loading the larger VLM.
    encoder.model.to("cpu")
    del encoder.model
    gc.collect()
    if encoder.torch.cuda.is_available():
        encoder.torch.cuda.empty_cache()


def import_pairs(concepts, directory, shared_directory=None):
    from r2p.pairwise_loader import PairwiseDistinctFeaturesLoader, SharedFeaturesLoader
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Pairwise directory not found: {directory}")
    loader = PairwiseDistinctFeaturesLoader(str(directory))
    shared = SharedFeaturesLoader(str(Path(shared_directory).expanduser())) if shared_directory else None
    graph = ContrastGraph({c.name: c for c in concepts})
    for a, b in combinations(concepts, 2):
        data = loader.load_pairwise_features(a.name, b.name)
        if not data:
            raise ValueError(f"Missing or invalid features for {a.name} / {b.name}.")
        if any(not isinstance(data[key], list) for key in
               ("concept1_distinct_features", "concept2_distinct_features")):
            raise ValueError("Pairwise features must be JSON arrays, not plain strings.")
        common = shared.load_shared_features(a.name, b.name) if shared else []
        if common is not None and (not isinstance(common, list) or
                                   any(not isinstance(cue, str) for cue in common)):
            raise ValueError("Shared features must be a JSON array of strings.")
        graph.add(PairEvidence(a.name, b.name, tuple(data["concept1_distinct_features"]),
                              tuple(data["concept2_distinct_features"]), tuple(common or [])))
    return graph


def build(args):
    from r2p.retriever import ClipEncoder
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Output already exists; choose a new database file.")
    records = load_support(args.manifest or args.r2p_db, r2p=bool(args.r2p_db), image_root=args.image_root)
    clip_model = args.clip_model
    if Path(clip_model).expanduser().is_dir():
        clip_model = str(Path(clip_model).expanduser().resolve())
    encoder = ClipEncoder(clip_model, args.device)
    embeddings = encoder.encode([r.image for r in records], args.batch_size)
    release_encoder(encoder)
    concepts = [Concept(r.name, r.image, e) for r, e in zip(records, embeddings)]
    backend = None
    descriptions = {}
    for record in records:
        if record.info:
            descriptions[record.name] = record.info
        else:
            backend = backend or make_backend(args)
            descriptions[record.name] = backend.describe(record.image)
    if args.skip_graph:
        graph = ContrastGraph({c.name: c for c in concepts})
    elif args.pairwise_dir:
        graph = import_pairs(concepts, args.pairwise_dir, args.shared_dir)
    else:
        backend = backend or make_backend(args)
        graph = build_graph(concepts, backend.extract)
    save_database(output, graph, descriptions, clip_model)
    print(f"Saved local database: {output} ({len(concepts)} concepts, {len(graph.pairs)} pairs)")


def infer(args):
    from r2p.retriever import ClipEncoder
    if args.output and Path(args.output).expanduser().exists():
        raise FileExistsError("Output already exists; choose a new result file.")
    graph, descriptions, clip_model = load_database(Path(args.database).expanduser())
    query = resolve_image(args.query, Path.cwd())
    for node in graph.nodes.values():
        resolve_image(node.support_image, Path.cwd())
    encoder = ClipEncoder(clip_model, args.device)
    query_embedding = encoder.encode([query])[0]
    release_encoder(encoder)
    candidates = retrieve_top3(query_embedding, list(graph.nodes.values()))
    if args.method == "recog":
        for a, b in combinations(candidates, 2):
            graph.evidence(a.concept.name, b.concept.name)
    elif any(not descriptions.get(c.concept.name) for c in candidates):
        raise ValueError("The database lacks descriptions needed for R2P.")
    backend = make_backend(args)
    result = {"method": args.method,
              "candidates": [{"name": c.concept.name, "similarity": c.similarity} for c in candidates]}
    if args.method == "recog":
        prediction = run_league(query, candidates, graph, backend.judge(graph))
        result.update(winner=prediction.winner, wins=prediction.wins,
                      matches=[asdict(m) for m in prediction.matches])
    else:
        winner, response = backend.select_r2p(query, candidates, descriptions)
        result.update(winner=winner, reasoning=response)
    if args.output:
        write_json(Path(args.output).expanduser(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def validate(args):
    graph, descriptions, clip_model = load_database(Path(args.database).expanduser())
    for node in graph.nodes.values():
        resolve_image(node.support_image, Path.cwd())
    missing = [pair for pair in combinations(graph.nodes, 2) if tuple(sorted(pair)) not in graph.pairs]
    if args.method == "recog" and missing:
        raise ValueError(f"The complete graph is missing {len(missing)} pair(s).")
    if args.method == "r2p" and any(not descriptions.get(n) for n in graph.nodes):
        raise ValueError("R2P needs a description for every concept.")
    print(f"Valid local {args.method} DB: {len(graph.nodes)} concepts, {len(graph.pairs)} pairs; encoder: {clip_model}")


def model_arguments(parser):
    parser.add_argument("--vlm-model", default="openbmb/MiniCPM-o-2_6", help="Hugging Face ID or local MiniCPM-o checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)


def create_parser():
    parser = argparse.ArgumentParser(prog="recog", description="R2P and ReCoG with private local databases")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("build", help="Build a private local database from support images")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Your local support manifest JSON")
    source.add_argument("--r2p-db", help="Your existing R2P concept_dict database JSON")
    prepare.add_argument("--image-root", help="Base directory for relative support paths")
    prepare.add_argument("--output", required=True, help="New local DB JSON; will not overwrite")
    prepare.add_argument("--clip-model", default="openai/clip-vit-large-patch14-336")
    prepare.add_argument("--batch-size", type=int, default=8)
    graph = prepare.add_mutually_exclusive_group()
    graph.add_argument("--pairwise-dir", help="Import local pairwise feature files")
    graph.add_argument("--skip-graph", action="store_true", help="Build R2P descriptions and embeddings only")
    prepare.add_argument("--shared-dir", help="Local shared-feature directory; requires --pairwise-dir")
    model_arguments(prepare)
    prepare.set_defaults(action=build)
    recognition = sub.add_parser("recognize", help="Recognize a query using a local database")
    recognition.add_argument("--database", required=True)
    recognition.add_argument("--query", required=True)
    recognition.add_argument("--method", choices=["recog", "r2p"], default="recog")
    recognition.add_argument("--output", help="Optional new private result JSON")
    model_arguments(recognition)
    recognition.set_defaults(action=infer)
    check = sub.add_parser("validate", help="Check a local DB without loading a model")
    check.add_argument("--database", required=True)
    check.add_argument("--method", choices=["recog", "r2p"], default="recog")
    check.set_defaults(action=validate)
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if getattr(args, "shared_dir", None) and not getattr(args, "pairwise_dir", None):
        parser.error("--shared-dir requires --pairwise-dir")
    if min(getattr(args, "max_new_tokens", 1), getattr(args, "batch_size", 1)) < 1:
        parser.error("Token and batch sizes must be positive")
    try:
        args.action(args)
    except (ValueError, KeyError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
