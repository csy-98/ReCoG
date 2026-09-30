# 🚀 Run R2P and ReCoG

The repository includes real CLIP encoding and MiniCPM-o inference integration.
Supply your own local images and metadata; no database or concept inventory is
bundled. Both methods retrieve up to three concepts from the same support set.

## 🧰 Installation

Create an isolated Python 3.10 environment, then install from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
recog --help
```

The environment used to create `.venv` must be Python 3.10. The runtime extra
pins PyTorch 2.3.1, Transformers 4.44.2, and supporting libraries. The
[official MiniCPM-o 2.6 model card](https://huggingface.co/openbmb/MiniCPM-o-2_6#usage)
documents this model's Transformers requirement and vision-only initialization.
The loader uses SDPA, enables vision, and disables audio/TTS model components.

CUDA is the default device; choose `--dtype float16` on GPUs without bfloat16
support. The large VLM needs sufficient GPU memory. CPU with `--device cpu
--dtype float32` is selectable but has not been performance-tested. If your GPU
requires a newer PyTorch/CUDA combination, use a compatible environment and
validate it separately; the pinned installation has not been exercised here.

For reading the core or running offline tests without model dependencies:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

## 🌱 Private input formats

Prepare a local JSON file whose root object has a `concepts` array. Each entry
contains the following fields. The repository intentionally includes no manifest
with a real concept list.

| Field | Meaning |
| :--- | :--- |
| `name` | Unique, nonempty concept identifier |
| `image` | One support-image path, relative to the manifest or absolute |
| `info` | Optional object containing the concept's unary description, such as `general` and `distinct features` |

If `info` is empty or omitted, MiniCPM generates a unary description for the R2P
mode. Support images and embeddings also serve as the graph nodes for ReCoG.
Use one support image per concept. Image paths must point to actual local files.
`--image-root` supplies a different base for relative image paths.

## 🪴 Build a local database

```bash
recog build \
  --manifest /path/to/support-manifest.json \
  --output private/database.json
```

The command encodes support images with CLIP, creates any missing unary
descriptions, and extracts both directional cue lists for every concept pair.
The complete graph costs one VLM call per unordered pair. CLIP GPU memory is
released before loading MiniCPM. An existing output file is never overwritten;
choose a new output filename for a rebuild.

To prepare only R2P descriptions and embeddings, add `--skip-graph`. Such a DB
supports `--method r2p`; build a graph before using `--method recog`.

### Import your existing R2P data

```bash
recog build \
  --r2p-db /path/to/local-r2p-database.json \
  --pairwise-dir /path/to/local-pairwise-features \
  --output private/database.json
```

The importer reads `concept_dict`, each concept's `image` and `info`, and the
optional `path_to_concept` mapping. If `image` is a list, it uses its first entry.
If the image field is absent, it consults `path_to_concept`. Relative paths use
the DB file's directory or `--image-root`; stale absolute paths must be corrected
in your local metadata before import.

Pair feature filenames follow `{a}_vs_{b}_distinct.json`, with concept identifiers
sorted alphabetically. Each contains `{a}_distinct_features` and
`{b}_distinct_features` arrays of nonempty strings. Raw-output-wrapped JSON is
also supported by the included loader. Missing or invalid pair evidence causes
an error. Add `--shared-dir /path/to/local-shared-features` to read optional
`{a}_vs_{b}_shared.json` files containing `shared_features` arrays.

Without `--pairwise-dir`, the command extracts the pair features with MiniCPM.
Neither input files nor concept inventories are copied into the Git repository
by these commands. Generated data remains in your chosen output location.

## 🔎 Validate and recognize

```bash
# Checks paths, embeddings, and complete graph coverage; does not load models.
recog validate --database private/database.json --method recog

# ReCoG: graph-conditioned pairwise league.
recog recognize \
  --database private/database.json \
  --query /path/to/query.jpg \
  --method recog \
  --output private/recog-result.json

# R2P: reason over the retrieved concepts' unary descriptions.
recog recognize \
  --database private/database.json \
  --query /path/to/query.jpg \
  --method r2p \
  --output private/r2p-result.json
```

The R2P mode uses the included R2P text-option prompt with the query image and
retrieved unary descriptions. The ReCoG mode uses the original pairwise prompt
with query/A/B images and the graph's directional cues. Both parse structured
answers and return the selected concept with candidate retrieval scores; ReCoG
also returns the match winners and win table.

JSON result files are local outputs. Console output can also contain model
initialization messages; use `--output` when consuming results programmatically.
An invalid VLM answer is reported as an error instead of silently becoming a win.

## 🧩 Model paths and decoding

Defaults are `openai/clip-vit-large-patch14-336` and `openbmb/MiniCPM-o-2_6`.
The model loader uses the model repository's Python implementation through
`trust_remote_code=True`. To use your already-downloaded checkpoints:

```bash
recog build \
  --manifest /path/to/support-manifest.json \
  --clip-model /path/to/clip-checkpoint \
  --vlm-model /path/to/MiniCPM-o-2_6 \
  --output private/database.json

recog recognize \
  --database private/database.json \
  --query /path/to/query.jpg \
  --vlm-model /path/to/MiniCPM-o-2_6
```

The local DB records the CLIP model identifier/path, ensuring recognition loads
that encoder for the query. Keep a local checkpoint at that path, or update the
private DB's model path when moving machines. Support image paths saved in the DB
are absolute and must likewise remain valid.

Generation uses `sampling=False`. `--max-new-tokens` defaults to 512;
increase it if the model truncates JSON. `--seed`, `--device`, and `--dtype` are
available for both build and recognition. No model is loaded for CLI help or
`validate`.

## 🔒 What stays out of Git

DB files and their inventories, real images, generated pair/shared features,
embeddings, indices, results, and weights remain local. `.gitignore` covers these
artifacts, and the package configuration includes only Python modules from
`recog` and `r2p`. Input formats are documented here without publishing a DB list.

## 📝 Verification status

Offline tests cover DB loading/round-trip, directional cue mapping, retrieval,
league ties, malformed input, and build/recognize wiring using test doubles.
Runtime imports and CLI help were also checked. No real-model inference, dataset
example, or benchmark was executed. These checks do not establish GPU memory
requirements, model accuracy, or end-to-end behavior with downloaded weights.
