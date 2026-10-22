# Download FINGER

FINGER is distributed at **[ssun0626/FINGER on Hugging Face](https://huggingface.co/datasets/ssun0626/FINGER)**.
Release `v1.0.0` contains the following independent subsets:

| Subset | Concepts | Support (`train`) | Query (`test`) | Original folders |
| --- | ---: | ---: | ---: | --- |
| `cub_easy` | 20 | 20 | 380 | `CUB_easy/` |
| `cub_hard` | 20 | 20 | 380 | `CUB_hard/` |
| `clevr_v2` | 20 | 20 | 180 | `CLEVR/` |

There are 1,000 benchmark images in total. `train` is the support set for personalization;
it does not imply that the VLM must be trained. There is no validation split.

## Download ZIP archives

- [CUB-Easy](https://huggingface.co/datasets/ssun0626/FINGER/resolve/v1.0.0/downloads/CUB_easy.zip?download=true)
- [CUB-Hard](https://huggingface.co/datasets/ssun0626/FINGER/resolve/v1.0.0/downloads/CUB_hard.zip?download=true)
- [CLEVR (V2)](https://huggingface.co/datasets/ssun0626/FINGER/resolve/v1.0.0/downloads/CLEVR.zip?download=true)

Or download all three archives from the terminal:

```bash
python -m pip install -U huggingface_hub
hf download ssun0626/FINGER --repo-type dataset --revision v1.0.0 \
  --include 'downloads/*.zip' --local-dir finger-release
mkdir -p data
unzip finger-release/downloads/CUB_easy.zip -d data
unzip finger-release/downloads/CUB_hard.zip -d data
unzip finger-release/downloads/CLEVR.zip -d data
```

Each extracted dataset folder contains:

```text
<dataset>/
├── train/<concept_id>/...
├── test/<concept_id>/...
├── mapping.json
├── train_test_val_seed_23.json
└── MANIFEST.csv
```

Additional metadata is retained where available. All paths in the split JSON are
relative to the extracted dataset folder, for example `train/fai/fai_train.jpg`.
Join them with the dataset root when an existing evaluation script expects an absolute path.
The ZIP layout preserves the original filenames and image bytes.

## Load in Python

```bash
python -m pip install -U datasets Pillow
```

```python
from datasets import load_dataset

dataset = load_dataset("ssun0626/FINGER", "cub_easy", revision="v1.0.0")
support = dataset["train"]  # 20 examples
queries = dataset["test"]   # 380 examples
example = queries[0]
print(example["concept_id"], example["image_path"])
image = example["image"]    # PIL image
```

Choose `cub_hard` or `clevr_v2` to load the other subsets. Their Parquet files contain
the original image bytes, anonymous concept IDs, and original relative image paths.
The original species/scene mappings are separate metadata files; they are not part of the image input.

## Version and integrity

- Pin `revision="v1.0.0"` to reproduce this release instead of following `main`.
- Each ZIP contains a `MANIFEST.csv` with file sizes and SHA-256 checksums.
- The Hub's [SHA256SUMS](https://huggingface.co/datasets/ssun0626/FINGER/blob/v1.0.0/SHA256SUMS)
  covers the archives, Parquet files, and metadata copies.
- The [dataset card](https://huggingface.co/datasets/ssun0626/FINGER) describes source attribution
  and source dataset terms.

The CLEVR source available for this release is named **`clevrV2`**; its generation
metadata internally uses `clevr_hard_v1`. Other CLEVR rendering/difficulty variants
are not included. This packaging verifies file integrity and split consistency;
it does not independently establish identity with the final paper's CLEVR assets.

For inference and support-manifest formats, see the [execution guide](running.md).
