# AnimalCLEF 2026: reproducible animal identity discovery

Code, experiment records, and a [Chinese paper](paper/animalclef2026_paper.pdf) for the [AnimalCLEF 2026 Kaggle competition](https://www.kaggle.com/competitions/animal-clef-2026). The task is to group test photographs by individual animal across four species. Three species provide labelled reference images; Texas horned lizards do not. The official metric is Adjusted Rand Index (ARI). The best recorded submission scores **0.30642 public / 0.39072 private ARI**.

## Method

1. Extract L2-normalized [MegaDescriptor-T-224](https://huggingface.co/BVRA/MegaDescriptor-T-224) and [MiewID-msv3](https://huggingface.co/conservationxlabs/miewid-msv3) embeddings; concatenate them with equal cosine-similarity weight.
2. For salamanders and Texas horned lizards, match SIFT features in each image's 30 nearest fusion-embedding neighbors. Strong local matches raise the corresponding pair similarity.
3. Cluster test images independently by species using average-linkage clustering. For the three species with references, attach a test cluster to a known identity only when its median best-reference score passes both an absolute threshold and a margin over the runner-up.
4. Emit `cluster_<dataset>_<number>` submission labels in the sample-submission row order. Cluster numbers are arbitrary; the partition determines ARI.

The code also retains single-model and mutual-kNN graph variants so the reported ablations can be reproduced. The graph variant did not improve private ARI in this experiment.

## Setup

Python 3.12 is recommended. From this directory:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Download the competition data after accepting its rules on Kaggle. Put `metadata.csv`, `sample_submission.csv`, and `images/` under `data/`. Competition images are intentionally excluded from Git.

```bash
.venv/bin/python animalclef.py features --data data --model mega --features features.npz
.venv/bin/python animalclef.py features --data data --model miewid --features miewid_features.npz
.venv/bin/python animalclef.py features --data data --model fusion --features fusion05_features.npz --mega-features features.npz --miewid-features miewid_features.npz
PYTHONPATH=. .venv/bin/python experiments/build_sift_cache.py --data data --features fusion05_features.npz --species SalamanderID2025 --output salamander_sift30.npz
PYTHONPATH=. .venv/bin/python experiments/build_sift_cache.py --data data --features fusion05_features.npz --species TexasHornedLizards --output thl_sift30.npz
.venv/bin/python animalclef.py validate --data data --model fusion --features fusion05_features.npz --local-cache salamander_sift30.npz --seed 2026 --config results/reproduced_seed2026.json
.venv/bin/python animalclef.py submit --data data --model fusion --features fusion05_features.npz --local-cache salamander_sift30.npz --local-cache thl_sift30.npz --config results/fusion05_sift_thl085.json --output submission.csv
```

Repeat validation with seeds `2027` and `2028`. The checked-in JSON files in `results/` contain the original three-seed measurements. `results/fusion05_sift_thl085.json` is the selected submission configuration; its horned-lizard threshold was compared on the Kaggle leaderboard because that species has no labels in the provided data.

Use `--batch-size 4` if feature extraction runs out of GPU memory. A CPU fallback is available, but feature extraction will take longer. The model checkpoint is downloaded from Hugging Face on the first run.

## Validation protocol

For each of the three labelled species, identities are divided into calibration (75%) and held-out (25%) pools. Within each pool, some identities provide reference photos and remaining photos become queries; other identities appear only as queries and simulate novel animals. Cluster and known-identity attachment thresholds are selected using calibration identities. The reported `holdout_ari` is computed on disjoint identities and is the primary local metric. `pooled_holdout_ari_three_species` pools those hold-out queries, but omits Texas horned lizards and is not the official competition score. The Texas horned lizard threshold is transferred from the median of the other three species because the competition supplies no labelled reference identities for it; this is an explicit limitation.

Each validation JSON records the seed, model, thresholds, query counts, and scores. Do not interpret calibration ARI as an unbiased estimate. No test image was hand-labelled, in accordance with the [competition rules](https://www.kaggle.com/competitions/animal-clef-2026/rules).

## Measured results

Three-seed identity-disjoint held-out ARI (mean; the paper reports standard deviations):

| Method | Lynx | Salamander | Sea turtle | Pooled, three labelled species |
| --- | ---: | ---: | ---: | ---: |
| MegaDescriptor | 0.251 | 0.097 | 0.496 | 0.422 |
| MiewID | 0.264 | 0.152 | 0.775 | 0.637 |
| Equal-weight fusion | 0.256 | 0.162 | 0.835 | 0.648 |
| Fusion + mutual-kNN graph | 0.278 | 0.113 | 0.742 | 0.619 |
| Fusion + Salamander SIFT | 0.256 | 0.222 | 0.835 | 0.648 |

The selected submission reached **0.30642 public / 0.39072 private ARI**. The [submission log](results/submissions.csv) includes the other leaderboard scores. The initial horned-lizard threshold of 0.725 collapsed 274 images into just six clusters; threshold 0.85 with local matches produced 214 clusters. Threshold 0.85 **without** horned-lizard SIFT scored 0.34872 private ARI, isolating the gain from the local matcher. Thresholds 0.825 and 0.875 with SIFT scored 0.36086 and 0.39003, respectively. This is leaderboard-guided model selection, not independent validation. Mutual-kNN graph clustering for lynx improved public ARI but reduced private ARI, so the final configuration uses average linkage.

The [paper source](paper/main.tex) and [PDF](paper/animalclef2026_paper.pdf) document the method, protocol, ablations, and limitations.

## Data and model rights

Follow the [competition rules](https://www.kaggle.com/competitions/animal-clef-2026/rules), the SeaTurtleID2022 terms, and each external model or dataset license. Do not upload competition images or labels to this public repository.
