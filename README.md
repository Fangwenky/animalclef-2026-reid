# AnimalCLEF 2026: identity discovery baseline

Reproducible baseline for the [AnimalCLEF 2026 Kaggle competition](https://www.kaggle.com/competitions/animal-clef-2026). The task is to group test photographs by individual animal across four species. Three species provide labelled reference images; Texas horned lizards do not. The official metric is Adjusted Rand Index (ARI).

## Method

1. Extract L2-normalized [MegaDescriptor-T-224](https://huggingface.co/BVRA/MegaDescriptor-T-224) embeddings from every image.
2. For each species, use average-linkage hierarchical clustering on test-image cosine distances.
3. For species with references, attach a test cluster to a known identity only when its median best-reference score passes both an absolute threshold and a margin over the runner-up. Otherwise it remains a novel cluster.
4. Convert identity groups to the required `cluster_<dataset>_<number>` submission labels. Cluster numbers are arbitrary; only the grouping matters for ARI.

The baseline deliberately uses one global descriptor. Segmentation, local keypoint matching, fusion, and fine-tuning should be added only after a measured validation gain. The published [AnimalCLEF26 winning solution](https://github.com/MIA-AI-Team/AnimalCLEF26) supports those as promising next experiments, but its reported scores are not results of this repository.

## Setup

Python 3.11 or 3.12 is recommended. From this directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Download the competition data after accepting its rules on Kaggle. Put `metadata.csv`, `sample_submission.csv`, and `images/` under `data/`. Competition images are intentionally excluded from Git.

```bash
.venv/bin/python animalclef.py features --data data --features features.npz
.venv/bin/python animalclef.py validate --data data --features features.npz --config results/baseline.json
.venv/bin/python animalclef.py submit --data data --features features.npz --config results/baseline.json --output submission.csv
```

Use `--batch-size 4` if feature extraction runs out of GPU memory. A CPU fallback is available, but feature extraction will take longer. The model checkpoint is downloaded from Hugging Face on the first run.

## Validation protocol

For each of the three labelled species, identities are divided into calibration (75%) and held-out (25%) pools. Within each pool, some identities provide reference photos and remaining photos become queries; other identities appear only as queries and simulate novel animals. Cluster and known-identity attachment thresholds are selected using calibration identities. The reported `holdout_ari` is computed on disjoint identities and is the primary local metric. `pooled_holdout_ari_three_species` pools those hold-out queries, but omits Texas horned lizards and is not the official competition score. The Texas horned lizard threshold is transferred from the median of the other three species because the competition supplies no labelled reference identities for it; this is an explicit limitation.

`results/baseline.json` records the seed, model, thresholds, query counts, and scores. Do not interpret the calibration score as an unbiased estimate. No test image is hand-labelled, in accordance with the [competition rules](https://www.kaggle.com/competitions/animal-clef-2026/rules).

## Experiment sequence

| Run | Change | Decision rule |
| --- | --- | --- |
| B0 | MegaDescriptor, average linkage, known attachment | Establish valid submission and local ARI |
| B1 | Animal crop/segmentation | Keep only if held-out ARI improves |
| B2 | Local feature reranking of global top-k pairs | Keep only if held-out ARI improves |
| B3 | Fused similarities and species-specific graph clustering | Keep only if held-out ARI improves |
| B4 | Species-aware fine-tuning | Attempt only if earlier errors justify compute |

For every run, record code commit, configuration, per-species calibration and held-out ARI, cluster count, runtime, and a short error analysis. The paper in `paper/` must be updated from actual run artifacts; no benchmark number is assumed here.

## Data and model rights

Follow the [competition rules](https://www.kaggle.com/competitions/animal-clef-2026/rules), the SeaTurtleID2022 terms, and each external model or dataset license. Do not upload competition images or labels to this public repository.
