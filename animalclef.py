"""Reproducible global-descriptor baseline for AnimalCLEF 2026."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score


SPECIES = (
    "LynxID2025",
    "SalamanderID2025",
    "SeaTurtleID2022",
    "TexasHornedLizards",
)
MODEL = "hf-hub:BVRA/MegaDescriptor-T-224"
MIEWID = "conservationxlabs/miewid-msv3"
MIEWID_REVISION = "4f1d7f2b521149e5fe34bb85f377248ce9971a7d"


def metadata(root):
    frame = pd.read_csv(root / "metadata.csv")
    required = {"image_id", "path", "split", "identity"}
    if not required.issubset(frame):
        raise ValueError(f"metadata.csv missing columns: {sorted(required - set(frame))}")
    frame["image_id"] = frame["image_id"].astype(str)
    if frame["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in metadata.csv")
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["dataset_key"] = frame["path"].astype(str).str.replace("\\", "/", regex=False).str.extract(
        r"(?:^|/)images/([^/]+)/"
    )[0]
    if frame["dataset_key"].isna().any():
        if "dataset" not in frame:
            raise ValueError("Cannot infer species dataset from image paths")
        frame["dataset_key"] = frame["dataset_key"].fillna(frame["dataset"])
    unexpected = set(frame["dataset_key"]) - set(SPECIES)
    if unexpected:
        raise ValueError(f"Unexpected dataset names: {sorted(unexpected)}")
    return frame


def features(root, frame, output, batch_size, model_name):
    import torch
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if model_name == MODEL:
        import timm
        from timm.data import create_transform, resolve_model_data_config

        model = timm.create_model(MODEL, pretrained=True, num_classes=0).to(device).eval()
        transform = create_transform(**resolve_model_data_config(model), is_training=False)
    else:
        from torchvision import transforms as T
        from transformers import AutoModel

        model = AutoModel.from_pretrained(MIEWID, revision=MIEWID_REVISION,
                                          trust_remote_code=True).to(device).eval()
        transform = T.Compose([
            T.Resize((440, 440)), T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            tensors = []
            for path in batch["path"]:
                image_path = root / str(path)
                with Image.open(image_path) as image:
                    tensors.append(transform(image.convert("RGB")))
            output_tensor = model(torch.stack(tensors).to(device))
            output_tensor = torch.nn.functional.normalize(output_tensor, dim=1)
            vectors.append(output_tensor.cpu().numpy().astype(np.float32))
            print(f"Embedded {min(start + batch_size, len(frame))}/{len(frame)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, image_id=np.asarray(frame["image_id"].tolist(), dtype=str),
                        embeddings=np.concatenate(vectors), model=model_name)
    print(f"Saved {output}")


def load_features(path, frame, model_name):
    with np.load(path) as saved:
        ids = saved["image_id"].astype(str)
        if saved["model"].item() != model_name or not np.array_equal(ids, frame["image_id"].to_numpy()):
            raise ValueError("Feature cache does not match model or metadata order")
        return saved["embeddings"].astype(np.float32)


def hierarchy(vectors):
    if len(vectors) < 2:
        return None
    return linkage(pdist(vectors, metric="cosine"), method="average")


def clusters(tree, size, threshold):
    if tree is None:
        return np.ones(size, dtype=int)
    # Cosine distance is 2 * (1 - similarity on the [0, 1] scale).
    return fcluster(tree, t=2 * (1 - threshold), criterion="distance")


def identity_scores(query, reference, reference_ids):
    if len(reference) == 0:
        return np.array([], dtype=str), np.empty((len(query), 0))
    similarities = (query @ reference.T + 1) / 2
    reference_ids = reference_ids.astype(str)
    identities = np.unique(reference_ids)
    per_identity = np.column_stack(
        [similarities[:, reference_ids == identity].max(axis=1) for identity in identities]
    )
    return identities, per_identity


def predict(query, reference, reference_ids, groups, attach_threshold, margin=0.03, prepared=None):
    """Assign each query cluster to a confident known identity or a new identity."""
    result = np.array([f"new:{g}" for g in groups], dtype=object)
    identities, per_identity = prepared if prepared is not None else identity_scores(query, reference, reference_ids)
    if len(identities) == 0:
        return result
    for group in np.unique(groups):
        scores = np.median(per_identity[groups == group], axis=0)
        best = int(np.argmax(scores))
        runner_up = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else 0.0
        if scores[best] >= attach_threshold and scores[best] - runner_up >= margin:
            result[groups == group] = f"known:{identities[best]}"
    return result


def validation_case(frame, vectors, random):
    """Within an identity-disjoint pool, create known references and novel queries."""
    names = frame["identity"].astype(str).to_numpy()
    unique = np.unique(names)
    eligible = [name for name in unique if np.sum(names == name) >= 2]
    random.shuffle(eligible)
    known = set(eligible[: max(1, int(len(eligible) * 0.6))])
    reference_indices, query_indices = [], []
    for name in unique:
        indices = np.flatnonzero(names == name)
        random.shuffle(indices)
        if name in known:
            cut = max(1, len(indices) // 2)
            reference_indices.extend(indices[:cut])
            query_indices.extend(indices[cut:])
        else:
            query_indices.extend(indices)
    query_indices = np.array(query_indices, dtype=int)
    reference_indices = np.array(reference_indices, dtype=int)
    return {
        "query": vectors[query_indices],
        "truth": names[query_indices],
        "reference": vectors[reference_indices],
        "reference_ids": names[reference_indices],
        "tree": hierarchy(vectors[query_indices]),
        "identity_scores": identity_scores(vectors[query_indices], vectors[reference_indices], names[reference_indices]),
        "cluster_cache": {},
    }


def evaluate(case, cluster_threshold, attach_threshold):
    if cluster_threshold not in case["cluster_cache"]:
        case["cluster_cache"][cluster_threshold] = clusters(case["tree"], len(case["query"]), cluster_threshold)
    groups = case["cluster_cache"][cluster_threshold]
    labels = predict(case["query"], case["reference"], case["reference_ids"], groups,
                     attach_threshold, prepared=case["identity_scores"])
    return float(adjusted_rand_score(case["truth"], labels))


def pair_errors(truth, labels):
    def pairs(counts):
        return sum(size * (size - 1) // 2 for size in counts.values())

    true_pairs = pairs(Counter(truth))
    predicted_pairs = pairs(Counter(labels))
    correct_pairs = pairs(Counter(zip(truth, labels)))
    return {
        "false_merge_pairs": predicted_pairs - correct_pairs,
        "false_split_pairs": true_pairs - correct_pairs,
    }


def validate(frame, vectors, output, seed, model_name):
    random = np.random.default_rng(seed)
    report = {"seed": seed, "model": model_name, "species": {}}
    pooled_truth, pooled_prediction = [], []
    for species in SPECIES[:-1]:
        subset = frame[(frame["dataset_key"] == species) & (frame["split"] == "train") & frame["identity"].notna()]
        if subset.empty:
            raise ValueError(f"No labelled training images for {species}")
        ids = subset["identity"].astype(str).unique()
        if len(ids) < 8:
            raise ValueError(f"Need at least eight identities to validate {species}")
        random.shuffle(ids)
        cut = max(4, int(len(ids) * 0.75))
        calibration = subset[subset["identity"].astype(str).isin(ids[:cut])]
        holdout = subset[subset["identity"].astype(str).isin(ids[cut:])]
        index = {value: i for i, value in enumerate(frame["image_id"])}
        cal_vectors = vectors[[index[value] for value in calibration["image_id"]]]
        hold_vectors = vectors[[index[value] for value in holdout["image_id"]]]
        cal_case = validation_case(calibration, cal_vectors, random)
        hold_case = validation_case(holdout, hold_vectors, random)
        candidates = (
            (round(cluster, 3), round(attach, 3))
            for cluster in np.arange(0.55, 0.951, 0.025)
            for attach in np.arange(0.65, 0.951, 0.05)
        )
        best_cluster, best_attach = max(candidates, key=lambda pair: evaluate(cal_case, *pair))
        best_cluster, best_attach = float(best_cluster), float(best_attach)
        result = {
            "cluster_threshold": best_cluster,
            "attach_threshold": best_attach,
            "calibration_ari": evaluate(cal_case, best_cluster, best_attach),
            "holdout_ari": evaluate(hold_case, best_cluster, best_attach),
            "calibration_queries": len(cal_case["query"]),
            "holdout_queries": len(hold_case["query"]),
        }
        report["species"][species] = result
        hold_groups = clusters(hold_case["tree"], len(hold_case["query"]), best_cluster)
        hold_labels = predict(hold_case["query"], hold_case["reference"],
                              hold_case["reference_ids"], hold_groups, best_attach,
                              prepared=hold_case["identity_scores"])
        result.update(pair_errors(hold_case["truth"], hold_labels))
        result["holdout_predicted_clusters"] = len(np.unique(hold_labels))
        known_mask = np.isin(hold_case["truth"], hold_case["reference_ids"])
        result["known_attach_accuracy"] = float(np.mean(
            hold_labels[known_mask] == np.char.add("known:", hold_case["truth"][known_mask].astype(str))
        )) if known_mask.any() else None
        result["novel_false_attach_rate"] = float(np.mean([
            label.startswith("known:") for label in hold_labels[~known_mask]
        ])) if (~known_mask).any() else None
        pooled_truth.extend(f"{species}:{label}" for label in hold_case["truth"])
        pooled_prediction.extend(f"{species}:{label}" for label in hold_labels)
        print(species, result, flush=True)
    report["pooled_holdout_ari_three_species"] = float(adjusted_rand_score(pooled_truth, pooled_prediction))
    # No labelled Texas horned lizard identities are supplied; transfer a starting threshold.
    report["species"][SPECIES[-1]] = {
        "cluster_threshold": float(np.median([report["species"][name]["cluster_threshold"] for name in SPECIES[:-1]])),
        "attach_threshold": None,
        "holdout_ari": None,
        "note": "Transferred threshold; no species-specific labelled validation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}")


def submit(root, frame, vectors, config_path, output):
    config = json.loads(config_path.read_text())["species"]
    predictions = {}
    for species in SPECIES:
        test = frame[(frame["dataset_key"] == species) & (frame["split"] == "test")]
        train = frame[(frame["dataset_key"] == species) & (frame["split"] == "train") & frame["identity"].notna()]
        if test.empty:
            raise ValueError(f"No test images for {species}")
        positions = {value: i for i, value in enumerate(frame["image_id"])}
        query = vectors[[positions[value] for value in test["image_id"]]]
        reference = vectors[[positions[value] for value in train["image_id"]]]
        groups = clusters(hierarchy(query), len(query), config[species]["cluster_threshold"])
        labels = predict(query, reference, train["identity"].astype(str).to_numpy(), groups,
                         config[species]["attach_threshold"] or 1.0)
        numbering = {label: index for index, label in enumerate(dict.fromkeys(labels))}
        for image_id, label in zip(test["image_id"], labels):
            predictions[image_id] = f"cluster_{species}_{numbering[label]}"
        print(f"{species}: {len(test)} images, {len(numbering)} clusters", flush=True)
    sample = pd.read_csv(root / "sample_submission.csv")
    if list(sample.columns) != ["image_id", "cluster"]:
        raise ValueError("Unexpected sample submission columns")
    ids = sample["image_id"].astype(str)
    if len(ids) != len(predictions) or set(ids) != set(predictions):
        raise ValueError("Prediction image IDs differ from sample submission")
    sample["cluster"] = ids.map(predictions)
    sample.to_csv(output, index=False)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["features", "validate", "submit"])
    parser.add_argument("--data", type=Path, required=True, help="Directory containing metadata.csv")
    parser.add_argument("--features", type=Path, default=Path("features.npz"))
    parser.add_argument("--config", type=Path, default=Path("results/baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", choices=["mega", "miewid"], default="mega")
    args = parser.parse_args()
    frame = metadata(args.data)
    model_name = MODEL if args.model == "mega" else MIEWID
    if args.command == "features":
        features(args.data, frame, args.features, args.batch_size, model_name)
    elif args.command == "validate":
        validate(frame, load_features(args.features, frame, model_name), args.config, args.seed, model_name)
    else:
        if json.loads(args.config.read_text())["model"] != model_name:
            raise ValueError("Config and --model refer to different feature extractors")
        submit(args.data, frame, load_features(args.features, frame, model_name), args.config, args.output)


if __name__ == "__main__":
    main()
