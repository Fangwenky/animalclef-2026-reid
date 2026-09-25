"""Measure whether local SIFT matches help Salamander retrieval."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from animalclef import metadata, load_features


def sift_descriptors(path, detector):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(path)
    height, width = image.shape
    scale = min(1.0, 600 / max(height, width))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, desc = detector.detectAndCompute(image, None)
    return desc


def match_score(left, right, matcher):
    if left is None or right is None or len(left) < 2 or len(right) < 2:
        return 0
    pairs = matcher.knnMatch(left, right, k=2)
    return sum(a.distance < 0.75 * b.distance for a, b in pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, default=Path("fusion05_features.npz"))
    parser.add_argument("--species", default="SalamanderID2025")
    args = parser.parse_args()
    frame = metadata(args.data)
    vectors = load_features(args.features, frame, "fusion-mega-0.50")
    mask = (frame.dataset_key == args.species) & (frame.split == "train")
    subset = frame[mask].reset_index(drop=True)
    vectors = vectors[np.flatnonzero(mask)]
    scores = (vectors @ vectors.T + 1) / 2
    np.fill_diagonal(scores, -1)
    rng = np.random.default_rng(42)
    selected = rng.choice(len(subset), size=min(200, len(subset)), replace=False)
    top = np.argsort(scores[selected], axis=1)[:, -10:]
    pairs = sorted({tuple(sorted((int(i), int(j)))) for i, row in zip(selected, top) for j in row})
    needed = sorted({i for pair in pairs for i in pair})
    detector = cv2.SIFT_create(nfeatures=1000)
    descriptors = {}
    for n, i in enumerate(needed, 1):
        descriptors[i] = sift_descriptors(args.data / subset.path[i], detector)
        if n % 100 == 0:
            print("descriptors", n, "/", len(needed), flush=True)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    truth = []
    global_scores = []
    local_scores = []
    for n, (i, j) in enumerate(pairs, 1):
        truth.append(subset.identity[i] == subset.identity[j])
        global_scores.append(scores[i, j])
        local_scores.append(match_score(descriptors[i], descriptors[j], matcher))
        if n % 500 == 0:
            print("pairs", n, "/", len(pairs), flush=True)
    print("species", args.species, "pairs", len(pairs), "positives", sum(truth))
    print("global_auc", roc_auc_score(truth, global_scores))
    print("local_auc", roc_auc_score(truth, local_scores))
    for label in (False, True):
        values = np.asarray(local_scores)[np.asarray(truth) == label]
        print(label, "local_quantiles", np.quantile(values, [0.25, 0.5, 0.75, 0.9, 0.99]))


if __name__ == "__main__":
    main()
