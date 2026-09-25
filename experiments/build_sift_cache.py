"""Compute SIFT match counts for global-descriptor nearest-neighbor pairs."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from animalclef import load_features, metadata
from experiments.sift_probe import match_score, sift_descriptors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, default=Path("fusion05_features.npz"))
    parser.add_argument("--species", default="SalamanderID2025")
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = metadata(args.data)
    vectors = load_features(args.features, frame, "fusion-mega-0.50")
    mask = frame.dataset_key == args.species
    subset = frame[mask].reset_index(drop=True)
    vectors = vectors[np.flatnonzero(mask)]
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -2)
    k = min(args.neighbors, len(subset) - 1)
    near = np.argpartition(similarity, -k, axis=1)[:, -k:]
    pairs = sorted({tuple(sorted((i, int(j)))) for i, row in enumerate(near) for j in row})
    detector = cv2.SIFT_create(nfeatures=1000)
    descriptors = []
    for i, path in enumerate(subset.path, 1):
        descriptors.append(sift_descriptors(args.data / path, detector))
        if i % 250 == 0:
            print("descriptors", i, "/", len(subset), flush=True)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = np.empty(len(pairs), dtype=np.int16)
    for n, (i, j) in enumerate(pairs):
        matches[n] = match_score(descriptors[i], descriptors[j], matcher)
        if (n + 1) % 5000 == 0:
            print("pairs", n + 1, "/", len(pairs), flush=True)
    ids = subset.image_id.astype(str).to_numpy(dtype=str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, image_id=ids, row=np.array([p[0] for p in pairs]),
                        col=np.array([p[1] for p in pairs]), matches=matches,
                        species=args.species, neighbors=k)
    print("saved", args.output, "pairs", len(pairs))


if __name__ == "__main__":
    main()
