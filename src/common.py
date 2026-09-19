"""Label vocabulary and cache loading -- no torch, so the audit and submission
tooling stay runnable without a deep-learning install."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

CLASSES = ["Benign", "Malignant", "Normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ImageNet statistics collapsed to one channel: the cached images are grayscale and
# get replicated across RGB, so a single mean/std is all we need.
IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226


def load_cache(path: str):
    d = np.load(path, allow_pickle=True)
    y = np.array([CLASS_TO_IDX[v] for v in d["train_labels"]], dtype=np.int64)
    return d["x_train"], y, d["x_test"], d["train_ids"], d["test_ids"]


def make_folds(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Stratified fold assignment.

    The organisers guarantee a *patient-disjoint* train/test split but ship no
    patient id, and near-duplicate search over the training set finds no view pairs
    to group on (see HANDOFF.md), so plain stratification is the best available
    approximation. If patient ids ever surface, swap in StratifiedGroupKFold.
    """
    folds = np.zeros(len(y), dtype=np.int64)
    for f, (_, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(y, y)):
        folds[va] = f
    return folds
