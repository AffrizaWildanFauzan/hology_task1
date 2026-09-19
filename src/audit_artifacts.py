"""Audit the dataset for acquisition artifacts that correlate with the label.

Finding this early changes where the effort goes, so it runs before any modelling.
It fits deliberately weak classifiers on signals that carry no diagnostic meaning
-- JPEG colour mode, file size, global intensity statistics -- and reports how much
of the label they recover. Anything they recover is the dataset's provenance
showing through, not breast pathology.

Nothing here feeds the submission. See the "Acquisition artifact" section of
HANDOFF.md for what we do about it and why.

Run:  python src/audit_artifacts.py
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, load_cache  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def raw_file_features(image_ids, image_dir: str) -> pd.DataFrame:
    rows = []
    for name in image_ids:
        path = os.path.join(image_dir, name)
        with Image.open(path) as im:
            rows.append(dict(image_id=name, mode=im.mode, width=im.size[0], height=im.size[1],
                             bytes=os.path.getsize(path)))
    return pd.DataFrame(rows)


def pixel_features(images: np.ndarray) -> np.ndarray:
    feats = []
    for im in images:
        tissue = im[im > 5].astype(np.float32)
        hist = np.histogram(tissue, bins=32, range=(0, 256), density=True)[0]
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, 3)
        grad = np.hypot(gx, gy)[im > 5]
        feats.append(np.concatenate([hist, [tissue.mean(), tissue.std(), grad.mean(),
                                            grad.std(), (im > 5).mean()]]))
    return np.stack(feats)


def probe(name: str, features: np.ndarray, y: np.ndarray) -> float:
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, C=0.3, class_weight="balanced"))
    pred = cross_val_predict(clf, features, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    score = f1_score(y, pred, average="macro")
    print(f"\n### {name}\nmacro F1 = {score:.4f}   (chance ~0.33)")
    print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    print(confusion_matrix(y, pred))
    binary = (y == CLASSES.index("Normal")).astype(int)
    pred_b = cross_val_predict(clf, features, binary, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"Normal-vs-abnormal accuracy from this signal alone = {(pred_b == binary).mean():.4f}")
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=ROOT)
    ap.add_argument("--cache", default=os.path.join(ROOT, "artifacts", "cache.npz"))
    args = ap.parse_args()

    x_train, y, _, train_ids, _ = load_cache(args.cache)
    raw = raw_file_features(train_ids, os.path.join(args.data_dir, "train_images"))
    labels = pd.Series([CLASSES[i] for i in y], index=raw.index)

    print("=" * 72)
    print("RAW FILE PROPERTIES vs LABEL")
    print("=" * 72)
    print("\nJPEG colour mode x label:")
    print(pd.crosstab(raw["mode"], labels))
    print("\nfile size (bytes) by label:")
    print(raw.groupby(labels.values).bytes.agg(["count", "mean", "median"]).round(0))
    print("\nimage dimensions by label:")
    print(raw.groupby(labels.values)[["width", "height"]].agg(["min", "max"]))

    meta = np.c_[(raw["mode"].values == "L").astype(float), np.log(raw.bytes.values),
                 raw.width.values, raw.height.values]
    probe("PROBE A -- file metadata only (colour mode, size, dimensions)", meta, y)
    probe("PROBE B -- global pixel statistics after our preprocessing", pixel_features(x_train), y)

    print("\n" + "=" * 72)
    print("Read this as: whatever PROBE A recovers is provenance, not pathology.")
    print("The submission pipeline never sees file metadata -- preprocess.py collapses")
    print("every image to one channel and equalises contrast per image, which is what")
    print("drops PROBE B well below PROBE A. See HANDOFF.md.")
    print("=" * 72)


if __name__ == "__main__":
    main()
