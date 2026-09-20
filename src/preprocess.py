"""Preprocess full-field digital mammograms into a compact cached array.

The raw images are ~3540x4740 JPEGs. Decoding them on every epoch would dominate
training time, so this module does the expensive, deterministic work exactly once
and writes the result to a single .npz cache:

  1. force single-channel grayscale (the raw set mixes 'L' and 'RGB' JPEGs -- see
     the artifact note in HANDOFF.md; collapsing the channel makes the encoding
     mode invisible to the model)
  2. drop the burned-in view annotations ("R MLO", "L CC", scanner labels) and the
     detector border, then crop to the breast itself
  3. normalise laterality so every breast points the same way
  4. CLAHE, then resize to a fixed training resolution

Run:  python src/preprocess.py --size 512 384
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _breast_mask(small: np.ndarray) -> np.ndarray:
    """Binary mask of the breast on a downscaled image.

    Otsu separates tissue from the black detector background. The burned-in text
    labels also survive the threshold, so we keep only the largest connected
    component -- the breast is by far the biggest bright object, and the labels sit
    in a corner disconnected from it.
    """
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Opening removes thin text strokes and the 1-2px detector frame; closing then
    # fills the speckle inside the tissue so the component stays in one piece.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.ones_like(small, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _orient_left(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mirror the image so the chest wall (dense side) is always on the left."""
    half = img.shape[1] // 2
    if mask[:, :half].sum() < mask[:, half:].sum():
        return cv2.flip(img, 1)
    return img


def load_and_crop(path: str, out_h: int, out_w: int, scale: int = 8) -> np.ndarray:
    """Read one mammogram and return a normalised (out_h, out_w) uint8 array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    small = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale),
                       interpolation=cv2.INTER_AREA)
    mask_small = _breast_mask(small)

    ys, xs = np.where(mask_small > 0)
    y0, y1 = ys.min() * scale, min((ys.max() + 1) * scale, img.shape[0])
    x0, x1 = xs.min() * scale, min((xs.max() + 1) * scale, img.shape[1])

    # Zero everything outside the breast so leftover annotations cannot leak in,
    # then crop to the bounding box.
    mask_full = cv2.resize(mask_small, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    img = img * mask_full
    crop = img[y0:y1, x0:x1]
    crop = _orient_left(crop, mask_full[y0:y1, x0:x1])

    crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


def build_cache(image_ids, image_dir: str, out_h: int, out_w: int) -> np.ndarray:
    out = np.zeros((len(image_ids), out_h, out_w), dtype=np.uint8)
    t0 = time.time()
    for i, name in enumerate(image_ids):
        out[i] = load_and_crop(os.path.join(image_dir, name), out_h, out_w)
        if (i + 1) % 25 == 0 or i + 1 == len(image_ids):
            print(f"  {i + 1}/{len(image_ids)}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", nargs=2, type=int, default=[512, 384],
                    metavar=("H", "W"), help="cached resolution")
    ap.add_argument("--data-dir", default=ROOT)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "cache.npz"))
    args = ap.parse_args()
    h, w = args.size

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    print(f"train: {len(train)} images -> {h}x{w}")
    x_train = build_cache(train.image_id, os.path.join(args.data_dir, "train_images"), h, w)
    print(f"test: {len(test)} images -> {h}x{w}")
    x_test = build_cache(test.image_id, os.path.join(args.data_dir, "test_images"), h, w)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        x_train=x_train, x_test=x_test,
        train_ids=train.image_id.values, test_ids=test.image_id.values,
        train_labels=train.label.values, size=np.array([h, w]),
    )
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
