"""Score a Hugging Face classifier on the training set without fine-tuning it.

This exists to settle one question with a number instead of an argument: does
`hugging-science/breast-cancer-detector-2` -- trained on breast *ultrasound*, with
mammography listed as out of scope on its own model card -- actually read our
mammograms? Its label order (benign, malignant, normal) matches CLASSES exactly, so
its predictions can be compared to our labels directly.

Reading the result:
  ~0.33  the checkpoint transfers nothing; its head is noise on this modality
  ~0.50  some transfer, still far below a fine-tuned ImageNet backbone
  >0.60  genuinely competitive, worth building on

This only reads train.csv labels. It writes no submission -- zero-shot output is a
diagnostic, not a candidate.

Run:  python src/zeroshot_hf.py --model hugging-science/breast-cancer-detector-2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, IMAGENET_MEAN, IMAGENET_STD, load_cache  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hugging-science/breast-cancer-detector-2")
    ap.add_argument("--cache", default=os.path.join(ROOT, "artifacts", "cache.npz"))
    ap.add_argument("--img-size", nargs=2, type=int, default=[224, 224], metavar=("H", "W"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import cv2
    from model import HFClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y, _, _, _ = load_cache(args.cache)

    model = HFClassifier(args.model, len(CLASSES), keep_head=True).to(device).eval()
    print(f"{args.model} on {len(x)} training images at {args.img_size[0]}x{args.img_size[1]}, "
          f"device={device}")

    h, w = args.img_size
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), args.batch_size):
            batch = np.stack([cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
                              for im in x[i:i + args.batch_size]]).astype(np.float32) / 255.0
            batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
            t = torch.from_numpy(batch)[:, None].repeat(1, 3, 1, 1).to(device)
            probs.append(model(t).softmax(1).float().cpu().numpy())
    prob = np.concatenate(probs)
    pred = prob.argmax(1)

    score = f1_score(y, pred, average="macro")
    print(f"\nzero-shot macro F1 = {score:.4f}   (chance ~0.33)")
    print("\n" + classification_report(y, pred, target_names=CLASSES, digits=3, zero_division=0))
    print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    for row in confusion_matrix(y, pred):
        print("   ", row)

    counts = np.bincount(pred, minlength=len(CLASSES))
    print("\nprediction spread: " + "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
    if counts.max() > 0.8 * len(y):
        print("  -> the checkpoint collapses onto one class here, which is what a model")
        print("     reading the wrong imaging modality looks like.")


if __name__ == "__main__":
    main()
