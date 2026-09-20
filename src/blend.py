"""Find the best blend of several runs' saved probabilities, and write a submission.

Runs on a laptop in seconds -- it only reads the .npy files a training run already
wrote, so trying an ensemble costs no GPU time and no submission slot.

Why blending is worth a look here: the ViT and resnet34 runs fail differently.
resnet34 catches almost every Normal (recall 0.963) but floods the class with false
positives (precision 0.554), while the ViT is balanced but misses a third of the
Normals. One is stronger at Normal-vs-abnormal, the other at Benign-vs-Malignant.
Averaging models that make the *same* mistakes buys nothing; averaging models that
make different ones is the whole point.

Everything is judged out-of-fold, and the weights are also evaluated cross-fitted, so
the reported gain is one that can carry to the test set rather than the gain of
fitting and scoring on the same rows.

Run:
  python src/blend.py --runs artifacts/r34 /path/to/vit_run --out submission.csv

A "run" is either a directory holding oof.npy and test.npy, or an explicit
oof.npy:test.npy pair.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, load_cache  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_run(spec: str):
    """Accept either a directory or an explicit 'oof.npy:test.npy' pair."""
    if ":" in spec and not os.path.isdir(spec):
        oof_path, test_path = spec.split(":", 1)
    else:
        candidates = [("oof.npy", "test.npy"), ("oof.npy", "test_prob.npy"),
                      ("oof_vit.npy", "test_prob_vit.npy"), ("oof_v2.npy", "test_prob_v2.npy")]
        for a, b in candidates:
            if os.path.exists(os.path.join(spec, a)) and os.path.exists(os.path.join(spec, b)):
                oof_path, test_path = os.path.join(spec, a), os.path.join(spec, b)
                break
        else:
            raise FileNotFoundError(f"no oof/test pair found in {spec}")
    oof, test = np.load(oof_path), np.load(test_path)
    # A run saved with a different normalisation would silently dominate the blend.
    oof = oof / oof.sum(1, keepdims=True)
    test = test / test.sum(1, keepdims=True)
    return oof, test, os.path.basename(spec.rstrip("/"))


def simplex_grid(n: int, step: float = 0.1):
    """All non-negative weight vectors over n runs that sum to 1, on a coarse grid."""
    ticks = int(round(1.0 / step))
    for combo in itertools.product(range(ticks + 1), repeat=n):
        if sum(combo) == ticks:
            yield np.array(combo, dtype=float) / ticks


def best_weights(oofs, y, step: float = 0.1):
    best_w, best = None, -1.0
    for w in simplex_grid(len(oofs), step):
        s = f1_score(y, sum(wi * o for wi, o in zip(w, oofs)).argmax(1), average="macro")
        if s > best:
            best_w, best = w, s
    return best_w, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache", default=os.path.join(ROOT, "artifacts", "cache.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "submission_blend.csv"))
    ap.add_argument("--step", type=float, default=0.1, help="weight grid resolution")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    _, y, _, _, test_ids = load_cache(args.cache)
    oofs, tests, names = [], [], []
    for spec in args.runs:
        o, t, name = load_run(spec)
        if o.shape[0] != len(y):
            raise ValueError(f"{name}: oof has {o.shape[0]} rows, expected {len(y)}")
        oofs.append(o)
        tests.append(t)
        names.append(name)

    print("individual runs, out-of-fold:")
    for name, o in zip(names, oofs):
        print(f"  {name:<28} macro F1 {f1_score(y, o.argmax(1), average='macro'):.4f}")
    solo_best = max(f1_score(y, o.argmax(1), average="macro") for o in oofs)

    w, blend_score = best_weights(oofs, y, args.step)
    print("\nbest blend on OOF:")
    for name, wi in zip(names, w):
        print(f"  {name:<28} weight {wi:.2f}")
    print(f"  macro F1 {blend_score:.4f}   (best single run {solo_best:.4f}, "
          f"gain {blend_score - solo_best:+.4f})")

    # Cross-fitted: pick the weights on four folds, score the fifth. Searching a
    # weight grid against the same rows you then report is how a blend looks better
    # than it is, and with 212 rows that self-flattery is easily worth a few points.
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=0)
    pred_cf = np.empty(len(y), dtype=np.int64)
    for tr, va in skf.split(y, y):
        w_tr, _ = best_weights([o[tr] for o in oofs], y[tr], args.step)
        pred_cf[va] = sum(wi * o[va] for wi, o in zip(w_tr, oofs)).argmax(1)
    honest = f1_score(y, pred_cf, average="macro")
    print(f"\ncross-fitted blend macro F1 {honest:.4f}   "
          f"(vs {solo_best:.4f} for the best single run)")

    if honest <= solo_best:
        print("  -> blending does NOT beat the best single run out-of-fold. Ship that")
        print("     run on its own; the apparent gain above is weight-search overfitting.")
    else:
        print("  -> blending holds up out-of-fold. Worth a submission slot.")

    blend_oof = sum(wi * o for wi, o in zip(w, oofs))
    blend_test = sum(wi * t for wi, t in zip(w, tests))
    print("\nper-class F1 (blend, in-sample -- indicative only):")
    pred = blend_oof.argmax(1)
    print("  " + "  ".join(
        f"{c}={f1_score(y, pred, average=None, labels=[i])[0]:.3f}" for i, c in enumerate(CLASSES)))
    print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    for row in confusion_matrix(y, pred):
        print("   ", row)

    sub = pd.DataFrame({"image_id": test_ids,
                        "label": [CLASSES[i] for i in blend_test.argmax(1)]})
    sub.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(sub.label.value_counts().to_string())


if __name__ == "__main__":
    main()
