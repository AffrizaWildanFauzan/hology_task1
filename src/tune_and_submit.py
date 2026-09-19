"""Turn fold probabilities into a submission, and report how much to trust it.

Two jobs:

1. Class-prior tuning. Macro F1 weights the three classes equally while the
   training set does not (52 Benign vs 80/80), so plain argmax systematically
   under-predicts Benign. Rescaling the probabilities by a per-class weight before
   argmax recovers most of that. The weights are fitted on out-of-fold predictions
   and -- importantly -- also evaluated *cross-fitted*, so the reported gain is not
   the gain of fitting on the same data it is scored on.

2. An honest error bar. The test set is 54 images and the public leaderboard is
   ~30% of it, i.e. about 16 images. A single image moving is worth several macro
   F1 points there. This script prints the resulting spread so the public score is
   read as noise rather than as signal.

Run:  python src/tune_and_submit.py --run artifacts/run --out submission.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, load_cache  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fit_weights(prob: np.ndarray, y: np.ndarray, rounds: int = 6,
                grid: np.ndarray | None = None) -> np.ndarray:
    """Coordinate ascent on per-class multipliers, maximising macro F1."""
    if grid is None:
        grid = np.exp(np.linspace(-1.2, 1.2, 49))
    w = np.ones(len(CLASSES))
    best = f1_score(y, (prob * w).argmax(1), average="macro")
    for _ in range(rounds):
        improved = False
        for c in range(len(CLASSES)):
            base = w[c]
            for g in grid:
                w[c] = g
                score = f1_score(y, (prob * w).argmax(1), average="macro")
                if score > best + 1e-9:
                    best, base, improved = score, g, True
            w[c] = base
        if not improved:
            break
    return w / w.mean()


def cross_fitted_predictions(prob: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Tuned predictions where each row's weights were fitted without that row.

    Scoring these is the only honest estimate of what class-prior tuning buys on
    unseen data; fitting and scoring the weights on the same OOF rows flatters
    itself by a few points.
    """
    pred = np.empty(len(y), dtype=np.int64)
    for f in np.unique(folds):
        va = folds == f
        pred[va] = (prob[va] * fit_weights(prob[~va], y[~va])).argmax(1)
    return pred


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        scores[i] = f1_score(y[idx], pred[idx], average="macro", zero_division=0)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def public_lb_spread(y: np.ndarray, pred: np.ndarray, n_public: int, n: int = 4000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Simulate scoring on a public slice of `n_public` images drawn from OOF quality."""
    rng = np.random.default_rng(seed)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.choice(len(y), n_public, replace=False)
        scores[i] = f1_score(y[idx], pred[idx], average="macro", zero_division=0)
    return float(np.percentile(scores, 5)), float(scores.mean()), float(np.percentile(scores, 95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=[os.path.join(ROOT, "artifacts", "run")],
                    help="one or more run directories; their probabilities are averaged")
    ap.add_argument("--cache", default=os.path.join(ROOT, "artifacts", "cache.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "submission.csv"))
    ap.add_argument("--no-tune", action="store_true", help="plain argmax, skip class-prior tuning")
    ap.add_argument("--n-test", type=int, default=54)
    args = ap.parse_args()

    _, y, _, _, test_ids = load_cache(args.cache)
    oof = np.mean([np.load(os.path.join(r, "oof.npy")) for r in args.run], axis=0)
    test = np.mean([np.load(os.path.join(r, "test.npy")) for r in args.run], axis=0)
    folds = np.load(os.path.join(args.run[0], "folds.npy"))

    pred_plain = oof.argmax(1)
    pred_cf = cross_fitted_predictions(oof, y, folds)
    plain = f1_score(y, pred_plain, average="macro")
    tuned = f1_score(y, pred_cf, average="macro")

    print("=" * 68)
    print("HONEST ESTIMATES (nothing below is fitted on the rows it is scored on)")
    print("=" * 68)
    print(f"  OOF macro F1, plain argmax        : {plain:.4f}")
    print(f"  OOF macro F1, tuned (cross-fitted): {tuned:.4f}   (gain {tuned - plain:+.4f})")

    use_tuning = (not args.no_tune) and tuned > plain
    # Decide *whether* to tune from the cross-fitted comparison above, then fit the
    # shipped weights on all of the OOF -- three scalars on 212 rows, so fitting them
    # on everything is worth the extra data and cannot inflate the estimate we report.
    w = fit_weights(oof, y) if use_tuning else np.ones(len(CLASSES))
    honest_pred = pred_cf if use_tuning else pred_plain
    honest = tuned if use_tuning else plain

    lo, hi = bootstrap_ci(y, honest_pred)
    n_public = max(1, round(0.30 * args.n_test))
    p5, mean, p95 = public_lb_spread(y, honest_pred, n_public)
    print(f"  expected macro F1 on the test set : {honest:.4f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  simulated PUBLIC LB ({n_public} images)    : mean {mean:.3f}, "
          f"90% range [{p5:.3f}, {p95:.3f}]")
    print("    -> a swing inside that range is measurement noise, not model quality.")
    print("       Select models on OOF, never on the public leaderboard.")

    print("\n  per-class F1: " + "  ".join(
        f"{c}={f1_score(y, honest_pred, average=None, labels=[i])[0]:.3f}"
        for i, c in enumerate(CLASSES)))
    print("  confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    for row in confusion_matrix(y, honest_pred):
        print("   ", row)

    print("\n" + "=" * 68)
    if use_tuning:
        print(f"shipping class-prior tuning, weights {np.round(w, 3)} "
              f"for {CLASSES}")
        print(f"  (in-sample OOF with these weights is "
              f"{f1_score(y, (oof * w).argmax(1), average='macro'):.4f} -- OPTIMISTIC, "
              f"do not quote it)")
    else:
        print("class-prior tuning did not help out-of-fold -- shipping plain argmax.")

    pred_test = (test * w).argmax(1)
    sub = pd.DataFrame({"image_id": test_ids, "label": [CLASSES[i] for i in pred_test]})
    sub.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(sub.label.value_counts().to_string())


if __name__ == "__main__":
    main()
