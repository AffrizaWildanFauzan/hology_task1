"""Combine several runs' saved probabilities: weighted blend, stacking, or both.

Methods 52 (stacking) and 73 (temperature scaling) from the method list, applied to
the .npy files a training run already wrote. Everything here is CPU-only and takes
seconds, so an extra combination costs no GPU time and no submission slot.

Why these two in particular, from what the runs measured:

  * Stacking (52). A weighted average applies one weight per model to every image.
    A meta-classifier can instead learn that resnet34 is the one to trust for Normal
    (recall 0.963) while the ViT is the one to trust for Benign (0.577 vs 0.327).
    With 212 rows only a tiny meta-model is safe, so it is multinomial logistic
    regression on the nine probabilities, heavily regularised.

  * Temperature scaling (73). The blend's test predictions came out 46% Benign while
    its out-of-fold predictions were 25.5%, against a 24.5% prior. Sharpening or
    softening each model before combining lets an over-confident model stop dragging
    the blend, and the temperature is fitted out-of-fold like everything else.

Every option is scored cross-fitted -- fitted on four folds, scored on the fifth --
and the script ships whichever wins on that comparison, not on the fit itself.

Run:
  python src/stack.py --runs artifacts/r34 oof_vit.npy:test_vit.npy --out submission.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, load_cache  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-9


def load_run(spec: str):
    if ":" in spec and not os.path.isdir(spec):
        oof_path, test_path = spec.split(":", 1)
    else:
        for a, b in [("oof.npy", "test.npy"), ("oof.npy", "test_prob.npy")]:
            if os.path.exists(os.path.join(spec, a)):
                oof_path, test_path = os.path.join(spec, a), os.path.join(spec, b)
                break
        else:
            raise FileNotFoundError(f"no oof/test pair in {spec}")
    oof, test = np.load(oof_path).astype(np.float64), np.load(test_path).astype(np.float64)
    name = os.path.basename(oof_path).replace("oof_", "").replace(".npy", "")
    return oof / oof.sum(1, keepdims=True), test / test.sum(1, keepdims=True), name


def apply_temperature(prob: np.ndarray, t: float) -> np.ndarray:
    """Re-sharpen (t<1) or soften (t>1) a probability vector."""
    logit = np.log(np.clip(prob, EPS, None)) / t
    e = np.exp(logit - logit.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_temperatures(oofs, y, grid=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
    """One temperature per model, chosen greedily on macro F1 of the mean blend."""
    temps = [1.0] * len(oofs)
    best = f1_score(y, np.mean(oofs, axis=0).argmax(1), average="macro")
    for i in range(len(oofs)):
        for t in grid:
            cand = list(temps)
            cand[i] = t
            s = f1_score(y, np.mean([apply_temperature(o, c) for o, c in zip(oofs, cand)],
                                    axis=0).argmax(1), average="macro")
            if s > best + 1e-9:
                best, temps = s, cand
    return temps


def simplex_grid(n, step=0.1):
    ticks = int(round(1 / step))
    for combo in itertools.product(range(ticks + 1), repeat=n):
        if sum(combo) == ticks:
            yield np.array(combo, dtype=float) / ticks


def best_weights(oofs, y, step=0.1):
    best_w, best = None, -1.0
    for w in simplex_grid(len(oofs), step):
        s = f1_score(y, sum(wi * o for wi, o in zip(w, oofs)).argmax(1), average="macro")
        if s > best:
            best_w, best = w, s
    return best_w


def stack_features(probs):
    """Log-probabilities of every model, side by side, as meta-features."""
    return np.hstack([np.log(np.clip(p, EPS, None)) for p in probs])


def fit_stacker(oofs, y, C: float):
    # Heavy regularisation and balanced weights: 212 rows against 3 x n_models
    # features is little enough data that an unconstrained meta-model just memorises.
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    clf.fit(stack_features(oofs), y)
    return clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache", default=os.path.join(ROOT, "artifacts", "cache.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "submission_stack.csv"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    _, y, _, _, test_ids = load_cache(args.cache)
    oofs, tests, names = [], [], []
    for spec in args.runs:
        o, t, n = load_run(spec)
        oofs.append(o)
        tests.append(t)
        names.append(n)

    print("individual runs, out-of-fold:")
    for n, o in zip(names, oofs):
        print(f"  {n:<32} {f1_score(y, o.argmax(1), average='macro'):.4f}")
    solo = max(f1_score(y, o.argmax(1), average="macro") for o in oofs)

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=0)
    results = {}

    # --- weighted blend, cross-fitted ---
    pred = np.empty(len(y), dtype=np.int64)
    for tr, va in skf.split(y, y):
        w = best_weights([o[tr] for o in oofs], y[tr])
        pred[va] = sum(wi * o[va] for wi, o in zip(w, oofs)).argmax(1)
    results["blend"] = f1_score(y, pred, average="macro")

    # --- temperature-scaled blend, cross-fitted ---
    pred = np.empty(len(y), dtype=np.int64)
    for tr, va in skf.split(y, y):
        temps = fit_temperatures([o[tr] for o in oofs], y[tr])
        scaled = [apply_temperature(o, t) for o, t in zip(oofs, temps)]
        w = best_weights([o[tr] for o in scaled], y[tr])
        pred[va] = sum(wi * o[va] for wi, o in zip(w, scaled)).argmax(1)
    results["blend+temperature"] = f1_score(y, pred, average="macro")

    # --- stacking, cross-fitted, at a few regularisation strengths ---
    for C in (0.05, 0.2, 1.0):
        pred = np.empty(len(y), dtype=np.int64)
        for tr, va in skf.split(y, y):
            clf = fit_stacker([o[tr] for o in oofs], y[tr], C)
            pred[va] = clf.predict(stack_features([o[va] for o in oofs]))
        results[f"stack(C={C})"] = f1_score(y, pred, average="macro")

    print("\ncross-fitted comparison (nothing scored on the rows it was fitted on):")
    print(f"  {'best single run':<32} {solo:.4f}")
    for k, v in results.items():
        print(f"  {k:<32} {v:.4f}")

    best_name = max(results, key=results.get)
    if results[best_name] <= solo:
        print(f"\nnothing beat the best single run ({solo:.4f}). Ship that run on its own.")
        return

    print(f"\nwinner: {best_name} at {results[best_name]:.4f}")

    # Refit the winner on all of the OOF rows and apply it to the test probabilities.
    if best_name.startswith("stack"):
        C = float(best_name.split("=")[1].rstrip(")"))
        clf = fit_stacker(oofs, y, C)
        oof_final = clf.predict_proba(stack_features(oofs))
        test_final = clf.predict_proba(stack_features(tests))
    elif best_name == "blend+temperature":
        temps = fit_temperatures(oofs, y)
        print(f"  temperatures: " + ", ".join(f"{n}={t}" for n, t in zip(names, temps)))
        so = [apply_temperature(o, t) for o, t in zip(oofs, temps)]
        st = [apply_temperature(t_, t) for t_, t in zip(tests, temps)]
        w = best_weights(so, y)
        print(f"  weights: " + ", ".join(f"{n}={wi:.2f}" for n, wi in zip(names, w)))
        oof_final = sum(wi * o for wi, o in zip(w, so))
        test_final = sum(wi * t for wi, t in zip(w, st))
    else:
        w = best_weights(oofs, y)
        print(f"  weights: " + ", ".join(f"{n}={wi:.2f}" for n, wi in zip(names, w)))
        oof_final = sum(wi * o for wi, o in zip(w, oofs))
        test_final = sum(wi * t for wi, t in zip(w, tests))

    print("\nconfusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    for row in confusion_matrix(y, oof_final.argmax(1)):
        print("   ", row)
    print("per-class F1: " + "  ".join(
        f"{c}={f1_score(y, oof_final.argmax(1), average=None, labels=[i])[0]:.3f}"
        for i, c in enumerate(CLASSES)))

    sub = pd.DataFrame({"image_id": test_ids,
                        "label": [CLASSES[i] for i in test_final.argmax(1)]})
    sub.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(sub.label.value_counts().to_string())


if __name__ == "__main__":
    main()
