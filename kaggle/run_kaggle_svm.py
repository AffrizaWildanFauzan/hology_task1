"""HoloMine Task 1 -- frozen feature extraction + SVM (classical ML).

Paste into one Kaggle notebook cell. Internet ON (the backbones come from the Hub).
Prints a sklearn classification_report and the macro F1 at every stage.

WHY THIS SCRIPT EXISTS, AND WHAT IT IS NOT
------------------------------------------
This follows the notebook that reports 0.95 on breast images: extract features, hand
them to an SVM. It does NOT reproduce that 0.95, and it cannot, for three reasons
that are worth stating up front so nobody is surprised by the number at the bottom:

  1. That notebook runs on Dataset_BUSI_with_GT -- breast *ultrasound*. Ours is
     full-field digital mammography. Different modality, different appearance.
  2. Its feature step is
         masked_image = cv2.bitwise_and(image, image, mask=mask)
     where `mask` is the ground-truth lesion segmentation shipped with BUSI. It
     computes HOG on a picture of the lesion, with the lesion already located for it.
     We are given no masks, so the hardest part of our problem -- finding the lesion
     in a 3540x4740 image -- is the part that notebook is handed for free.
  3. It scores a single random train_test_split(test_size=0.2) over 399 balanced
     images. Ours is a patient-disjoint split of 212. Those are different questions,
     and the random one is the easier of the two.

So the comparison is not apples to apples. What IS worth having: an SVM on frozen
features fails differently from a fine-tuned network, and blending two models that
fail differently is the only thing measured to pay on this dataset (+0.07 OOF).
That is the reason to run this -- as a blend partner for the ViT run, not as a
replacement for it. The DIAGNOSTIC block at the end reproduces the notebook's own
evaluation protocol on our data so you can see the gap for yourself.

No external data, no LLM/VLM, no AutoML, no Ultralytics. Publicly available
pretrained backbones, used frozen as feature extractors, which the rules permit.

Runtime: ~8-15 min on a P100/T4 (most of it is the breast-crop preprocessing).
"""
from __future__ import annotations

import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/holomine-breast-cancer-classification-task-1"
OUT_CSV = "/kaggle/working/submission_svm.csv"
WORK = "/kaggle/working"

CLASSES = ["Benign", "Malignant", "Normal"]
CACHE_H, CACHE_W = 512, 384
MEAN, STD = 0.449, 0.226
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_FOLDS = 5
SEED = 0            # same split as run_kaggle_vit.py, so the .npy files blend row-wise
BATCH_SIZE = 16
PCA_DIM = 96        # 212 training rows against a 2048-dim feature vector is hopeless
                    # without this; 96 keeps ~all the variance and lets the RBF kernel
                    # see distances that mean something.

# Each extractor is frozen -- no gradient step is taken anywhere in this script.
# `kind` is how the weights are loaded, not what the model is.
EXTRACTORS = [
    dict(tag="hog",       kind="hog"),
    dict(tag="bc2_vit",   kind="hf",   repo="hugging-science/breast-cancer-detector-2"),
    dict(tag="resnet50",  kind="timm", repo="resnet50"),
    dict(tag="convnext",  kind="timm", repo="convnext_tiny"),
]

# The SVM grid. Small, because 212 rows will happily memorise anything larger.
SVM_GRID = {
    "svc__C": [0.1, 1.0, 10.0, 100.0],
    "svc__gamma": ["scale", 0.01, 0.001],
    "svc__kernel": ["rbf", "linear"],
}

# Probabilities saved by another run, blended in at the end. Point this at the ViT
# run's outputs to get the combination this script is actually for.
BLEND_WITH = []     # e.g. [("/kaggle/input/prev/oof_vit.npy",
                    #        "/kaggle/input/prev/test_prob_vit.npy")]

# ---------------------------------------------------------------------------
# Preprocessing -- identical to run_kaggle_vit.py so the caches are interchangeable
# ---------------------------------------------------------------------------


def breast_mask(small):
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.ones_like(small, dtype=np.uint8)
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def load_and_crop(path, scale=8):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale),
                       interpolation=cv2.INTER_AREA)
    m = breast_mask(small)
    ys, xs = np.where(m > 0)
    mask_full = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    img = img * mask_full
    y0, y1 = ys.min() * scale, (ys.max() + 1) * scale
    x0, x1 = xs.min() * scale, (xs.max() + 1) * scale
    crop, sub = img[y0:y1, x0:x1], mask_full[y0:y1, x0:x1]
    if sub[:, :sub.shape[1] // 2].sum() < sub[:, sub.shape[1] // 2:].sum():
        crop = cv2.flip(crop, 1)
    crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
    return cv2.resize(crop, (CACHE_W, CACHE_H), interpolation=cv2.INTER_AREA)


def build_cache(ids, image_dir):
    out = np.zeros((len(ids), CACHE_H, CACHE_W), dtype=np.uint8)
    for i, name in enumerate(ids):
        out[i] = load_and_crop(os.path.join(image_dir, name))
    return out


# ---------------------------------------------------------------------------
# Feature extraction -- every backbone frozen, eval mode, no_grad
# ---------------------------------------------------------------------------


class PlainDataset(Dataset):
    def __init__(self, images, size, flip=False):
        self.images, self.size, self.flip = images, size, flip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i].astype(np.float32) / 255.0
        if self.flip:
            img = img[:, ::-1].copy()
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray((img - MEAN) / STD))[None].repeat(3, 1, 1)


def hog_features(images):
    """The notebook's feature, minus the ground-truth mask it applies first.

    That mask is the difference between "describe the lesion" and "describe the
    breast". We only get to do the second one, which is why HOG alone is weak here.

    Falls back to OpenCV's HOG when scikit-image is missing. Same 9 orientations,
    8x8 cells, 2x2 blocks, L2-Hys -- 8100 dims either way.
    """
    try:
        from skimage.feature import hog

        def describe(small):
            return hog(small, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm="L2-Hys")
    except ImportError:
        describe = _hog_numpy

    return np.asarray([describe(cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA))
                       for img in images], dtype=np.float32)


def _hog_numpy(small, bins=9, cell=8, block=2, eps=1e-7):
    """HOG without scikit-image, for an environment that does not ship it.

    Same recipe: unsigned gradient orientation into 9 bins, 8x8 cells, 2x2 blocks,
    L2-Hys normalisation. Kept as a fallback only -- when skimage is present it is
    used instead, so the features match the reference implementation.
    """
    g = small.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0) * (bins / 180.0)
    lo = np.floor(ang).astype(np.int32) % bins           # linear vote between the two
    hi = (lo + 1) % bins                                 # neighbouring bins
    frac = ang - np.floor(ang)

    h, w = g.shape
    ny, nx = h // cell, w // cell
    hist = np.zeros((ny, nx, bins), dtype=np.float32)
    for b in range(bins):
        v = mag * ((lo == b) * (1.0 - frac) + (hi == b) * frac)
        hist[:, :, b] = v[:ny * cell, :nx * cell].reshape(ny, cell, nx, cell).sum((1, 3))

    out = []
    for y in range(ny - block + 1):
        for x in range(nx - block + 1):
            v = hist[y:y + block, x:x + block].ravel()
            v = v / np.sqrt((v * v).sum() + eps ** 2)
            v = np.clip(v, 0, 0.2)                       # the "Hys" clip, then renorm
            out.append(v / np.sqrt((v * v).sum() + eps ** 2))
    return np.concatenate(out)


@torch.no_grad()
def deep_features(model, images, size):
    """Mean of the original and the horizontally flipped view (cheap feature TTA)."""
    model.eval()
    total = None
    for flip in (False, True):
        chunks = []
        for xb in DataLoader(PlainDataset(images, size, flip), batch_size=BATCH_SIZE):
            xb = xb.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                chunks.append(model(xb).float().cpu().numpy())
        f = np.concatenate(chunks)
        total = f if total is None else total + f
    return (total / 2.0).astype(np.float32)


class HFEncoder(nn.Module):
    """CLS token concatenated with the mean patch token, for a ViT-style backbone."""

    def __init__(self, repo):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(repo)
        self.interp = "vit" in self.model.config.model_type

    def forward(self, x):
        kw = dict(interpolate_pos_encoding=True) if self.interp else {}
        h = self.model(pixel_values=x, **kw).last_hidden_state
        if h.dim() == 4:                       # conv backbone: (B, C, H, W)
            return h.mean((2, 3))
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)


def build_extractor(spec):
    if spec["kind"] == "hf":
        return HFEncoder(spec["repo"]).to(DEVICE), 224
    import timm
    m = timm.create_model(spec["repo"], pretrained=True, num_classes=0)
    return m.to(DEVICE), 224


def extract(spec, x_train, x_test):
    """Returns (train_features, test_features), or (None, None) if the load failed.

    Isolated on purpose: a backbone that will not download should cost this run one
    feature set, not the whole script.
    """
    t = time.time()
    try:
        if spec["kind"] == "hog":
            f_tr, f_te = hog_features(x_train), hog_features(x_test)
        else:
            model, size = build_extractor(spec)
            f_tr = deep_features(model, x_train, size)
            f_te = deep_features(model, x_test, size)
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    except Exception as exc:                   # noqa: BLE001 -- report and carry on
        print(f"  [{spec['tag']}] SKIPPED: {type(exc).__name__}: {exc}", flush=True)
        return None, None
    print(f"  [{spec['tag']}] {f_tr.shape[1]} dims  ({time.time() - t:.0f}s)", flush=True)
    return f_tr, f_te


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------


def make_pipeline(n_features):
    """Scale -> PCA -> RBF SVM. class_weight balanced because macro F1 does not care
    that Benign has 52 rows against 80 and 80."""
    steps = [("scale", StandardScaler())]
    if n_features > PCA_DIM:
        steps.append(("pca", PCA(n_components=PCA_DIM, whiten=True, random_state=SEED)))
    steps.append(("svc", SVC(probability=True, class_weight="balanced", random_state=SEED)))
    return Pipeline(steps)


def cv_svm(f_tr, f_te, y, tag):
    """Out-of-fold probabilities from an SVM whose hyper-parameters are chosen
    INSIDE each fold.

    The grid search never sees the rows it is scored on. That matters: fitting the
    grid once on all 212 rows and reporting its own CV score is how a notebook gets
    to report 0.95, and it is not a number that transfers.
    """
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_prob = np.zeros((len(f_te), 3), dtype=np.float32)
    picked = []
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(f_tr, y):
        gs = GridSearchCV(make_pipeline(f_tr.shape[1]), SVM_GRID,
                          scoring="f1_macro", cv=4, n_jobs=-1, refit=True)
        gs.fit(f_tr[tr], y[tr])
        oof[va] = gs.predict_proba(f_tr[va])
        test_prob += gs.predict_proba(f_te) / N_FOLDS
        picked.append({k.replace("svc__", ""): v for k, v in gs.best_params_.items()})
    score = f1_score(y, oof.argmax(1), average="macro")
    print(f"  [{tag}] OOF macro F1 = {score:.4f}")
    print(f"         params per fold: {picked}")
    return oof, test_prob, score


def report(y, pred, title):
    """The classification report and macro F1, printed for every stage."""
    print("\n" + "-" * 68)
    print(title)
    print("-" * 68)
    print(classification_report(y, pred, target_names=CLASSES, digits=4, zero_division=0))
    print(f"macro F1 = {f1_score(y, pred, average='macro'):.4f}")
    print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    for row in confusion_matrix(y, pred):
        print("   ", row)


def fit_blend(probs, y):
    """Coordinate ascent on per-model weights, maximising macro F1."""
    w = np.ones(len(probs)) / len(probs)
    best = f1_score(y, sum(wi * p for wi, p in zip(w, probs)).argmax(1), average="macro")
    for _ in range(6):
        improved = False
        for i in range(len(probs)):
            base = w[i]
            for g in np.linspace(0.0, 2.0, 21):
                w[i] = g
                s = f1_score(y, sum(wi * p for wi, p in zip(w, probs)).argmax(1),
                             average="macro")
                if s > best + 1e-9:
                    best, base, improved = s, g, True
            w[i] = base
        if not improved:
            break
    return w / max(w.sum(), 1e-9)


def crossfit_blend(probs, y):
    """What the blend scores when its weights are fitted without seeing the rows.

    A weight vector tuned on all 212 rows and then scored on those same rows is a
    fitted number, not an estimate. This is the honest version.
    """
    pred = np.empty(len(y), dtype=np.int64)
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(y, y):
        w = fit_blend([p[tr] for p in probs], y[tr])
        pred[va] = sum(wi * p[va] for wi, p in zip(w, probs)).argmax(1)
    return pred


def notebook_protocol_diagnostic(feats, y):
    """Score our data the way the 0.95 notebook scores its own.

    One random 80/20 split, grid searched on the training part, accuracy on the other
    20%. Run purely so the gap between this and the OOF number above is visible rather
    than argued about. Do NOT use this to pick anything.
    """
    print("\n" + "=" * 68)
    print("DIAGNOSTIC -- the notebook's own protocol, on our data")
    print("=" * 68)
    print("single random 80/20 split, not patient-disjoint. 43 test rows.")
    for tag, (f_tr, _) in feats.items():
        a, b, ya, yb = train_test_split(f_tr, y, test_size=0.2, random_state=0, stratify=y)
        gs = GridSearchCV(make_pipeline(f_tr.shape[1]), SVM_GRID,
                          scoring="f1_macro", cv=4, n_jobs=-1)
        gs.fit(a, ya)
        p = gs.predict(b)
        print(f"  [{tag}] accuracy={(p == yb).mean():.4f}  "
              f"macro F1={f1_score(yb, p, average='macro'):.4f}")
    print("A number from 43 rows of a random split moves by ~0.05 per image. It is")
    print("not comparable to the 5-fold OOF above, and neither is the notebook's 0.95.")


# ---------------------------------------------------------------------------


def main():
    t0 = time.time()
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    y = np.array([CLASSES.index(v) for v in train.label], dtype=np.int64)

    print(f"device={DEVICE}  folds={N_FOLDS}  pca={PCA_DIM}  "
          f"extractors={[e['tag'] for e in EXTRACTORS]}")
    print("preprocessing...", flush=True)
    x_train = build_cache(train.image_id, f"{DATA_DIR}/train_images")
    x_test = build_cache(test.image_id, f"{DATA_DIR}/test_images")
    print(f"  done in {time.time() - t0:.0f}s  {x_train.shape} {x_test.shape}\n", flush=True)

    print("=" * 68)
    print("FEATURE EXTRACTION (all backbones frozen)")
    print("=" * 68)
    feats = {}
    for spec in EXTRACTORS:
        f_tr, f_te = extract(spec, x_train, x_test)
        if f_tr is not None:
            feats[spec["tag"]] = (f_tr, f_te)
    if not feats:
        raise RuntimeError("every extractor failed -- check the internet switch")

    print("\n" + "=" * 68)
    print("SVM PER FEATURE SET")
    print("=" * 68)
    oofs, tests, tags = [], [], []
    for tag, (f_tr, f_te) in feats.items():
        oof, tp, _ = cv_svm(f_tr, f_te, y, tag)
        report(y, oof.argmax(1), f"classification report -- SVM on {tag} features (OOF)")
        np.save(f"{WORK}/oof_svm_{tag}.npy", oof)
        np.save(f"{WORK}/test_svm_{tag}.npy", tp)
        oofs.append(oof)
        tests.append(tp)
        tags.append(tag)

    # Concatenating every feature set into one SVM is the obvious move and usually the
    # wrong one at 212 rows -- it hands the kernel one enormous vector instead of
    # several opinions. Averaging the probabilities keeps them separable.
    print("\n" + "=" * 68)
    print("BLEND ACROSS FEATURE SETS")
    print("=" * 68)
    if len(oofs) > 1:
        eq = sum(oofs) / len(oofs)
        print(f"  equal weights      : {f1_score(y, eq.argmax(1), average='macro'):.4f}")
        cf = f1_score(y, crossfit_blend(oofs, y), average="macro")
        print(f"  tuned, cross-fitted: {cf:.4f}   <-- the honest estimate")
        best_solo = max(f1_score(y, o.argmax(1), average="macro") for o in oofs)
        if cf > max(best_solo, f1_score(y, eq.argmax(1), average="macro")):
            w = fit_blend(oofs, y)
            print(f"  shipping tuned weights {dict(zip(tags, np.round(w, 3)))}")
        else:
            w = np.ones(len(oofs)) / len(oofs)
            print("  tuning does not survive cross-fitting; shipping equal weights")
        oof = sum(wi * p for wi, p in zip(w, oofs))
        test_prob = sum(wi * p for wi, p in zip(w, tests))
    else:
        oof, test_prob = oofs[0], tests[0]
    report(y, oof.argmax(1), "classification report -- SVM blend (OOF)")

    # The point of the whole script: an SVM on frozen features fails on different
    # images than a fine-tuned network does, so the two average to something better
    # than either. Only shipped if it survives cross-fitting.
    for i, (oof_path, test_path) in enumerate(BLEND_WITH):
        try:
            o_ext, t_ext = np.load(oof_path), np.load(test_path)
        except Exception as exc:               # noqa: BLE001
            print(f"  external source {i} unreadable ({exc}); skipped")
            continue
        solo = f1_score(y, oof.argmax(1), average="macro")
        ext_solo = f1_score(y, o_ext.argmax(1), average="macro")
        cf = f1_score(y, crossfit_blend([oof, o_ext], y), average="macro")
        print(f"\n  + {os.path.basename(oof_path)}: alone {ext_solo:.4f}, "
              f"SVM alone {solo:.4f}, blended cross-fitted {cf:.4f}")
        if cf > max(solo, ext_solo):
            w = fit_blend([oof, o_ext], y)
            print(f"    -> shipping the blend, weights {np.round(w, 3)}")
            oof = w[0] * oof + w[1] * o_ext
            test_prob = w[0] * test_prob + w[1] * t_ext
            report(y, oof.argmax(1), "classification report -- SVM + external (OOF)")
        elif ext_solo > solo:
            # The blend did not help, and the external source is the better of the
            # two. Ship THAT, not the SVM. Shipping the weaker model because the
            # blend failed would be worse than not running this script at all --
            # which is exactly what an earlier version of this branch did.
            print("    -> blend does not help, and the external source is stronger "
                  "alone; shipping the external source unchanged")
            oof, test_prob = o_ext, t_ext
            report(y, oof.argmax(1), "classification report -- external alone (OOF)")
        else:
            print("    -> does not beat the SVM out-of-fold; keeping the SVM")

    np.save(f"{WORK}/oof_svm.npy", oof)
    np.save(f"{WORK}/test_svm.npy", test_prob)

    notebook_protocol_diagnostic(feats, y)

    final_oof = f1_score(y, oof.argmax(1), average="macro")
    print("\n" + "=" * 68)
    print(f"FINAL OOF macro F1 = {final_oof:.4f}")
    print("  ^ compare this against the ViT run's OOF, not against the public LB,")
    print("    which is scored on ~16 images.")
    if final_oof < 0.45:
        print("\n  *** DO NOT SUBMIT THIS FILE. ***")
        print("  A three-class macro F1 floors around 0.333, so this is barely above")
        print("  chance. Either a backbone failed to download (check the SKIPPED lines")
        print("  above) or frozen features alone are not enough here. Set BLEND_WITH to")
        print("  the ViT run's .npy files, or just submit the ViT run's own CSV.")
    print("=" * 68)

    sub = pd.DataFrame({"image_id": test.image_id,
                        "label": [CLASSES[i] for i in test_prob.argmax(1)]})
    sub.to_csv(OUT_CSV, index=False)
    assert len(sub) == len(test) and sub.image_id.is_unique
    assert set(sub.label) <= set(CLASSES)
    print(f"\nwrote {OUT_CSV} in {time.time() - t0:.0f}s")
    print(sub.label.value_counts().to_string())


if __name__ == "__main__":
    main()
