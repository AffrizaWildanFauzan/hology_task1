"""Self-contained Kaggle notebook script -- HoloMine Task 1 (Hology 9.0).

Paste into one Kaggle notebook cell, or add this repo as a Kaggle Dataset and
`%run` it. Nothing is downloaded at run time except the ImageNet weights that
torchvision/timm ship, so it works with the internet switch off as long as the
weights come from a Kaggle "Models"/dataset mount (set WEIGHTS_DIR below) --
otherwise turn internet on for the first run.

No external data, no LLM/VLM, no AutoML, no Ultralytics. ImageNet-pretrained
backbones only, which the rules permit.

Pipeline: breast crop -> CV fine-tune -> flip TTA -> class-prior tuning -> CSV.
Expect ~10-20 min on a P100/T4 for the default settings.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/holomine-breast-cancer-classification-task-1"
OUT_CSV = "/kaggle/working/submission.csv"
WEIGHTS_DIR = None            # set to a Kaggle mount to run with internet off

IMG_H, IMG_W = 512, 384
MODEL_NAME = "resnet34"
N_FOLDS = 5
SEEDS = [0, 1]                # repeated CV; more seeds = steadier estimate
EPOCHS = 25
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-2
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.99

CLASSES = ["Benign", "Malignant", "Normal"]
MEAN, STD = 0.449, 0.226
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if WEIGHTS_DIR:
    os.environ["TORCH_HOME"] = WEIGHTS_DIR

# ----------------------------------------------------------------------------
# Preprocessing: crop the breast out of the 3540x4740 full-field image
# ----------------------------------------------------------------------------


def breast_mask(small):
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.ones_like(small, dtype=np.uint8)
    # The breast is the largest bright component; burned-in "R MLO" style labels
    # sit in a corner, disconnected, and are dropped here.
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def load_and_crop(path, scale=8):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # also collapses the mixed L/RGB encoding
    small = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_AREA)
    m = breast_mask(small)
    ys, xs = np.where(m > 0)
    mask_full = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    img = img * mask_full
    crop = img[ys.min() * scale:(ys.max() + 1) * scale, xs.min() * scale:(xs.max() + 1) * scale]
    sub = mask_full[ys.min() * scale:(ys.max() + 1) * scale, xs.min() * scale:(xs.max() + 1) * scale]
    if sub[:, :sub.shape[1] // 2].sum() < sub[:, sub.shape[1] // 2:].sum():
        crop = cv2.flip(crop, 1)                    # normalise laterality
    crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
    return cv2.resize(crop, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def build_cache(ids, image_dir):
    out = np.zeros((len(ids), IMG_H, IMG_W), dtype=np.uint8)
    for i, name in enumerate(ids):
        out[i] = load_and_crop(os.path.join(image_dir, name))
    return out

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------


def rand_affine(img, rng):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-12, 12), rng.uniform(0.88, 1.12))
    m[0, 2] += rng.uniform(-0.05, 0.05) * w
    m[1, 2] += rng.uniform(-0.05, 0.05) * h
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


class MammoDataset(Dataset):
    def __init__(self, images, labels, train, seed=0):
        self.images, self.labels, self.train = images, labels, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.train:
            rng = self.rng
            if rng.random() < 0.5:
                img = img[:, ::-1].copy()
            if rng.random() < 0.8:
                img = rand_affine(img, rng)
            img = img.astype(np.float32) / 255.0
            if rng.random() < 0.7:
                img = np.clip((img - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.08, 0.08), 0, 1)
            if rng.random() < 0.3:
                h, w = img.shape
                for _ in range(rng.integers(1, 4)):
                    ch, cw = int(h * rng.uniform(.06, .16)), int(w * rng.uniform(.06, .16))
                    y0, x0 = rng.integers(0, h - ch), rng.integers(0, w - cw)
                    img[y0:y0 + ch, x0:x0 + cw] = 0.0
        else:
            img = img.astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray((img - MEAN) / STD))[None].repeat(3, 1, 1)
        return x if self.labels is None else (x, int(self.labels[i]))

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


def build_model():
    try:
        import timm
        return timm.create_model(MODEL_NAME, pretrained=True, num_classes=3, drop_rate=DROPOUT)
    except Exception:
        import torchvision.models as tvm
        m = getattr(tvm, MODEL_NAME)(weights="DEFAULT")
        if hasattr(m, "fc"):
            m.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.fc.in_features, 3))
        else:
            in_f = m.classifier[-1].in_features
            m.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(in_f, 3))
        return m


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def copy_to(self, model):
        own = model.state_dict()
        model.load_state_dict({k: v.to(dtype=own[k].dtype) for k, v in self.shadow.items()})


@torch.no_grad()
def predict(model, images):
    model.eval()
    out = []
    for xb in DataLoader(MammoDataset(images, None, False), batch_size=BATCH_SIZE):
        xb = xb.to(DEVICE)
        p = (model(xb).softmax(1) + model(torch.flip(xb, dims=[3])).softmax(1)) / 2
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def train_fold(x_tr, y_tr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model().to(DEVICE)
    counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    # macro F1 weights all three classes equally while the data does not (52 Benign
    # vs 80/80), so the loss is inverse-frequency weighted to match the metric.
    crit = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (3 * counts), device=DEVICE),
                               label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(MammoDataset(x_tr, y_tr, True, seed), batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=len(x_tr) > BATCH_SIZE, num_workers=2, pin_memory=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=len(loader) * EPOCHS,
                                                pct_start=0.25)
    ema = EMA(model, EMA_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
    ema.copy_to(model)          # EMA weights generalise better at this sample size
    return model

# ----------------------------------------------------------------------------
# Class-prior tuning
# ----------------------------------------------------------------------------


def fit_weights(prob, y):
    grid = np.exp(np.linspace(-1.2, 1.2, 49))
    w = np.ones(3)
    best = f1_score(y, (prob * w).argmax(1), average="macro")
    for _ in range(6):
        improved = False
        for c in range(3):
            base = w[c]
            for g in grid:
                w[c] = g
                s = f1_score(y, (prob * w).argmax(1), average="macro")
                if s > best + 1e-9:
                    best, base, improved = s, g, True
            w[c] = base
        if not improved:
            break
    return w / w.mean()


def main():
    t0 = time.time()
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    y = np.array([CLASSES.index(v) for v in train.label], dtype=np.int64)

    print("preprocessing...", flush=True)
    x_train = build_cache(train.image_id, f"{DATA_DIR}/train_images")
    x_test = build_cache(test.image_id, f"{DATA_DIR}/test_images")
    print(f"  done in {time.time() - t0:.0f}s  {x_train.shape} {x_test.shape}", flush=True)

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_prob = np.zeros((len(test), 3), dtype=np.float32)
    n_runs = 0
    for seed in SEEDS:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        for f, (tr, va) in enumerate(skf.split(y, y)):
            model = train_fold(x_train[tr], y[tr], seed * 100 + f)
            va_prob = predict(model, x_train[va])
            oof[va] += va_prob
            test_prob += predict(model, x_test)
            n_runs += 1
            print(f"  seed {seed} fold {f}: macroF1 "
                  f"{f1_score(y[va], va_prob.argmax(1), average='macro'):.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    oof /= len(SEEDS)
    test_prob /= n_runs

    # Cross-fitted check: fit the weights on 4 folds, score the 5th, so the gain
    # reported is one that can actually carry over to the test set.
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEEDS[0])
    pred_cf = np.empty(len(y), dtype=np.int64)
    for tr, va in skf.split(y, y):
        pred_cf[va] = (oof[va] * fit_weights(oof[tr], y[tr])).argmax(1)
    plain = f1_score(y, oof.argmax(1), average="macro")
    tuned = f1_score(y, pred_cf, average="macro")
    print(f"\nOOF macro F1  plain={plain:.4f}  tuned(cross-fit)={tuned:.4f}")

    w = fit_weights(oof, y) if tuned > plain else np.ones(3)
    print("class weights:", np.round(w, 3))
    print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    print(confusion_matrix(y, (oof * w).argmax(1)))

    pred = (test_prob * w).argmax(1)
    sub = pd.DataFrame({"image_id": test.image_id, "label": [CLASSES[i] for i in pred]})
    sub.to_csv(OUT_CSV, index=False)
    np.save("/kaggle/working/oof.npy", oof)
    np.save("/kaggle/working/test_prob.npy", test_prob)
    print(f"\nwrote {OUT_CSV} in {time.time() - t0:.0f}s")
    print(sub.label.value_counts().to_string())


if __name__ == "__main__":
    main()
