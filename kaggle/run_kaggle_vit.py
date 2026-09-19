"""HoloMine Task 1 -- single-model pipeline on hugging-science/breast-cancer-detector-2.

Paste into one Kaggle notebook cell. Needs the internet switch ON (the checkpoint is
pulled from Hugging Face); for an offline run, attach the repo as a Kaggle Model input
and set HF_MODEL_DIR to its path.

READ THIS FIRST. That checkpoint is a ViT-base fine-tuned on breast *ultrasound*
(BUSI, ~1,578 images), and its own model card lists mammography under Out-of-Scope.
The 94.46% accuracy it advertises is ultrasound accuracy and does not carry over here.
Whether it beats a plain ImageNet backbone on mammograms is an open question, so this
script answers it for you: STAGE 1 scores the checkpoint zero-shot before spending any
time on training. If that number is near chance (~0.33), stop and use run_kaggle.py
with resnet34 instead.

One thing genuinely lines up: the checkpoint's label order is benign, malignant,
normal -- exactly our class order -- so its trained 3-class head is kept as a warm
start rather than thrown away.

No external data, no LLM/VLM, no AutoML, no Ultralytics. A publicly available
pretrained classifier, which the rules permit.

Runtime: ~3 min for stage 1, ~25-40 min for stage 2 on a P100/T4.
"""
from __future__ import annotations

import os
import re
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config -- everything you would want to change lives here
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/holomine-breast-cancer-classification-task-1"
OUT_CSV = "/kaggle/working/submission.csv"

MODEL_ID = "hugging-science/breast-cancer-detector-2"
HF_MODEL_DIR = None        # local path to the repo, for a run with internet off
KEEP_HEAD = True           # reuse the checkpoint's 3-class head (label order matches)

CACHE_H, CACHE_W = 512, 384    # resolution the breast crops are cached at

# Each variant is a full cross-validated run of the same checkpoint, and they are
# blended at the end. This is the one thing measured to pay here: blending two models
# that fail differently was worth +0.07 out-of-fold, while every single-model tweak
# tried came in under 0.02 -- below the 0.023 spread between identical runs.
#
# Resolution is what makes them differ. 224x224 is the size this ViT was pretrained
# at, so its position embeddings are used as-is; 384x288 interpolates them and keeps
# more of the lesion detail that 224 throws away. The two disagree on different
# images, which is the whole point. Differing seeds alone would not: averaging two
# 384x288 runs gave 0.6259, between the two rather than above them.
#
# Cut this to one entry if the GPU budget is tight; each costs about 6 minutes.
VARIANTS = [
    dict(h=384, w=288, seed=0),
    dict(h=224, w=224, seed=1),
]

N_FOLDS = 5
EPOCHS = 12                # ViT-base overfits 212 images fast; more is not better
BATCH_SIZE = 8
LR = 3e-5                  # ViT fine-tuning needs a far lower LR than a CNN would
WEIGHT_DECAY = 0.05
DROPOUT = 0.1
LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.99
RUN_STAGE_1 = True         # zero-shot check before training

# ---------------------------------------------------------------------------
# Optional methods
#
# The defaults below reproduce the configuration that scored OOF 0.6401 and 0.6172
# on two runs. That is deliberate. Every previous attempt to switch several of these
# on at once made things worse -- mixup and a balanced sampler together cost this
# model 0.052 and collapsed a resnet34 to 0.1272 -- so each is a separate flag.
# Change ONE, rerun, and compare the OOF line. The fold-to-fold spread here is about
# 0.15, so a change worth under 0.02 cannot be told apart from noise on one run.
# ---------------------------------------------------------------------------

# Method 31/138 -- test-time augmentation. Horizontal flip is always applied, so this
# is 2 views by default, (1.0, 0.9) gives 4 and (1.0, 0.95, 0.9, 0.85) gives 8. Costs
# inference time only, never training, and averaging rarely hurts. Try this first.
TTA_SCALES = (1.0,)

# Method 29 -- snapshot ensemble. Keep a checkpoint every N epochs over the last third
# of training and average its predictions with the EMA weights. Several points on one
# trajectory that disagree usefully, at no extra training cost. 0 disables it.
SNAPSHOT_EPOCHS = 0

# Method 3/131 -- gradual unfreezing. Less compelling here than elsewhere: this
# checkpoint's head is reused rather than randomly initialised, so there is no
# gradient shock to protect the backbone from. Kept available for comparison.
WARMUP_EPOCHS = 0          # epochs training the head alone before anything unfreezes
GRADUAL_UNFREEZE = False   # after the warmup, release one block at a time

# Method 36 (default) or 40. Logit adjustment adds tau * log(prior) to the logits
# instead of re-weighting the loss, which targets macro F1 without changing a class's
# effective sample size -- so it does not stack with other imbalance corrections the
# way the sampler did when it collapsed an earlier run.
LOSS = "weighted_ce"       # "weighted_ce" or "logit_adjust"
LOGIT_ADJUST_TAU = 1.0

# Method 51/73 -- blend this run with probabilities saved by another one, sweeping a
# mixing weight and a temperature. Blending two models that fail differently was worth
# +0.09 out-of-fold here, more than any single-model change tried. The winner is
# re-checked cross-fitted and only shipped if it survives that.
BLEND_WITH = []            # e.g. [("/kaggle/input/prev/oof_r34.npy",
                           #        "/kaggle/input/prev/test_r34.npy")]

# ---------------------------------------------------------------------------
# A second opinion from a mammography-trained checkpoint
#
# abdullahtahir/resnet50-mammography-birads is the only public checkpoint found that
# matches this task on all three axes: it is mammography (not ultrasound, not
# histopathology), it has our exact three classes, and it reports 84.5% accuracy /
# 0.9665 AUC over 5-fold CV on 5,662 King Abdulaziz University images.
#
# It ships as a Keras .keras file, so it cannot join the PyTorch training loop. It
# does not need to: it is used untouched, predicting on our images without ever
# seeing our labels, which makes its predictions out-of-sample by construction and
# therefore honest to blend against our out-of-fold ones.
#
# Two things are unknown and therefore measured rather than assumed:
#
#   * Its class ORDER. The model card lists "Normal (BI-RADS 1), Benign (BI-RADS 2-3),
#     Malignant (BI-RADS 4-5)", which is not our order. All six permutations are
#     scored on the 212 labelled training images and the best is taken -- that is
#     aligning indices against known labels, not fitting a model.
#   * Its PREPROCESSING, which the card does not document. Both plausible
#     conventions are tried and the better one reported.
#
# Download the .keras file into a Kaggle Dataset and point KERAS_MODEL_PATH at it,
# or leave it None to skip this entirely.
# ---------------------------------------------------------------------------
KERAS_MODEL_PATH = None    # e.g. "/kaggle/input/mammo-birads/resnet50_mammography_best.keras"

CLASSES = ["Benign", "Malignant", "Normal"]
MEAN, STD = 0.449, 0.226
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Preprocessing: crop the breast out of the 3540x4740 full-field image
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
    # The breast is the largest bright component; the burned-in "R MLO" style labels
    # sit in a corner, disconnected, and are dropped here. That matters twice over for
    # this checkpoint, whose model card calls out text overlays as out of scope.
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def load_and_crop(path, scale=8):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # collapses the mixed L/RGB encoding
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
        crop = cv2.flip(crop, 1)                   # normalise laterality
    crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
    return cv2.resize(crop, (CACHE_W, CACHE_H), interpolation=cv2.INTER_AREA)


def build_cache(ids, image_dir):
    out = np.zeros((len(ids), CACHE_H, CACHE_W), dtype=np.uint8)
    for i, name in enumerate(ids):
        out[i] = load_and_crop(os.path.join(image_dir, name))
    return out

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def rand_affine(img, rng):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-12, 12), rng.uniform(0.88, 1.12))
    m[0, 2] += rng.uniform(-0.05, 0.05) * w
    m[1, 2] += rng.uniform(-0.05, 0.05) * h
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


class MammoDataset(Dataset):
    """Geometry-heavy, photometry-light: lesion appearance is the signal."""

    def __init__(self, images, labels, train, size, seed=0, scale=1.0, flip=False):
        self.images, self.labels, self.train = images, labels, train
        self.size = size                         # (h, w) this variant wants
        self.scale, self.flip = scale, flip      # fixed transforms, for TTA
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
                img = np.clip((img - 0.5) * rng.uniform(0.85, 1.15) + 0.5
                              + rng.uniform(-0.08, 0.08), 0, 1)
            if rng.random() < 0.3:
                h, w = img.shape
                for _ in range(rng.integers(1, 4)):
                    ch, cw = int(h * rng.uniform(.06, .16)), int(w * rng.uniform(.06, .16))
                    y0, x0 = rng.integers(0, h - ch), rng.integers(0, w - cw)
                    img[y0:y0 + ch, x0:x0 + cw] = 0.0
        else:
            img = img.astype(np.float32) / 255.0
            if self.flip:
                img = img[:, ::-1].copy()
            if self.scale != 1.0:
                h, w = img.shape                       # centre zoom, size fixed after
                ch, cw = int(h * self.scale), int(w * self.scale)
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                img = img[y0:y0 + ch, x0:x0 + cw]
        if img.shape != self.size:
            img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray((img - MEAN) / STD))[None].repeat(3, 1, 1)
        return x if self.labels is None else (x, int(self.labels[i]))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HFClassifier(nn.Module):
    """Wraps the transformers classifier so it returns a plain logits tensor."""

    def __init__(self, repo_id, num_classes=3, dropout=DROPOUT, keep_head=False):
        super().__init__()
        from transformers import AutoModelForImageClassification

        kwargs = {} if keep_head else dict(num_labels=num_classes, ignore_mismatched_sizes=True)
        self.model = AutoModelForImageClassification.from_pretrained(repo_id, **kwargs)
        if keep_head:
            got = self.model.classifier.out_features
            if got != num_classes:
                raise ValueError(f"{repo_id} head has {got} classes, expected {num_classes}")
        else:
            in_f = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Dropout(dropout),
                                                  nn.Linear(in_f, num_classes))
        self.interp = "vit" in self.model.config.model_type

    def forward(self, x):
        if self.interp:
            # ViT position embeddings are tied to 224x224; resample them so the model
            # can take the larger crops small mammographic lesions need.
            return self.model(pixel_values=x, interpolate_pos_encoding=True).logits
        return self.model(pixel_values=x).logits


def build_model(keep_head=KEEP_HEAD):
    return HFClassifier(HF_MODEL_DIR or MODEL_ID, 3, DROPOUT, keep_head)


class EMA:
    """Averaging the weight trajectory beats the last step at this sample size."""

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
def predict(model, images, size):
    """Average over every flip x scale view (method 31/138)."""
    model.eval()
    total = None
    for scale in TTA_SCALES:
        for flip in (False, True):
            loader = DataLoader(MammoDataset(images, None, False, size,
                                             scale=scale, flip=flip),
                                batch_size=BATCH_SIZE)
            out = []
            for xb in loader:
                xb = xb.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                    out.append(model(xb).softmax(1).float().cpu().numpy())
            p = np.concatenate(out)
            total = p if total is None else total + p
    return total / (len(TTA_SCALES) * 2)


def train_fold(x_tr, y_tr, size, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model().to(DEVICE)

    # macro F1 weights all three classes equally while the data does not
    # (52 Benign vs 80/80), so the loss has to make up the difference.
    counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    if LOSS == "logit_adjust":
        log_prior = torch.tensor(np.log(counts / counts.sum()), device=DEVICE,
                                 dtype=torch.float32)
        base = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        def crit(logits, target):
            return base(logits + LOGIT_ADJUST_TAU * log_prior, target)
    else:
        crit = nn.CrossEntropyLoss(
            weight=torch.tensor(counts.sum() / (3 * counts), device=DEVICE),
            label_smoothing=LABEL_SMOOTHING)

    # Block index per parameter, for gradual unfreezing. ViT blocks are
    # `encoder.layer.N`; the embeddings count as block 0.
    depth_of, max_block = {}, 1
    for n, _ in model.named_parameters():
        m = re.search(r"layer\.(\d+)\.", n)
        d = (int(m.group(1)) + 1) if m else (0 if "embed" in n else None)
        depth_of[n] = d
        if d is not None:
            max_block = max(max_block, d)

    def set_trainable(unfrozen_from):
        """Unfreeze the head plus every block at or above `unfrozen_from`; None = head only."""
        for n, p in model.named_parameters():
            if "classifier" in n:
                p.requires_grad_(True)
                continue
            d = depth_of[n]
            p.requires_grad_(unfrozen_from is not None
                             and (max_block if d is None else d) >= unfrozen_from)

    def unfreeze_plan(epoch):
        if epoch < WARMUP_EPOCHS:
            return None
        if not GRADUAL_UNFREEZE:
            return 0
        span = max(1, EPOCHS - WARMUP_EPOCHS)
        return max(0, int(round(max_block * (1.0 - (epoch - WARMUP_EPOCHS) / span))))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(MammoDataset(x_tr, y_tr, True, size, seed),
                        batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=len(x_tr) > BATCH_SIZE, num_workers=2, pin_memory=True)
    steps = max(1, len(loader)) * max(1, EPOCHS - WARMUP_EPOCHS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.25)
    ema = EMA(model, EMA_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    n_ok = n_skip = 0

    snapshots = []                       # method 29
    snapshot_from = int(EPOCHS * 2 / 3)

    for epoch in range(EPOCHS):
        warming = epoch < WARMUP_EPOCHS
        set_trainable(unfreeze_plan(epoch))
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # A dropped scale means GradScaler skipped this step (fp16 overflow).
            # Advancing the schedule or the EMA on a step that never happened is how a
            # run quietly ends up undertrained.
            if scaler.get_scale() >= scale_before:
                if not warming and sched.last_epoch < steps - 1:
                    sched.step()
                ema.update(model)
                n_ok += 1
            else:
                n_skip += 1

        if (SNAPSHOT_EPOCHS and epoch >= snapshot_from
                and (epoch - snapshot_from) % SNAPSHOT_EPOCHS == 0):
            snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})

    set_trainable(0)
    if n_skip > 0.1 * max(n_ok + n_skip, 1):
        print(f"    WARNING: fp16 overflow skipped {n_skip}/{n_ok + n_skip} optimizer "
              f"steps -- lower LR", flush=True)
    ema.copy_to(model)
    return model, snapshots


def confidence_report(prob):
    """Flag a fold that produced near-uniform probabilities.

    A three-class softmax floors at 0.333. Two earlier runs posted a plausible-looking
    macro F1 while sitting at a median max-prob under 0.42 -- they had not learned
    anything, and only this said so. A healthy fold sits near 0.7.
    """
    med = float(np.median(prob.max(1)))
    flag = "   <-- NEAR-UNIFORM, this fold did not train" if med < 0.45 else ""
    print(f"    confidence: median max-prob {med:.3f}, "
          f"{(prob.max(1) > 0.5).mean():.0%} above 0.5{flag}", flush=True)


# ---------------------------------------------------------------------------
# Class-prior tuning
# ---------------------------------------------------------------------------


def fit_weights(prob, y):
    """Coordinate ascent on per-class multipliers, maximising macro F1."""
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

# ---------------------------------------------------------------------------


def _temper(prob, t):
    """Re-sharpen (t<1) or soften (t>1) a probability vector."""
    logit = np.log(np.clip(prob, 1e-9, None)) / t
    e = np.exp(logit - logit.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def try_blend(oof, test_prob, y):
    """Blend with another run's saved probabilities, if it survives cross-fitting.

    Sweeps the mixing weight and a temperature on the other run (methods 51 and 73).
    A run that is systematically more confident otherwise dominates the argmax
    whatever weight it is given, and tempering lets the weight mean what it says.
    Picking both against the same 212 rows that then score them is worth several
    points of self-flattery, so the winner is re-checked fold by fold first.
    """
    if not BLEND_WITH:
        return oof, test_prob

    solo = f1_score(y, oof.argmax(1), average="macro")
    best_oof, best_test, best_cf = oof, test_prob, solo

    def search(a_oof, b_oof, yy):
        best, arg = -1.0, (1.0, 1.0)
        for temp in (0.5, 0.75, 1.0, 1.5, 2.0):
            tb = _temper(b_oof, temp)
            for a in np.arange(0.0, 1.01, 0.1):
                sc = f1_score(yy, (a * a_oof + (1 - a) * tb).argmax(1), average="macro")
                if sc > best:
                    best, arg = sc, (a, temp)
        return arg

    for oof_path, test_path in BLEND_WITH:
        if not (os.path.exists(oof_path) and os.path.exists(test_path)):
            print(f"  blend source missing, skipped: {oof_path}")
            continue
        o2 = np.load(oof_path).astype(np.float64)
        t2 = np.load(test_path).astype(np.float64)
        o2 /= o2.sum(1, keepdims=True)
        t2 /= t2.sum(1, keepdims=True)

        pred = np.empty(len(y), dtype=np.int64)
        for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(y, y):
            a, temp = search(oof[tr], o2[tr], y[tr])
            pred[va] = (a * oof[va] + (1 - a) * _temper(o2, temp)[va]).argmax(1)
        cf = f1_score(y, pred, average="macro")
        name = os.path.basename(oof_path)
        print(f"  blend with {name}: cross-fitted {cf:.4f}  (this model alone {solo:.4f})")
        if cf > best_cf + 1e-9:
            best_cf = cf
            a, temp = search(oof, o2, y)
            print(f"    -> shipping it, weight {a:.1f} here, temperature {temp} on {name}")
            best_oof = a * oof + (1 - a) * _temper(o2, temp)
            best_test = a * test_prob + (1 - a) * _temper(t2, temp)

    if best_oof is oof:
        print("  no blend beat this model alone out-of-fold; shipping it unblended")
    return best_oof, best_test



# ---------------------------------------------------------------------------
# Mammography-trained second opinion (Keras)
# ---------------------------------------------------------------------------


def keras_second_opinion(x_train, x_test, y):
    """Predict with the mammography Keras checkpoint, aligning its classes to ours.

    Returns (oof_like, test_prob) or (None, None) if it is not configured or cannot
    be read. The predictions on x_train are out-of-sample by construction: this model
    was trained on a different dataset and never sees our labels, so blending its
    output against our out-of-fold predictions is honest.
    """
    if not KERAS_MODEL_PATH or not os.path.exists(KERAS_MODEL_PATH):
        return None, None

    import itertools
    import tensorflow as tf

    print("=" * 68)
    print("SECOND OPINION -- abdullahtahir/resnet50-mammography-birads (Keras)")
    print("=" * 68)
    model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)

    # Read the expected input geometry off the model rather than guessing it.
    shape = model.input_shape
    h, w, c = (shape[1] or 224), (shape[2] or 224), (shape[3] or 3)
    n_out = int(model.output_shape[-1])
    print(f"  input {h}x{w}x{c}, {n_out} outputs")
    if n_out != len(CLASSES):
        print(f"  -> {n_out} outputs cannot map onto {len(CLASSES)} classes; skipping")
        return None, None

    def run(images, mode):
        batch = np.stack([cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
                          for im in images]).astype(np.float32)
        batch = np.repeat(batch[..., None], c, axis=-1)
        if mode == "caffe":
            # What tf.keras.applications.resnet50.preprocess_input does: BGR order
            # and per-channel ImageNet mean subtraction, no scaling.
            batch = batch[..., ::-1] - np.array([103.939, 116.779, 123.68], np.float32)
        else:
            batch = batch / 255.0
        out = model.predict(batch, batch_size=16, verbose=0)
        out = np.asarray(out, dtype=np.float64)
        if not np.allclose(out.sum(1), 1.0, atol=1e-3):      # logits, not probabilities
            e = np.exp(out - out.max(1, keepdims=True))
            out = e / e.sum(1, keepdims=True)
        return out / out.sum(1, keepdims=True)

    # The card documents neither the preprocessing nor the class order, so both are
    # resolved against the 212 labelled images instead of assumed.
    best = None
    for mode in ("caffe", "scale"):
        prob = run(x_train, mode)
        for perm in itertools.permutations(range(len(CLASSES))):
            sc = f1_score(y, prob[:, perm].argmax(1), average="macro")
            if best is None or sc > best[0]:
                best = (sc, mode, perm)
    score, mode, perm = best
    print(f"  best: preprocessing={mode}  class order={perm}  macro F1 {score:.4f}"
          f"   (chance ~0.33)")
    print(f"    reading its outputs as " +
          ", ".join(f"{CLASSES[i]}<-idx{p}" for i, p in enumerate(perm)))

    if score < 0.40:
        print("  -> at chance on our images; the domain gap is too wide. Skipping.")
        return None, None

    oof_like = run(x_train, mode)[:, perm]
    test_prob = run(x_test, mode)[:, perm]
    confidence_report(oof_like)
    print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    for row in confusion_matrix(y, oof_like.argmax(1)):
        print("   ", row)
    np.save("/kaggle/working/oof_keras_mammo.npy", oof_like)
    np.save("/kaggle/working/test_keras_mammo.npy", test_prob)
    print()
    del model
    return oof_like, test_prob


def stage_1_zeroshot(x_train, y, size):
    """Score the checkpoint untouched. Cheap, and it decides whether stage 2 is worth it."""
    print("=" * 68)
    print("STAGE 1 -- zero-shot, no fine-tuning")
    print("=" * 68)
    model = build_model(keep_head=True).to(DEVICE).eval()
    pred = predict(model, x_train, size).argmax(1)
    score = f1_score(y, pred, average="macro")
    print(f"zero-shot macro F1 on the 212 training images = {score:.4f}   (chance ~0.33)")
    counts = np.bincount(pred, minlength=3)
    print("prediction spread: " + "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
    if counts.max() > 0.8 * len(y):
        print("  -> collapsed onto one class. That is what a checkpoint reading the")
        print("     wrong imaging modality looks like, and it says little: this one")
        print("     scored 0.1922 here and still fine-tuned to 0.64.")
    print()
    del model
    torch.cuda.empty_cache()
    return score


def run_variant(variant, x_train, y, x_test, t0):
    """One full cross-validated run at this variant's resolution."""
    size = (variant["h"], variant["w"])
    seed = variant["seed"]
    tag = f"{variant['h']}x{variant['w']}_s{seed}"
    print("=" * 68)
    print(f"VARIANT {tag}   (native 224x224 uses the position embeddings as-is; "
          f"anything larger interpolates them)")
    print("=" * 68)

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_prob = np.zeros((len(x_test), 3), dtype=np.float32)
    scores = []
    for f, (tr, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True,
                                                 random_state=seed).split(y, y)):
        model, snaps = train_fold(x_train[tr], y[tr], size, seed * 100 + f)
        # EMA weights plus each snapshot, averaged: one training run, several points
        # on its trajectory, and they disagree in useful ways.
        va_prob = predict(model, x_train[va], size)
        te_prob = predict(model, x_test, size)
        if snaps:
            ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            for snap in snaps:
                model.load_state_dict(snap)
                va_prob += predict(model, x_train[va], size)
                te_prob += predict(model, x_test, size)
            model.load_state_dict(ema_state)
            va_prob /= (1 + len(snaps))
            te_prob /= (1 + len(snaps))
        confidence_report(va_prob)
        oof[va] = va_prob
        test_prob += te_prob
        s = f1_score(y[va], va_prob.argmax(1), average="macro")
        scores.append(s)
        print(f"  fold {f}: macroF1 {s:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    test_prob /= N_FOLDS
    print(f"  OOF macro F1 {f1_score(y, oof.argmax(1), average='macro'):.4f}   "
          f"fold spread {min(scores):.3f}-{max(scores):.3f}\n")
    np.save(f"/kaggle/working/oof_vit_{tag}.npy", oof)
    np.save(f"/kaggle/working/test_vit_{tag}.npy", test_prob)
    return oof, test_prob, tag


def blend_variants(oofs, tests, tags, y):
    """Weighted blend across variants, with the weights checked cross-fitted."""
    solo = [f1_score(y, o.argmax(1), average="macro") for o in oofs]
    for tag, sc in zip(tags, solo):
        print(f"  {tag:<24} OOF {sc:.4f}")
    best_i = int(np.argmax(solo))
    if len(oofs) == 1:
        return oofs[0], tests[0]

    grid = [np.array(w) / 10 for w in np.ndindex(*(11,) * len(oofs)) if sum(w) == 10]

    def search(subset, yy):
        best, arg = -1.0, grid[0]
        for w in grid:
            sc = f1_score(yy, sum(wi * o for wi, o in zip(w, subset)).argmax(1),
                          average="macro")
            if sc > best:
                best, arg = sc, w
        return arg

    # Pick the weights on four folds, score the fifth. Searching a grid against the
    # same 212 rows it is then reported on is worth several points of self-flattery.
    pred = np.empty(len(y), dtype=np.int64)
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(y, y):
        w = search([o[tr] for o in oofs], y[tr])
        pred[va] = sum(wi * o[va] for wi, o in zip(w, oofs)).argmax(1)
    cf = f1_score(y, pred, average="macro")
    print(f"\n  cross-fitted blend {cf:.4f}   vs best single {solo[best_i]:.4f}")

    if cf <= solo[best_i]:
        print(f"  -> blend does NOT hold up; shipping {tags[best_i]} alone")
        return oofs[best_i], tests[best_i]
    w = search(oofs, y)
    print("  -> shipping the blend, weights " +
          ", ".join(f"{t}={wi:.2f}" for t, wi in zip(tags, w)))
    return (sum(wi * o for wi, o in zip(w, oofs)),
            sum(wi * t for wi, t in zip(w, tests)))


def main():
    t0 = time.time()
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    y = np.array([CLASSES.index(v) for v in train.label], dtype=np.int64)

    print(f"device={DEVICE}  model={MODEL_ID}  keep_head={KEEP_HEAD}")
    print(f"variants={[(v['h'], v['w'], v['seed']) for v in VARIANTS]}  folds={N_FOLDS}  "
          f"epochs={EPOCHS}  loss={LOSS}  tta_views={len(TTA_SCALES) * 2}")
    print(f"snapshot_every={SNAPSHOT_EPOCHS}  warmup={WARMUP_EPOCHS}  "
          f"gradual_unfreeze={GRADUAL_UNFREEZE}  blend_sources={len(BLEND_WITH)}")
    print("preprocessing...", flush=True)
    x_train = build_cache(train.image_id, f"{DATA_DIR}/train_images")
    x_test = build_cache(test.image_id, f"{DATA_DIR}/test_images")
    print(f"  done in {time.time() - t0:.0f}s  {x_train.shape} {x_test.shape}\n", flush=True)

    if RUN_STAGE_1:
        stage_1_zeroshot(x_train, y, (VARIANTS[0]["h"], VARIANTS[0]["w"]))

    oofs, tests, tags = [], [], []
    for variant in VARIANTS:
        o, t, tag = run_variant(variant, x_train, y, x_test, t0)
        oofs.append(o)
        tests.append(t)
        tags.append(tag)

    print("=" * 68)
    print("BLEND ACROSS VARIANTS")
    print("=" * 68)
    oof, test_prob = blend_variants(oofs, tests, tags, y)
    np.save("/kaggle/working/oof_vit.npy", oof)
    np.save("/kaggle/working/test_prob_vit.npy", test_prob)

    k_oof, k_test = keras_second_opinion(x_train, x_test, y)
    if k_oof is not None:
        # The Keras model fails differently from a fine-tuned ViT -- different
        # architecture, different training set, different modality history -- which is
        # the only reason averaging them is worth anything.
        solo = f1_score(y, oof.argmax(1), average="macro")
        best_a, best_sc = None, solo
        pred = np.empty(len(y), dtype=np.int64)
        for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(y, y):
            a_best, s_best = 1.0, -1.0
            for a in np.arange(0.0, 1.01, 0.1):
                sc = f1_score(y[tr], (a * oof[tr] + (1 - a) * k_oof[tr]).argmax(1),
                              average="macro")
                if sc > s_best:
                    a_best, s_best = a, sc
            pred[va] = (a_best * oof[va] + (1 - a_best) * k_oof[va]).argmax(1)
        cf = f1_score(y, pred, average="macro")
        print(f"  ViT + mammography checkpoint: cross-fitted {cf:.4f}  "
              f"(ViT alone {solo:.4f})")
        if cf > solo:
            for a in np.arange(0.0, 1.01, 0.1):
                sc = f1_score(y, (a * oof + (1 - a) * k_oof).argmax(1), average="macro")
                if sc > best_sc:
                    best_a, best_sc = a, sc
            if best_a is not None:
                print(f"    -> shipping it, weight {best_a:.1f} on the ViT")
                oof = best_a * oof + (1 - best_a) * k_oof
                test_prob = best_a * test_prob + (1 - best_a) * k_test
        else:
            print("    -> does not hold up out-of-fold; keeping the ViT alone")

    oof, test_prob = try_blend(oof, test_prob, y)

    # Cross-fitted: weights for each fold's rows are fitted only on the other folds,
    # so the gain reported is one that can actually carry over to the test set.
    pred_cf = np.empty(len(y), dtype=np.int64)
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(y, y):
        pred_cf[va] = (oof[va] * fit_weights(oof[tr], y[tr])).argmax(1)
    plain = f1_score(y, oof.argmax(1), average="macro")
    tuned = f1_score(y, pred_cf, average="macro")

    print("\n" + "=" * 68)
    print(f"FINAL OOF  plain={plain:.4f}   class-prior tuned(cross-fitted)={tuned:.4f}")
    print("  ^ THIS is the number to compare across runs. Not the public LB, which is")
    print("    scored on ~16 images -- six teams sit on the identical score there.")
    w = fit_weights(oof, y) if tuned > plain else np.ones(3)
    print(f"class weights {np.round(w, 3)} for {CLASSES}")
    final = (oof * w).argmax(1)
    print("per-class F1: " + "  ".join(
        f"{c}={f1_score(y, final, average=None, labels=[i])[0]:.3f}"
        for i, c in enumerate(CLASSES)))
    print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    for row in confusion_matrix(y, final):
        print("   ", row)

    sub = pd.DataFrame({"image_id": test.image_id,
                        "label": [CLASSES[i] for i in (test_prob * w).argmax(1)]})
    sub.to_csv(OUT_CSV, index=False)
    assert len(sub) == len(test) and sub.image_id.is_unique
    assert set(sub.label) <= set(CLASSES)
    print(f"\nwrote {OUT_CSV} in {time.time() - t0:.0f}s")
    print(sub.label.value_counts().to_string())


if __name__ == "__main__":
    main()
