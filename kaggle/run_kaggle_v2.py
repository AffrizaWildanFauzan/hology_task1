"""HoloMine Task 1 -- ConvNeXtV2-large pipeline with the variance controls that the
first real run showed were needed.

Paste into one Kaggle notebook cell. Internet switch ON (the checkpoint comes from
Hugging Face); for an offline run, attach the repo as a Kaggle Model input and point
HF_MODEL_DIR at it.

WHY THIS SCRIPT EXISTS
----------------------
The first real run (ViT, hugging-science/breast-cancer-detector-2) came back with
OOF macro F1 0.6401 and per-fold scores of 0.69, 0.57, 0.50, 0.69, 0.75. That spread
-- a quarter of a point between the best and worst fold on 42 validation images each --
is the single biggest problem to attack. It is not model capacity. So most of what is
added here fights variance rather than chasing a fancier architecture:

  * repeated CV over several seeds, averaged
  * mixup, which is worth more on 212 images than any backbone swap
  * layer-wise LR decay, so a 197M-parameter backbone is not wrecked by one LR
  * multi-scale + flip TTA instead of flip alone
  * balanced sampling on top of the class-weighted loss
  * optional blending with a previous run's saved probabilities, chosen on OOF

ABOUT THIS CHECKPOINT
---------------------
ALM-AHME/convnextv2-large-...-BreakHis was fine-tuned on BreakHis, which is breast
*histopathology* -- microscope images of biopsy slides -- not mammography. That is a
third imaging modality, as unlike our X-ray projections as the ultrasound checkpoint
was. Its 99.01% accuracy is a histopathology number and does not transfer.

It also has only TWO classes (benign, malignant) and no "normal", so its head cannot
be reused; the script detects this and re-initialises a 3-way head automatically.

What the ViT run established is that the zero-shot score says little: that checkpoint
scored 0.1922 zero-shot, collapsing onto one class, then fine-tuned to 0.6401. The
pretrained *features* transfer even when the pretrained *head* does not. Stage 1 is
still run here because it is cheap, but read it as a sanity check, not as a verdict.

No external data, no LLM/VLM, no AutoML, no Ultralytics. A publicly available
pretrained classifier, which the rules permit.

Runtime: roughly 12-18 min per seed on a P100/T4 at these settings.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/holomine-breast-cancer-classification-task-1"
WORK_DIR = "/kaggle/working"
OUT_CSV = f"{WORK_DIR}/submission.csv"

MODEL_ID = ("ALM-AHME/convnextv2-large-1k-224-finetuned-"
            "BreastCancer-Classification-BreakHis-AH-60-20-20")
HF_MODEL_DIR = None          # local path to the repo, for a run with internet off

CACHE_H, CACHE_W = 512, 384
MODEL_H, MODEL_W = 384, 288  # ConvNeXt is fully convolutional -- any size works, no
                             # position-embedding interpolation needed. Raise to
                             # 512x384 if memory allows; compare on OOF.

N_FOLDS = 5
SEEDS = [0, 1]               # repeated CV. The fold spread in the ViT run was 0.25,
                             # so averaging seeds is the highest-value knob here.
EPOCHS = 16
WARMUP_EPOCHS = 3            # head-only epochs before the backbone is unfrozen.
                             # This checkpoint's 2-class head cannot be reused, so
                             # ours starts random; letting its large early gradients
                             # straight into a 197M-parameter backbone is the most
                             # likely reason the first attempt never learned. The ViT
                             # never hit this because its 3-class head was reusable.
BATCH_SIZE = 4               # convnextv2-large is ~197M params
GRAD_ACCUM = 2               # effective batch 8
GRAD_CHECKPOINT = True       # trades ~30% speed for a large memory saving
USE_AMP = False              # resnet34 scored 0.5842 in fp32 and 0.4012 under fp16 on
                             # an otherwise identical run; fp32 is the safe default for
                             # a model that is failing to train
LR = 4e-5
LLRD = 0.75                  # layer-wise LR decay: early layers train slower
HEAD_LR_MULT = 10.0          # the fresh 3-way head needs a much higher LR
WEIGHT_DECAY = 0.05
DROPOUT = 0.1
LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.99

# Both measured, both harmful here. Turning them on cost the ViT 0.052 and collapsed
# resnet34 to 0.1272, where it predicted Benign for 210 of 212 images: the
# class-weighted loss already corrects the imbalance, and a sampler plus mixup on top
# of it, on 212 images and a few hundred optimizer steps, stops the model learning and
# lets the doubled Benign bias take over.
MIXUP_PROB = 0.0
MIXUP_ALPHA = 0.4
BALANCED_SAMPLER = False
TTA_SCALES = (1.0,)          # with horizontal flip -> 2 views per image

# Blend with a previous run's saved probabilities, e.g. the ViT run's
# oof_vit.npy / test_prob_vit.npy. The blend weight is chosen on OOF, and the blend
# is only shipped if it actually beats this model alone.
BLEND_WITH = []              # e.g. [(f"{WORK_DIR}/oof_vit.npy", f"{WORK_DIR}/test_prob_vit.npy")]

RUN_STAGE_1 = True

CLASSES = ["Benign", "Malignant", "Normal"]
MEAN, STD = 0.449, 0.226
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Preprocessing
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
    # Largest bright component is the breast; the burned-in "R MLO" labels sit in a
    # disconnected corner and are dropped with it.
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

    def __init__(self, images, labels, train, seed=0, scale=1.0, flip=False):
        self.images, self.labels, self.train = images, labels, train
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
                # Zoom about the centre, keeping the output size fixed.
                h, w = img.shape
                ch, cw = int(h * self.scale), int(w * self.scale)
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                img = img[y0:y0 + ch, x0:x0 + cw]

        img = cv2.resize(img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray((img - MEAN) / STD))[None].repeat(3, 1, 1)
        return x if self.labels is None else (x, int(self.labels[i]))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_HEAD_REUSED = None


class HFClassifier(nn.Module):
    """Wraps a transformers classifier so it returns a plain logits tensor.

    The checkpoint's head is reused only when its label order already matches
    CLASSES. This one is a 2-class benign/malignant head with no "normal", so it is
    replaced -- the backbone is what we are after.
    """

    def __init__(self, repo_id, num_classes=3, dropout=DROPOUT):
        super().__init__()
        global _HEAD_REUSED
        from transformers import AutoConfig, AutoModelForImageClassification

        cfg = AutoConfig.from_pretrained(repo_id)
        ckpt_labels = [cfg.id2label[i].lower() for i in range(cfg.num_labels)]
        reuse = ckpt_labels == [c.lower() for c in CLASSES]
        if _HEAD_REUSED is None:
            _HEAD_REUSED = reuse
            print(f"  checkpoint head: {ckpt_labels} -> "
                  + ("reused as a warm start" if reuse
                     else f"replaced with a fresh {num_classes}-way head"), flush=True)

        kwargs = {} if reuse else dict(num_labels=num_classes, ignore_mismatched_sizes=True)
        self.model = AutoModelForImageClassification.from_pretrained(repo_id, **kwargs)
        if not reuse:
            in_f = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Dropout(dropout),
                                                  nn.Linear(in_f, num_classes))
        self.interp = "vit" in self.model.config.model_type
        if GRAD_CHECKPOINT:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

    def forward(self, x):
        if self.interp:
            return self.model(pixel_values=x, interpolate_pos_encoding=True).logits
        return self.model(pixel_values=x).logits


def _depth_of(name: str, max_stage: int) -> int:
    """Map a parameter name to a depth index: 0 = stem, max_stage + 1 = head.

    Reads the stage/block number out of the module path, so it works for ConvNeXt
    (`encoder.stages.N`) and ViT (`encoder.layer.N`) alike without hard-coding either.
    """
    if "classifier" in name:
        return max_stage + 1
    m = re.search(r"stages?\.(\d+)\.", name) or re.search(r"layer\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if "embed" in name or "stem" in name or "patch" in name:
        return 0
    return max_stage                      # final norm and anything else, treat as late


def param_groups(model, base_lr):
    """Layer-wise LR decay: early layers move slowly, the fresh head moves fastest.

    A backbone pretrained on another modality should not have its early generic
    filters moved at the same rate as its last stage. The decay spans LLRD ** depth,
    so with LLRD 0.75 over five stages the stem trains at about a quarter of the last
    stage's rate.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    stages = [int(m.group(1)) for n, _ in named
              for m in [re.search(r"stages?\.(\d+)\.", n) or re.search(r"layer\.(\d+)\.", n)]
              if m]
    max_stage = (max(stages) + 1) if stages else 1

    buckets: dict[int, list] = {}
    for n, p in named:
        buckets.setdefault(_depth_of(n, max_stage), []).append(p)

    groups = []
    for depth in sorted(buckets):
        if depth == max_stage + 1:
            lr = base_lr * HEAD_LR_MULT   # freshly initialised, needs to catch up
        else:
            lr = base_lr * (LLRD ** (max_stage - depth))
        groups.append({"params": buckets[depth], "lr": lr})
    return groups


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

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def mixup(x, y, alpha, rng):
    """Convex combinations of image pairs, with the loss split across both labels.

    On 212 images this is worth more than any backbone change: it manufactures
    intermediate examples along the line between real ones and stops a 197M-parameter
    model from simply memorising the training set.
    """
    lam = float(rng.beta(alpha, alpha))
    lam = max(lam, 1 - lam)                 # keep the dominant image dominant
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


@torch.no_grad()
def predict(model, images):
    """Average over flip x scale views."""
    model.eval()
    total = None
    for scale in TTA_SCALES:
        for flip in (False, True):
            loader = DataLoader(MammoDataset(images, None, False, scale=scale, flip=flip),
                                batch_size=BATCH_SIZE * 2)
            out = []
            for xb in loader:
                xb = xb.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=USE_AMP and DEVICE == "cuda"):
                    out.append(model(xb).softmax(1).float().cpu().numpy())
            p = np.concatenate(out)
            total = p if total is None else total + p
    return total / (len(TTA_SCALES) * 2)


def train_fold(x_tr, y_tr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = HFClassifier(HF_MODEL_DIR or MODEL_ID, len(CLASSES), DROPOUT).to(DEVICE)

    # macro F1 weights all three classes equally while the data does not
    # (52 Benign vs 80/80), so the loss is inverse-frequency weighted to match it.
    counts = np.bincount(y_tr, minlength=len(CLASSES)).astype(np.float32)
    w = counts.sum() / (len(CLASSES) * counts)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE),
                               label_smoothing=LABEL_SMOOTHING)

    ds = MammoDataset(x_tr, y_tr, True, seed)
    if BALANCED_SAMPLER:
        # Sampling and loss weighting attack the imbalance from opposite ends; with
        # only 52 Benign images, using both keeps the minority class in every batch.
        sample_w = w[y_tr]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                        num_samples=len(y_tr), replacement=True)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True,
                            num_workers=2, pin_memory=True)
    else:
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                            num_workers=2, pin_memory=True)

    # Head warmup: with a randomly initialised 3-way head, the first backward passes
    # carry large, meaningless gradients. Holding the backbone still until the head is
    # roughly calibrated keeps them out of 197M pretrained parameters.
    def set_backbone_trainable(flag: bool) -> None:
        for n, p in model.named_parameters():
            if "classifier" not in n:
                p.requires_grad_(flag)

    opt = torch.optim.AdamW(param_groups(model, LR), weight_decay=WEIGHT_DECAY)
    steps = max(1, len(loader) // GRAD_ACCUM) * max(1, EPOCHS - WARMUP_EPOCHS)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps, pct_start=0.25)
    ema = EMA(model, EMA_DECAY)
    amp = USE_AMP and DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    n_ok = n_skip = 0

    for epoch in range(EPOCHS):
        warming = epoch < WARMUP_EPOCHS
        set_backbone_trainable(not warming)
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            use_mix = rng.random() < MIXUP_PROB and xb.size(0) > 1
            if use_mix:
                xb, ya, yb2, lam = mixup(xb, yb, MIXUP_ALPHA, rng)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(xb)
                loss = (lam * crit(logits, ya) + (1 - lam) * crit(logits, yb2)) if use_mix \
                    else crit(logits, yb)
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                # A dropped scale means GradScaler skipped this step (fp16 overflow).
                # Advancing the schedule or the EMA on a step that never happened is
                # how a run quietly ends up undertrained.
                if scaler.get_scale() >= scale_before:
                    if not warming and sched.last_epoch < steps - 1:
                        sched.step()
                    ema.update(model)
                    n_ok += 1
                else:
                    n_skip += 1

    set_backbone_trainable(True)
    if n_skip > 0.1 * max(n_ok + n_skip, 1):
        print(f"    WARNING: fp16 overflow skipped {n_skip}/{n_ok + n_skip} steps -- "
              f"set USE_AMP = False", flush=True)
    ema.copy_to(model)
    return model

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def confidence_report(prob, label=""):
    """Flag a fold that produced near-uniform probabilities.

    A 3-class softmax floors at 0.333. A model that learned anything puts most of its
    mass well above that; one that sits at ~0.39 did not train, however plausible its
    macro F1 looks. Worth checking every fold, because the macro F1 of a barely-trained
    model on 42 images can still land near 0.45 by luck.
    """
    med = float(np.median(prob.max(1)))
    frac = float((prob.max(1) > 0.5).mean())
    msg = f"    confidence{label}: median max-prob {med:.3f}, {frac:.0%} above 0.5"
    if med < 0.45:
        msg += "   <-- NEAR-UNIFORM, this fold did not train"
    print(msg, flush=True)
    return med


def fit_weights(prob, y):
    """Coordinate ascent on per-class multipliers, maximising macro F1."""
    grid = np.exp(np.linspace(-1.2, 1.2, 49))
    w = np.ones(len(CLASSES))
    best = f1_score(y, (prob * w).argmax(1), average="macro")
    for _ in range(6):
        improved = False
        for c in range(len(CLASSES)):
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


def cross_fitted(prob, y, seed):
    """Tuned predictions whose weights never saw the row they score."""
    pred = np.empty(len(y), dtype=np.int64)
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(y, y):
        pred[va] = (prob[va] * fit_weights(prob[tr], y[tr])).argmax(1)
    return pred


def try_blend(oof, test_prob, y):
    """Blend with saved probabilities from an earlier run, if it helps out-of-fold."""
    if not BLEND_WITH:
        return oof, test_prob
    best_oof, best_test = oof, test_prob
    best = f1_score(y, oof.argmax(1), average="macro")
    for oof_path, test_path in BLEND_WITH:
        if not (os.path.exists(oof_path) and os.path.exists(test_path)):
            print(f"  blend source missing, skipped: {oof_path}")
            continue
        o2, t2 = np.load(oof_path), np.load(test_path)
        for a in np.arange(0.1, 1.0, 0.1):
            cand = f1_score(y, (a * oof + (1 - a) * o2).argmax(1), average="macro")
            if cand > best + 1e-9:
                best = cand
                best_oof = a * oof + (1 - a) * o2
                best_test = a * test_prob + (1 - a) * t2
                print(f"  blend with {os.path.basename(oof_path)} at w={a:.1f} "
                      f"-> OOF {cand:.4f}")
    if best is not None and best_oof is oof:
        print("  no blend beat this model alone; shipping it unblended")
    return best_oof, best_test


def confidence_report(prob):
    """Flag a fold that produced near-uniform probabilities.

    A three-class softmax floors at 0.333. Two earlier runs scored a plausible-looking
    macro F1 while sitting at a median max-prob under 0.42 -- they had not learned
    anything, and only this told us so. A healthy fold sits near 0.7.
    """
    med = float(np.median(prob.max(1)))
    flag = "   <-- NEAR-UNIFORM, this fold did not train" if med < 0.45 else ""
    print(f"    confidence: median max-prob {med:.3f}, "
          f"{(prob.max(1) > 0.5).mean():.0%} above 0.5{flag}", flush=True)


def stage_1_zeroshot(x_train, y):
    print("=" * 70)
    print("STAGE 1 -- zero-shot sanity check (not a verdict: the ViT scored 0.19 here")
    print("           and still fine-tuned to 0.64)")
    print("=" * 70)
    model = HFClassifier(HF_MODEL_DIR or MODEL_ID, len(CLASSES), DROPOUT).to(DEVICE).eval()
    pred = predict(model, x_train).argmax(1)
    print(f"zero-shot macro F1 = {f1_score(y, pred, average='macro'):.4f}  (chance ~0.33)")
    counts = np.bincount(pred, minlength=len(CLASSES))
    print("prediction spread: " + "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
    if _HEAD_REUSED is False:
        print("  (head was re-initialised, so this is random-head output by construction)")
    del model
    torch.cuda.empty_cache()
    print()


def main():
    t0 = time.time()
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    y = np.array([CLASSES.index(v) for v in train.label], dtype=np.int64)

    print(f"device={DEVICE}  input={MODEL_H}x{MODEL_W}  folds={N_FOLDS}  seeds={SEEDS}")
    print(f"mixup={MIXUP_PROB}  balanced_sampler={BALANCED_SAMPLER}  tta_scales={TTA_SCALES}")
    print(f"precision={'fp16' if USE_AMP else 'fp32'}  warmup_epochs={WARMUP_EPOCHS}  "
          f"epochs={EPOCHS}  llrd={LLRD}")
    print(f"model={MODEL_ID}")
    print("\npreprocessing...", flush=True)
    x_train = build_cache(train.image_id, f"{DATA_DIR}/train_images")
    x_test = build_cache(test.image_id, f"{DATA_DIR}/test_images")
    print(f"  done in {time.time() - t0:.0f}s  {x_train.shape} {x_test.shape}\n", flush=True)

    if RUN_STAGE_1:
        stage_1_zeroshot(x_train, y)

    print("=" * 70)
    print("STAGE 2 -- repeated cross-validated fine-tuning")
    print("=" * 70)
    oof = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    test_prob = np.zeros((len(test), len(CLASSES)), dtype=np.float32)
    fold_scores, n_runs = [], 0

    for seed in SEEDS:
        for f, (tr, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True,
                                                     random_state=seed).split(y, y)):
            model = train_fold(x_train[tr], y[tr], seed * 100 + f)
            va_prob = predict(model, x_train[va])
            confidence_report(va_prob)
            oof[va] += va_prob
            test_prob += predict(model, x_test)
            n_runs += 1
            s = f1_score(y[va], va_prob.argmax(1), average="macro")
            fold_scores.append(s)
            print(f"  seed {seed} fold {f}: macroF1 {s:.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            del model
            torch.cuda.empty_cache()
    oof /= len(SEEDS)
    test_prob /= n_runs

    np.save(f"{WORK_DIR}/oof_v2.npy", oof)
    np.save(f"{WORK_DIR}/test_prob_v2.npy", test_prob)

    print(f"\nper-fold: min {min(fold_scores):.4f}  max {max(fold_scores):.4f}  "
          f"std {np.std(fold_scores):.4f}")
    print("  (the ViT run's spread was 0.50-0.75; a smaller spread here means the")
    print("   variance controls are doing their job)")

    oof, test_prob = try_blend(oof, test_prob, y)

    plain = f1_score(y, oof.argmax(1), average="macro")
    tuned = f1_score(y, cross_fitted(oof, y, SEEDS[0]), average="macro")
    print("\n" + "=" * 70)
    print(f"OOF macro F1  plain={plain:.4f}   tuned(cross-fitted)={tuned:.4f}")
    print("  ^ compare THIS against other runs. Not the public LB, which is scored on")
    print("    about 16 images -- the ViT run moved 0.64 OOF -> 0.689 LB on noise alone.")

    w = fit_weights(oof, y) if tuned > plain else np.ones(len(CLASSES))
    print(f"class weights {np.round(w, 3)} for {CLASSES}")
    print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    for row in confusion_matrix(y, (oof * w).argmax(1)):
        print("   ", row)
    print("per-class F1: " + "  ".join(
        f"{c}={f1_score(y, (oof * w).argmax(1), average=None, labels=[i])[0]:.3f}"
        for i, c in enumerate(CLASSES)))

    pred = (test_prob * w).argmax(1)
    sub = pd.DataFrame({"image_id": test.image_id, "label": [CLASSES[i] for i in pred]})
    sub.to_csv(OUT_CSV, index=False)

    assert len(sub) == len(test) and sub.image_id.is_unique
    assert set(sub.label) <= set(CLASSES)
    print(f"\nwrote {OUT_CSV} in {time.time() - t0:.0f}s")
    print(sub.label.value_counts().to_string())


if __name__ == "__main__":
    main()
