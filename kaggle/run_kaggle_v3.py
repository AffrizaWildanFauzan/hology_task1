"""HoloMine Task 1 -- deadline run: mammography-pretrained ViT + diverse blend.

One Kaggle cell, internet ON, T4. Budgeted to finish in ~30-35 min.

WHAT CHANGED FROM THE LAST RUN, AND WHY
---------------------------------------
1. A mammography-pretrained backbone is now the lead model.
   The previous lead, hugging-science/breast-cancer-detector-2, is a ViT fine-tuned
   on breast ULTRASOUND; its own model card lists mammography under Out-of-Scope.
   BTX24/vit-base-patch16-224-in21k-finetuned-hongrui_mammogram_v_1 is the same
   architecture (ViT-base/16, 85.8M) fine-tuned on mammograms instead. Its head is
   4-way BIRADS, so it is replaced with a fresh 3-way head; the body is what we want.

2. resnet34 now loads TORCHVISION weights, not timm's.
   Four runs with identical hyperparameters line up by checkpoint source, not by
   precision: torchvision -> OOF 0.5842 (healthy, median max-prob 0.757); timm
   (a1_in1k) -> 0.4410 / 0.4012 / 0.3762, every fold near-uniform. src/model.py fell
   back to a GitHub mirror of the torchvision checkpoint because the local proxy
   blocked the hub; the Kaggle script had internet and silently got a different
   model under the same name. kind="tv" pins the healthy one.

3. Multi-seed. Fold spread last run was 0.502-0.696. A single seed's OOF is noisier
   than every difference being argued about. Seeds are per-model so the budget can
   be spent where it pays.

4. Blend guard. Last run the weight search handed a BROKEN model (OOF 0.40) a weight
   of 0.70, because near-uniform probabilities have tiny dynamic range and act as a
   lucky perturbation rather than a vote. Models now must clear both an OOF floor
   and a confidence floor before they may enter the blend.

5. Scale TTA (1.0, 0.9) on top of flip -> 4 views. num_workers=0, because both
   workers held identical RNG state that reset every epoch, so augmentation
   diversity was a fraction of what was intended.

6. Time budget + incremental submission. A submission file is rewritten after every
   model finishes, so a dead session or an overrun still leaves a valid CSV.

READING THE OUTPUT
------------------
Compare FINAL OOF against 0.6372 (blend run) and 0.6549 (ViT-alone run). Do NOT
compare public LB scores: on ~16 images, two identical models differ by >=0.025
about 89% of the time and by >=0.10 about 58% of the time.

No external data, no LLM/VLM, no AutoML, no Ultralytics. Publicly available
pretrained classifiers only, which the rules permit.
"""
from __future__ import annotations

import itertools
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

T0 = time.time()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/competitions/holomine-breasts-cancer-classification-task-1"
WORK_DIR = "/kaggle/working"
OUT_CSV = f"{WORK_DIR}/submission.csv"

CACHE_H, CACHE_W = 512, 384
N_FOLDS = 5

# Wall-clock ceiling for TRAINING. A model whose estimate does not fit in what is
# left is skipped, not started; the blend then ships what did finish.
TIME_BUDGET_S = 2400          # 40 min

VIT_MAMMO = "BTX24/vit-base-patch16-224-in21k-finetuned-hongrui_mammogram_v_1"
VIT_US = "hugging-science/breast-cancer-detector-2"

# Ordered cheapest-risk first: the known-good model runs before the new one, so a
# valid submission exists early. est_s is a T4 estimate used by the time guard.
MODELS = [
    dict(name=VIT_US, kind="hf", h=384, w=288, lr=3e-5, epochs=12, batch=8,
         dropout=0.1, wd=0.05, seeds=[0, 1], amp=True, est_s=700),
    dict(name=VIT_MAMMO, kind="hf", h=384, w=288, lr=3e-5, epochs=12, batch=8,
         dropout=0.1, wd=0.05, seeds=[0, 1], amp=True, est_s=700),
    dict(name="resnet34", kind="tv", h=512, w=384, lr=3e-4, epochs=25, batch=16,
         dropout=0.3, wd=1e-2, seeds=[0], amp=True, est_s=300),
]

LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.99
TTA_SCALES = (1.0, 0.9)       # with flip -> 4 views per image
BLEND_OOF_FLOOR = 0.08        # a model more than this below the best is not blended
BLEND_CONF_FLOOR = 0.55       # nor one whose median max-prob says it never trained

CLASSES = ["Benign", "Malignant", "Normal"]
MEAN, STD = 0.449, 0.226
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    # The breast is the largest bright component; the burned-in "R MLO" labels sit in
    # a disconnected corner and are dropped with it.
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
        self.size = size                       # (h, w) this model wants
        self.scale, self.flip = scale, flip    # fixed transforms, for TTA
        # num_workers=0, so one RNG in one process: the stream actually advances
        # across epochs instead of being duplicated per worker and reset each epoch.
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

        img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray((img - MEAN) / STD))[None].repeat(3, 1, 1)
        return x if self.labels is None else (x, int(self.labels[i]))

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class HFClassifier(nn.Module):
    """Wraps a transformers classifier so it returns a plain logits tensor."""

    def __init__(self, repo_id, num_classes=3, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModelForImageClassification

        cfg = AutoConfig.from_pretrained(repo_id)
        ckpt = [cfg.id2label[i].lower() for i in range(cfg.num_labels)]
        reuse = ckpt == [c.lower() for c in CLASSES]
        print(f"    checkpoint labels {ckpt} -> "
              f"{'reusing head' if reuse else 'fresh 3-way head'}", flush=True)
        kwargs = {} if reuse else dict(num_labels=num_classes, ignore_mismatched_sizes=True)
        self.model = AutoModelForImageClassification.from_pretrained(repo_id, **kwargs)
        if not reuse:
            in_f = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Dropout(dropout),
                                                  nn.Linear(in_f, num_classes))
        # ViT position embeddings are tied to 224x224; resampling them lets the model
        # take the larger crops that small mammographic lesions need.
        self.interp = "vit" in self.model.config.model_type

    def forward(self, x):
        if self.interp:
            return self.model(pixel_values=x, interpolate_pos_encoding=True).logits
        return self.model(pixel_values=x).logits


def _torchvision_backbone(spec, weights):
    import torchvision.models as tvm
    m = getattr(tvm, spec["name"])(weights=weights)
    if hasattr(m, "fc"):
        m.fc = nn.Sequential(nn.Dropout(spec["dropout"]),
                             nn.Linear(m.fc.in_features, len(CLASSES)))
    else:
        m.classifier = nn.Sequential(nn.Dropout(spec["dropout"]),
                                     nn.Linear(m.classifier[-1].in_features, len(CLASSES)))
    return m


def build_model(spec):
    if spec["kind"] == "hf":
        return HFClassifier(spec["name"], len(CLASSES), spec["dropout"])
    if spec["kind"] == "tv":
        # Pinned deliberately. timm's default resnet34 tag is a different checkpoint
        # (a1_in1k) and every Kaggle run that used it collapsed to near-uniform.
        return _torchvision_backbone(spec, "DEFAULT")
    import timm
    return timm.create_model(spec["name"], pretrained=True, num_classes=len(CLASSES),
                             drop_rate=spec["dropout"])


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


@torch.no_grad()
def predict(model, images, spec):
    amp = spec.get("amp", True) and DEVICE == "cuda"
    model.eval()
    total = None
    for scale in TTA_SCALES:
        for flip in (False, True):
            loader = DataLoader(
                MammoDataset(images, None, False, (spec["h"], spec["w"]),
                             scale=scale, flip=flip),
                batch_size=spec["batch"] * 2, num_workers=0)
            out = []
            for xb in loader:
                xb = xb.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=amp):
                    out.append(model(xb).softmax(1).float().cpu().numpy())
            p = np.concatenate(out)
            total = p if total is None else total + p
    return total / (len(TTA_SCALES) * 2)


def train_fold(x_tr, y_tr, spec, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    amp = spec.get("amp", True) and DEVICE == "cuda"
    model = build_model(spec).to(DEVICE)

    # macro F1 weights all three classes equally while the data does not
    # (52 Benign vs 80/80), so the loss is inverse-frequency weighted to match it.
    counts = np.bincount(y_tr, minlength=len(CLASSES)).astype(np.float32)
    cls_w = counts.sum() / (len(CLASSES) * counts)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, device=DEVICE),
                               label_smoothing=LABEL_SMOOTHING)

    ds = MammoDataset(x_tr, y_tr, True, (spec["h"], spec["w"]), seed)
    loader = DataLoader(ds, batch_size=spec["batch"], shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=DEVICE == "cuda")

    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["wd"])
    steps = max(1, len(loader)) * spec["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=spec["lr"], total_steps=steps,
                                                pct_start=0.25)
    ema = EMA(model, EMA_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    n_ok = n_skip = 0

    for _ in range(spec["epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # A dropped scale means GradScaler skipped this step (fp16 overflow);
            # advancing the schedule or the EMA on a step that never happened is
            # not free, even though it was not what broke the earlier runs.
            if scaler.get_scale() >= scale_before:
                if sched.last_epoch < steps - 1:
                    sched.step()
                ema.update(model)
                n_ok += 1
            else:
                n_skip += 1

    if n_skip > 0.1 * max(n_ok + n_skip, 1):
        print(f"    WARNING: fp16 overflow skipped {n_skip}/{n_ok + n_skip} steps -- "
              f"lower this model's lr", flush=True)
    ema.copy_to(model)
    return model


def run_model(spec, x_train, y, x_test):
    tag = f"{spec['name'].split('/')[-1]}_{spec['h']}x{spec['w']}"
    print("=" * 76)
    print(f"{tag}  |  lr={spec['lr']}  epochs={spec['epochs']}  seeds={spec['seeds']}")
    print("=" * 76, flush=True)
    oof = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    test_prob = np.zeros((len(x_test), len(CLASSES)), dtype=np.float32)
    n_runs, scores, confs = 0, [], []
    t0 = time.time()
    for seed in spec["seeds"]:
        for f, (tr, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True,
                                                     random_state=seed).split(y, y)):
            model = train_fold(x_train[tr], y[tr], spec, seed * 100 + f)
            va_prob = predict(model, x_train[va], spec)
            med = float(np.median(va_prob.max(1)))
            confs.append(med)
            oof[va] += va_prob
            test_prob += predict(model, x_test, spec)
            n_runs += 1
            s = f1_score(y[va], va_prob.argmax(1), average="macro")
            scores.append(s)
            flag = "  <-- NEAR-UNIFORM, did not train" if med < 0.45 else ""
            print(f"  seed {seed} fold {f}: macroF1 {s:.4f}  conf {med:.3f}"
                  f"  ({time.time() - t0:.0f}s){flag}", flush=True)
            del model
            torch.cuda.empty_cache()
    oof /= len(spec["seeds"])
    test_prob /= n_runs
    solo = f1_score(y, oof.argmax(1), average="macro")
    conf = float(np.median(confs))
    print(f"  OOF macro F1 {solo:.4f}   fold spread {min(scores):.3f}-{max(scores):.3f}"
          f"   median conf {conf:.3f}\n", flush=True)
    np.save(f"{WORK_DIR}/oof_{tag}.npy", oof)
    np.save(f"{WORK_DIR}/test_{tag}.npy", test_prob)
    return oof, test_prob, tag, solo, conf

# ---------------------------------------------------------------------------
# Blending and class-prior tuning
# ---------------------------------------------------------------------------


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
    return best_w, best


def fit_class_weights(prob, y):
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


def finalize(oofs, tests, tags, solos, confs, y, test, final=False):
    """Blend what survives the guards, tune the class prior, write the CSV.

    Called after every model, so an overrun or a dead session still leaves a file.
    """
    best = max(solos)
    keep = [i for i in range(len(oofs))
            if solos[i] >= best - BLEND_OOF_FLOOR and confs[i] >= BLEND_CONF_FLOOR]
    if not keep:
        keep = [int(np.argmax(solos))]
    dropped = [tags[i] for i in range(len(tags)) if i not in keep]
    if dropped and final:
        print(f"  excluded from blend (OOF or confidence floor): {dropped}")

    o_k = [oofs[i] for i in keep]
    t_k = [tests[i] for i in keep]
    g_k = [tags[i] for i in keep]
    s_k = [solos[i] for i in keep]
    bi = int(np.argmax(s_k))

    if len(o_k) > 1:
        w, raw = best_weights(o_k, y)
        # Pick the weights on four folds, score the fifth. Searching a weight grid
        # against the same 212 rows you then report is how a blend looks better than
        # it is.
        pred_cf = np.empty(len(y), dtype=np.int64)
        for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(y, y):
            w_tr, _ = best_weights([o[tr] for o in o_k], y[tr])
            pred_cf[va] = sum(wi * o[va] for wi, o in zip(w_tr, o_k)).argmax(1)
        honest = f1_score(y, pred_cf, average="macro")
        if final:
            print("  weights on full OOF: " + ", ".join(
                f"{g}={wi:.2f}" for g, wi in zip(g_k, w)) + f"  -> {raw:.4f}")
            print(f"  cross-fitted blend {honest:.4f}  vs best single "
                  f"{s_k[bi]:.4f} ({g_k[bi]})")
        if honest > s_k[bi]:
            oof = sum(wi * o for wi, o in zip(w, o_k))
            test_prob = sum(wi * t for wi, t in zip(w, t_k))
            chosen = "blend " + "+".join(f"{g}:{wi:.2f}" for g, wi in zip(g_k, w) if wi > 0)
        else:
            oof, test_prob, chosen = o_k[bi], t_k[bi], g_k[bi] + " (solo)"
    else:
        oof, test_prob, chosen = o_k[0], t_k[0], g_k[0] + " (solo)"

    plain = f1_score(y, oof.argmax(1), average="macro")
    pred_cf = np.empty(len(y), dtype=np.int64)
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(y, y):
        pred_cf[va] = (oof[va] * fit_class_weights(oof[tr], y[tr])).argmax(1)
    tuned = f1_score(y, pred_cf, average="macro")
    cw = fit_class_weights(oof, y) if tuned > plain else np.ones(len(CLASSES))

    sub = pd.DataFrame({"image_id": test.image_id,
                        "label": [CLASSES[i] for i in (test_prob * cw).argmax(1)]})
    assert len(sub) == len(test) and sub.image_id.is_unique
    assert set(sub.label) <= set(CLASSES)
    sub.to_csv(OUT_CSV, index=False)

    score = max(plain, tuned)
    if final:
        print("\n" + "=" * 76)
        print(f"SHIPPING: {chosen}")
        print(f"FINAL OOF   plain={plain:.4f}   class-prior tuned (cross-fitted)={tuned:.4f}")
        print(f"  compare against: 0.6372 (blend run)   0.6549 (ViT-alone run)")
        print(f"  do NOT compare public LB: on ~16 images two identical models")
        print(f"  differ by >=0.025 about 89% of the time.")
        print(f"class weights {np.round(cw, 3)} for {CLASSES}")
        final_pred = (oof * cw).argmax(1)
        print("per-class F1: " + "  ".join(
            f"{c}={f1_score(y, final_pred, average=None, labels=[i])[0]:.3f}"
            for i, c in enumerate(CLASSES)))
        print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
        for row in confusion_matrix(y, final_pred):
            print("   ", row)
        print(f"\nwrote {OUT_CSV}")
        print(sub.label.value_counts().to_string())
    else:
        print(f"  [interim submission written: {chosen}, OOF {score:.4f}]\n", flush=True)
    return score


def main():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    # test.csv ships with CRLF line endings while the other two use LF; pandas
    # handles it, but strip anyway so nothing downstream sees a trailing \r.
    test["image_id"] = test.image_id.astype(str).str.strip()
    train["image_id"] = train.image_id.astype(str).str.strip()
    y = np.array([CLASSES.index(v) for v in train.label], dtype=np.int64)

    print(f"device={DEVICE}  folds={N_FOLDS}  models={len(MODELS)}  "
          f"budget={TIME_BUDGET_S}s")
    print("preprocessing...", flush=True)
    x_train = build_cache(train.image_id, f"{DATA_DIR}/train_images")
    x_test = build_cache(test.image_id, f"{DATA_DIR}/test_images")
    print(f"  done in {time.time() - T0:.0f}s  {x_train.shape} {x_test.shape}\n", flush=True)

    oofs, tests, tags, solos, confs = [], [], [], [], []
    for spec in MODELS:
        left = TIME_BUDGET_S - (time.time() - T0)
        if left < spec["est_s"]:
            print(f"SKIP {spec['name']}: {left:.0f}s left, needs ~{spec['est_s']}s\n",
                  flush=True)
            continue
        try:
            o, t, tag, solo, conf = run_model(spec, x_train, y, x_test)
        except Exception as exc:
            print(f"  FAILED ({type(exc).__name__}: {exc}); continuing without it\n",
                  flush=True)
            continue
        oofs.append(o), tests.append(t), tags.append(tag)
        solos.append(solo), confs.append(conf)
        finalize(oofs, tests, tags, solos, confs, y, test, final=False)

    if not oofs:
        raise RuntimeError("no model finished; nothing to submit")

    print("=" * 76)
    print("BLEND")
    print("=" * 76)
    for tag, s, c in zip(tags, solos, confs):
        print(f"  {tag:<46} OOF {s:.4f}  conf {c:.3f}")
    finalize(oofs, tests, tags, solos, confs, y, test, final=True)
    print(f"total {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
