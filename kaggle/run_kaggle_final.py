"""HoloMine Task 1 -- two complementary backbones, blended, in one notebook run.

Paste into one Kaggle cell. Internet switch ON. Expect 15-25 min on a P100/T4.

WHY THIS SHAPE
--------------
Three runs are on record:

  ViT (ultrasound-pretrained, fine-tuned)   OOF 0.6401   public LB 0.68888
  resnet34 local, fp32                      OOF 0.5842
  resnet34 Kaggle, fp16                     OOF 0.4410   (undertrained, see below)

The two healthy runs fail in opposite directions. resnet34 catches almost every
Normal (recall 0.963) but floods the class with false positives (precision 0.554),
and is the better of the two at telling Benign from Malignant (0.800 vs 0.737). The
ViT is the better one at Normal-vs-abnormal (0.792 vs 0.693) and recovers nearly
twice as many Benign cases (0.577 vs 0.327). Averaging models that make the same
mistakes buys nothing; these do not.

So this script trains both and blends them, choosing the blend weights out-of-fold
and checking them cross-fitted. If the blend does not survive that check it ships the
better single model instead and says so.

It also carries the fix for what broke the earlier resnet34 run: under fp16,
GradScaler skips any step whose gradients overflowed, and the loop used to advance
the LR schedule and the EMA anyway. The per-fold confidence line is there to catch a
repeat immediately -- a three-class softmax floors at 0.333, and that run sat at a
median 0.388 with nothing above 0.8, while a healthy one sits near 0.76.

No external data, no LLM/VLM, no AutoML, no Ultralytics. Publicly available
pretrained classifiers, which the rules permit.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/kaggle/input/competitions/holomine-breasts-cancer-classification-task-1"
WORK_DIR = "/kaggle/working"
OUT_CSV = f"{WORK_DIR}/submission.csv"

CACHE_H, CACHE_W = 512, 384
N_FOLDS = 5
SEEDS = [0]                  # add 1 to halve the fold-to-fold noise, at double the time

# Each entry trains its own cross-validated ensemble. Learning rates differ by an
# order of magnitude between the two families on purpose: 3e-4 wrecked resnet34 under
# fp16, and a CNN learning rate would wreck a ViT outright.
MODELS = [
    dict(name="resnet34", kind="timm", h=512, w=384, lr=1e-4, epochs=25,
         batch=16, accum=1, dropout=0.3, wd=1e-2),
    dict(name="hugging-science/breast-cancer-detector-2", kind="hf", h=384, w=288,
         lr=3e-5, epochs=12, batch=8, accum=1, dropout=0.1, wd=0.05),
]

LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.99
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.4
BALANCED_SAMPLER = True
TTA_SCALES = (1.0, 0.9)      # with flip -> 4 views per image
GRAD_CHECKPOINT = False      # turn on if a larger backbone runs out of memory

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
        kwargs = {} if reuse else dict(num_labels=num_classes, ignore_mismatched_sizes=True)
        self.model = AutoModelForImageClassification.from_pretrained(repo_id, **kwargs)
        if not reuse:
            in_f = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Dropout(dropout),
                                                  nn.Linear(in_f, num_classes))
        # ViT position embeddings are tied to 224x224; resampling them lets the model
        # take the larger crops that small mammographic lesions need.
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


# Mirrors of the same ImageNet checkpoints, for a notebook run with the internet
# switch off (or a host that blocks download.pytorch.org).
MIRROR = {"resnet34": "https://github.com/huggingface/pytorch-image-models/releases/"
                      "download/v0.1-weights/resnet34-43635321.pth"}


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
    try:
        import timm
        return timm.create_model(spec["name"], pretrained=True, num_classes=len(CLASSES),
                                 drop_rate=spec["dropout"])
    except Exception:
        pass
    try:
        return _torchvision_backbone(spec, "DEFAULT")
    except Exception:
        pass

    url = MIRROR.get(spec["name"])
    if url is None:
        raise RuntimeError(f"cannot fetch pretrained weights for {spec['name']}; "
                           f"turn the notebook's internet switch on")
    cached = "/tmp/" + url.rsplit("/", 1)[-1]
    if not os.path.exists(cached):
        torch.hub.download_url_to_file(url, cached, progress=False)
    model = _torchvision_backbone(spec, None)
    state = torch.load(cached, map_location="cpu", weights_only=True)
    own = model.state_dict()
    # Drop the 1000-way ImageNet head; ours is 3-way and freshly initialised.
    state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    missing, _ = model.load_state_dict(state, strict=False)
    body = [k for k in missing if not any(h in k for h in ("fc.", "classifier."))]
    if body:
        raise RuntimeError(f"backbone keys did not load for {spec['name']}: {sorted(body)[:8]}")
    return model


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
    """Convex combinations of image pairs, loss split across both labels.

    On 212 images this is worth more than any backbone change: it manufactures
    examples along the line between real ones and blocks outright memorisation.
    """
    lam = float(rng.beta(alpha, alpha))
    lam = max(lam, 1 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


@torch.no_grad()
def predict(model, images, spec):
    model.eval()
    total = None
    for scale in TTA_SCALES:
        for flip in (False, True):
            loader = DataLoader(
                MammoDataset(images, None, False, (spec["h"], spec["w"]),
                             scale=scale, flip=flip),
                batch_size=spec["batch"] * 2)
            out = []
            for xb in loader:
                xb = xb.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                    out.append(model(xb).softmax(1).float().cpu().numpy())
            p = np.concatenate(out)
            total = p if total is None else total + p
    return total / (len(TTA_SCALES) * 2)


def train_fold(x_tr, y_tr, spec, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(spec).to(DEVICE)

    # macro F1 weights all three classes equally while the data does not
    # (52 Benign vs 80/80), so the loss is inverse-frequency weighted to match it.
    counts = np.bincount(y_tr, minlength=len(CLASSES)).astype(np.float32)
    cls_w = counts.sum() / (len(CLASSES) * counts)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, device=DEVICE),
                               label_smoothing=LABEL_SMOOTHING)

    ds = MammoDataset(x_tr, y_tr, True, (spec["h"], spec["w"]), seed)
    if BALANCED_SAMPLER:
        # Sampling and loss weighting attack the imbalance from opposite ends; with
        # only 52 Benign images, both together keep the minority class in every batch.
        sampler = WeightedRandomSampler(torch.as_tensor(cls_w[y_tr], dtype=torch.double),
                                        num_samples=len(y_tr), replacement=True)
        loader = DataLoader(ds, batch_size=spec["batch"], sampler=sampler, drop_last=True,
                            num_workers=2, pin_memory=True)
    else:
        loader = DataLoader(ds, batch_size=spec["batch"], shuffle=True, drop_last=True,
                            num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["wd"])
    steps = max(1, len(loader) // spec["accum"]) * spec["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=spec["lr"], total_steps=steps,
                                                pct_start=0.25)
    ema = EMA(model, EMA_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    n_ok = n_skip = 0

    for _ in range(spec["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            use_mix = rng.random() < MIXUP_PROB and xb.size(0) > 1
            if use_mix:
                xb, ya, yb2, lam = mixup(xb, yb, MIXUP_ALPHA, rng)
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                logits = model(xb)
                loss = (lam * crit(logits, ya) + (1 - lam) * crit(logits, yb2)) if use_mix \
                    else crit(logits, yb)
                loss = loss / spec["accum"]
            scaler.scale(loss).backward()
            if (step + 1) % spec["accum"] == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                # A dropped scale means GradScaler skipped this step (fp16 overflow).
                # Advancing the schedule or the EMA on a step that never happened is
                # exactly how the earlier resnet34 run ended up near-uniform.
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


def confidence_report(prob):
    """Flag a fold that produced near-uniform probabilities.

    A three-class softmax floors at 0.333. The broken resnet34 run sat at a median
    0.388 with nothing above 0.8; a healthy run sits near 0.76. Its macro F1 still
    looked plausible, which is why this is checked directly.
    """
    med = float(np.median(prob.max(1)))
    flag = "   <-- NEAR-UNIFORM, this fold did not train" if med < 0.45 else ""
    print(f"    confidence: median max-prob {med:.3f}, "
          f"{(prob.max(1) > 0.5).mean():.0%} above 0.5{flag}", flush=True)


def run_model(spec, x_train, y, x_test):
    tag = spec["name"].split("/")[-1]
    print("=" * 72)
    print(f"{tag}  |  {spec['h']}x{spec['w']}  lr={spec['lr']}  epochs={spec['epochs']}")
    print("=" * 72)
    oof = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    test_prob = np.zeros((len(x_test), len(CLASSES)), dtype=np.float32)
    n_runs, scores = 0, []
    t0 = time.time()
    for seed in SEEDS:
        for f, (tr, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True,
                                                     random_state=seed).split(y, y)):
            model = train_fold(x_train[tr], y[tr], spec, seed * 100 + f)
            va_prob = predict(model, x_train[va], spec)
            confidence_report(va_prob)
            oof[va] += va_prob
            test_prob += predict(model, x_test, spec)
            n_runs += 1
            s = f1_score(y[va], va_prob.argmax(1), average="macro")
            scores.append(s)
            print(f"  seed {seed} fold {f}: macroF1 {s:.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            del model
            torch.cuda.empty_cache()
    oof /= len(SEEDS)
    test_prob /= n_runs
    print(f"  OOF macro F1 {f1_score(y, oof.argmax(1), average='macro'):.4f}   "
          f"fold spread {min(scores):.3f}-{max(scores):.3f}\n")
    np.save(f"{WORK_DIR}/oof_{tag}.npy", oof)
    np.save(f"{WORK_DIR}/test_{tag}.npy", test_prob)
    return oof, test_prob, tag

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


def main():
    t0 = time.time()
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    y = np.array([CLASSES.index(v) for v in train.label], dtype=np.int64)

    print(f"device={DEVICE}  folds={N_FOLDS}  seeds={SEEDS}  models={len(MODELS)}")
    print("preprocessing...", flush=True)
    x_train = build_cache(train.image_id, f"{DATA_DIR}/train_images")
    x_test = build_cache(test.image_id, f"{DATA_DIR}/test_images")
    print(f"  done in {time.time() - t0:.0f}s  {x_train.shape} {x_test.shape}\n", flush=True)

    oofs, tests, tags = [], [], []
    for spec in MODELS:
        o, t, tag = run_model(spec, x_train, y, x_test)
        oofs.append(o)
        tests.append(t)
        tags.append(tag)

    print("=" * 72)
    print("BLEND")
    print("=" * 72)
    solo = [f1_score(y, o.argmax(1), average="macro") for o in oofs]
    for tag, s in zip(tags, solo):
        print(f"  {tag:<34} OOF {s:.4f}")
    best_solo_i = int(np.argmax(solo))

    if len(oofs) > 1:
        w, blend_score = best_weights(oofs, y)
        print("\n  best weights on OOF: " + ", ".join(
            f"{tag}={wi:.2f}" for tag, wi in zip(tags, w)) + f"  -> {blend_score:.4f}")

        # Pick the weights on four folds, score the fifth. Searching a weight grid
        # against the same 212 rows you then report is how a blend looks better than
        # it is.
        pred_cf = np.empty(len(y), dtype=np.int64)
        for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(y, y):
            w_tr, _ = best_weights([o[tr] for o in oofs], y[tr])
            pred_cf[va] = sum(wi * o[va] for wi, o in zip(w_tr, oofs)).argmax(1)
        honest = f1_score(y, pred_cf, average="macro")
        print(f"  cross-fitted blend OOF {honest:.4f}  vs best single {solo[best_solo_i]:.4f}")

        if honest > solo[best_solo_i]:
            print("  -> blend holds up out-of-fold; shipping it")
            oof = sum(wi * o for wi, o in zip(w, oofs))
            test_prob = sum(wi * t for wi, t in zip(w, tests))
        else:
            print(f"  -> blend does NOT hold up; shipping {tags[best_solo_i]} alone")
            oof, test_prob = oofs[best_solo_i], tests[best_solo_i]
    else:
        oof, test_prob = oofs[0], tests[0]

    plain = f1_score(y, oof.argmax(1), average="macro")
    pred_cf = np.empty(len(y), dtype=np.int64)
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(y, y):
        pred_cf[va] = (oof[va] * fit_class_weights(oof[tr], y[tr])).argmax(1)
    tuned = f1_score(y, pred_cf, average="macro")
    cw = fit_class_weights(oof, y) if tuned > plain else np.ones(len(CLASSES))

    print("\n" + "=" * 72)
    print(f"FINAL OOF  plain={plain:.4f}   class-prior tuned(cross-fitted)={tuned:.4f}")
    print("  ^ compare THIS across runs. The public LB is ~16 images: the gap between")
    print("    rank 5 and rank 16 there is one or two images, mostly luck.")
    print(f"class weights {np.round(cw, 3)} for {CLASSES}")
    final = (oof * cw).argmax(1)
    print("per-class F1: " + "  ".join(
        f"{c}={f1_score(y, final, average=None, labels=[i])[0]:.3f}"
        for i, c in enumerate(CLASSES)))
    print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
    for row in confusion_matrix(y, final):
        print("   ", row)

    sub = pd.DataFrame({"image_id": test.image_id,
                        "label": [CLASSES[i] for i in (test_prob * cw).argmax(1)]})
    sub.to_csv(OUT_CSV, index=False)
    assert len(sub) == len(test) and sub.image_id.is_unique
    assert set(sub.label) <= set(CLASSES)
    print(f"\nwrote {OUT_CSV} in {time.time() - t0:.0f}s")
    print(sub.label.value_counts().to_string())


if __name__ == "__main__":
    main()
