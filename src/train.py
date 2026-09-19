"""Cross-validated fine-tuning.

Writes three things to --out-dir, which is everything the rest of the pipeline needs:
  oof.npy   (n_train, 3) out-of-fold probabilities
  test.npy  (n_test, 3)  fold-averaged test probabilities
  folds.npy (n_train,)   fold assignment, so OOF scores stay reproducible

Run:  python src/train.py --model resnet34 --folds 5 --epochs 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLASSES, MammoDataset, load_cache, make_folds  # noqa: E402
from model import EMA, build_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, images: np.ndarray, device: str, batch_size: int, tta: bool,
            out_size=None) -> np.ndarray:
    model.eval()
    loader = DataLoader(MammoDataset(images, None, train=False, out_size=out_size),
                        batch_size=batch_size)
    out = []
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb).softmax(1)
        if tta:  # horizontal flip is the only label-preserving view worth averaging
            logits = (logits + model(torch.flip(xb, dims=[3])).softmax(1)) / 2
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def train_fold(x_tr, y_tr, x_va, y_va, args, device, seed):
    set_seed(seed)
    model = build_model(args.model, len(CLASSES), pretrained=not args.no_pretrained,
                        dropout=args.dropout, keep_head=args.keep_head).to(device)

    # Benign is the minority class (52 vs 80/80) and macro F1 weights all three
    # equally, so the loss is inverse-frequency weighted to match the metric.
    counts = np.bincount(y_tr, minlength=len(CLASSES)).astype(np.float32)
    weights = torch.tensor((counts.sum() / (len(CLASSES) * counts)), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(MammoDataset(x_tr, y_tr, train=True, seed=seed,
                                           out_size=args.img_size),
                              batch_size=args.batch_size, shuffle=True, drop_last=len(x_tr) > args.batch_size,
                              num_workers=args.workers, pin_memory=device == "cuda")
    steps = max(1, len(train_loader)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.25)
    ema = EMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            running += loss.item()
        if args.verbose and (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch + 1:>3}/{args.epochs}  loss {running / len(train_loader):.4f}", flush=True)

    # The EMA weights are what we evaluate and ship; the raw last-step weights are
    # noticeably noisier at this sample size.
    ema.copy_to(model)
    va_prob = predict(model, x_va, device, args.batch_size, args.tta, args.img_size)
    return model, va_prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "artifacts", "cache.npz"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts", "run"))
    ap.add_argument("--model", default="resnet34",
                    help='timm/torchvision name, or "hf:<repo_id>" for a Hugging Face classifier')
    ap.add_argument("--img-size", nargs=2, type=int, default=None, metavar=("H", "W"),
                    help="resize the cached crops to this before the model (default: cache size)")
    ap.add_argument("--keep-head", action="store_true",
                    help="keep the checkpoint's own classifier as a warm start; only valid "
                         "when its label order matches CLASSES")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="repeat the whole CV with these seeds and average")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tta", action="store_true", default=True)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.img_size is not None:
        args.img_size = tuple(args.img_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    x_train, y, x_test, train_ids, test_ids = load_cache(args.cache)
    print(f"device={device}  model={args.model}  train={x_train.shape}  test={x_test.shape}")

    oof = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    test_prob = np.zeros((len(x_test), len(CLASSES)), dtype=np.float32)
    n_runs = 0
    t0 = time.time()

    for seed in args.seeds:
        folds = make_folds(y, args.folds, seed)
        for f in range(args.folds):
            tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
            model, va_prob = train_fold(x_train[tr], y[tr], x_train[va], y[va], args, device, seed * 100 + f)
            oof[va] += va_prob
            test_prob += predict(model, x_test, device, args.batch_size, args.tta, args.img_size)
            n_runs += 1
            score = f1_score(y[va], va_prob.argmax(1), average="macro")
            print(f"  seed {seed} fold {f}: macroF1 {score:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            del model

    oof /= len(args.seeds)
    test_prob /= n_runs
    oof_score = f1_score(y, oof.argmax(1), average="macro")
    print(f"\nOOF macro F1 (argmax) = {oof_score:.4f}")

    np.save(os.path.join(args.out_dir, "oof.npy"), oof)
    np.save(os.path.join(args.out_dir, "test.npy"), test_prob)
    np.save(os.path.join(args.out_dir, "folds.npy"), make_folds(y, args.folds, args.seeds[0]))
    with open(os.path.join(args.out_dir, "config.json"), "w") as fh:
        json.dump({**vars(args), "oof_macro_f1": float(oof_score), "device": device}, fh, indent=2)
    print(f"wrote predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
