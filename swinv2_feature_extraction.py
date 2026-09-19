# =========================================================
# Feature Extraction dengan Swin-V2 Large fine-tuned breast cancer
#   repo: keanteng/swin-v2-large-ft-breast-cancer-classification-0603
# Untuk klasifikasi 3 kelas: Normal, Benign, Malignant
#
# Kenapa versi lama gagal load:
#   1. config.json di repo itu BUKAN config transformers (tidak ada
#      "architectures"/"hidden_size"/"depths"), cuma catatan buatan tangan.
#      Jadi AutoModel.from_pretrained / AutoConfig pasti error.
#   2. Nama "swinv2_base_window8_256" di timm tidak ada hubungannya dengan
#      bobot ini. Bobotnya = Swin-V2 LARGE (embed_dim 192) turunan
#      microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft,
#      plus head MLP custom 2 kelas (Has_Cancer vs Normal), 195.6M param.
#   3. model.safetensors itu state_dict mentah hasil torch.save/save_file dari
#      modul PyTorch custom, jadi nama layernya berprefix (mis. "backbone."/
#      "model."/"swinv2.") dan tidak akan pernah cocok dengan timm.
#
# Solusi di bawah: unduh state_dict-nya, deteksi sendiri gaya penamaan layer
# (transformers vs timm), cocokkan prefix secara otomatis, lalu rekonstruksi
# head MLP-nya dari bentuk tensor supaya probabilitas "Has_Cancer" hasil
# fine-tune bisa dipakai sebagai fitur tambahan.
#
# Kebutuhan: pip install -q transformers safetensors timm
# Cek struktur checkpoint tanpa menjalankan pipeline:
#     python swinv2_feature_extraction.py --inspect
# =========================================================
import os, re, gc, sys, random, warnings

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None          # mammogram full-field ukurannya besar


# =========================================================
# CONFIG
# =========================================================
class CFG:
    root        = "/kaggle/input/competitions/holomine-breasts-cancer-classification-task-1"
    train_csv   = f"{root}/train.csv"
    test_csv    = f"{root}/test.csv"
    train_dir   = f"{root}/train_images"
    test_dir    = f"{root}/test_images"
    out_dir     = "/kaggle/working"

    hf_repo     = "keanteng/swin-v2-large-ft-breast-cancer-classification-0603"
    hf_files    = ["model.safetensors", "pytorch_model.bin"]   # dicoba berurutan
    # arsitektur dasar sebelum fine-tune (dipakai untuk membangun kerangka model)
    base_model  = "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"
    timm_names  = [                                            # cadangan kalau
        "swinv2_large_window12to16_192to256",                  # bobotnya ternyata
        "swinv2_large_window12to24_192to384",                  # bergaya timm
        "swinv2_large_window12_192",
    ]

    img_size    = 256      # README model: "Input image size: 256x256"
    batch_size  = 8
    num_workers = 2

    use_head    = True     # pakai logit head fine-tune sebagai fitur tambahan
    hflip_tta   = True     # rata-rata fitur citra asli + cerminannya

    seed        = 42
    device      = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# AMBIL STATE_DICT DARI HUGGING FACE
# =========================================================
def download_state_dict():
    from huggingface_hub import hf_hub_download

    last_err = None
    for fname in CFG.hf_files:
        try:
            path = hf_hub_download(repo_id=CFG.hf_repo, filename=fname)
        except Exception as e:
            print(f"[hf] {fname} tidak ada / gagal diunduh ({type(e).__name__})")
            last_err = e
            continue
        print(f"[hf] berhasil: {path}")
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        return torch.load(path, map_location="cpu", weights_only=False)
    raise RuntimeError(f"Tidak ada file bobot yang bisa diunduh dari {CFG.hf_repo}: {last_err}")


def unwrap_state_dict(obj):
    """Checkpoint kadang dibungkus {'state_dict': ...} / {'model': ...}."""
    if isinstance(obj, dict) and not any(torch.is_tensor(v) for v in obj.values()):
        for key in ("state_dict", "model", "model_state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                return unwrap_state_dict(obj[key])
    return {k: v for k, v in obj.items() if torch.is_tensor(v)}


def summarize_keys(sd, n=8):
    print(f"\n[ckpt] {len(sd)} tensor, total param: {sum(v.numel() for v in sd.values())/1e6:.1f}M")
    prefixes = {}
    for k in sd:
        prefixes[k.split(".")[0]] = prefixes.get(k.split(".")[0], 0) + 1
    print("[ckpt] prefix level-1:", dict(sorted(prefixes.items(), key=lambda x: -x[1])[:10]))
    keys = list(sd.keys())
    for k in keys[:n]:
        print(f"    {k:70s} {tuple(sd[k].shape)}")
    print("    ...")
    for k in keys[-n:]:
        print(f"    {k:70s} {tuple(sd[k].shape)}")


# =========================================================
# COCOKKAN BOBOT KE KERANGKA MODEL
# =========================================================
def best_prefix(src_keys, target_keys):
    """
    Cari prefix yang harus dibuang supaya nama layer checkpoint cocok dengan
    model yang kita bangun. Menangani 'backbone.', 'model.', 'swinv2.',
    'module.', 'backbone.swinv2.', dst. tanpa perlu menebak.
    """
    cands = {""}
    for k in src_keys:
        parts = k.split(".")
        for i in range(1, min(4, len(parts))):
            cands.add(".".join(parts[:i]) + ".")
    best, best_hits = "", -1
    for p in sorted(cands):
        hits = sum(1 for k in src_keys
                   if k.startswith(p) and k[len(p):] in target_keys)
        if hits > best_hits:
            best, best_hits = p, hits
    return best, best_hits


class HFBackbone(nn.Module):
    """Swin-V2 versi transformers -> pooler_output."""
    def __init__(self, model):
        super().__init__()
        self.inner = model          # bobot dicocokkan ke sini, bukan ke wrapper
        self.num_features = getattr(model, "num_features", model.config.embed_dim * 8)

    def forward(self, x):
        out = self.inner(x)
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out.last_hidden_state.mean(1)


class TimmBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner = model
        self.num_features = model.num_features

    def forward(self, x):
        return self.inner(x)


def _hf_config():
    """Config Swin-V2 Large. Ambil dari Hub; kalau offline pakai nilai hardcoded."""
    from transformers import Swinv2Config
    try:
        cfg = Swinv2Config.from_pretrained(CFG.base_model)
    except Exception as e:
        print(f"[cfg] gagal ambil config base model ({type(e).__name__}), pakai default Large.")
        cfg = Swinv2Config(
            patch_size=4, num_channels=3, embed_dim=192,
            depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
            window_size=24, pretrained_window_sizes=[12, 12, 12, 6],
            mlp_ratio=4.0, image_size=384,
        )
    cfg.image_size = CFG.img_size      # SwinV2 pakai CPB, ganti resolusi aman
    return cfg


def candidate_backbones():
    """Kandidat kerangka model, urut dari yang paling mungkin."""
    def _hf():
        from transformers import Swinv2Model
        return HFBackbone(Swinv2Model(_hf_config()))

    yield "transformers/Swinv2Model", _hf

    import timm
    for name in CFG.timm_names:
        def _timm(name=name):
            return TimmBackbone(timm.create_model(
                name, pretrained=False, num_classes=0, img_size=CFG.img_size))
        yield f"timm/{name}", _timm


def build_backbone(sd):
    """Bangun backbone dan muat bobotnya; pilih kandidat dengan coverage terbaik."""
    best = None
    for label, factory in candidate_backbones():
        try:
            module = factory()
        except Exception as e:
            print(f"  - {label:45s} gagal dibuat ({type(e).__name__}: {e})")
            continue
        tgt = set(module.inner.state_dict().keys())
        prefix, hits = best_prefix(sd.keys(), tgt)
        cov = hits / max(len(tgt), 1)
        print(f"  - {label:45s} prefix={prefix!r:24s} cocok {hits}/{len(tgt)} ({cov:.1%})")
        if best is None or cov > best[0]:
            best = (cov, label, module, prefix)
        if cov > 0.99:
            break

    if best is None:
        raise RuntimeError("Tidak ada kerangka model yang bisa dibangun.")

    cov, label, module, prefix = best
    tgt = module.inner.state_dict()
    matched = {k[len(prefix):]: v for k, v in sd.items()
               if k.startswith(prefix) and k[len(prefix):] in tgt}

    # buang bobot yang namanya cocok tapi bentuknya beda (jangan diam-diam salah)
    shape_bad = [k for k, v in matched.items() if tuple(v.shape) != tuple(tgt[k].shape)]
    for k in shape_bad:
        matched.pop(k)
    if shape_bad:
        print(f"[warn] {len(shape_bad)} tensor dilewati karena bentuknya beda, contoh: {shape_bad[:3]}")

    missing, unexpected = module.inner.load_state_dict(matched, strict=False)
    loaded = len(tgt) - len(missing)
    print(f"\nBackbone terpilih : {label}")
    print(f"Prefix dibuang    : {prefix!r}")
    print(f"Bobot ter-load    : {loaded}/{len(tgt)} ({loaded/len(tgt):.1%})")
    if missing:
        print("  contoh missing  :", list(missing)[:5])
    if loaded / len(tgt) < 0.95:
        raise RuntimeError(
            "Bobot gagal dicocokkan (<95%). Pastikan 'transformers' terpasang "
            "(pip install -q transformers), lalu jalankan dengan argumen --inspect "
            "untuk melihat struktur checkpoint-nya."
        )

    used = {prefix + k for k in matched}
    leftover = {k: v for k, v in sd.items() if k not in used}
    return module, leftover


# =========================================================
# REKONSTRUKSI HEAD MLP CUSTOM (Has_Cancer vs Normal)
# =========================================================
def _natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def build_head(leftover, in_dim, activation="gelu", n_out=2):
    """
    Head-nya MLP custom, jadi kita susun ulang dari bentuk tensor yang tersisa:
    weight 2D -> Linear, weight 1D -> LayerNorm/BatchNorm1d, aktivasi disisipkan
    di antara dua Linear. Kalau rantai dimensinya tidak nyambung -> None.
    """
    groups = {}
    for k, v in leftover.items():
        parent, _, leaf = k.rpartition(".")
        groups.setdefault(parent, {})[leaf] = v

    act = {"gelu": nn.GELU, "relu": nn.ReLU}[activation]
    layers, new_sd, cur = [], {}, in_dim
    seen_linear = False

    for name in sorted(groups, key=_natkey):
        g = groups[name]
        w = g.get("weight")
        if w is None or w.dim() not in (1, 2):
            continue
        if w.dim() == 2:
            out_f, in_f = w.shape
            if in_f != cur:
                return None, f"dimensi tidak nyambung di '{name}': {in_f} != {cur}"
            if seen_linear:
                layers.append(act())
            idx = len(layers)
            layers.append(nn.Linear(in_f, out_f, bias="bias" in g))
            new_sd[f"{idx}.weight"] = w
            if "bias" in g:
                new_sd[f"{idx}.bias"] = g["bias"]
            cur, seen_linear = out_f, True
        else:
            if w.shape[0] != cur:
                continue
            idx = len(layers)
            if "running_mean" in g:
                layers.append(nn.BatchNorm1d(cur))
                new_sd[f"{idx}.running_mean"] = g["running_mean"]
                new_sd[f"{idx}.running_var"] = g["running_var"]
            else:
                layers.append(nn.LayerNorm(cur))
            new_sd[f"{idx}.weight"] = w
            if "bias" in g:
                new_sd[f"{idx}.bias"] = g["bias"]

    if not seen_linear:
        return None, "tidak ada layer Linear di sisa bobot"
    if cur != n_out:
        return None, f"output head {cur}, bukan {n_out}"

    head = nn.Sequential(*layers)
    missing, unexpected = head.load_state_dict(new_sd, strict=False)
    if missing:
        return None, f"head kurang bobot: {list(missing)[:3]}"
    return head.eval(), f"{len(layers)} layer, aktivasi {activation}"


# =========================================================
# DATA
# =========================================================
tfm = T.Compose([
    T.Resize((CFG.img_size, CFG.img_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class BreastDataset(Dataset):
    def __init__(self, df, img_dir, with_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, image_id):
        p = os.path.join(self.img_dir, str(image_id))
        if os.path.exists(p):
            return p
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            q = os.path.join(self.img_dir, str(image_id) + ext)
            if os.path.exists(q):
                return q
        raise FileNotFoundError(f"Image not found: {image_id}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self._resolve_path(row["image_id"])).convert("RGB")
        x = tfm(img)
        return (x, int(row["target"])) if self.with_label else (x, -1)


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    use_amp = (device == "cuda")
    for imgs, lbls in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            f = model(imgs)
            if CFG.hflip_tta:
                f = (f + model(torch.flip(imgs, dims=[3]))) / 2
        feats.append(f.float().cpu().numpy())
        labels.append(np.asarray(lbls))
    return np.concatenate(feats, 0), np.concatenate(labels, 0)


# =========================================================
# LINEAR PROBE
# =========================================================
def make_clf(name, seed=CFG.seed):
    if name.startswith("LogReg"):
        C = float(name.split("_C")[1])
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, C=C,
                                                class_weight="balanced"))
    return make_pipeline(StandardScaler(),
                         SVC(kernel="linear", C=1.0, class_weight="balanced",
                             probability=True, random_state=seed))


CLF_NAMES = ["LogReg_C0.01", "LogReg_C0.1", "LogReg_C1.0", "SVM_linear"]


# =========================================================
# PIPELINE
# =========================================================
def main(inspect_only=False):
    seed_everything(CFG.seed)
    print("Device:", CFG.device, "| torch:", torch.__version__)

    sd = unwrap_state_dict(download_state_dict())
    summarize_keys(sd)
    if inspect_only:
        return

    print("\n=== Mencocokkan bobot ke kerangka model ===")
    backbone, leftover = build_backbone(sd)
    feat_dim = backbone.num_features
    backbone = backbone.to(CFG.device).eval()
    print("Dimensi fitur backbone:", feat_dim)
    print("Sisa bobot (kandidat head):", list(leftover.keys())[:10])

    heads = {}
    if CFG.use_head:
        for act in ("gelu", "relu"):
            h, info = build_head(leftover, feat_dim, activation=act)
            print(f"[head/{act}] {'OK - ' + info if h is not None else 'gagal - ' + info}")
            if h is not None:
                heads[act] = h
    del sd
    gc.collect()

    # ---------------- DATA ----------------
    train_df = pd.read_csv(CFG.train_csv)
    test_df  = pd.read_csv(CFG.test_csv)
    classes   = sorted(train_df["label"].unique())
    class2idx = {c: i for i, c in enumerate(classes)}
    idx2class = {i: c for c, i in class2idx.items()}
    train_df["target"] = train_df["label"].map(class2idx)
    num_classes = len(classes)
    print("\nTrain:", train_df.shape, "| Test:", test_df.shape, "| Classes:", class2idx)

    train_loader = DataLoader(BreastDataset(train_df, CFG.train_dir, True),
                              batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True)
    test_loader  = DataLoader(BreastDataset(test_df, CFG.test_dir, False),
                              batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True)

    print("\n=== Extracting train features ===")
    X_train, y_train = extract_features(backbone, train_loader, CFG.device)
    print("X_train:", X_train.shape)

    print("\n=== Extracting test features ===")
    X_test, _ = extract_features(backbone, test_loader, CFG.device)
    print("X_test:", X_test.shape)

    del backbone
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- fitur tambahan dari head fine-tune ----------------
    # Head dilatih untuk Has_Cancer vs Normal, persis memisahkan Normal dari
    # (Benign + Malignant) di soal ini. Aktivasi head aslinya tidak tercatat di
    # repo, jadi dipilih lewat AUC di data train; kalau dua-duanya lemah, dibuang.
    if heads:
        y_bin = (np.array([idx2class[t] for t in y_train]) != "Normal").astype(int)
        scored = {}
        for act, h in heads.items():
            with torch.no_grad():
                lg = h(torch.from_numpy(X_train).float()).numpy()
            auc = roc_auc_score(y_bin, lg[:, 0] - lg[:, 1])   # kelas 0 = Has_Cancer
            scored[act] = (auc, h)
            print(f"[head/{act}] AUC Has_Cancer vs Normal di train: {auc:.4f}")
        best_act = max(scored, key=lambda a: abs(scored[a][0] - 0.5))
        auc, h = scored[best_act]
        if abs(auc - 0.5) < 0.05:
            print("[head] terlalu lemah (AUC ~0.5), fitur head tidak dipakai.")
        else:
            print(f"[head] dipakai: aktivasi {best_act} (AUC {auc:.4f})")
            with torch.no_grad():
                lg_tr = h(torch.from_numpy(X_train).float()).numpy()
                lg_te = h(torch.from_numpy(X_test).float()).numpy()
            X_train = np.hstack([X_train, lg_tr])
            X_test  = np.hstack([X_test,  lg_te])
            print("X_train + head:", X_train.shape)

    os.makedirs(CFG.out_dir, exist_ok=True)
    np.save(f"{CFG.out_dir}/X_train.npy", X_train)
    np.save(f"{CFG.out_dir}/y_train.npy", y_train)
    np.save(f"{CFG.out_dir}/X_test.npy", X_test)

    # ---------------- 5-FOLD CV ----------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    results = {}
    print("\n===== 5-FOLD CV (Linear Probe) =====")
    for name in CLF_NAMES:
        s = cross_val_score(make_clf(name), X_train, y_train, cv=cv,
                            scoring="f1_macro", n_jobs=-1)
        results[name] = s.mean()
        print(f"{name:14s} | Macro F1: {s.mean():.4f} +/- {s.std():.4f}")

    best_name = max(results, key=results.get)
    print(f"\n=== Best classifier: {best_name} (F1={results[best_name]:.4f}) ===")

    # ---------------- OOF (pakai classifier terbaik, bukan hardcode) ----------------
    oof = np.zeros((len(y_train), num_classes))
    for tr, va in cv.split(X_train, y_train):
        clf = make_clf(best_name)
        clf.fit(X_train[tr], y_train[tr])
        oof[va] = clf.predict_proba(X_train[va])
    oof_pred = oof.argmax(1)

    print("\n===== OOF SUMMARY =====")
    print("Macro F1     :", f1_score(y_train, oof_pred, average="macro"))
    print("Balanced Acc :", balanced_accuracy_score(y_train, oof_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_train, oof_pred))
    print(classification_report(y_train, oof_pred, target_names=classes))

    # ---------------- FIT FINAL & SUBMISSION ----------------
    final = make_clf(best_name)
    final.fit(X_train, y_train)
    pred = final.predict_proba(X_test).argmax(1)

    sub = pd.DataFrame({"image_id": test_df["image_id"],
                        "label": [idx2class[i] for i in pred]})
    sub.to_csv(f"{CFG.out_dir}/submission.csv", index=False)
    print("\nSaved submission.csv")
    print(sub.head())
    print(sub["label"].value_counts())


if __name__ == "__main__":
    main(inspect_only="--inspect" in sys.argv)
