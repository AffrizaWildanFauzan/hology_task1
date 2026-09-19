# =========================================================
# Feature extraction + linear probe untuk klasifikasi mammogram 3 kelas
# (Normal / Benign / Malignant).
#
# Backbone yang didukung (pilih di CFG.backbones):
#   "mammo_clip" -> shawn24/Mammo-CLIP  (default)
#        Pre-trained-checkpoints/b2-model-best-epoch-10.tar (EfficientNet-B2)
#        Pre-trained-checkpoints/b5-model-best-epoch-7.tar  (EfficientNet-B5)
#        CLIP mammogram-specific (paper arXiv:2405.12255). Bobotnya hasil
#        torch.save dari repo batmanlab/Mammo-CLIP: {"model": state_dict,
#        "config": ...} dengan prefix "image_encoder.", BUKAN format
#        from_pretrained -> AutoModel/timm.from_pretrained pasti gagal.
#   "swinv2"     -> keanteng/swin-v2-large-ft-breast-cancer-classification-0603
#        Swin-V2 Large + head MLP 2 kelas (Has_Cancer vs Normal).
#        config.json-nya bukan config transformers, dan bobotnya state_dict
#        mentah berprefix -> AutoModel juga pasti gagal.
#
# Isi CFG.backbones dengan dua-duanya untuk menggabungkan fiturnya:
#     backbones = ["mammo_clip", "swinv2"]
#
# Kebutuhan : pip install -q timm transformers safetensors
# Inspeksi  : python feature_extraction.py --inspect
# =========================================================
import os, re, gc, sys, random, pickle, warnings, subprocess

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

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

    backbones   = ["mammo_clip"]        # atau ["mammo_clip", "swinv2"]

    # --- Mammo-CLIP ---
    mc_repo     = "shawn24/Mammo-CLIP"
    mc_variant  = "b2"                  # "b2" (1.4 GB) atau "b5" (1.65 GB, lebih kuat)
    mc_files    = {"b2": "Pre-trained-checkpoints/b2-model-best-epoch-10.tar",
                   "b5": "Pre-trained-checkpoints/b5-model-best-epoch-7.tar"}
    mc_arch     = {"b2": "tf_efficientnet_b2_ns", "b5": "tf_efficientnet_b5_ns"}
    mc_size     = (912, 544)            # (H, W); paper pakai (1520, 912)
    mc_crop     = True                  # buang latar hitam di sekitar payudara

    # --- Swin-V2 fine-tuned ---
    sw_repo     = "keanteng/swin-v2-large-ft-breast-cancer-classification-0603"
    sw_files    = ["model.safetensors", "pytorch_model.bin"]
    sw_base     = "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"
    sw_timm     = ["swinv2_large_window12to16_192to256",
                   "swinv2_large_window12to24_192to384",
                   "swinv2_large_window12_192"]
    sw_size     = 256                   # README model: "Input image size: 256x256"
    sw_use_head = True                  # logit head fine-tune jadi fitur tambahan

    batch_size  = 8
    num_workers = 2
    hflip_tta   = True                  # rata-rata fitur citra asli + cerminannya

    seed        = 42
    device      = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# UTIL CHECKPOINT
# =========================================================
class _Stub:
    """Placeholder untuk objek config (omegaconf/argparse) yang tak bisa di-unpickle."""
    def __init__(self, *a, **k): pass
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


class _tolerant_pickle:                 # shim "pickle_module" untuk torch.load
    Unpickler = _TolerantUnpickler
    @staticmethod
    def load(f, **kw):
        return _TolerantUnpickler(f, **kw).load()


def robust_torch_load(path):
    """
    Checkpoint Mammo-CLIP menyimpan config hydra/omegaconf berdampingan dengan
    bobot, jadi weights_only=True (default torch>=2.6) selalu gagal. Urutan:
      1. torch.load biasa (weights_only=False)
      2. pasang modul yang hilang (mis. omegaconf), ulangi
      3. unpickler toleran: objek asing jadi stub, tensornya tetap utuh
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        missing = (e.name or "").split(".")[0]
        print(f"[load] modul '{missing}' belum ada, mencoba pip install ...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", missing], check=True)
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e2:
            print(f"[load] pip install gagal ({e2}), pakai unpickler toleran.")
    except Exception as e:
        print(f"[load] torch.load standar gagal ({type(e).__name__}), pakai unpickler toleran.")
    return torch.load(path, map_location="cpu", weights_only=False,
                      pickle_module=_tolerant_pickle)


def find_state_dict(obj, _depth=0):
    """Cari dict-of-tensors terbesar di dalam checkpoint (model / state_dict / ...)."""
    if isinstance(obj, dict):
        if sum(1 for v in obj.values() if torch.is_tensor(v)) > 10:
            return obj
        if _depth < 2:
            best, best_n = None, 0
            for key in ("model", "state_dict", "model_state_dict", "net", "ema", "module"):
                if key in obj:
                    cand = find_state_dict(obj[key], _depth + 1)
                    if cand is not None and len(cand) > best_n:
                        best, best_n = cand, len(cand)
            if best is not None:
                return best
            for v in obj.values():
                cand = find_state_dict(v, _depth + 1)
                if cand is not None and len(cand) > best_n:
                    best, best_n = cand, len(cand)
            return best
    return None


def summarize_keys(sd, n=6):
    total = sum(v.numel() for v in sd.values()) / 1e6
    print(f"[ckpt] {len(sd)} tensor | {total:.1f}M param")
    pref = {}
    for k in sd:
        pref[k.split(".")[0]] = pref.get(k.split(".")[0], 0) + 1
    print("[ckpt] prefix level-1:", dict(sorted(pref.items(), key=lambda x: -x[1])[:10]))
    keys = list(sd.keys())
    for k in keys[:n] + ["..."] + keys[-n:]:
        print(f"    {k:70s} {tuple(sd[k].shape) if k in sd else ''}")


def best_prefix(src_keys, target_keys):
    """
    Cari prefix yang harus dibuang supaya nama layer checkpoint cocok dengan
    kerangka model yang kita bangun: 'image_encoder.', 'image_encoder.model.',
    'backbone.', 'swinv2.', 'module.', dst. tanpa perlu menebak.
    """
    cands = {""}
    for k in src_keys:
        parts = k.split(".")
        for i in range(1, min(4, len(parts))):
            cands.add(".".join(parts[:i]) + ".")
    best, best_hits = "", -1
    for p in sorted(cands):
        hits = sum(1 for k in src_keys if k.startswith(p) and k[len(p):] in target_keys)
        if hits > best_hits:
            best, best_hits = p, hits
    return best, best_hits


# =========================================================
# WRAPPER BACKBONE
# Bobot dicocokkan/dimuat ke .inner, bukan ke wrapper, supaya nama layer
# tidak kegeser prefix wrapper-nya.
# =========================================================
class TimmBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner = model
        self.num_features = model.num_features

    def forward(self, x):
        return self.inner(x)


class HFBackbone(nn.Module):
    """Swin-V2 versi transformers -> pooler_output."""
    def __init__(self, model):
        super().__init__()
        self.inner = model
        self.num_features = getattr(model, "num_features", model.config.embed_dim * 8)

    def forward(self, x):
        out = self.inner(x)
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out.last_hidden_state.mean(1)


def match_and_load(candidates, sd, min_cov=0.95):
    """
    candidates: iterable of (label, factory). Bangun tiap kandidat, hitung
    berapa persen bobotnya bisa dicocokkan, pilih yang terbaik, lalu muat.
    Mengembalikan (module, leftover_keys).
    """
    best = None
    for label, factory in candidates:
        try:
            module = factory()
        except Exception as e:
            print(f"  - {label:45s} gagal dibuat ({type(e).__name__}: {e})")
            continue
        tgt = set(module.inner.state_dict().keys())
        prefix, hits = best_prefix(sd.keys(), tgt)
        cov = hits / max(len(tgt), 1)
        print(f"  - {label:45s} prefix={prefix!r:22s} cocok {hits}/{len(tgt)} ({cov:.1%})")
        if best is None or cov > best[0]:
            best = (cov, label, module, prefix)
        if cov > 0.99:
            break

    if best is None:
        raise RuntimeError("Tidak ada kerangka model yang bisa dibangun. "
                           "Cek dependensi (timm / transformers).")

    cov, label, module, prefix = best
    tgt = module.inner.state_dict()
    matched = {k[len(prefix):]: v for k, v in sd.items()
               if k.startswith(prefix) and k[len(prefix):] in tgt}

    bad = [k for k, v in matched.items() if tuple(v.shape) != tuple(tgt[k].shape)]
    for k in bad:
        matched.pop(k)
    if bad:
        print(f"[warn] {len(bad)} tensor dilewati karena bentuknya beda, contoh: {bad[:3]}")

    missing, _ = module.inner.load_state_dict(matched, strict=False)
    loaded = len(tgt) - len(missing)
    print(f"Backbone terpilih : {label}")
    print(f"Prefix dibuang    : {prefix!r}")
    print(f"Bobot ter-load    : {loaded}/{len(tgt)} ({loaded/len(tgt):.1%})")
    if missing:
        print("  contoh missing  :", list(missing)[:5])
    if loaded / len(tgt) < min_cov:
        raise RuntimeError(
            f"Bobot gagal dicocokkan (<{min_cov:.0%}). Jalankan dengan --inspect "
            "untuk melihat struktur checkpoint-nya."
        )

    used = {prefix + k for k in matched}
    leftover = {k: v for k, v in sd.items() if k not in used}
    return module, leftover


# =========================================================
# PREPROCESSING
# =========================================================
# Statistik mammogram yang dipakai Mammo-CLIP (setelah min-max per citra)
MAMMO_MEAN, MAMMO_STD = 0.3089279, 0.25053555408335154


def crop_breast_region(arr, thr_ratio=0.08, pad=8):
    """Potong latar hitam. Kalau hasilnya tidak masuk akal, kembalikan citra utuh."""
    try:
        thr = arr.min() + (arr.max() - arr.min()) * thr_ratio
        mask = arr > thr
        rows = np.where(mask.sum(axis=1) > mask.shape[1] * 0.01)[0]
        cols = np.where(mask.sum(axis=0) > mask.shape[0] * 0.01)[0]
        if rows.size == 0 or cols.size == 0:
            return arr
        r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, arr.shape[0] - 1)
        c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, arr.shape[1] - 1)
        out = arr[r0:r1 + 1, c0:c1 + 1]
        if out.size < arr.size * 0.05 or min(out.shape) < 32:
            return arr
        return out
    except Exception:
        return arr


def make_mammoclip_transform(in_chans):
    """Grayscale -> min-max per citra -> normalisasi statistik mammogram."""
    h, w = CFG.mc_size

    def _tf(img):
        arr = np.asarray(img.convert("L"))
        if CFG.mc_crop:
            arr = crop_breast_region(arr)
        arr = np.asarray(Image.fromarray(arr).resize((w, h), Image.BILINEAR),
                         dtype=np.float32)
        arr -= arr.min()
        m = arr.max()
        arr /= m if m > 1e-6 else 1.0
        arr = (arr - MAMMO_MEAN) / MAMMO_STD
        x = torch.from_numpy(arr).unsqueeze(0)
        return x.repeat(3, 1, 1) if in_chans == 3 else x

    return _tf


def make_swinv2_transform():
    """RGB + normalisasi ImageNet, sesuai base model Swin-V2-nya."""
    tf = T.Compose([
        T.Resize((CFG.sw_size, CFG.sw_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return lambda img: tf(img.convert("RGB"))


# =========================================================
# BACKBONE BUILDERS
# =========================================================
def _cfg_value(cfg, *path):
    cur = cfg
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif hasattr(cur, p):
            cur = getattr(cur, p)
        else:
            return None
    return cur if isinstance(cur, (str, int, float)) else None


def load_mammoclip(inspect_only=False):
    from huggingface_hub import hf_hub_download

    fname = CFG.mc_files[CFG.mc_variant]
    print(f"\n=== Mammo-CLIP: mengunduh {CFG.mc_repo}/{fname} ===")
    ckpt = robust_torch_load(hf_hub_download(repo_id=CFG.mc_repo, filename=fname))
    if isinstance(ckpt, dict):
        print("[ckpt] key level atas:", list(ckpt.keys())[:10])

    full = find_state_dict(ckpt)
    if full is None:
        raise RuntimeError("Tidak menemukan state_dict di dalam checkpoint Mammo-CLIP.")
    full = {k: v for k, v in full.items() if torch.is_tensor(v)}
    summarize_keys(full)
    if inspect_only:
        return None, None

    # ambil image encoder saja; sisanya text encoder BERT + proyeksi CLIP
    img_sd = {k: v for k, v in full.items() if k.startswith("image_encoder.")}
    if not img_sd:
        img_sd = {k: v for k, v in full.items()
                  if not any(t in k for t in ("text_encoder", "text_projection",
                                              "logit_scale", "bert", "tokenizer"))}
    print(f"[ckpt] tensor image encoder: {len(img_sd)}")

    # jumlah channel input dibaca langsung dari bobot conv pertama
    stem = next((v for k, v in img_sd.items()
                 if k.endswith("conv_stem.weight") or k.endswith("conv1.weight")), None)
    in_chans = int(stem.shape[1]) if stem is not None else 1
    cfg_name = _cfg_value(ckpt.get("config"), "model", "image_encoder", "name") \
        if isinstance(ckpt, dict) else None
    print(f"[ckpt] in_chans={in_chans} | nama encoder di config: {cfg_name}")

    names = []
    for n in (cfg_name, CFG.mc_arch[CFG.mc_variant]):
        if n:
            n = str(n).replace("-detect", "")
            names += [n, n.replace("_ns", ".ns_jft_in1k")]
    names = list(dict.fromkeys(names))

    def _factory(name):
        return lambda: TimmBackbone(timm.create_model(
            name, pretrained=False, num_classes=0, in_chans=in_chans))

    print("=== Mencocokkan bobot ke kerangka model ===")
    backbone, _ = match_and_load([(f"timm/{n}", _factory(n)) for n in names], img_sd)
    del ckpt, full, img_sd
    gc.collect()
    return backbone, make_mammoclip_transform(in_chans)


def load_swinv2(inspect_only=False):
    from huggingface_hub import hf_hub_download

    sd, last_err = None, None
    for fname in CFG.sw_files:
        try:
            path = hf_hub_download(repo_id=CFG.sw_repo, filename=fname)
        except Exception as e:
            print(f"[hf] {fname} tidak ada / gagal diunduh ({type(e).__name__})")
            last_err = e
            continue
        print(f"[hf] berhasil: {path}")
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(path)
        else:
            sd = find_state_dict(robust_torch_load(path))
        break
    if sd is None:
        raise RuntimeError(f"Tidak ada bobot yang bisa diunduh dari {CFG.sw_repo}: {last_err}")

    sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    summarize_keys(sd)
    if inspect_only:
        return None, None, None

    def _hf():
        from transformers import Swinv2Config, Swinv2Model
        try:
            cfg = Swinv2Config.from_pretrained(CFG.sw_base)
        except Exception as e:
            print(f"[cfg] gagal ambil config base model ({type(e).__name__}), pakai default Large.")
            cfg = Swinv2Config(patch_size=4, num_channels=3, embed_dim=192,
                               depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                               window_size=24, pretrained_window_sizes=[12, 12, 12, 6],
                               mlp_ratio=4.0, image_size=384)
        cfg.image_size = CFG.sw_size      # SwinV2 pakai CPB, ganti resolusi aman
        return HFBackbone(Swinv2Model(cfg))

    def _timm(name):
        return lambda: TimmBackbone(timm.create_model(
            name, pretrained=False, num_classes=0, img_size=CFG.sw_size))

    cands = [("transformers/Swinv2Model", _hf)]
    cands += [(f"timm/{n}", _timm(n)) for n in CFG.sw_timm]

    print("=== Mencocokkan bobot ke kerangka model ===")
    backbone, leftover = match_and_load(cands, sd)
    print("Sisa bobot (kandidat head):", list(leftover.keys())[:10])
    del sd
    gc.collect()
    return backbone, make_swinv2_transform(), leftover


# =========================================================
# REKONSTRUKSI HEAD MLP CUSTOM (Swin-V2: Has_Cancer vs Normal)
# =========================================================
def _natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def build_head(leftover, in_dim, activation="gelu", n_out=2):
    """
    Head-nya MLP custom yang tidak terdokumentasi, jadi disusun ulang dari bentuk
    tensor yang tersisa: weight 2D -> Linear, weight 1D -> LayerNorm/BatchNorm1d,
    aktivasi disisipkan di antara dua Linear. Rantai dimensi tak nyambung -> None.
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
    missing, _ = head.load_state_dict(new_sd, strict=False)
    if missing:
        return None, f"head kurang bobot: {list(missing)[:3]}"
    return head.eval(), f"{len(layers)} layer, aktivasi {activation}"


def head_features(leftover, feat_dim, X_train, X_test, y_labels):
    """
    Head Swin-V2 dilatih Has_Cancer vs Normal, persis memisahkan Normal dari
    (Benign + Malignant) di soal ini. Aktivasi aslinya tidak tercatat di repo,
    jadi dipilih lewat AUC di train; kalau dua-duanya lemah, fiturnya dibuang.
    """
    scored = {}
    for act in ("gelu", "relu"):
        h, info = build_head(leftover, feat_dim, activation=act)
        print(f"[head/{act}] {'OK - ' + info if h is not None else 'gagal - ' + info}")
        if h is None:
            continue
        with torch.no_grad():
            lg = h(torch.from_numpy(X_train).float()).numpy()
        auc = roc_auc_score((y_labels != "Normal").astype(int), lg[:, 0] - lg[:, 1])
        print(f"[head/{act}] AUC Has_Cancer vs Normal di train: {auc:.4f}")
        scored[act] = (auc, h)

    if not scored:
        return None, None
    best_act = max(scored, key=lambda a: abs(scored[a][0] - 0.5))
    auc, h = scored[best_act]
    if abs(auc - 0.5) < 0.05:
        print("[head] terlalu lemah (AUC ~0.5), fitur head tidak dipakai.")
        return None, None
    print(f"[head] dipakai: aktivasi {best_act} (AUC {auc:.4f})")
    with torch.no_grad():
        return (h(torch.from_numpy(X_train).float()).numpy(),
                h(torch.from_numpy(X_test).float()).numpy())


# =========================================================
# DATA
# =========================================================
class BreastDataset(Dataset):
    def __init__(self, df, img_dir, transform, with_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
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
        x = self.transform(Image.open(self._resolve_path(row["image_id"])))
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


def run_backbone(name, train_df, test_df):
    """Muat satu backbone, ekstrak fitur train+test, lalu bebaskan memorinya."""
    if name == "mammo_clip":
        backbone, tfm = load_mammoclip()
        leftover = None
    elif name == "swinv2":
        backbone, tfm, leftover = load_swinv2()
    else:
        raise ValueError(f"Backbone tidak dikenal: {name}")

    feat_dim = backbone.num_features
    backbone = backbone.to(CFG.device).eval()
    print("Dimensi fitur:", feat_dim)

    loaders = [DataLoader(BreastDataset(df, d, tfm, lbl),
                          batch_size=CFG.batch_size, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True)
               for df, d, lbl in ((train_df, CFG.train_dir, True),
                                  (test_df, CFG.test_dir, False))]

    print(f"\n=== [{name}] ekstraksi fitur train ===")
    X_train, y_train = extract_features(backbone, loaders[0], CFG.device)
    print(f"\n=== [{name}] ekstraksi fitur test ===")
    X_test, _ = extract_features(backbone, loaders[1], CFG.device)
    print(f"[{name}] X_train {X_train.shape} | X_test {X_test.shape}")

    del backbone
    gc.collect()
    torch.cuda.empty_cache()
    return X_train, y_train, X_test, leftover, feat_dim


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
    print("Device:", CFG.device, "| torch:", torch.__version__, "| backbones:", CFG.backbones)

    if inspect_only:
        for name in CFG.backbones:
            print(f"\n########## {name} ##########")
            (load_mammoclip if name == "mammo_clip" else load_swinv2)(inspect_only=True)
        return

    # ---------------- DATA ----------------
    train_df = pd.read_csv(CFG.train_csv)
    test_df  = pd.read_csv(CFG.test_csv)
    classes   = sorted(train_df["label"].unique())
    class2idx = {c: i for i, c in enumerate(classes)}
    idx2class = {i: c for c, i in class2idx.items()}
    train_df["target"] = train_df["label"].map(class2idx)
    num_classes = len(classes)
    print("Train:", train_df.shape, "| Test:", test_df.shape, "| Classes:", class2idx)

    # ---------------- FITUR ----------------
    os.makedirs(CFG.out_dir, exist_ok=True)
    tr_blocks, te_blocks, y_train = [], [], None

    for name in CFG.backbones:
        print(f"\n########## {name} ##########")
        Xtr, y, Xte, leftover, feat_dim = run_backbone(name, train_df, test_df)
        y_train = y
        np.save(f"{CFG.out_dir}/X_train_{name}.npy", Xtr)
        np.save(f"{CFG.out_dir}/X_test_{name}.npy", Xte)
        tr_blocks.append(Xtr); te_blocks.append(Xte)

        if name == "swinv2" and CFG.sw_use_head and leftover:
            y_lbl = np.array([idx2class[t] for t in y])
            h_tr, h_te = head_features(leftover, feat_dim, Xtr, Xte, y_lbl)
            if h_tr is not None:
                tr_blocks.append(h_tr); te_blocks.append(h_te)

    X_train = np.hstack(tr_blocks)
    X_test  = np.hstack(te_blocks)
    np.save(f"{CFG.out_dir}/X_train.npy", X_train)
    np.save(f"{CFG.out_dir}/y_train.npy", y_train)
    np.save(f"{CFG.out_dir}/X_test.npy", X_test)
    print("\nFitur gabungan:", X_train.shape, "|", X_test.shape)

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
