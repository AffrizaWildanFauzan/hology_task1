# =========================================================
# Feature extraction + linear probe untuk klasifikasi mammogram 3 kelas
# (Normal / Benign / Malignant) dengan data latih kecil (212 citra).
#
# Backbone (pilih di CFG.backbones):
#   "mammo_clip" -> shawn24/Mammo-CLIP  (default, b5)
#        Pre-trained-checkpoints/b{2,5}-model-best-epoch-*.tar
#        Bobotnya torch.save dari repo batmanlab/Mammo-CLIP:
#        {"model": state_dict, "config": ...} dengan prefix "image_encoder.",
#        BUKAN format from_pretrained -> AutoModel/timm pasti gagal.
#   "swinv2"     -> keanteng/swin-v2-large-ft-breast-cancer-classification-0603
#        Swin-V2 Large + head MLP 2 kelas (Has_Cancer vs Normal). config.json-nya
#        bukan config transformers dan bobotnya state_dict mentah berprefix.
#   "hf_auto"    -> model HF standar apa pun lewat AutoModel, mis.
#        microsoft/rad-dino atau facebook/dinov2-large (atur CFG.auto_repo).
#
# Gabungkan beberapa backbone: backbones = ["mammo_clip", "swinv2"]
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
import timm

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None          # mammogram full-field ukurannya besar

try:
    from scipy import ndimage as _ndi
except Exception:
    _ndi = None
try:
    import cv2 as _cv2
except Exception:
    _cv2 = None


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
    # cache citra hasil preprocessing: taruh di luar working dir supaya tidak
    # ikut tersimpan sebagai output notebook (bisa ratusan MB)
    cache_dir   = "/tmp/mammo_cache"

    backbones   = ["mammo_clip"]

    # --- Mammo-CLIP ---
    mc_repo     = "shawn24/Mammo-CLIP"
    mc_variant  = "b5"                  # "b5" (1.65 GB) atau "b2" (1.4 GB)
    mc_files    = {"b2": "Pre-trained-checkpoints/b2-model-best-epoch-10.tar",
                   "b5": "Pre-trained-checkpoints/b5-model-best-epoch-7.tar"}
    mc_arch     = {"b2": "tf_efficientnet_b2_ns", "b5": "tf_efficientnet_b5_ns"}
    mc_size     = (1520, 912)           # (H, W) sesuai paper Mammo-CLIP

    # --- Swin-V2 fine-tuned ---
    sw_repo     = "keanteng/swin-v2-large-ft-breast-cancer-classification-0603"
    sw_files    = ["model.safetensors", "pytorch_model.bin"]
    sw_base     = "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"
    sw_timm     = ["swinv2_large_window12to16_192to256",
                   "swinv2_large_window12to24_192to384",
                   "swinv2_large_window12_192"]
    sw_size     = (256, 256)
    sw_use_head = True                  # logit head fine-tune jadi fitur tambahan

    # --- model HF standar (AutoModel) ---
    auto_repo   = "microsoft/rad-dino"
    auto_size   = (518, 518)

    # --- preprocessing ---
    crop_breast = True                  # Otsu + komponen terbesar + bbox
    mask_breast = True                  # nolkan piksel di luar payudara (buang
                                        # label teks "L MLO"/"R CC" yang bisa
                                        # jadi sinyal palsu)
    fix_lateral = True                  # samakan arah payudara (kiri/kanan)
    fix_invert  = True                  # balik citra yang polaritasnya terbalik
    clip_pct    = (1.0, 99.0)           # truncation normalization
    clahe       = False                 # butuh cv2; uji dulu lewat CV

    # --- augmentasi tingkat fitur ---
    n_aug       = 4                     # salinan ter-augmentasi per citra train
    hflip_tta   = True

    # --- evaluasi ---
    cv_splits   = 5
    cv_repeats  = 5                     # 212 sampel -> 1x 5-fold terlalu berderau
    try_hier    = True                  # bandingkan flat vs bertingkat

    batch_size  = 4                     # resolusi tinggi -> batch kecil
    num_workers = 2
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


def best_prefix(src_sd, target_sd):
    """
    Cari prefix yang harus dibuang supaya nama layer checkpoint cocok dengan
    kerangka model: 'image_encoder.', 'image_encoder.model.', 'backbone.',
    'swinv2.', 'module.', dst. Sebuah key dihitung cocok hanya kalau nama DAN
    bentuk tensornya sama, supaya persentasenya tidak menipu.
    """
    cands = {""}
    for k in src_sd:
        parts = k.split(".")
        for i in range(1, min(4, len(parts))):
            cands.add(".".join(parts[:i]) + ".")

    def _hits(p):
        n = 0
        for k, v in src_sd.items():
            if k.startswith(p):
                t = target_sd.get(k[len(p):])
                if t is not None and tuple(t.shape) == tuple(v.shape):
                    n += 1
        return n

    best, best_hits = "", -1
    for p in sorted(cands):
        h = _hits(p)
        if h > best_hits:
            best, best_hits = p, h
    return best, best_hits


def positional_match(src_sd, tgt_sd):
    """
    Fallback kalau nama layer tidak cocok sama sekali (bobot disimpan dari
    implementasi lain). Urutan tensor di state_dict = urutan registrasi modul,
    jadi dua implementasi arsitektur yang sama punya DERET BENTUK identik.
    Hanya diterima kalau deretnya sama persis, supaya arsitektur beda ditolak.
    """
    s_items, t_items = list(src_sd.items()), list(tgt_sd.items())
    if len(s_items) != len(t_items) or not s_items:
        return None
    for (_, sv), (_, tv) in zip(s_items, t_items):
        if tuple(sv.shape) != tuple(tv.shape):
            return None
    return {tk: sv for (_, sv), (tk, _) in zip(s_items, t_items)}


def _diagnose(sd, module, prefix, n=12):
    tgt_sd = module.inner.state_dict()
    tgt, src = list(tgt_sd.keys()), list(sd.keys())
    print("\n" + "=" * 60)
    print("DIAGNOSTIK PENCOCOKAN BOBOT (salin seluruh blok ini kalau mau dibantu)")
    print("=" * 60)
    print(f"[src] {len(src)} key di checkpoint, prefix terbaik = {prefix!r}")
    for k in src[:n]:
        print(f"    src  {k:70s} {tuple(sd[k].shape)}")
    print("    ...")
    for k in src[-3:]:
        print(f"    src  {k:70s} {tuple(sd[k].shape)}")
    print(f"[tgt] {len(tgt)} key di kerangka model")
    for k in tgt[:n]:
        print(f"    tgt  {k:70s} {tuple(tgt_sd[k].shape)}")
    print("    ...")
    for k in tgt[-3:]:
        print(f"    tgt  {k:70s} {tuple(tgt_sd[k].shape)}")
    print("=" * 60 + "\n")


class TimmBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner = model
        self.num_features = model.num_features

    def forward(self, x):
        return self.inner(x)


class HFBackbone(nn.Module):
    """Model transformers -> pooler_output (fallback: rata-rata token)."""
    def __init__(self, model, num_features=None):
        super().__init__()
        self.inner = model
        self.num_features = num_features or getattr(
            model, "num_features", getattr(model.config, "hidden_size", None))

    def forward(self, x):
        out = self.inner(x)
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out.last_hidden_state.mean(1)


def match_and_load(candidates, sd, min_cov=0.95):
    """
    candidates: iterable of (label, factory). Cocokkan lewat nama; kalau gagal,
    coba pemetaan posisional. Pilih kandidat terbaik lalu muat bobotnya.
    """
    best = None
    for label, factory in candidates:
        try:
            module = factory()
        except Exception as e:
            print(f"  - {label:45s} gagal dibuat ({type(e).__name__}: {e})")
            continue
        tgt = module.inner.state_dict()
        prefix, hits = best_prefix(sd, tgt)
        cov, mode, pmap = hits / max(len(tgt), 1), "nama", None
        print(f"  - {label:45s} prefix={prefix!r:22s} cocok {hits}/{len(tgt)} ({cov:.1%})")
        if cov < min_cov:
            pmap = positional_match(sd, tgt)
            if pmap is not None:
                cov, mode = 1.0, "posisional"
                print(f"    -> nama tidak cocok, tapi deret bentuk tensornya identik "
                      f"({len(pmap)} tensor): pakai pemetaan posisional")
        if best is None or cov > best[0]:
            best = (cov, label, module, prefix, mode, pmap)
        if cov > 0.99:
            break

    if best is None:
        raise RuntimeError("Tidak ada kerangka model yang bisa dibangun. "
                           "Cek dependensi (timm / transformers).")

    cov, label, module, prefix, mode, pmap = best
    tgt = module.inner.state_dict()

    if mode == "posisional":
        print("[warn] Bobot dipasang berdasarkan URUTAN tensor, bukan nama layer. "
              "Deret bentuknya sama persis, tapi pastikan hasilnya masuk akal.")
        matched, used = pmap, set(sd.keys())
    else:
        matched = {k[len(prefix):]: v for k, v in sd.items()
                   if k.startswith(prefix) and k[len(prefix):] in tgt}
        bad = [k for k, v in matched.items() if tuple(v.shape) != tuple(tgt[k].shape)]
        for k in bad:
            matched.pop(k)
        if bad:
            print(f"[warn] {len(bad)} tensor dilewati karena bentuknya beda, contoh: {bad[:3]}")
        used = {prefix + k for k in matched}

    missing, _ = module.inner.load_state_dict(matched, strict=False)
    loaded = len(tgt) - len(missing)
    print(f"Backbone terpilih : {label}")
    print(f"Cara pencocokan   : {mode}" + (f" (prefix {prefix!r})" if mode == "nama" else ""))
    print(f"Bobot ter-load    : {loaded}/{len(tgt)} ({loaded/len(tgt):.1%})")
    if missing:
        print("  contoh missing  :", list(missing)[:5])
    if loaded / len(tgt) < min_cov:
        _diagnose(sd, module, prefix)
        raise RuntimeError(f"Bobot gagal dicocokkan (<{min_cov:.0%}). "
                           "Lihat blok DIAGNOSTIK di atas.")

    leftover = {k: v for k, v in sd.items() if k not in used}
    return module, leftover


# =========================================================
# BACKBONE BUILDERS
# tiap builder -> (backbone, spec) dengan spec = {size, chans, mean, std, leftover}
# =========================================================
MAMMO_MEAN, MAMMO_STD = 0.3089279, 0.25053555408335154
IMNET_MEAN, IMNET_STD = 0.449, 0.226      # rata-rata 3 kanal ImageNet (citra grayscale)


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

    img_sd = {k: v for k, v in full.items() if k.startswith("image_encoder.")}
    if not img_sd:
        img_sd = {k: v for k, v in full.items()
                  if not any(t in k for t in ("text_encoder", "text_projection",
                                              "logit_scale", "bert", "tokenizer"))}
    print(f"[ckpt] tensor image encoder: {len(img_sd)}")

    stem = next((v for k, v in img_sd.items()
                 if k.endswith("conv_stem.weight") or k.endswith("conv1.weight")), None)
    if stem is None:
        stem = next((v for v in img_sd.values() if v.dim() == 4), None)
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
    return backbone, {"size": CFG.mc_size, "chans": in_chans,
                      "mean": MAMMO_MEAN, "std": MAMMO_STD, "leftover": None}


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
        return None, None

    def _hf():
        from transformers import Swinv2Config, Swinv2Model
        try:
            cfg = Swinv2Config.from_pretrained(CFG.sw_base)
        except Exception as e:
            print(f"[cfg] gagal ambil config base ({type(e).__name__}), pakai default Large.")
            cfg = Swinv2Config(patch_size=4, num_channels=3, embed_dim=192,
                               depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                               window_size=24, pretrained_window_sizes=[12, 12, 12, 6],
                               mlp_ratio=4.0, image_size=384)
        cfg.image_size = CFG.sw_size[0]   # SwinV2 pakai CPB, ganti resolusi aman
        return HFBackbone(Swinv2Model(cfg))

    def _timm(name):
        return lambda: TimmBackbone(timm.create_model(
            name, pretrained=False, num_classes=0, img_size=CFG.sw_size[0]))

    cands = [("transformers/Swinv2Model", _hf)]
    cands += [(f"timm/{n}", _timm(n)) for n in CFG.sw_timm]

    print("=== Mencocokkan bobot ke kerangka model ===")
    backbone, leftover = match_and_load(cands, sd)
    print("Sisa bobot (kandidat head):", list(leftover.keys())[:10])
    del sd
    gc.collect()
    return backbone, {"size": CFG.sw_size, "chans": 3,
                      "mean": IMNET_MEAN, "std": IMNET_STD, "leftover": leftover}


def load_hf_auto(inspect_only=False):
    """Model HF standar (rad-dino, dinov2, ...) - repo-nya normal, jadi langsung
    AutoModel.from_pretrained tanpa akrobat pencocokan bobot."""
    from transformers import AutoModel
    print(f"\n=== AutoModel: {CFG.auto_repo} ===")
    model = AutoModel.from_pretrained(CFG.auto_repo)
    if inspect_only:
        print("[cfg]", model.config.__class__.__name__,
              "hidden_size:", getattr(model.config, "hidden_size", None))
        return None, None
    bb = HFBackbone(model)
    print("Dimensi fitur:", bb.num_features)
    size, mean, std = CFG.auto_size, IMNET_MEAN, IMNET_STD
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(CFG.auto_repo)
        s = getattr(proc, "crop_size", None) or getattr(proc, "size", None)
        if isinstance(s, dict):
            h = s.get("height") or s.get("shortest_edge")
            w = s.get("width") or s.get("shortest_edge")
            if h and w:
                size = (int(h), int(w))
        if getattr(proc, "image_mean", None):
            mean = float(np.mean(proc.image_mean)); std = float(np.mean(proc.image_std))
        print(f"[proc] size={size} mean={mean:.3f} std={std:.3f}")
    except Exception as e:
        print(f"[proc] processor tidak terbaca ({type(e).__name__}), pakai default.")
    return bb, {"size": size, "chans": 3, "mean": mean, "std": std, "leftover": None}


LOADERS = {"mammo_clip": load_mammoclip, "swinv2": load_swinv2, "hf_auto": load_hf_auto}


# =========================================================
# PREPROCESSING MAMMOGRAM
# =========================================================
def otsu_threshold(arr):
    """Ambang Otsu dari histogram 256-bin (numpy murni)."""
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return int(np.argmax(sigma_b))


def breast_mask_bbox(arr, pad=16, work_max=1024):
    """
    Otsu -> komponen terhubung terbesar -> (mask, bounding box). Komponen
    terbesar memisahkan payudara dari label teks dan marker yang menempel di
    pinggir; mask-nya dipakai untuk menolkan semua yang bukan payudara.

    Mask dihitung di resolusi kerja kecil lalu diperbesar lagi: label +
    dilatasi di citra 17 MP makan waktu detikan per citra, sedangkan bentuk
    payudara tidak butuh presisi sebesar itu.
    """
    scale = min(1.0, work_max / max(arr.shape))
    if scale < 1.0:
        sh, sw = max(int(arr.shape[0] * scale), 1), max(int(arr.shape[1] * scale), 1)
        small = np.asarray(Image.fromarray(arr).resize((sw, sh), Image.BILINEAR))
    else:
        small = arr

    thr = otsu_threshold(small)
    mask = small > max(thr, 1)
    if mask.sum() < small.size * 0.01:
        return None, None
    if _ndi is not None:
        lab, n = _ndi.label(mask)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            mask = lab == int(np.argmax(sizes))
        # sedikit dilatasi supaya tepi payudara tidak ikut terpotong
        mask = _ndi.binary_dilation(mask, iterations=4)

    if mask.shape != arr.shape:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (arr.shape[1], arr.shape[0]), Image.NEAREST)) > 127

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None, None
    r0, r1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad, arr.shape[0] - 1)
    c0, c1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad, arr.shape[1] - 1)
    if (r1 - r0) * (c1 - c0) < arr.size * 0.05 or min(r1 - r0, c1 - c0) < 32:
        return mask, None
    return mask, (r0, r1, c0, c1)


def looks_inverted(arr):
    """
    Polaritas terbalik (MONOCHROME1). Sengaja konservatif: pada mammogram yang
    normal minimal tiga dari empat pojok adalah latar hitam, jadi baru dianggap
    terbalik kalau mayoritas pojok benar-benar terang. Versi sebelumnya
    membandingkan pinggir vs tengah dan salah pada 62 dari 212 citra - sisi
    dinding dada menyentuh tepi, jadi 'pinggir' justru jaringan yang terang.
    """
    h, w = arr.shape
    ph, pw = max(h // 10, 1), max(w // 10, 1)
    corners = [arr[:ph, :pw], arr[:ph, -pw:], arr[-ph:, :pw], arr[-ph:, -pw:]]
    bright = sum(1 for c in corners if np.median(c) > 200)
    return bright >= 3


def normalize_laterality(arr):
    """
    Mammogram ada yang payudara kiri ada yang kanan. Samakan arahnya (dinding
    dada selalu di kiri) supaya model tidak perlu belajar dua cermin dari data
    yang cuma 212 citra.
    """
    w = arr.shape[1]
    third = max(w // 3, 1)
    if arr[:, -third:].mean() > arr[:, :third].mean():
        return np.ascontiguousarray(arr[:, ::-1])
    return arr


def apply_clahe(arr):
    if _cv2 is None:
        return arr
    return _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(arr)


def resize_pad(arr, out_h, out_w):
    """Resize dengan menjaga rasio lalu pad, supaya payudara tidak gepeng."""
    h, w = arr.shape
    scale = min(out_h / h, out_w / w)
    nh, nw = max(int(round(h * scale)), 1), max(int(round(w * scale)), 1)
    img = Image.fromarray(arr).resize((nw, nh), Image.BILINEAR)
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    out[:nh, :nw] = np.asarray(img)      # rata kiri-atas: dinding dada tetap di kiri
    return out


def base_preprocess(path, out_h, out_w):
    """Dijalankan sekali per citra, hasilnya di-cache: uint8 (out_h, out_w)."""
    arr = np.asarray(Image.open(path).convert("L"))
    if CFG.fix_invert and looks_inverted(arr):
        arr = 255 - arr
    if CFG.crop_breast or CFG.mask_breast:
        mask, box = breast_mask_bbox(arr)
        if mask is not None and CFG.mask_breast:
            arr = np.where(mask, arr, 0)
        if box is not None and CFG.crop_breast:
            r0, r1, c0, c1 = box
            arr = arr[r0:r1 + 1, c0:c1 + 1]
    if CFG.fix_lateral:
        arr = normalize_laterality(arr)
    if CFG.clahe:
        arr = apply_clahe(arr)
    return resize_pad(arr, out_h, out_w)


def augment(arr, rng):
    """Augmentasi ringan di atas citra yang sudah dinormalisasi arahnya."""
    img = Image.fromarray(arr)
    h, w = arr.shape
    if rng.random() < 0.7:
        img = img.rotate(rng.uniform(-10, 10), resample=Image.BILINEAR, fillcolor=0)
    if rng.random() < 0.7:                      # random resized crop
        s = rng.uniform(0.85, 1.0)
        ch, cw = int(h * s), int(w * s)
        top, left = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
        img = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)
    out = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.7:                      # brightness / contrast
        out = out * rng.uniform(0.9, 1.1) + rng.uniform(-12, 12)
    # tidak ada flip horizontal di sini: arah payudara sudah dinormalisasi dan
    # hflip-TTA membuat fitur invarian terhadap cermin, jadi flip cuma buang jatah
    return np.clip(out, 0, 255).astype(np.uint8)


def to_tensor(arr, spec):
    """Truncation normalization + normalisasi statistik backbone."""
    x = arr.astype(np.float32)
    lo, hi = np.percentile(x, CFG.clip_pct[0]), np.percentile(x, CFG.clip_pct[1])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x = (x - spec["mean"]) / spec["std"]
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    return t.repeat(3, 1, 1) if spec["chans"] == 3 else t


# =========================================================
# DATASET (membaca cache memmap hasil base_preprocess)
# =========================================================
def resolve_path(img_dir, image_id):
    p = os.path.join(img_dir, str(image_id))
    if os.path.exists(p):
        return p
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        q = os.path.join(img_dir, str(image_id) + ext)
        if os.path.exists(q):
            return q
    raise FileNotFoundError(f"Image not found: {image_id}")


def build_cache(df, img_dir, out_h, out_w, path):
    """
    Base preprocessing (decode JPEG besar, crop, lateralitas) dijalankan SEKALI
    dan disimpan sebagai memmap, karena setiap pass augmentasi akan membacanya
    ulang. Memmap juga berarti worker DataLoader berbagi memori yang sama.
    """
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r")
        if arr.shape == (len(df), out_h, out_w):
            print(f"[cache] pakai {path} {arr.shape}")
            return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                    shape=(len(df), out_h, out_w))
    for i, image_id in enumerate(tqdm(df["image_id"].tolist(), desc="preprocess", leave=False)):
        arr[i] = base_preprocess(resolve_path(img_dir, image_id), out_h, out_w)
    arr.flush()
    print(f"[cache] tulis {path} {arr.shape}")
    return np.load(path, mmap_mode="r")


class CachedDataset(Dataset):
    def __init__(self, cache, spec, aug_id=0, seed=0):
        self.cache, self.spec, self.aug_id, self.seed = cache, spec, aug_id, seed

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        arr = np.asarray(self.cache[i])
        if self.aug_id > 0:
            arr = augment(arr, np.random.default_rng((self.seed, self.aug_id, i)))
        return to_tensor(arr, self.spec), i


@torch.no_grad()
def extract_features(model, cache, spec, aug_id, desc):
    """Ekstraksi fitur dengan penurunan batch otomatis kalau VRAM tidak cukup."""
    bs = CFG.batch_size
    while True:
        try:
            loader = DataLoader(CachedDataset(cache, spec, aug_id, CFG.seed),
                                batch_size=bs, shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True)
            feats = []
            use_amp = (CFG.device == "cuda")
            for imgs, _ in tqdm(loader, desc=desc, leave=False):
                imgs = imgs.to(CFG.device, non_blocking=True)
                with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                    f = model(imgs)
                    if CFG.hflip_tta:
                        f = (f + model(torch.flip(imgs, dims=[3]))) / 2
                feats.append(f.float().cpu().numpy())
            return np.concatenate(feats, 0)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or bs <= 1:
                raise
            bs = max(bs // 2, 1)
            print(f"[oom] VRAM tidak cukup, ulangi dengan batch_size={bs}")
            torch.cuda.empty_cache()
            gc.collect()


def l2norm(X):
    """Tiap blok fitur dinormalisasi supaya satu backbone tidak mendominasi."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-8)


# =========================================================
# HEAD SWIN-V2 (Has_Cancer vs Normal) SEBAGAI FITUR TAMBAHAN
# =========================================================
def _natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def build_head(leftover, in_dim, activation="gelu", n_out=2):
    """Head MLP custom disusun ulang dari bentuk tensor yang tersisa."""
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


def pick_head(leftover, feat_dim, X_ref, y_is_abnormal):
    """Aktivasi head tidak tercatat di repo; pilih lewat AUC, buang kalau lemah."""
    scored = {}
    for act in ("gelu", "relu"):
        h, info = build_head(leftover, feat_dim, activation=act)
        print(f"[head/{act}] {'OK - ' + info if h is not None else 'gagal - ' + info}")
        if h is None:
            continue
        with torch.no_grad():
            lg = h(torch.from_numpy(X_ref).float()).numpy()
        auc = roc_auc_score(y_is_abnormal, lg[:, 0] - lg[:, 1])
        print(f"[head/{act}] AUC Has_Cancer vs Normal di train: {auc:.4f}")
        scored[act] = (auc, h)
    if not scored:
        return None
    act = max(scored, key=lambda a: abs(scored[a][0] - 0.5))
    auc, h = scored[act]
    if abs(auc - 0.5) < 0.05:
        print("[head] terlalu lemah (AUC ~0.5), tidak dipakai.")
        return None
    print(f"[head] dipakai: aktivasi {act} (AUC {auc:.4f})")
    return h


def head_logits(head, X):
    with torch.no_grad():
        return head(torch.from_numpy(X).float()).numpy()


# =========================================================
# KLASIFIKASI
# =========================================================
def make_clf(name, seed=CFG.seed):
    if name.startswith("LogReg"):
        C = float(name.split("_C")[1])
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=C,
                                                class_weight="balanced"))
    return make_pipeline(StandardScaler(),
                         SVC(kernel="linear", C=1.0, class_weight="balanced",
                             probability=True, random_state=seed))


CLF_NAMES = ["LogReg_C0.01", "LogReg_C0.1", "LogReg_C1.0", "SVM_linear"]


class FlatModel:
    def __init__(self, clf_name):
        self.clf_name = clf_name

    def fit(self, X, y, n_classes):
        self.n_classes = n_classes
        self.clf = make_clf(self.clf_name).fit(X, y)
        return self

    def predict_proba(self, X):
        p = np.zeros((len(X), self.n_classes))
        p[:, self.clf.classes_] = self.clf.predict_proba(X)
        return p


class HierModel:
    """
    Dua tahap: Normal vs abnormal dulu (pemisahan ini sudah paling kuat), lalu
    Benign vs Malignant hanya di antara yang abnormal.
    """
    def __init__(self, clf_name, normal_idx):
        self.clf_name, self.normal_idx = clf_name, normal_idx

    def fit(self, X, y, n_classes):
        self.n_classes = n_classes
        self.stage1 = make_clf(self.clf_name).fit(X, (y != self.normal_idx).astype(int))
        m = y != self.normal_idx
        self.abnormal = np.unique(y[m])
        self.stage2 = make_clf(self.clf_name).fit(X[m], y[m]) if len(self.abnormal) > 1 else None
        return self

    def predict_proba(self, X):
        p_abn = self.stage1.predict_proba(X)[:, 1]
        p = np.zeros((len(X), self.n_classes))
        p[:, self.normal_idx] = 1.0 - p_abn
        if self.stage2 is None:
            p[:, self.abnormal[0]] = p_abn
        else:
            p2 = self.stage2.predict_proba(X)
            for j, c in enumerate(self.stage2.classes_):
                p[:, c] = p_abn * p2[:, j]
        return p


def repeated_cv(model_factory, X_clean, y_clean, X_aug, g_aug, y_aug,
                n_classes, n_splits, n_repeats, seed):
    """
    CV berulang dengan baris augmentasi dikelompokkan: salinan ter-augmentasi
    sebuah citra SELALU ikut fold citra aslinya (kalau tidak, citra yang sama
    muncul di train dan validasi -> skor bocor dan terlalu optimistis).
    Validasi selalu pada baris asli saja.
    """
    scores, oof_sum = [], np.zeros((len(y_clean), n_classes))
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        oof = np.zeros((len(y_clean), n_classes))
        for tr, va in cv.split(X_clean, y_clean):
            Xtr, ytr = X_clean[tr], y_clean[tr]
            if X_aug is not None and len(X_aug):
                m = np.isin(g_aug, tr)
                Xtr = np.vstack([Xtr, X_aug[m]])
                ytr = np.concatenate([ytr, y_aug[m]])
            model = model_factory().fit(Xtr, ytr, n_classes)
            oof[va] = model.predict_proba(X_clean[va])
        scores.append(f1_score(y_clean, oof.argmax(1), average="macro"))
        oof_sum += oof
    return float(np.mean(scores)), float(np.std(scores)), oof_sum / n_repeats


# =========================================================
# PIPELINE
# =========================================================
def main(inspect_only=False):
    seed_everything(CFG.seed)
    print("Device:", CFG.device, "| torch:", torch.__version__, "| backbones:", CFG.backbones)

    if inspect_only:
        for name in CFG.backbones:
            print(f"\n########## {name} ##########")
            LOADERS[name](inspect_only=True)
        return

    # ---------------- DATA ----------------
    train_df = pd.read_csv(CFG.train_csv)
    test_df  = pd.read_csv(CFG.test_csv)
    classes   = sorted(train_df["label"].unique())
    class2idx = {c: i for i, c in enumerate(classes)}
    idx2class = {i: c for c, i in class2idx.items()}
    y_clean = train_df["label"].map(class2idx).to_numpy()
    n_classes = len(classes)
    print("Train:", train_df.shape, "| Test:", test_df.shape, "| Classes:", class2idx)

    os.makedirs(CFG.out_dir, exist_ok=True)
    tr_blocks, te_blocks, aug_blocks = [], [], []

    # ---------------- FITUR ----------------
    for name in CFG.backbones:
        print(f"\n########## {name} ##########")
        backbone, spec = LOADERS[name]()
        feat_dim = backbone.num_features
        backbone = backbone.to(CFG.device).eval()
        h, w = spec["size"]
        print(f"Dimensi fitur: {feat_dim} | input {h}x{w} | {spec['chans']} channel")

        os.makedirs(CFG.cache_dir, exist_ok=True)
        tag = f"{name}_{CFG.mc_variant if name == 'mammo_clip' else ''}{h}x{w}"
        cache_tr = build_cache(train_df, CFG.train_dir, h, w,
                               f"{CFG.cache_dir}/cache_{tag}_train.npy")
        cache_te = build_cache(test_df, CFG.test_dir, h, w,
                               f"{CFG.cache_dir}/cache_{tag}_test.npy")

        Xtr = extract_features(backbone, cache_tr, spec, 0, f"{name} train")
        Xte = extract_features(backbone, cache_te, spec, 0, f"{name} test")
        Xaug = [extract_features(backbone, cache_tr, spec, k, f"{name} aug {k}")
                for k in range(1, CFG.n_aug + 1)]
        print(f"[{name}] X_train {Xtr.shape} | X_test {Xte.shape} | aug {len(Xaug)}x")

        head = None
        if name == "swinv2" and CFG.sw_use_head and spec["leftover"]:
            head = pick_head(spec["leftover"], feat_dim, Xtr,
                             (np.array([idx2class[t] for t in y_clean]) != "Normal").astype(int))

        tr_blocks.append(l2norm(Xtr))
        te_blocks.append(l2norm(Xte))
        aug_blocks.append(np.vstack([l2norm(a) for a in Xaug]) if Xaug else None)
        if head is not None:
            tr_blocks.append(head_logits(head, Xtr))
            te_blocks.append(head_logits(head, Xte))
            if Xaug:
                aug_blocks.append(np.vstack([head_logits(head, a) for a in Xaug]))

        del backbone
        gc.collect()
        torch.cuda.empty_cache()

    X_clean = np.hstack(tr_blocks)
    X_test  = np.hstack(te_blocks)
    aug_blocks = [b for b in aug_blocks if b is not None]
    if aug_blocks and CFG.n_aug > 0:
        X_aug = np.hstack(aug_blocks)
        g_aug = np.tile(np.arange(len(y_clean)), CFG.n_aug)
        y_aug = np.tile(y_clean, CFG.n_aug)
    else:
        X_aug, g_aug, y_aug = None, None, None

    np.save(f"{CFG.out_dir}/X_train.npy", X_clean)
    np.save(f"{CFG.out_dir}/y_train.npy", y_clean)
    np.save(f"{CFG.out_dir}/X_test.npy", X_test)
    print("\nFitur:", X_clean.shape, "| test:", X_test.shape,
          "| baris augmentasi:", 0 if X_aug is None else len(X_aug))

    # ---------------- PEMILIHAN MODEL ----------------
    strategies = {"flat": lambda n: (lambda: FlatModel(n))}
    if CFG.try_hier and "Normal" in class2idx:
        ni = class2idx["Normal"]
        strategies["hier"] = lambda n: (lambda: HierModel(n, ni))

    print(f"\n===== CV {CFG.cv_splits}-fold x {CFG.cv_repeats} ulangan (macro F1) =====")
    results = {}
    for sname, sfactory in strategies.items():
        for cname in CLF_NAMES:
            mean, std, oof = repeated_cv(sfactory(cname), X_clean, y_clean,
                                         X_aug, g_aug, y_aug, n_classes,
                                         CFG.cv_splits, CFG.cv_repeats, CFG.seed)
            results[(sname, cname)] = (mean, std, oof)
            print(f"{sname:5s} {cname:14s} | {mean:.4f} +/- {std:.4f}")

    best_key = max(results, key=lambda k: results[k][0])
    best_mean, best_std, best_oof = results[best_key]
    print(f"\n=== Terbaik: {best_key[0]} + {best_key[1]} "
          f"(macro F1 {best_mean:.4f} +/- {best_std:.4f}) ===")
    print(f"[catatan] sebaran antar-ulangan +/-{best_std:.3f} adalah lantai derai "
          f"Anda; selisih di bawah itu jangan dianggap peningkatan.")

    # ---------------- OOF ----------------
    oof_pred = best_oof.argmax(1)
    print("\n===== OOF SUMMARY =====")
    print("Macro F1     :", f1_score(y_clean, oof_pred, average="macro"))
    print("Balanced Acc :", balanced_accuracy_score(y_clean, oof_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_clean, oof_pred))
    print(classification_report(y_clean, oof_pred, target_names=classes))

    # ---------------- FIT FINAL & SUBMISSION ----------------
    sname, cname = best_key
    Xfit, yfit = X_clean, y_clean
    if X_aug is not None:
        Xfit = np.vstack([X_clean, X_aug])
        yfit = np.concatenate([y_clean, y_aug])
    final = strategies[sname](cname)().fit(Xfit, yfit, n_classes)
    pred = final.predict_proba(X_test).argmax(1)

    sub = pd.DataFrame({"image_id": test_df["image_id"],
                        "label": [idx2class[i] for i in pred]})
    sub.to_csv(f"{CFG.out_dir}/submission.csv", index=False)
    print("\nSaved submission.csv")
    print(sub.head())
    print(sub["label"].value_counts())


if __name__ == "__main__":
    main(inspect_only="--inspect" in sys.argv)
