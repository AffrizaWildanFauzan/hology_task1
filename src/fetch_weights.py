"""Fetch ImageNet weights from a GitHub mirror.

torchvision and timm both pull their checkpoints from hosts that some sandboxed
or air-gapped environments block (download.pytorch.org, huggingface.co). The
timm weights GitHub release mirrors the same ImageNet-1k checkpoints, so this
module fetches from there and loads them by hand.

On Kaggle none of this is needed -- `pretrained=True` works directly. This exists
so the pipeline is runnable and verifiable off-Kaggle too.

Run:  python src/fetch_weights.py resnet34
"""
from __future__ import annotations

import os
import sys

import torch

MIRROR = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/"

# Filenames as published in that release; the hash suffix is part of the name.
MIRRORED = {
    # torchvision-format state dict
    "resnet34": "resnet34-43635321.pth",
    # timm-format state dicts
    "efficientnet_b0": "efficientnet_b0_ra-3dd342df.pth",
    "tf_efficientnet_b0": "tf_efficientnet_b0_aa-827b6e33.pth",
}

CACHE = os.path.expanduser("~/.cache/hology_weights")


def fetch(name: str) -> str:
    if name not in MIRRORED:
        raise KeyError(f"no mirror for {name}; available: {sorted(MIRRORED)}")
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, MIRRORED[name])
    if not os.path.exists(path):
        torch.hub.download_url_to_file(MIRROR + MIRRORED[name], path, progress=False)
    return path


def load_into(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Load mirrored ImageNet weights, tolerating the replaced classifier head."""
    state = torch.load(fetch(name), map_location="cpu", weights_only=True)

    # Drop the 1000-way ImageNet head; ours is 3-way and freshly initialised.
    own = model.state_dict()
    state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}

    missing, _ = model.load_state_dict(state, strict=False)
    body_missing = [k for k in missing if not any(h in k for h in ("fc.", "classifier."))]
    if body_missing:
        raise RuntimeError(f"backbone keys did not load for {name}: {sorted(body_missing)[:8]}")
    print(f"  loaded mirrored ImageNet weights for {name} "
          f"({len(state)} tensors, head re-initialised)", flush=True)
    return model


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["resnet34"]:
        print(arg, "->", fetch(arg))
