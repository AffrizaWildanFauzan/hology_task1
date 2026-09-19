"""Backbone construction.

Only ImageNet-pretrained classifiers from timm/torchvision are used, which the
competition rules allow explicitly ("pretrained model ... yang tersedia secara
publik"). No detection models and nothing from Ultralytics.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _replace_head(model: nn.Module, num_classes: int, dropout: float) -> nn.Module:
    """torchvision puts the classifier under different attribute names per family."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif hasattr(model, "classifier"):
        head = model.classifier
        in_f = head[-1].in_features if isinstance(head, nn.Sequential) else head.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"unrecognised head layout for {type(model).__name__}")
    return model


def _from_hub(name: str, num_classes: int, dropout: float) -> nn.Module:
    """Normal path: let timm or torchvision download the checkpoint."""
    try:
        import timm
        return timm.create_model(name, pretrained=True, num_classes=num_classes, drop_rate=dropout)
    except ImportError:
        pass
    import torchvision.models as tvm
    return _replace_head(getattr(tvm, name)(weights="DEFAULT"), num_classes, dropout)


def _from_mirror(name: str, num_classes: int, dropout: float) -> nn.Module:
    """Fallback for environments that block download.pytorch.org / huggingface.co."""
    from fetch_weights import load_into

    try:
        import timm
        model = timm.create_model(name, pretrained=False, num_classes=num_classes, drop_rate=dropout)
    except ImportError:
        import torchvision.models as tvm
        model = _replace_head(getattr(tvm, name)(weights=None), num_classes, dropout)
    return load_into(model, name)


def build_model(name: str = "resnet34", num_classes: int = 3, pretrained: bool = True,
                dropout: float = 0.3) -> nn.Module:
    if not pretrained:
        try:
            import timm
            return timm.create_model(name, pretrained=False, num_classes=num_classes, drop_rate=dropout)
        except ImportError:
            import torchvision.models as tvm
            return _replace_head(getattr(tvm, name)(weights=None), num_classes, dropout)

    try:
        return _from_hub(name, num_classes, dropout)
    except Exception as exc:  # network blocked, or the checkpoint host is down
        print(f"  pretrained download failed ({type(exc).__name__}); trying GitHub mirror", flush=True)
        return _from_mirror(name, num_classes, dropout)


class EMA:
    """Exponential moving average of weights.

    With 212 images a run swings a lot between epochs; averaging the trajectory is
    a cheap way to land on a flatter, better-generalising point than the last step.
    """

    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def copy_to(self, model: nn.Module) -> None:
        own = model.state_dict()
        model.load_state_dict({k: v.to(dtype=own[k].dtype) for k, v in self.shadow.items()})
