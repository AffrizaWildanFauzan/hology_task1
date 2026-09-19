"""Dataset, augmentation and fold construction for the mammogram cache."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from common import CLASSES, CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD, load_cache, make_folds  # noqa: F401


def _rand_affine(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    import cv2
    h, w = img.shape
    angle = rng.uniform(-12, 12)
    scale = rng.uniform(0.88, 1.12)
    tx, ty = rng.uniform(-0.05, 0.05, 2) * [w, h]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


class MammoDataset(Dataset):
    """Serves cached uint8 crops as 3-channel float tensors.

    Augmentation is deliberately geometry-heavy and photometry-light: lesion
    appearance (mass margins, microcalcification clusters) is the signal, so we
    keep intensity distortions mild enough not to erase it.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray | None, train: bool, seed: int = 0):
        self.images = images
        self.labels = labels
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        rng = self.rng
        if rng.random() < 0.5:
            img = img[:, ::-1].copy()
        if rng.random() < 0.8:
            img = _rand_affine(img, rng)
        img = img.astype(np.float32) / 255.0
        if rng.random() < 0.7:  # brightness / contrast jitter around the mid grey
            img = np.clip((img - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.08, 0.08), 0, 1)
        if rng.random() < 0.3:  # coarse dropout, forces the model off any single region
            h, w = img.shape
            for _ in range(rng.integers(1, 4)):
                ch, cw = int(h * rng.uniform(0.06, 0.16)), int(w * rng.uniform(0.06, 0.16))
                y0, x0 = rng.integers(0, h - ch), rng.integers(0, w - cw)
                img[y0:y0 + ch, x0:x0 + cw] = 0.0
        return img

    def __getitem__(self, i: int):
        img = self.images[i]
        img = self._augment(img) if self.train else img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(img))[None].repeat(3, 1, 1)
        if self.labels is None:
            return x
        return x, int(self.labels[i])
