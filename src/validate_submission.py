"""Check a submission against the format the organisers specify, before uploading.

A malformed CSV wastes one of the day's submission slots, and on a 54-image test
set every slot is expensive. Run this on every file before it goes to Kaggle.

Run:  python src/validate_submission.py submission.csv
"""
from __future__ import annotations

import sys

import pandas as pd

VALID = {"Normal", "Benign", "Malignant"}


def validate(sub_path: str, test_path: str = "test.csv") -> bool:
    ok = True

    def fail(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"  FAIL  {msg}")

    test = pd.read_csv(test_path)
    sub = pd.read_csv(sub_path)
    print(f"checking {sub_path} against {test_path}")

    if list(sub.columns) != ["image_id", "label"]:
        fail(f"columns are {list(sub.columns)}, must be exactly ['image_id', 'label']")
    if len(sub) != len(test):
        fail(f"{len(sub)} rows, expected {len(test)}")
    if sub.image_id.duplicated().any():
        fail(f"duplicate image_id: {sub.image_id[sub.image_id.duplicated()].tolist()}")

    missing = set(test.image_id) - set(sub.image_id)
    extra = set(sub.image_id) - set(test.image_id)
    if missing:
        fail(f"{len(missing)} image_id from test.csv are absent, e.g. {sorted(missing)[:5]}")
    if extra:
        fail(f"{len(extra)} image_id are not in test.csv, e.g. {sorted(extra)[:5]}")

    bad = set(sub.label.dropna().unique()) - VALID
    if bad:
        fail(f"labels outside {sorted(VALID)}: {sorted(bad)}")
    if sub.label.isna().any():
        fail(f"{int(sub.label.isna().sum())} missing labels")

    print("\n  prediction counts:")
    for label, n in sub.label.value_counts().items():
        print(f"    {label:<10} {n:>3}  ({n / len(sub):.1%})")
    if sub.label.nunique() < 3:
        print("  WARN  not all three classes are predicted -- macro F1 caps at 2/3 "
              "if a class is present in the truth but never predicted")

    print("\nPASS -- safe to upload" if ok else "\nDO NOT UPLOAD")
    return ok


if __name__ == "__main__":
    sys.exit(0 if validate(*sys.argv[1:] or ["submission.csv"]) else 1)
