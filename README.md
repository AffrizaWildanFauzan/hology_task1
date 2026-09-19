# HoloMine Breast Cancer Classification — Task 1 (Hology 9.0)

Klasifikasi citra mammogram full-field digital ke tiga kelas: `Normal`, `Benign`,
`Malignant`. Metrik lomba: **macro F1** pada 54 citra uji.

**Baca [`HANDOFF.md`](HANDOFF.md) lebih dulu.** Dokumen itu berisi seluruh konteks
lomba, temuan analisis data (termasuk satu artefak akuisisi yang penting), alasan
di balik setiap keputusan desain, dan daftar periksa untuk mereview pekerjaan ini.

## Struktur

| File | Fungsi |
|---|---|
| `src/preprocess.py` | Potong region payudara dari citra 3540×4740, buang anotasi burned-in, samakan lateralitas, CLAHE, cache ke `.npz` |
| `src/common.py` | Konstanta kelas, loader cache, pembentukan fold (tanpa torch) |
| `src/data.py` | `Dataset` PyTorch + augmentasi |
| `src/model.py` | Backbone ImageNet-pretrained + EMA |
| `src/train.py` | Fine-tune cross-validation, simpan probabilitas OOF & test |
| `src/tune_and_submit.py` | Tuning class-prior untuk macro F1, error bar, tulis `submission.csv` |
| `src/audit_artifacts.py` | Audit artefak dataset — **jalankan dan baca hasilnya** |
| `kaggle/run_kaggle.py` | Versi satu-file untuk dijalankan di notebook Kaggle (GPU) |

## Cara menjalankan

```bash
pip install -r requirements.txt

python src/preprocess.py --size 512 384          # ~35 detik, sekali saja
python src/audit_artifacts.py                    # baca temuannya
python src/train.py --model resnet34 --folds 5 --seeds 0 1 --epochs 25
python src/tune_and_submit.py --run artifacts/run --out submission.csv
```

Di Kaggle (GPU), cukup jalankan `kaggle/run_kaggle.py` — satu file, tanpa
dependensi pada repo ini.

## Kepatuhan aturan

Tanpa data eksternal. Tanpa LLM/VLM/AutoML/generative AI dalam pipeline. Tanpa
model Ultralytics. Hanya backbone ImageNet-pretrained dari torchvision/timm, yang
secara eksplisit diizinkan panitia. Rinciannya ada di `HANDOFF.md`.
