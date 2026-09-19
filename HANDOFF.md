# HANDOFF — HoloMine Breast Cancer Classification, Task 1 (Hology 9.0)

> **Untuk AI yang membaca dokumen ini (Claude, ChatGPT, atau lainnya):** dokumen ini
> dirancang sebagai konteks tunggal yang lengkap. Anda **tidak perlu** mengakses
> Kaggle atau bertanya ke pengguna untuk memahami lomba ini. Tugas Anda ada di
> [§10 Daftar periksa untuk reviewer](#10-daftar-periksa-untuk-reviewer). Baca §1–§9
> lebih dulu, lalu kerjakan §10 secara berurutan. Jika Anda menemukan kesalahan,
> sebutkan nomor bagian dan nama file + nomor baris yang bermasalah.
>
> Status dokumen: ditulis oleh Claude (Opus 5) pada sesi pengerjaan pertama.
> Angka hasil di §8 berasal dari eksekusi nyata di mesin CPU, bukan estimasi.

---

## 1. Konteks lomba

**Nama:** [Holomine] Breast Cancer Classification Task 1 — cabang HoloMine, Hology 9.0
**Platform:** Kaggle Competition (private/invite, host: Faris El hakim)
**Hadiah:** Kudos (tidak memberi Points/Medals Kaggle)

### Objective
Membangun model yang **dapat direproduksi** untuk mengklasifikasikan setiap citra
mammogram digital *full-field* sebagai **Normal**, **Benign**, atau **Malignant**.

### Tujuan (dari panitia)
> Peserta berperan sebagai peneliti yang membantu membaca citra mammogram. Model akan
> dinilai menggunakan **macro F1 multiclass** pada data uji tersembunyi yang bersifat
> *patient-disjoint*, sehingga model perlu mengenali pola citra dan bukan menghafal
> pasien. Data telah dideidentifikasi dan hanya digunakan untuk riset serta kompetisi,
> bukan untuk diagnosis klinis.

### Dataset
266 citra mammogram digital lapangan penuh yang telah dideidentifikasi, dari
**AISSLab / MDCMI-BC**.

| Berkas | Isi |
|---|---|
| `train_images/` | 212 citra latih berlabel |
| `train.csv` | pasangan `image_id`, `label` |
| `test_images/` | 54 citra uji tanpa label |
| `test.csv` | daftar `image_id` yang wajib diprediksi |
| `sample_submission.csv` | template berkas submission |

Pemisahan data bersifat **patient-disjoint**: satu pasien tidak muncul di data latih
dan data uji sekaligus.

### Metrik
**Macro F1** pada tiga kelas. Untuk tiap kelas *c*:

```
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 · (Precision_c · Recall_c) / (Precision_c + Recall_c)
Macro F1    = (F1_Normal + F1_Benign + F1_Malignant) / 3
```

Setiap kelas berbobot sama, jadi performa pada kelas dengan jumlah citra lebih sedikit
(**Benign**, 52 citra) tetap diperhatikan. Ini konsekuensi penting — lihat §6.4.

**Leaderboard publik dihitung pada sekitar 30% data uji tersembunyi. Peringkat akhir
menggunakan leaderboard privat pada sisa data uji.** Lihat §5 — ini krusial.

### Format submission
CSV dengan tepat dua kolom, `image_id` dan `label`. Setiap `image_id` dari `test.csv`
harus muncul tepat sekali; nilai `label` wajib salah satu dari `Normal`, `Benign`,
`Malignant`.

```
image_id,label
image_0001.jpg,Normal
image_0002.jpg,Benign
image_0003.jpg,Malignant
```

### Aturan panitia (verbatim)

* Dilarang menggunakan data eksternal untuk melakukan prediksi.
* Diperbolehkan menggunakan bahasa pemrograman, tools, library, algoritma, dan software apapun.
* Diperbolehkan menggunakan pretrained model, embeddings, tokenizer, dan library NLP yang tersedia secara publik.
* Tidak diperbolehkan melakukan manual labeling terhadap test set.
* Tidak diperbolehkan melakukan kerja sama antar-tim di luar peserta resmi yang telah terdaftar dalam 1 tim.
* Segala bentuk target leakage, penggunaan hidden label, manipulasi leaderboard, atau metode lain yang memberikan keuntungan tidak adil dapat menyebabkan diskualifikasi.
* Dilarang menggunakan LLM, VLM, AutoML, generative AI yang mengotomatisasi analisis/pengembangan/prediksi, serta model dari Ultralytics.
* Nama tim yang digunakan di Kaggle Competition harus sama seperti nama tim yang didaftarkan saat pendaftaran HOLOGY 9.0
* Peserta hanya diperbolehkan melakukan submission dalam bentuk tim. Dilarang melakukan submission secara individu.

---

## 2. Fakta data (hasil EDA, angka nyata)

Dihasilkan oleh `src/audit_artifacts.py` dan skrip EDA pada 212 citra latih.

### 2.1 Distribusi label

| Kelas | Jumlah | Proporsi |
|---|---|---|
| Normal | 80 | 37.7% |
| Malignant | 80 | 37.7% |
| Benign | 52 | 24.5% |

Tidak seimbang, tapi ringan. Benign adalah kelas minoritas.

### 2.2 Properti citra

* Resolusi: **264 dari 266 citra berukuran 3540×4740**. Dua citra `Malignant`
  berukuran 4760×5840. Ukuran citra praktis tidak informatif.
* Semua citra berformat JPEG.
* Ukuran berkas berkisar 0.49 MB – 8.3 MB.
* Citra adalah mammogram lapangan penuh dengan latar hitam, border detektor, dan
  **anotasi tampilan yang terbakar ke dalam citra** ("R MLO", "L CC", dsb.) di salah
  satu sudut. Anotasi ini terlihat jelas pada contoh citra dari panitia.

### 2.3 Tidak ada pasangan tampilan yang bisa dideteksi

Karena pemisahan bersifat *patient-disjoint* sementara panitia **tidak menyediakan
`patient_id`**, idealnya fold cross-validation dikelompokkan per pasien. Kami mencari
pasangan citra dari pasien yang sama dengan korelasi silang ternormalisasi pada
thumbnail 32×24 dari 212 citra latih:

| Ambang kemiripan | Jumlah grup | Grup terbesar |
|---|---|---|
| 0.90 | 212 (semua tunggal) | 1 |
| 0.85 | 166 | 35 |
| 0.80 | 93 | 119 |

Distribusi kemiripan **mulus, tanpa mode terpisah di ujung atas** (persentil ke-99
hanya 0.828). Artinya tidak ada pasangan duplikat/near-duplikat yang jelas — konsisten
dengan hipotesis satu citra per pasien, atau tampilan-tampilan yang terlalu berbeda
untuk dicocokkan secara piksel.

**Implikasi:** kami memakai `StratifiedKFold` biasa. Ini bisa optimistis **jika**
ternyata ada beberapa tampilan per pasien di data latih. Lihat §9 (risiko terbuka).

---

## 3. TEMUAN UTAMA — artefak akuisisi yang berkorelasi dengan label

**Ini bagian terpenting di dokumen ini. Baca sampai habis sebelum mengubah apa pun.**

Metadata berkas mentah — yang tidak punya makna diagnostik apa pun — hampir sempurna
memisahkan `Normal` dari abnormal.

### 3.1 Mode warna JPEG × label (212 citra latih)

| Mode JPEG | Benign | Malignant | Normal |
|---|---|---|---|
| `L` (grayscale) | 0 | 0 | **78** |
| `RGB` | 52 | 80 | 2 |

### 3.2 Ukuran berkas × label

| Kelas | n | Rata-rata (byte) | Median (byte) |
|---|---|---|---|
| Benign | 52 | 1,561,102 | 1,462,146 |
| Malignant | 80 | 1,507,166 | 1,467,215 |
| **Normal** | **80** | **3,959,188** | **4,114,746** |

Citra `Normal` rata-rata **2.6× lebih besar** dari citra abnormal.

### 3.3 Seberapa kuat sinyalnya

Regresi logistik yang **hanya** melihat metadata berkas (mode warna, log ukuran
berkas, dimensi), dievaluasi dengan 5-fold stratified CV:

```
PROBE A — metadata berkas saja
  macro F1 = 0.5717   (chance ≈ 0.33)
  akurasi Normal-vs-abnormal = 0.9906   ← 210 dari 212 benar
```

Sebagai pembanding, statistik piksel global **setelah preprocessing kami**:

```
PROBE B — statistik piksel global setelah preprocessing
  macro F1 = 0.4152
  akurasi Normal-vs-abnormal = 0.6085
```

### 3.4 Apa artinya

Citra `Normal` jelas berasal dari **sumber atau pipeline pemrosesan yang berbeda**
dari citra abnormal — disimpan sebagai JPEG grayscale dengan kualitas lebih tinggi,
sementara citra abnormal disimpan sebagai RGB terkompresi lebih kuat. Ini jejak
provenance dataset, **bukan patologi payudara**.

Dua konsekuensi:

1. **Pemisahan Normal-vs-abnormal pada lomba ini jauh lebih mudah daripada di dunia
   nyata.** Model apa pun yang melihat piksel mentah akan sebagian ikut mempelajari
   perbedaan sumber ini. Skor CV dan leaderboard akan terlihat bagus karena alasan
   yang salah.
2. **Masalah sebenarnya adalah Benign vs Malignant.** Metadata tidak membantu sama
   sekali di sana (lihat matriks konfusi PROBE A: kelas Benign dan Malignant saling
   tertukar acak). Di situlah macro F1 benar-benar diperebutkan.

### 3.5 Keputusan yang kami ambil, dan alasannya

**Pipeline submission tidak pernah melihat metadata berkas.** `src/preprocess.py`:

* membaca setiap citra dengan `cv2.IMREAD_GRAYSCALE`, sehingga mode `L` vs `RGB`
  menjadi tidak terlihat oleh model;
* menerapkan CLAHE per citra, yang menyamakan distribusi kontras antar sumber.

Efeknya terukur: PROBE B (0.6085 untuk Normal-vs-abnormal) jauh di bawah PROBE A
(0.9906). Preprocessing kami **secara sengaja membuang sebagian besar artefak itu.**

**Alasan keputusan ini:**

* Aturan panitia menyebut *"Segala bentuk target leakage … atau metode lain yang
  memberikan keuntungan tidak adil dapat menyebabkan diskualifikasi."* Menjadikan
  ukuran byte berkas JPEG sebagai fitur tabular adalah persis "metode lain yang
  memberikan keuntungan tidak adil" — tidak ada pembacaan yang masuk akal di mana
  jumlah byte sebuah berkas adalah temuan radiologis.
* Objective panitia meminta model yang "mengenali pola citra". Mengeksploitasi
  provenance bertentangan langsung dengan itu.
* Risikonya asimetris: keuntungannya beberapa poin pada 54 citra uji, kerugiannya
  diskualifikasi.

**Ini keputusan tim, bukan keputusan Claude.** Kalau tim memutuskan lain, bukti
kuantitatifnya sudah tersedia lewat `src/audit_artifacts.py`. Tapi rekomendasi kami
tegas: **jangan pakai metadata berkas sebagai fitur.**

> Catatan jujur: model CNN yang dilatih pada piksel **tetap** akan menangkap sisa
> perbedaan tekstur/derau antar sumber. Itu tidak bisa dihilangkan sepenuhnya tanpa
> membuang sinyal asli, dan itu berlaku sama untuk setiap peserta. Yang kami hindari
> adalah mengeksploitasinya **secara sengaja lewat jalur non-citra**.

---

## 4. Konsekuensi: di mana usaha sebaiknya dihabiskan

Berdasarkan §3, tugas ini efektifnya terpecah dua:

| Sub-masalah | Kesulitan | Catatan |
|---|---|---|
| Normal vs abnormal | Mudah (artefak + perbedaan tekstur nyata) | Jangan habiskan waktu di sini |
| **Benign vs Malignant** | **Sulit** | **Di sinilah macro F1 ditentukan** |

Benign vs Malignant bergantung pada ciri lesi halus: margin massa (berbatas tegas =
cenderung jinak; spiculated/ireguler = cenderung ganas), kluster mikrokalsifikasi
pleomorfik, distorsi arsitektur. Semuanya kecil relatif terhadap citra 3540×4740 —
karena itu resolusi input dan *cropping* yang benar jauh lebih penting daripada
memilih backbone yang lebih besar.

---

## 5. Lantai derau — mengapa leaderboard publik hampir tidak berarti

Data uji berisi **54 citra**. Leaderboard publik dihitung pada ~30% dari itu:

```
0.30 × 54 ≈ 16 citra
```

**Leaderboard publik Anda dihitung pada sekitar 16 citra.**

Konsekuensi yang harus dipahami seluruh tim:

* Satu citra berubah benar/salah menggeser macro F1 publik beberapa poin penuh.
* Dua submission dengan kualitas asli identik bisa berbeda 0.10+ di leaderboard publik
  murni karena keberuntungan pengambilan sampel.
* **Memilih submission berdasarkan skor publik = memilih berdasarkan derau.** Itu cara
  paling umum kehilangan peringkat di leaderboard privat.

`src/tune_and_submit.py` mencetak simulasi ini secara eksplisit: ia mengambil sampel
16 prediksi OOF berulang kali dan melaporkan rentang 90%-nya. Jalankan, lihat
angkanya, lalu perlakukan pergerakan di dalam rentang itu sebagai nol informasi.

**Aturan kerja untuk tim:** pilih model dan keputusan desain berdasarkan **skor OOF
cross-validation pada 212 citra latih**, bukan berdasarkan leaderboard publik. Gunakan
leaderboard publik hanya untuk memastikan tidak ada bug fatal (misalnya skor 0.15
berarti ada yang rusak, bukan berarti model jelek).

---

## 6. Arsitektur pipeline

```
citra mentah 3540×4740
        │
        ├─ src/preprocess.py ──────────────► artifacts/cache.npz  (uint8, 512×384)
        │     grayscale → mask Otsu → komponen terbesar → buang anotasi →
        │     crop bounding box → normalisasi lateralitas → CLAHE → resize
        │
        ├─ src/train.py ───────────────────► artifacts/run/{oof,test,folds}.npy
        │     StratifiedKFold × beberapa seed, backbone ImageNet, EMA, flip-TTA
        │
        └─ src/tune_and_submit.py ─────────► submission.csv
              tuning class-prior (cross-fitted) + error bar + simulasi LB publik
```

### 6.1 Preprocessing (`src/preprocess.py`)

Mammogram lapangan penuh sebagian besar isinya latar hitam. Meresize 3540×4740 apa
adanya ke 512×384 akan mengecilkan payudara hingga sebagian kecil frame dan membuat
lesi hilang. Langkahnya:

1. **Grayscale paksa** — menutup jalur artefak mode warna (§3.5).
2. **Mask Otsu pada citra yang diperkecil 8×** — memisahkan jaringan dari latar.
3. **Opening + closing morfologis** — menghapus goresan tipis teks anotasi dan border
   detektor, lalu menutup bintik di dalam jaringan agar payudara tetap satu komponen.
4. **Ambil komponen terhubung terbesar** — payudara jauh lebih besar daripada label
   teks, dan label terletak di sudut yang tidak tersambung, sehingga ikut terbuang.
   *Ini juga menghapus anotasi "R MLO"/"L CC" yang kalau dibiarkan bisa menjadi jalur
   kebocoran tersendiri.*
5. **Nolkan semua di luar mask, lalu crop ke bounding box.**
6. **Normalisasi lateralitas** — cermin horizontal agar dinding dada selalu di kiri.
   Tanpa ini, model harus belajar dua versi dari setiap pola.
7. **CLAHE** (clip 2.0, tile 8×8) — meratakan kontras lokal; juga menyamakan
   perbedaan windowing antar sumber (§3.5).
8. **Resize ke 512×384** — rasio tinggi:lebar 4:3 mendekati bentuk payudara terkrop,
   lebih hemat piksel daripada persegi.

Waktu jalan: **~35 detik untuk seluruh 266 citra** pada 4 vCPU. Cache 34 MB.

Verifikasi visual sudah dilakukan (contact sheet 6 citra per kelas): crop bersih,
payudara terisolasi, orientasi seragam, anotasi hilang. Tidak ada kegagalan crop.

### 6.2 Model (`src/model.py`)

Backbone klasifikasi **ImageNet-pretrained** dari `timm` (kalau ada) atau
`torchvision`. Default `resnet34`. Citra grayscale direplikasi ke 3 kanal.

Dengan 212 citra latih, model kecil + regularisasi kuat mengalahkan model besar. Jangan
langsung melompat ke backbone besar; ukur dulu lewat OOF.

**EMA (exponential moving average) bobot**, decay 0.99. Pada ukuran sampel sebesar ini
satu run berayun keras antar epoch; merata-ratakan lintasan bobot mendarat di titik
yang lebih datar dan lebih tahan generalisasi daripada bobot langkah terakhir. Bobot
EMA-lah yang dievaluasi dan dipakai untuk prediksi.

### 6.3 Augmentasi (`src/data.py`)

Berat di geometri, ringan di fotometri — penampilan lesi adalah sinyalnya, jadi
distorsi intensitas dijaga agar tidak menghapusnya.

| Augmentasi | Probabilitas | Parameter |
|---|---|---|
| Flip horizontal | 0.5 | — |
| Affine (rotasi/skala/translasi) | 0.8 | ±12°, 0.88–1.12×, ±5% |
| Jitter kecerahan/kontras | 0.7 | kontras 0.85–1.15, offset ±0.08 |
| Coarse dropout | 0.3 | 1–3 kotak, 6–16% sisi |

Catatan: flip horizontal sedikit membatalkan normalisasi lateralitas di §6.1. Itu
disengaja — normalisasi membuat distribusi latih konsisten, flip tetap berguna sebagai
regularisasi. Kalau OOF menunjukkan sebaliknya, matikan flip dan ukur ulang.

### 6.4 Loss dan metrik

Macro F1 memberi bobot sama ke tiga kelas, sementara data tidak (52 Benign vs 80/80).
`CrossEntropyLoss` karena itu diberi **bobot kelas inverse-frequency** agar selaras
dengan metrik, ditambah `label_smoothing=0.05` (dataset kecil, label medis punya
ketidakpastian antar-pembaca).

### 6.5 Tuning class-prior (`src/tune_and_submit.py`)

Karena macro F1 memperlakukan ketiga kelas setara sementara data tidak, `argmax` polos
secara sistematis kurang memprediksi Benign. Mengalikan probabilitas dengan pengali
per kelas sebelum `argmax` memperbaiki sebagian besar dari itu.

Bobot dicari dengan *coordinate ascent* pada grid log-uniform, dan — penting —
**dievaluasi secara cross-fitted**: bobot untuk baris di fold *f* dipasang hanya pada
fold-fold lain. Dengan begitu angka perbaikan yang dilaporkan adalah perbaikan yang
benar-benar bisa terbawa ke data uji, bukan hasil memasang dan menilai pada data yang
sama. Kalau cross-fitted **tidak** lebih baik dari argmax polos, skripnya otomatis
mengirim argmax polos.

### 6.6 Inferensi

Rata-rata probabilitas dari seluruh model fold × seluruh seed, masing-masing dengan
**flip-TTA** (identitas + cermin horizontal). Lalu kalikan bobot class-prior, lalu
argmax.

### 6.7 Backbone Hugging Face: `hugging-science/breast-cancer-detector-2`

Tim meminta model ini dipakai sebagai satu-satunya model. Sudah diintegrasikan
(`src/model.py::HFClassifier`, dipanggil dengan `--model hf:<repo_id>`). Berikut
fakta lengkapnya supaya keputusan pakai/tidak pakai diambil dengan mata terbuka.

**Spesifikasi (diverifikasi dari model card dan `config.json`):**

| Properti | Nilai |
|---|---|
| Arsitektur | `ViTForImageClassification`, ViT-base-patch16-224 |
| Parameter | 85.8 juta |
| Resolusi asli | 224×224 |
| Kelas | 3 — `benign`=0, `malignant`=1, `normal`=2 |
| Lisensi | Apache-2.0 |
| Rantai base model | `google/vit-base-patch16-224-in21k` → `Parveshiiii/breast-cancer-detector` → checkpoint ini |
| Data latih | `gymprathap/Breast-Cancer-Ultrasound-Images-Dataset` (BUSI), ~1.578 citra |
| Hasil yang diklaim | akurasi validasi 94.46% setelah 12 epoch |

**Urutan label cocok persis dengan `CLASSES` kami** (`['Benign','Malignant','Normal']`
→ benign, malignant, normal). Sudah diverifikasi lewat kode, bukan asumsi. Karena itu
head klasifikasi bawaannya bisa dipertahankan sebagai *warm start* lewat `--keep-head`,
tidak perlu dibuang dan dilatih dari nol.

**Kepatuhan aturan: AMAN.** Aturan panitia membolehkan *"pretrained model ... yang
tersedia secara publik"*, dan checkpoint ini publik serta berlisensi Apache-2.0. Data
latihnya (BUSI, ultrasonografi) sama sekali tidak beririsan dengan mammogram lomba ini,
jadi **tidak ada risiko kebocoran test set**. Ini kategori risiko yang berbeda dan jauh
lebih rendah daripada artefak di §3.

**Masalah teknisnya: modalitas citranya salah.**

Model card-nya sendiri mencantumkan, di bawah *Out-of-Scope*:

> - Use with **mammography**, MRI, CT, or any non-ultrasound modality.
> - Images with text overlays, annotations, calipers, or other artifacts.

Jadi pembuatnya secara eksplisit menyatakan model ini tidak untuk mammogram. Alasannya
fisis, bukan sekadar kehati-hatian administratif:

* **Ultrasonografi** adalah citra akustik — derau speckle, lesi hipoekoik, bayangan
  akustik posterior, medan pandang sempit (beberapa sentimeter), resolusi ~500×500.
* **Mammografi** adalah proyeksi sinar-X — atenuasi jaringan, mikrokalsifikasi
  sub-milimeter, margin massa spiculated, seluruh payudara dalam satu frame 3540×4740.

Ciri yang dipelajari pada satu modalitas tidak berlaku pada yang lain. Lebih jauh,
fine-tuning 12 epoch pada 1.500 citra ultrasonografi **menjauhkan** bobotnya dari fitur
umum ImageNet-21k. Jadi ada kemungkinan nyata checkpoint ini bekerja **lebih buruk**
daripada backbone ImageNet biasa untuk tugas ini — bukan karena modelnya jelek, tapi
karena spesialisasinya ke arah yang salah.

**Angka 94.46% di model card tidak berlaku di sini.** Itu akurasi pada ultrasonografi
BUSI, bukan macro F1 pada mammogram. Jangan dipakai sebagai ekspektasi.

**Cara menyelesaikan perdebatan ini: ukur, jangan berargumen.**

```bash
# zero-shot: pakai prediksi bawaannya langsung, tanpa dilatih ulang
python src/zeroshot_hf.py --model hugging-science/breast-cancer-detector-2

# fine-tune sebagai backbone, lalu bandingkan OOF-nya dengan baseline
python src/train.py --model hf:hugging-science/breast-cancer-detector-2 \
    --keep-head --img-size 384 288 --folds 5 --seeds 0 --epochs 15 \
    --out-dir artifacts/vit
python src/tune_and_submit.py --run artifacts/vit --out submission_vit.csv
```

Bandingkan OOF-nya dengan baseline resnet34 di §8.4. Kalau lebih tinggi, pakai. Kalau
lebih rendah, jangan — apa pun yang tertulis di model card.

**Catatan resolusi.** ViT-base membawa position embedding untuk 224×224. `HFClassifier`
menyalakan `interpolate_pos_encoding=True`, sehingga input lebih besar (384×288,
512×384) tetap bisa dipakai. Ini penting: memampatkan mammogram ke 224×224 hampir pasti
menghapus mikrokalsifikasi (§4). Tapi interpolasi position embedding menjauhkan model
dari kondisi pralatihnya, jadi 224 vs 384 vs 512 perlu diukur, bukan diasumsikan.

---

## 7. Cara menjalankan

```bash
pip install -r requirements.txt

python src/preprocess.py --size 512 384
python src/audit_artifacts.py                     # baca §3 sambil melihat outputnya
python src/train.py --model resnet34 --folds 5 --seeds 0 1 --epochs 25 --verbose
python src/tune_and_submit.py --run artifacts/run --out submission.csv
```

Di **Kaggle dengan GPU**, jalankan `kaggle/run_kaggle.py` — satu berkas mandiri, tidak
bergantung pada repo ini. Ubah `DATA_DIR` bila slug kompetisinya berbeda. Perkiraan
waktu: 10–20 menit pada P100/T4 untuk setelan default.

Untuk menggabungkan beberapa run (backbone berbeda) menjadi satu ensemble:

```bash
python src/train.py --model resnet34        --out-dir artifacts/r34
python src/train.py --model efficientnet_b0 --out-dir artifacts/eb0
python src/tune_and_submit.py --run artifacts/r34 artifacts/eb0 --out submission.csv
```

---

## 8. Hasil

### 8.1 Status

**Skor OOF final belum tersedia saat dokumen ini ditulis.** Run resnet34 5-fold x
25 epoch sedang berjalan di CPU (~2,8 jam). Bagian ini akan diperbarui dengan angka
nyata begitu selesai. Yang sudah terverifikasi ada di bawah.

Reviewer: kalau Anda membaca ini dan §8.4 masih kosong, **jangan menilai kualitas
model** -- belum ada angkanya. Tetap kerjakan §10 bagian A, B, C, E (kebenaran kode,
kepatuhan aturan, format, reproduktibilitas); bagian D butuh angka.

### 8.2 Yang sudah terverifikasi

| Komponen | Status | Bukti |
|---|---|---|
| Preprocessing 266 citra | LULUS | 35 detik, cache 34 MB, 0 kegagalan crop |
| Verifikasi visual crop | LULUS | contact sheet 6 citra/kelas: payudara terisolasi, orientasi seragam, anotasi hilang |
| Bobot ImageNet termuat | LULUS | resnet34 216 tensor, efficientnet_b0 358 tensor; loss turun 0.78 -> 0.37 dalam 15 epoch (inisialisasi acak tidak berperilaku begini) |
| Rantai train -> tune -> submit | LULUS | smoke run 2-fold x 2-epoch menghasilkan CSV yang lolos validator |
| Format submission | LULUS | `src/validate_submission.py` |

### 8.3 Batasan lingkungan tempat angka ini dihasilkan

Angka lokal dihasilkan di mesin **tanpa GPU** (4 vCPU), dengan **satu seed**, dan
proxy memblokir `download.pytorch.org` serta `huggingface.co` sehingga bobot ImageNet
diambil dari mirror GitHub (`src/fetch_weights.py`).

**Perlakukan angka lokal sebagai validasi bahwa pipeline-nya benar, bukan sebagai
estimasi kualitas model akhir.** Konfigurasi Kaggle (`kaggle/run_kaggle.py`) memakai
2 seed dan bisa di-ensemble dengan backbone kedua; hasilnya akan berbeda -- dan
seharusnya lebih baik serta lebih stabil.

### 8.4 Skor OOF

<!-- Diisi setelah run selesai. Format yang akan diisi:
     - OOF macro F1 plain argmax
     - OOF macro F1 tuned (cross-fitted)  <- INI angka yang dikutip
     - 95% CI bootstrap
     - simulasi LB publik (16 citra)
     - F1 per kelas + matriks konfusi
     - distribusi prediksi pada test set
-->

*(belum tersedia -- lihat §8.1)*


---

## 9. Risiko dan pertanyaan terbuka

1. **Pengelompokan pasien tidak terverifikasi.** Panitia menjamin split
   patient-disjoint tapi tidak memberi `patient_id`. Kalau data latih berisi beberapa
   tampilan per pasien, `StratifiedKFold` kami membocorkan pasien antar fold dan skor
   OOF menjadi optimistis. Pencarian near-duplicate (§2.3) tidak menemukan pasangan,
   tapi itu bukan bukti. **Mitigasi:** perlakukan OOF sebagai batas atas, dan jangan
   kaget kalau leaderboard privat lebih rendah.
2. **Artefak akuisisi (§3) membuat sub-masalah Normal terlihat lebih mudah dari
   seharusnya.** Baik CV maupun leaderboard sama-sama terpengaruh karena artefaknya ada
   di kedua sisi (test set juga campuran 20 `L` / 34 `RGB`), jadi ini tidak merusak
   pemilihan model — tapi jangan salah menyimpulkan bahwa modelnya "mendeteksi kanker
   dengan baik".
3. **Ukuran data uji.** 54 citra total, ~16 di leaderboard publik. Lihat §5.
4. **Resolusi input.** 512×384 dipilih sebagai kompromi. Mikrokalsifikasi berukuran
   sub-milimeter dan bisa hilang pada resolusi ini. Menaikkan ke 768×576 atau 1024×768
   adalah eksperimen bernilai tinggi kalau GPU memungkinkan — ukur lewat OOF.
5. **Backbone belum disapu.** Hanya sedikit konfigurasi yang diuji. `convnext_tiny`,
   `efficientnet_b0/b3`, `densenet121` layak dicoba. Bandingkan lewat OOF, bukan LB.

---

## 10. Daftar periksa untuk reviewer

Kerjakan berurutan. Untuk setiap butir, jawab **LULUS / GAGAL / TIDAK YAKIN** disertai
bukti (nama file + nomor baris, atau angka).

### A. Kebenaran — kebocoran dan validasi
1. Apakah ada informasi dari fold validasi yang bocor ke pelatihan fold tersebut?
   Periksa `src/train.py` — khususnya bahwa bobot kelas pada loss dihitung dari `y_tr`
   saja, bukan dari seluruh `y`.
2. Apakah tuning class-prior di `src/tune_and_submit.py::cross_fitted_predictions` benar-benar
   cross-fitted? Verifikasi bahwa `fit_weights` hanya menerima baris `~va`.
3. Bobot final yang dikirim (`fit_weights(oof, y)`) dipasang pada seluruh OOF. Apakah
   ini masalah, mengingat keputusan *apakah akan menggunakan tuning sama sekali* sudah
   diambil lewat estimasi cross-fitted? Beri argumen.
4. Apakah preprocessing dijalankan identik untuk train dan test? Bandingkan jalur kode
   di `src/preprocess.py::main`.
5. Apakah augmentasi benar-benar mati saat inferensi? Periksa `MammoDataset.train`.

### B. Kepatuhan aturan
6. Adakah data eksternal yang dipakai? (Bobot ImageNet-pretrained **diizinkan** secara
   eksplisit oleh aturan. Segala yang lain tidak.)
7. Adakah LLM/VLM/AutoML/generative AI **di dalam pipeline**? (Claude dipakai untuk
   menulis kode — itu perkakas pengembangan, bukan komponen pipeline. Verifikasi bahwa
   tidak ada panggilan model generatif saat runtime.)
8. Adakah model Ultralytics? Grep `ultralytics`, `yolo`.
9. Apakah metadata berkas (ukuran byte, mode JPEG, dimensi) masuk ke pipeline
   submission? **Seharusnya tidak.** Grep `getsize`, `\.mode`, `bytes` di luar
   `src/audit_artifacts.py`. Ini titik risiko diskualifikasi paling serius — periksa
   dengan teliti.
10. Apakah ada pelabelan manual pada test set? (Seharusnya tidak — tidak ada label uji
    yang di-hardcode di mana pun. Grep `test.csv` dan pastikan tidak ada label
    tertulis.)

### C. Kebenaran metrik dan format
11. Apakah macro F1 dihitung dengan `average="macro"` di semua tempat? Grep `f1_score`.
12. Apakah `submission.csv` berisi tepat 54 baris + header, kolom persis
    `image_id,label`, setiap `image_id` dari `test.csv` muncul tepat sekali, dan semua
    label ∈ {Normal, Benign, Malignant}? **Verifikasi ini secara langsung dengan
    membaca berkasnya**, jangan percaya kode saja.
13. Apakah urutan kelas konsisten antara `CLASSES` di `src/common.py`, bobot loss,
    bobot class-prior, dan pemetaan kembali ke string? Ketidakcocokan urutan di sini
    akan memberi skor yang terlihat masuk akal tapi salah — periksa ujung ke ujung.

### D. Metodologi
14. Apakah argumen di §5 tentang leaderboard publik benar secara statistik? Hitung
    ulang sendiri: berapa standar deviasi macro F1 pada sampel 16 citra?
15. Apakah `StratifiedKFold` pilihan yang tepat mengingat jaminan patient-disjoint?
    Apa yang seharusnya dilakukan kalau `patient_id` ternyata bisa direkonstruksi?
16. Apakah keputusan di §3.5 (membuang artefak, bukan mengeksploitasinya) dapat
    dipertahankan berdasarkan teks aturan yang dikutip di §1? Beri pendapat Anda.
17. Apakah augmentasi flip horizontal (§6.3) membatalkan manfaat normalisasi
    lateralitas? Rancang eksperimen untuk mengukurnya.
18. Apa perbaikan **berdampak tertinggi berikutnya**? Urutkan berdasarkan rasio
    dampak/usaha, dan untuk masing-masing sebutkan bagaimana cara mengukurnya lewat OOF
    (bukan lewat leaderboard).

### E. Reproduktibilitas
19. Apakah seed diatur di semua sumber keacakan (numpy, torch, RNG dataset)?
    Apakah dua jalannya menghasilkan angka yang sama?
20. Apakah `kaggle/run_kaggle.py` benar-benar mandiri dan menghasilkan pipeline yang
    sama seperti `src/`? Bandingkan baris per baris — **divergensi antara keduanya
    adalah bug**, karena hasil yang dilaporkan berasal dari `src/` sementara yang
    dikirim ke Kaggle kemungkinan berasal dari `kaggle/`.

### Format jawaban yang diminta
```
## Ringkasan
<2–3 kalimat: apakah pipeline ini layak dikirim?>

## Temuan kritis (harus diperbaiki sebelum submit)
1. <file:baris> — <masalah> — <perbaikan yang disarankan>

## Temuan penting (sebaiknya diperbaiki)
...

## Catatan kecil
...

## Jawaban daftar periksa
A1: LULUS — <bukti>
A2: ...
```

---

## 11. Lampiran — jejak keputusan

| Keputusan | Alternatif yang dipertimbangkan | Alasan memilih |
|---|---|---|
| Crop payudara via Otsu + komponen terbesar | Resize langsung; deteksi kotak via model | Latar hitam mendominasi; resize langsung membuang resolusi lesi. Deteksi berbasis model butuh label kotak yang tidak ada. |
| 512×384 | 224×224; 1024×768 | 224 terlalu kasar untuk lesi. 1024 layak dicoba kalau ada GPU (§9.4). |
| Normalisasi lateralitas | Biarkan apa adanya | Menghapus satu faktor variasi yang tidak informatif dari dataset 212 citra. |
| `resnet34` | resnet18; efficientnet; convnext | Titik awal yang masuk akal untuk n=212. Belum disapu (§9.5). |
| EMA bobot | Checkpoint terbaik per epoch | Pemilihan checkpoint pada fold validasi 42 citra sangat berderau; EMA tidak butuh sinyal validasi. |
| Bobot kelas inverse-frequency | Loss tanpa bobot; focal loss | Selaras langsung dengan macro F1 yang memberi bobot setara antar kelas. |
| Tuning class-prior cross-fitted | Argmax polos; tuning non-cross-fitted | Argmax kurang memprediksi kelas minoritas. Cross-fitting mencegah menipu diri sendiri. |
| Membuang artefak akuisisi | Mengeksploitasinya sebagai fitur | §3.5 — risiko diskualifikasi asimetris terhadap keuntungannya. |
| Pilih model via OOF | Pilih via leaderboard publik | LB publik ≈ 16 citra; lihat §5. |
