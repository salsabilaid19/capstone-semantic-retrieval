# Panduan Embedding & Kompatibilitas

## Kontrak embedding
- Model: `intfloat/multilingual-e5-small`, dimensi **384** (dicek otomatis; ubah `--expected-dim` hanya bila mengganti model).
- Teks dokumen = `"passage: " + document_text`; teks kueri = `"query: " + kueri`. Vektor float32, ternormalisasi (dot product = cosine).
- `src/embedding.py` adalah satu-satunya implementasi; notebook Colab dan `scripts/generate_embeddings.py` memakainya, jadi hasilnya sama.
- `model_config.json` menyimpan nama model, prefix, dimensi, jumlah dokumen, **hash urutan `doc_id`**, dan versi library pembuatnya.

## Membuat embedding
Pilih salah satu (dataset dipilih lewat folder: `artifacts` = contoh sintetis, `artifacts_real` = data nyata):
1. **Colab** (disarankan bila laptop lemah): `notebooks/generate_embeddings.ipynb`, set `DATASET`, unduh `embeddings.npy` + `model_config.json` ke folder artifacts.
2. **Lokal**: `pip install sentence-transformers` lalu `python -m scripts.generate_embeddings --artifacts-dir artifacts_real`.

Penulisan bersifat atomik: bila validasi gagal (NaN, tidak ternormalisasi, jumlah salah), berkas lama tidak diubah.

## Memeriksa kompatibilitas (WAJIB setelah membuat/menyalin embedding)
```powershell
python -m scripts.check_embeddings --artifacts-dir artifacts_real            # struktural, tanpa model
python -m scripts.check_embeddings --artifacts-dir artifacts_real --sanity   # dengan model asli
```
| Pemeriksaan | Menangkap |
|---|---|
| jumlah baris, hash `doc_id`, nama model, dimensi 384, NaN, norma | embedding dari dataset/model lain, dataset dibuat ulang |
| prefix & `normalize_embeddings` di config | konfigurasi tidak sesuai aplikasi |
| `--sanity` 1: encode ulang sampel dokumen, cosine ≥ 0.999 dengan vektor tersimpan | baris tertukar, prefix salah, teks/model berbeda (**lolos pemeriksaan struktural**, jadi hanya terdeteksi di sini) |
| `--sanity` 2: judul dokumen sebagai kueri → dokumennya di top-3 (≥ 80%) | model atau embedding tidak diskriminatif |

`--sanity` adalah uji kewajaran, **bukan evaluasi kualitas**. Ambang 0.999 dan 80% adalah pilihan praktis. Aplikasi sendiri hanya menjalankan pemeriksaan struktural saat memuat.

## Ukuran berkas
`embeddings.npy` ≈ jumlah dokumen × 384 × 4 byte: **500 dokumen ≈ 0,73 MB**, 10.000 dokumen ≈ 14,6 MB. Pada skala capstone (200–500 artikel) berkas ini kecil dan
aman di repositori; tidak perlu float16 (tidak diterapkan).

## Beban instalasi (terukur dari metadata PyPI, 27 Okt 2026)
- `sentence-transformers` terbaru 6.2.0 (butuh Python ≥ 3.10) bergantung pada `torch`, `transformers`, `scikit-learn`, `scipy`, `huggingface-hub`.
- Wheel `torch` 2.14.1 untuk Linux x86_64 / Python 3.12 saja berukuran **529 MB**, dan di Linux ia menarik tambahan `nvidia-*` (cuDNN, NCCL, ...) serta `triton`.
- Saat saya mencoba resolusi `pip install --dry-run` di sandbox pengembangan, unduhan memenuhi disk (±3 GB bebas) dan gagal dengan "No space left on device".
- **Belum diukur/diverifikasi:** ukuran wheel PyTorch versi CPU-only (tersedia dari indeks PyTorch, tidak terjangkau dari sandbox), ukuran unduhan model E5-small,
  dan pemakaian RAM saat berjalan. Angka-angka ini menentukan kelayakan hosting gratis dan akan diverifikasi di Tahap 7; **jangan mengasumsikan hosting gratis cukup**.
- Konsekuensi desain: jalur BM25 tidak butuh `sentence-transformers`, dan aplikasi publik otomatis fallback ke BM25 dengan pemberitahuan bila library/model tidak ada.

## Teruji vs belum
| Teruji | Belum teruji |
|---|---|
| 156 tes (model palsu): kontrak encode, validasi, penulisan atomik, pemeriksa, fallback tanpa library | `sentence-transformers` (versi mana pun) dan model E5 asli |
| Python 3.12: streamlit 1.30.0/pandas 2.1.4/numpy 1.26.4 **dan** streamlit 1.66.1/pandas 3.1.0/numpy 2.5.4 di venv bersih | Python 3.10/3.11; `sentence-transformers>=2.7` (batas bawah hanya perkiraan) |
| `requirements.txt` memuat semua import pihak ketiga dan tidak memuat paket tak terpakai (dites otomatis) | Pemasangan penuh `requirements.txt` termasuk PyTorch di lingkungan bersih |
