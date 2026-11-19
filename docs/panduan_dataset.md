# Panduan Membuat Dataset Paper Nyata

Semua perintah dijalankan di **komputer Anda** (sandbox pengembangan tidak punya akses ke penerbit/Crossref/Hugging Face).
Python 3.10+ dan `pip install -r requirements.txt`.

## 0. Siapkan identitas klien
Pakai email aktif agar penerbit/Crossref dapat menghubungi Anda bila ada masalah:
```powershell
$env:CONTACT_EMAIL = "anda@kampus.ac.id"
```

## 1. Periksa sumber (WAJIB sebelum panen)
```powershell
python -m scripts.check_sources --update-config
```
Contoh keluaran: `[OK] resti sampel=20 abstrak=100% kata_kunci=85% doi=100% <url>`.
- Sumber yang lolos ditandai `verified` dan `enabled=true`; URL yang bekerja disimpan.
- **Perhatikan `kata_kunci`**: jika rendah, Static/Adaptive QE akan jarang berfungsi pada sumber itu.
- Gagal? Buka situs jurnal di browser, cari tautan OAI di footer/"About", atau coba `<situs-jurnal>/oai?verb=Identify` sendiri,
  lalu perbaiki `oai_base_url` di `config/journals.csv`.

## 2. Menambah jurnal
Tambahkan satu baris di `config/journals.csv` (kolom dijelaskan di bawah). Cari jurnal lewat GARUDA/SINTA/DOAJ secara manual,
catat **lisensi** dari situs jurnal/DOAJ. Biarkan `enabled=false` sampai `check_sources` lolos.
| Kolom | Isi |
|---|---|
| `source_id` | ID pendek unik, huruf kecil |
| `name` | Nama jurnal (ditampilkan di hasil) |
| `oai_base_url` | Biasanya `https://situs/index.php/<path>/oai` |
| `oai_alt_urls` | Alternatif dipisah `\|` (mis. http vs https) |
| `license`, `license_evidence` | Lisensi dan dari mana Anda tahu |
| `verification` / `verified_on` | Diisi otomatis oleh `check_sources` |

## 3. Panen metadata
```powershell
python -m scripts.collect_oai --max-per-source 150
```
Opsi berguna: `--sources resti,ijccs`, `--from-date 2020-01-01`, `--no-topic-filter`, `--max-pages 2` (uji coba cepat).
Filter topik memakai `config/topic_terms.txt` (ubah sesuai kebutuhan). Hasil: `data/real/harvest/<source_id>.csv`.
Jeda default 1 detik/permintaan; **jangan** mengecilkan tanpa alasan.

## 4. (Opsional) Verifikasi/lengkapi dengan Crossref
```powershell
python -m scripts.crossref_enrich --input data/real/harvest/resti.csv --output data/real/harvest/resti.csv --mailto anda@kampus.ac.id
```
Kolom `Crossref_Title_Match` bernilai `mismatch(...)`? Tinjau barisnya manual, kemungkinan DOI salah.

## 5. Alternatif: impor CSV sendiri
```powershell
python -m scripts.import_csv --input paper_saya.csv --name manual_1 --data-source real --source-note "Disalin dari situs jurnal X, CC BY 4.0, 2026-10-27"
```

## 6. Bangun dataset (gabung, dedup, validasi)
```powershell
python -m scripts.build_dataset --kind real
```
Menulis `artifacts_real/documents_processed.csv`, `dataset_manifest.json`, dan `results/validation_real.md`.
**Jika ada error validasi, dataset tidak ditulis** (mis. DOI contoh, tanpa DOI/URL, tanpa catatan sumber, campuran real/sintetis).
Perbaiki sumbernya; `--force` hanya untuk keperluan debugging.

## 7. Embedding lalu jalankan aplikasi
Buka `notebooks/generate_embeddings.ipynb` di Colab, set `DATASET = 'real'`, jalankan, unduh `embeddings.npy` dan `model_config.json`
ke `artifacts_real/`. Lalu:
```powershell
$env:ARTIFACTS_DIR = "artifacts_real"; streamlit run app.py
```
Aplikasi menolak embedding yang tidak cocok dengan dokumen (hash `doc_id`).

## Yang tidak dijamin
- Akses OAI dapat diblokir atau berubah; hasil `check_sources` hanya valid pada saat dijalankan.
- Jumlah 200-500 artikel tidak dijamin: tergantung jumlah jurnal aktif dan filter topik. `validate` memberi peringatan bila di luar rentang.
- Deteksi bahasa (`language`) adalah heuristik sederhana; periksa distribusinya di laporan validasi.
