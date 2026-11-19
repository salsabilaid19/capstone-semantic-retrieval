# Sistem Temu Kembali Dokumen Akademik Berbasis Semantic Retrieval dengan Adaptive Query Expansion

Proyek capstone: mesin pencari artikel jurnal ilmiah (IJCCS, JNTETI, Jurnal RESTI) yang membandingkan empat metode pencarian dan memilih
tingkat perluasan kueri dari **kualitas hasil pencarian awal**.

## Empat metode yang dibandingkan
1. **BM25**: pencocokan kata (lexical) atas judul + abstrak + kata kunci.
2. **Semantic Retrieval**: embedding `intfloat/multilingual-e5-small` (384 dimensi), diurutkan menurut cosine similarity.
3. **Semantic + Static Query Expansion**: selalu mencoba menambah hingga 2 istilah dari kata kunci top-3 hasil awal.
4. **Semantic + Adaptive Query Expansion**: tingkat ekspansi dipilih dari kualitas hasil awal.

## Dataset penelitian (`artifacts_real/`)
900 artikel: IJCCS 300, JNTETI 300, Jurnal RESTI 300; Inggris 616, Indonesia 284. Artefak utama: `dataset_manifest.json`, `documents_processed.csv`,
`embeddings.npy` (900 x 384), `model_config.json` (memuat hash `doc_id` untuk memastikan embedding cocok dengan dokumen).
Folder `artifacts/` hanya 13 dokumen **sintetis** untuk uji coba; tidak boleh dipakai untuk klaim penelitian.
**Catatan kualitas data:** 2 dokumen JNTETI (sampul jurnal) berabstrak < 200 karakter, melanggar ketentuan penelitian; lihat `docs/audit_final.md`.

## Mekanisme Adaptive QE (yang benar-benar diimplementasikan)
Pencarian awal dengan kueri asli, lalu hasilnya dinilai dari **skor seluruh korpus** (skor cosine e5 sangat terkompresi, sehingga dipakai z-score, bukan ambang mutlak):

| Indikator | Fungsi |
|---|---|
| z skor top-1, z rata-rata top-5 | seberapa kuat hasil teratas menonjol dari korpus |
| selisih top-1 vs top-2, jarak top-1 ke top-10 | satu dokumen unggul jelas, atau sepuluh teratas datar |
| dukungan kandidat istilah | istilah dari kata kunci 5 dokumen teratas dipakai hanya bila muncul pada >= 2 dokumen teratas |
| jumlah kata kueri | kueri >= 6 kata dianggap spesifik, tingkat turun satu |

Hasil: **No Expansion** (0 istilah), **Light Expansion** (maks. 1), **Strong Expansion** (maks. 3). Bila kandidat tidak meyakinkan, tingkat diturunkan (Strong -> Light -> No).
Istilah generik, duplikat, atau yang sudah ada di kueri dibuang. Kueri asli, kueri akhir, istilah tambahan, indikator, dan alasan keputusan ditampilkan.
Pada korpus < 20 dokumen (z-score tidak bermakna) dipakai aturan panjang kueri lama dan hal itu dinyatakan.

**Keterbatasan:** ini heuristik berbasis aturan, bukan pendeteksi ambiguitas yang sempurna. Ambang awal (`AdaptiveConfig` di `src/query_expansion.py`)
belum dikalibrasi pada kueri nyata dengan model e5 asli. Kemanfaatan ekspansi hanya dapat dinilai lewat evaluasi dengan penilaian relevansi manusia.

## Evaluasi
Metrik: **Precision@K, nDCG@K (gain 0/1/2), MRR@K, Unjudged@K**. Perbandingan 4 metode, analisis per panjang kueri, analisis per keputusan No/Light/Strong,
uji randomisasi berpasangan, CI bootstrap 95%, koreksi Holm. Dokumen tanpa label dianggap tidak relevan; pooling menjaga Unjudged@K rendah.
**Penilaian relevansi belum tersedia** (`data/evaluation_queries.csv` dan `data/relevance_labels.csv` belum ada), sehingga evaluasi (lewat CLI) belum dapat dijalankan,
bukan angka. Sistem tidak membuat label atau hasil palsu. Alur: `docs/panduan_evaluasi.md`, `docs/panduan_anotasi.md`.

## Teknologi
Python, Streamlit (multi-halaman), sentence-transformers (e5), rank-bm25, pandas, NumPy, Altair, pytest.

## Struktur folder
```
app.py                  # entri aplikasi (st.navigation, tema)
views/                  # dashboard.py, search.py, about.py
src/                    # retrieval, query_expansion (Adaptive QE), evaluation (skrip CLI), validation, ui_common, ui_theme, charts, service, ...
scripts/                # make_pool, pool_to_labels, agreement, check_embeddings, validate_dataset, calibrate_adaptive, fix_short_abstracts, ...
tests/                  # 212 tes (pytest)
artifacts_real/         # dataset penelitian 900 dokumen + embeddings
artifacts/              # 13 dokumen sintetis (uji coba)
data/                   # templat kueri/label, draf kueri, panduan anotasi
docs/                   # panduan dan laporan (audit_final.md, panduan_evaluasi.md, ...)
assets/  .streamlit/    # logo dan tema
```

## Instalasi
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```
Pertama kali memakai pencarian semantic, model e5 (ratusan MB) diunduh otomatis. Tanpa `sentence-transformers`, aplikasi tetap berjalan dengan BM25 dan memberi pemberitahuan.

## Menjalankan aplikasi
```bash
streamlit run app.py
```
Buka http://localhost:8501. Data: `ARTIFACTS_DIR` bila diatur; selain itu `artifacts_real/` bila ada; selain itu `artifacts/` (sintetis).

## Menjalankan pengujian
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
Tes semantic memakai model palsu (alur kode); kualitas retrieval e5 tidak diuji oleh suite ini.

## Perintah penting
```bash
python -m scripts.check_embeddings --artifacts-dir artifacts_real          # validasi struktural embedding
python -m scripts.validate_dataset --artifacts-dir artifacts_real --kind real
python -m scripts.calibrate_adaptive --artifacts-dir artifacts_real --queries data/evaluation_queries.csv
python -m scripts.make_pool --artifacts-dir artifacts_real --depth 10      # pool anotasi
python -m scripts.pool_to_labels --input data/annotation/pool_terisi.csv --annotator "Nama"
python -m src.evaluation --artifacts artifacts_real --k 10
python -m scripts.fix_short_abstracts --artifacts-dir artifacts_real       # dry-run: laporan abstrak < 200 karakter
```
