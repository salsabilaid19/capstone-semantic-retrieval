# Audit Proyek & Rencana Perubahan (Tahap 1–2)

Tanggal audit: 26 Oktober 2026. Semua temuan di bawah diperiksa dengan menjalankan kode, bukan hanya membaca.

## A. Kondisi aktual

### Yang sudah berjalan (dipertahankan)
| Komponen | Status terverifikasi |
|---|---|
| `src/preprocessing.py` (pemetaan kolom PoP, pembersihan, dedup DOI/judul) | Berjalan pada 15 baris contoh, menghasilkan 13 dokumen |
| `src/retrieval.py` BM25 | Berjalan (lihat bug B1) |
| `src/retrieval.py` Semantic + `search_with_method` 4 metode | Belum diuji dengan model asli. Hanya diuji dengan model palsu oleh pengembang sebelumnya |
| `src/query_expansion.py` Static & Adaptive | Berjalan, 8 unit test lulus (`pytest`: 8 passed) |
| `src/evaluation.py` Precision/nDCG/MRR | Rumus benar. Belum ada file kueri & label sama sekali |
| `app.py` Streamlit | Hanya lolos kompilasi. Belum diuji di browser |
| Notebook Colab embedding | Sintaks valid, belum pernah dijalankan |

### Yang belum ada
- `artifacts/embeddings.npy` (di-.gitignore). Tanpa file ini aplikasi tidak bisa jalan sama sekali.
- Dataset nyata. `data/documents_raw.csv` berisi 15 baris **fiktif** (DOI `10.1000/sample.*`).
- `data/evaluation_queries.csv` dan `data/relevance_labels.csv`.
- Skrip pengumpulan data, validasi dataset, deployment, dan pemisahan UI publik vs eksperimen.

### Bug dan risiko yang terkonfirmasi
| ID | Temuan | Bukti | Dampak |
|---|---|---|---|
| B1 | Tokenizer BM25 hanya `lower().split()`. Tanda baca menempel ("banjir." ≠ "banjir") dan label template "Judul:/Abstrak:" ikut menjadi token | Kueri `banjir` mencocokkan 4 dari 5 dokumen yang memuat kata itu | Skor BM25 bias turun. Pembanding yang tidak adil terhadap metode semantic |
| B2 | BM25 mewajibkan `embeddings.npy` karena memakai `load_artifacts` yang sama | `FileNotFoundError` saat embeddings belum ada | BM25 tidak bisa dipakai mandiri. Pengguna tak bisa menguji tanpa embedding |
| B3 | Kolom `year` terbaca sebagai float jika ada tahun kosong | UI menampilkan `Tahun: 2021.0` | Tampilan salah pada data nyata (tahun sering kosong) |
| B4 | `doc_id` dibuat urut (`DOC0001…`) saat pembersihan | Artikel yang sama mendapat `DOC0002` lalu `DOC0001` setelah satu baris dibuang | **Label relevansi menjadi salah target** bila dataset dibuat ulang. Ini ancaman validitas evaluasi |
| B5 | Nama jurnal disimpan di `document_type` (fallback dari kolom Publication) | Kolom hanya berisi "Article"/"Conference paper" | Syarat UI "sumber jurnal" tidak terpenuhi. DOI tidak disimpan di data akhir |
| B6 | Tidak ada penanda asal data (nyata vs sintetis) dan tidak ada info lisensi/sumber | Tidak ada kolomnya | Melanggar syarat pemisahan data sintetis vs nyata |
| B7 | `muat_bm25(_documents_df)` memakai argumen berawalan `_` sehingga tidak ikut hash cache | Baca kode | Indeks BM25 tidak dibangun ulang bila data berubah dalam satu sesi server |
| B8 | Tidak ada pemeriksaan kecocokan embedding dengan model/dokumen selain jumlah baris | `load_artifacts` hanya membandingkan jumlah | Embedding dari data/model berbeda bisa lolos diam-diam |

### Risiko metodologis (bukan bug kode)
- **Adaptive QE hanya berbasis jumlah kata (1–3 kata → ekspansi).** Itu heuristik panjang kueri, bukan deteksi ambiguitas atau kualitas hasil awal. Harus dinyatakan jelas di dokumentasi dan di UI.
- **Static/Adaptive QE bergantung pada kolom `keywords`.** Pada data nyata (terutama OAI-PMH), kata kunci sering kosong atau tercampur bahasa. Jika kosong, ekspansi tidak terjadi dan metode menjadi identik dengan Semantic. Ini harus dilaporkan sebagai statistik per dataset.
- **Evaluasi rawan bias pooling.** Jika label hanya dibuat untuk hasil satu metode, metode lain dirugikan. Perlu pool gabungan dari keempat metode.

### Batas lingkungan kerja ini (jujur)
Sandbox ini hanya dapat menjangkau PyPI. Hugging Face, Crossref, GARUDA, dan endpoint OAI-PMH jurnal **tidak terjangkau**. Akibatnya:
- Dataset nyata **tidak bisa dikumpulkan di sini**. Skrip akan diuji dengan respons rekaman (fixture) yang jelas diberi label, dan pengambilan sungguhan harus dijalankan di komputer Anda.
- Model `multilingual-e5-small` asli tidak bisa diunduh di sini. Kompatibilitas embedding diuji dengan pemeriksaan struktural, dan uji end-to-end dengan model asli harus dijalankan di komputer Anda atau Colab.

## B. Rencana perubahan (urutan pengerjaan)

**Tahap 3A: Perbaikan inti** (B1–B8)
- Tokenizer BM25 baru (regex kata, buang label template, stopword Indonesia kecil; tanpa stemmer, dicatat sebagai keterbatasan).
- Pisahkan `load_documents()` dan `load_embeddings()`; BM25 jalan tanpa embedding.
- `doc_id` stabil: berbasis DOI ternormalisasi, fallback hash judul ternormalisasi.
- Skema baru: tambah `journal`, `doi`, `language`, `data_source` (`real`/`synthetic`), `source_note`. Tahun bertipe aman.
- `embedding_meta` di `model_config.json`: nama model, prefix, dimensi, jumlah dokumen, hash daftar `doc_id`. Divalidasi saat load.

**Tahap 3B: UI**
- `app.py` = pencarian publik sederhana (satu kotak cari, hasil: judul, abstrak, tahun, jurnal, tautan; banner jika data sintetis).
- `pages/2_Mode_Eksperimen.py` = bandingkan 4 metode berdampingan + detail ekspansi + catatan keterbatasan Adaptive QE.

**Tahap 4: Dataset**
- `scripts/collect_oai.py` (OAI-PMH, OJS `oai_dc`, daftar jurnal di konfigurasi), `scripts/crossref_enrich.py` (verifikasi/pelengkap), `scripts/validate_dataset.py`, `scripts/import_csv.py` (jalur alternatif).
- Pemisahan folder `data/real/` dan `data/synthetic/`; `data/DATA_SOURCES.md` untuk ketentuan penggunaan per sumber.
- GARUDA: status API resmi harus diverifikasi saat ada akses jaringan. Tidak diasumsikan ada.

**Tahap 5: Evaluasi**
- Templat kueri & label (kolom `annotator`, `label_source`), skrip pembuat pool dari 4 metode untuk dianotasi manual, uji signifikansi berpasangan. Tidak ada label buatan.

**Tahap 6: Uji & kompatibilitas**
- Tes baru untuk semua perbaikan, skrip pemeriksa kompatibilitas embedding, uji UI dengan `streamlit.testing`.

**Tahap 7: Dokumentasi & deployment**
- Panduan instalasi/penggunaan/evaluasi. Opsi hosting (mis. Hugging Face Spaces, Streamlit Community Cloud) dibandingkan setelah spesifikasi terbarunya diverifikasi lewat pencarian. Tidak ada klaim "gratis pasti cukup".
