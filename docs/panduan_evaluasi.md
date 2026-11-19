# Panduan Evaluasi Penelitian

Evaluasi yang valid butuh **dataset nyata**, **kueri yang dirancang manusia**, dan **label relevansi yang dinilai manusia**.
Aplikasi ini tidak membuat label atau hasil apa pun secara otomatis, dan `python -m src.evaluation` menolak dataset sintetis
(kecuali `--allow-synthetic`, hanya untuk menguji alur, dengan banner "BUKAN temuan penelitian").

## Alur kerja
1. **Dataset nyata** siap (`docs/panduan_dataset.md`) dan embedding dibuat.
2. **Susun kueri**: salin `data/templates/evaluation_queries_template.csv` ke `data/evaluation_queries.csv`.
   Kolom: `query_id`, `query`, `query_type` (opsional), `notes` (opsional).
3. **Buat pool anotasi** (komputer Anda, butuh model e5 asli):
   `python -m scripts.make_pool --artifacts-dir artifacts_real --depth 10`
   Hasil: `data/annotation/pool.csv` (untuk anotator) dan `pool_provenance.csv` (milik Anda; jangan diberikan ke anotator).
4. **Anotasi** manual (`docs/panduan_anotasi.md`): isi kolom `relevance` 0/1/2. Disarankan dua anotator independen:
   `python -m scripts.agreement --a pool_A.csv --b pool_B.csv --out data/annotation/perbedaan.csv`, lalu adjudikasi perbedaan.
5. **Ubah ke label**: `python -m scripts.pool_to_labels --input data/annotation/pool_final.csv --annotator "Nama"`
6. **Evaluasi**: `python -m src.evaluation --artifacts artifacts_real --k 10`

## Merancang kueri (menentukan validitas)
- Jumlah: usahakan **>= 30 kueri yang dapat dievaluasi** (praktik IR umumnya puluhan, sekitar 25-50). Di bawah itu, uji signifikansi berdaya rendah.
- **Seimbangkan panjang**: minimal ~10 kueri 1-3 kata dan ~10 kueri >3 kata. Adaptive QE memutuskan dari kualitas hasil awal (kueri >= 6 kata
  diturunkan satu tingkat), sehingga kedua kelompok panjang perlu terwakili agar perilaku Adaptive terlihat pada kueri spesifik maupun umum.
- **Draf kueri**: `data/templates/evaluation_queries_DRAFT_corpus900.csv` (30 kueri untuk korpus 900 dokumen) harus ditinjau tim; itu bukan penilaian relevansi.
- Tulis kueri **sebelum melihat hasil sistem**, sebagai pencari nyata (mis. dari kebutuhan informasi mahasiswa/dosen), bukan dengan
  menyalin kata kunci artikel di dataset. Kueri yang diturunkan dari kata kunci dokumen menguntungkan BM25 dan QE (QE mengambil istilah dari kata kunci) secara tidak realistis.
- Sertakan kueri pendek yang ambigu dan kueri panjang yang spesifik; catat jenisnya di `query_type`.
- Pastikan tiap kueri punya dokumen relevan di dataset. Kueri tanpa dokumen relevan dikecualikan dari rata-rata (dan dilaporkan).

## Metrik dan asumsinya (tulis di laporan)
| Metrik | Definisi di kode |
|---|---|
| Precision@k | dokumen relevan (relevance >= 1) pada k teratas, dibagi k |
| nDCG@k | gain = relevance (0/1/2), diskon log2; DCG ideal dari dokumen **berlabel** saja |
| MRR@k | 1 / peringkat dokumen relevan pertama dalam k teratas, 0 bila tak ada |
| unjudged@k | persentase hasil top-k yang tak punya label; **dianggap tidak relevan** oleh metrik di atas |

`unjudged@k` yang tinggi (> 20% memicu peringatan; ambang praktis, bukan standar) berarti metrik bisa meremehkan metode itu.
Pooling dengan `depth >= k` membuat unjudged@k mendekati 0 untuk keempat metode, tetapi **metode baru yang tidak ikut pool akan dirugikan**
(batasan pooling): jika menambah metode, buat pool ulang.

## Statistik
Perbandingan berpasangan per kueri (keempat pasangan utama): BM25 vs Semantic, Semantic vs Static, Semantic vs Adaptive, Static vs Adaptive.
Uji randomisasi berpasangan (sign-flip, dua sisi), CI bootstrap 95% untuk selisih rata-rata, **koreksi Holm** di dalam tiap metrik.
`n_changed` = jumlah kueri tempat dua metode memakai kueri akhir yang berbeda: hanya di sana QE bisa berpengaruh. Bila `n_changed` kecil,
selisih rata-rata kecil bukan bukti QE tidak berguna; datanya memang tidak memadai.
Hasil per segmen panjang kueri ada di `evaluation_by_segment.csv`, termasuk `expanded_rate` (seberapa sering QE benar-benar memperluas kueri).

## Analisis per keputusan Adaptive QE
`evaluation_by_decision.csv` memisahkan kueri menurut keputusan Adaptive (No / Light / Strong) dan membandingkan tiap metode pada kueri yang sama
(`delta_vs_semantic_*`). `evaluation_decision_distribution.csv` memuat jumlah kueri per tingkat. Kelompok < 10 kueri ditandai `small_group` (indikatif).

## Keterbatasan Adaptive QE yang harus dinyatakan
Adaptive QE = **aturan berbasis kualitas hasil awal**: z-score skor top-1 dan rata-rata top-5 terhadap seluruh korpus, selisih/sebaran skor teratas,
dukungan kandidat istilah (muncul pada >= 2 dari 5 dokumen teratas), dan spesifisitas kueri (>= 6 kata turun satu tingkat). Ini **bukan** deteksi ambiguitas
yang sempurna. Ambang awal (`AdaptiveConfig`) bersifat heuristik dan harus dikalibrasi pada kueri nyata (`python -m scripts.calibrate_adaptive`)
**tanpa memakai label evaluasi yang akan dilaporkan**. Bila ambang diubah setelah pool dibuat, buat ulang pool agar Unjudged@K tetap rendah.
Kemanfaatan ekspansi hanya dapat dinyatakan dari hasil evaluasi bermakna statistik, bukan dari desain metode.

## Reprodusibilitas
`results/evaluation_run.json` mencatat: hash dataset (`doc_ids_sha1`), hash berkas kueri dan label, model, k, seed, jumlah kueri dievaluasi/dikecualikan,
dan hasil validasi. Sertakan berkas kueri, label, dan JSON ini dalam lampiran penelitian.

## Yang tidak dapat disimpulkan dari evaluasi ini
- Hasil hanya berlaku untuk dataset, kueri, dan anotator tersebut; bukan klaim umum tentang semantic retrieval.
- Dengan <30 kueri atau segmen kecil, "tidak signifikan" bukan berarti "tidak ada beda".
- Satu anotator = risiko bias; laporkan kesepakatan (kappa) bila memakai dua.
