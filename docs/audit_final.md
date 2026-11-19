# Audit dan Status Akhir Proyek

Tanggal: 9 November 2026. Audit dilakukan atas ZIP yang diserahkan; dataset dan embedding asli tidak diubah.

## 1. Kondisi sebelum perubahan
| Area | Temuan |
|---|---|
| Dataset/embedding | Sesuai ketentuan: 900 dokumen (300/300/300), EN 616 / ID 284, e5-small, dim 384, hash doc_id cocok, tanpa NaN, norma 1 |
| Aturan abstrak >= 200 karakter | Tidak ditegakkan di kode; 2 dokumen melanggar |
| Default artifacts | `artifacts/` (13 dokumen sintetis, tanpa embedding) sehingga demo default bukan data penelitian |
| Adaptive QE | Aturan jumlah kata (<= 3 kata diperluas), biner; tidak menilai kualitas hasil |
| Evaluasi | Metrik, CI, uji signifikansi ada; tanpa analisis per keputusan; berkas kueri/label relevansi tidak ada |
| UI | Pencarian + Mode Eksperimen saja |
| Dokumentasi | `PROGRESS.md`/README usang (139 dokumen, tanpa embedding) |
| Tes | 162 lulus |

## 2. Temuan kualitas data (tidak diubah)
- **2 dokumen berabstrak < 200 karakter**: `DOC-26c0fe0352` ("Halaman Depan", 20 karakter) dan `DOC-35a5ce65cf` ("Halaman Belakang", 23 karakter), keduanya JNTETI 2020, sampul jurnal. **Keputusan peneliti diperlukan** (hapus -> 898 dokumen; ganti dengan 2 artikel JNTETI; atau pertahankan dengan catatan). Alat: `python -m scripts.fix_short_abstracts` (dry-run; `--apply --out` menulis salinan baru).
- 68 abstrak IJCCS berawalan "Abstrak"/"Abstract" menempel (artefak scraping); sudah ada di teks yang di-embed. UI menyembunyikannya hanya pada tampilan.
- Heuristik kasar menunjukkan ~27 dokumen (23 berlabel `en`, 4 berlabel `id`) yang bahasanya tampak tidak cocok dengan label; periksa sumber label bahasa.
- 2 rekaman IJCCS (2013) berteks identik dengan DOI berbeda (vektor embedding identik).
- Kata kunci tersedia pada 90% dokumen (sumber istilah ekspansi).

## 3. Perubahan
Lihat `RINGKASAN_PENGERJAAN.md`.

## 4. Keterbatasan yang diketahui
1. **Model e5 asli tidak dapat diunduh di lingkungan pengembangan**; jalur semantic teruji dengan model palsu dan model proksi uji (`tests/proxy_model.py`, bukan e5). Perilaku Adaptive QE dengan e5 asli pada kueri nyata belum diamati.
2. **Ambang Adaptive QE belum dikalibrasi** pada kueri nyata (nilai awal dari skala skor antar-dokumen). Jalur Strong teruji lewat skor terkontrol, bukan kueri nyata.
3. **Tidak ada penilaian relevansi**: evaluasi resmi belum dapat dijalankan; tidak ada angka hasil penelitian. Langkah: `docs/panduan_evaluasi.md`.
4. Kandidat istilah ekspansi bergantung pada kata kunci dokumen (campuran Indonesia/Inggris, sebagian frasa panjang); risiko query drift pada hasil awal yang lemah dikurangi, bukan dihilangkan, oleh syarat dukungan >= 2 dokumen.
5. Perubahan Adaptive QE mengubah perilaku metode ke-4 dibanding versi lama (kini bisa berbeda dari Static pada kueri pendek). Bila pool anotasi sudah ada dari versi lama, buat ulang.
