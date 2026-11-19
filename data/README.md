# Folder data

- `synthetic/` : data CONTOH FIKTIF untuk uji coba aplikasi (`sample_synthetic.csv`). Bukan artikel nyata.
- `real/harvest/` : hasil `scripts.collect_oai` (satu CSV per jurnal; tidak ikut git).
- `real/imports/` : hasil `scripts.import_csv` dengan `--data-source real`.
- `real/documents_raw.csv`, `synthetic/documents_raw.csv` : gabungan mentah yang dibuat `scripts.build_dataset` (jangan diedit tangan).
- `DATA_SOURCES.md` : sumber, lisensi, dan ketentuan penggunaan. Baca sebelum memublikasikan.

Cara membuat dataset nyata: `docs/panduan_dataset.md`. Kamus kolom: `docs/data_dictionary.md`.
