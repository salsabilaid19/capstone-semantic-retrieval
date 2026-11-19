# Ringkasan Pengerjaan

## Fitur yang diselesaikan
- Adaptive QE tiga tingkat (No/Light/Strong) berbasis kualitas hasil awal (`src/query_expansion.py`, `src/retrieval.py`).
- Analisis evaluasi per keputusan Adaptive dan sebaran keputusan (`src/evaluation.py`); logika halaman evaluasi tanpa data palsu (`src/eval_view.py`).
- Penegakan aturan abstrak >= 200 karakter di validator (`src/validation.py`).
- UI multi-halaman baru: Dashboard, Document Search (+ bandingkan 4 metode), Evaluation, About & Methodology (`app.py`, `views/`, `src/ui_theme.py`, `src/charts.py`, `.streamlit/config.toml`, `assets/`).
- Skrip baru: `scripts/calibrate_adaptive.py`, `scripts/fix_short_abstracts.py`. Draf 30 kueri: `data/templates/evaluation_queries_DRAFT_corpus900.csv`.

## Bug diperbaiki
- Default `artifacts/` menunjuk data sintetis; kini `ARTIFACTS_DIR` > `artifacts_real` > `artifacts`.
- Default CLI evaluasi menunjuk data sintetis.
- Aturan abstrak 200 karakter tidak ditegakkan.
- Peringatan validator usang (Static=Adaptive pada kueri pendek); target jumlah dokumen 200-500 diperluas ke 200-1000.
- Selama pengerjaan (pada kode baru): Dashboard crash pada kode bahasa tak dikenal; logo crash bila dijalankan dari folder lain; dua grafik kosong di dalam tab. Semuanya sudah diperbaiki dan punya tes.

## File yang diubah/ditambah
Diubah: `src/query_expansion.py`, `retrieval.py`, `evaluation.py`, `validation.py`, `service.py`, `ui_common.py`, `app.py`, `requirements*.txt`, `README.md`, `PROGRESS.md`, docs (`panduan_evaluasi`, `manual_pengguna`, `checklist_publikasi`, `test_plan`), tes lama (`test_app`, `test_validation`, `test_service`, `test_embeddings`).
Ditambah: `views/*`, `src/ui_theme.py`, `charts.py`, `eval_view.py`, `scripts/calibrate_adaptive.py`, `scripts/fix_short_abstracts.py`, `assets/*`, `.streamlit/config.toml`, `docs/audit_final.md`, tes baru (`test_adaptive_qe`, `test_evaluation_decision`, `test_real_artifacts`, `test_scripts_baru`, `proxy_model.py`).
Dihapus: `pages/1_Mode_Eksperimen.py` (fungsinya dipindah ke Document Search > "Bandingkan 4 metode").
Tidak diubah: seluruh isi `artifacts_real/` dan `artifacts/`.

## Hasil pengujian
`python -m pytest -q`: **212 tes lulus** (baseline awal 162). Tes semantic memakai model palsu; kualitas retrieval e5 tidak diuji.
Verifikasi visual keempat halaman dilakukan lewat browser headless (Chromium) pada artefak nyata; pencarian semantic memakai model proksi uji, bukan e5.

## Keterbatasan
Lihat `docs/audit_final.md` bagian 4.

## Menjalankan dari awal
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```
