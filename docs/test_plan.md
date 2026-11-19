# Test Plan

Status awal: **Belum diuji** (isi menjadi Lulus/Gagal saat pengujian). Bukti: tangkapan layar atau keluaran terminal.

| ID | Test case | Input | Expected result | Status | Bukti yang diambil |
|---|---|---|---|---|---|
| TC01 | Aplikasi berjalan | `streamlit run app.py` | Halaman terbuka di localhost:8501 dengan header dan sidebar | Belum diuji | Screenshot halaman |
| TC02 | Query kosong | Klik Cari tanpa isi | Muncul `st.warning`, tanpa error | Belum diuji | Screenshot |
| TC03 | BM25 kueri valid | `banjir`, BM25 | Daftar top-k terurut skor menurun | Belum diuji | Screenshot hasil |
| TC04 | Semantic | `banjir`, Semantic | Top-k tampil | Belum diuji | Screenshot |
| TC05 | Static QE | `banjir`, Static + detail aktif | Kueri akhir memuat maksimal 2 istilah tambahan | Belum diuji | Screenshot detail proses |
| TC06 | Adaptive kueri pendek | `banjir`, Adaptive | Tingkat (No/Light/Strong) sesuai indikator; alasan tampil | Belum diuji | Screenshot detail |
| TC07 | Adaptive kueri panjang | `sistem deteksi banjir menggunakan sensor IoT` | Kueri >= 6 kata diturunkan satu tingkat; alasan tampil | Belum diuji | Screenshot detail |
| TC08 | Hasil tidak ditemukan | Kueri tanpa kecocokan (BM25) | Muncul `st.info` | Belum diuji | Screenshot |
| TC09 | Artifact hilang | Pindahkan `embeddings.npy` sementara | `st.error` ramah, tanpa traceback | Belum diuji | Screenshot |
| TC10 | Link sumber | Dokumen dengan/tanpa URL | Link hanya muncul bila URL ada | Belum diuji | Screenshot |
| TC11 | Preprocessing | CSV dengan duplikat dan abstrak kosong | Duplikat dan baris tanpa abstrak terhapus | Belum diuji | Keluaran notebook/terminal |
| TC12 | Unit test | `pytest` | 8 test lulus | Lulus di lingkungan pengembang | Keluaran terminal |
| TC13 | Evaluasi offline | `python -m src.evaluation` | Dua file CSV hasil di `results/` | Belum diuji | Isi CSV + terminal |
