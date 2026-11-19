# Manual Pengguna

## Membuka aplikasi
1. Aktifkan virtual environment, lalu jalankan `streamlit run app.py` di folder proyek.
2. Browser terbuka di http://localhost:8501 (jika tidak, buka manual).

## Memilih metode (sidebar)
- **BM25**: pencocokan kata kunci.
- **Semantic Retrieval**: pencarian berdasarkan makna.
- **Semantic + Static QE**: kueri selalu ditambah hingga 2 istilah dari kata kunci top-3 hasil awal.
- **Semantic + Adaptive QE**: tingkat ekspansi (No / Light / Strong) dipilih dari kualitas hasil pencarian awal; alasan dan indikatornya ditampilkan.

Atur jumlah hasil (5–20) dan centang opsi detail proses/skor bila perlu.

## Memasukkan kueri
Ketik topik pada kolom input, lalu klik **Cari Dokumen**. Kueri kosong menampilkan peringatan.

## Membaca hasil
Tiap hasil menampilkan peringkat, judul, tahun/tipe/penulis, skor (jika diaktifkan), abstrak maksimal 350 karakter, kata kunci, dan tautan **Buka Sumber Dokumen** (hanya jika URL tersedia).
Aktifkan "Tampilkan detail proses pencarian" untuk melihat kueri asli, kueri akhir, istilah expansion, dan alasan keputusan.

## Contoh kueri
- Pendek: `banjir`, `analisis sentimen`, `temu balik informasi`
- Panjang: `sistem deteksi banjir menggunakan sensor IoT`

## Static QE vs Adaptive QE
- **Static QE** selalu mencoba ekspansi untuk semua kueri.
- **Adaptive QE** menilai skor hasil awal terhadap seluruh korpus lalu memilih No (0 istilah), Light (maks. 1), atau Strong (maks. 3) Expansion; sistem tidak mendeteksi ambiguitas secara sempurna.
Jika tidak ada kandidat istilah valid, kueri asli dipakai dan alasannya ditampilkan.

## Troubleshooting
| Masalah | Solusi |
|---|---|
| "Data belum tersedia" | Salin `documents_processed.csv` dan `embeddings.npy` ke folder `artifacts/` (lihat README). |
| "Data tidak konsisten" | Jumlah dokumen dan embedding berbeda; buat ulang embedding dari dataset yang sama. |
| Pencarian pertama lambat/gagal | Model sedang diunduh; pastikan internet aktif sekali lalu coba lagi. |
| `streamlit` tidak dikenali | Aktifkan virtual environment dan jalankan `pip install -r requirements.txt`. |
| Port 8501 terpakai | `streamlit run app.py --server.port 8502` |


## Halaman aplikasi
Dashboard (ringkasan dataset), Document Search (satu metode atau bandingkan 4 metode), Evaluation (metrik; tampil bila ada hasil evaluasi nyata), About & Methodology.
