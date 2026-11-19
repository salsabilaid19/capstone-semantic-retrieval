# Panduan Deployment

> Angka hosting di bawah berasal dari pencarian web pada **6 Nov 2026** (dokumen resmi Hugging Face dan Streamlit, serta
> artikel pihak ketiga). Batas layanan sering berubah: **cek ulang di dokumentasi resmi sebelum memutuskan.**
> `Dockerfile` belum pernah dibangun dan deployment belum pernah dicoba (lingkungan pengembangan tanpa Docker dan tanpa akses ke host).

## Fakta yang memengaruhi keputusan
| Host | Fakta (sumber & tanggal di atas) | Akibat untuk aplikasi ini |
|---|---|---|
| Streamlit Community Cloud | Gratis, butuh GitHub. Sumber daya bersama: sekitar 0,078-2 core CPU, **690 MB-2,7 GB RAM** (angka Feb 2024), tidur setelah 12 jam tanpa lalu lintas | RAM sempit untuk PyTorch + model e5 (belum diukur). **BM25 aman; semantic berisiko** |
| Hugging Face Spaces | CPU Basic: 2 vCPU, 16 GB RAM, 50 GB disk tidak permanen, tidur ±48 jam bila tidak dipakai. **Space Docker/Gradio berbasis komputasi kini butuh paket berbayar** (PRO, ±US$9/bulan, menurut dokumen HF dan satu artikel pihak ketiga); hanya Space Static yang gratis | Cukup untuk semantic penuh, tetapi tidak gratis untuk akun biasa. Tidak ada lagi opsi SDK Streamlit; Streamlit berjalan lewat Docker |
| Render (free) | Menurut satu artikel pihak ketiga: tidur setelah 15 menit, penyimpanan ephemeral | Belum diverifikasi ke dokumen resmi; RAM tidak diketahui |

## Pilihan arsitektur
**A. Demo publik ringan (disarankan untuk capstone yang butuh tautan gratis): Streamlit Community Cloud + BM25.**
1. Buat repositori GitHub. Commit `artifacts_real/documents_processed.csv` dan `dataset_manifest.json` **hanya bila lisensi penerbit mengizinkan** (lihat `docs/checklist_publikasi.md`).
2. Di cabang `deploy-lite`, ganti isi `requirements.txt` dengan `requirements-lite.txt`, dan atur variabel `ARTIFACTS_DIR=artifacts_real` di Secrets/Environment aplikasi.
3. Deploy `app.py`. Tanpa `embeddings.npy` dan tanpa library semantic, aplikasi menampilkan peringatan bahwa BM25 yang berjalan (perilaku sudah dites).
4. **Konsekuensi jujur:** demo ini TIDAK memperlihatkan Semantic Retrieval maupun Adaptive QE, yaitu inti judul penelitian. Katakan begitu di halaman demo dan laporan.

**B. Semantic penuh: Docker (HF Docker Space berbayar, VPS, atau Render berbayar).**
1. Buat embedding lebih dulu di komputer/Colab (`docs/panduan_embedding.md`), commit `embeddings.npy` (±0,2 MB untuk 139 dokumen) dan `model_config.json`.
2. Pakai `Dockerfile` di repo (port 7860). Untuk HF Docker Space, tambahkan YAML di `README.md` Space: `sdk: docker`, `app_port: 7860`.
3. Pantau RAM saat `streamlit run` pertama; model e5-small + PyTorch + Streamlit **belum diukur**.

**C. Demo langsung dari laptop** (paling aman untuk sidang): `streamlit run app.py` setelah model tersimpan di cache. Tidak perlu hosting.

## Pemecahan masalah
| Gejala | Penyebab umum | Tindakan |
|---|---|---|
| Aplikasi selalu BM25 dengan peringatan | `embeddings.npy` hilang/tidak cocok, atau library tidak terpasang | `python -m scripts.check_embeddings --artifacts-dir artifacts_real` |
| Build gagal kehabisan memori/disk | PyTorch versi CUDA terunduh | Pakai indeks CPU (lihat Dockerfile) atau opsi A |
| Aplikasi lambat pertama kali dibuka | Host tidur / model diunduh | Buka sekali sebelum demo |
