# Checklist Publikasi

- [ ] **Lisensi data**: dataset `artifacts_real` berasal dari ekspor Publish or Perish/Crossref berisi karya banyak penerbit (IEEE, Springer, Elsevier, dll.). Lisensi per artikel **belum diverifikasi**. Metadata Crossref umumnya dapat dipakai bebas, tetapi **abstrak** bisa tetap berhak cipta penerbit: periksa sebelum menaruh `documents_processed.csv` di repositori publik. Alternatif aman: publikasikan kode saja dan minta pengguna membuat dataset sendiri.
- [ ] Atribusi: hasil pencarian menampilkan jurnal dan tautan DOI asli (sudah).
- [ ] Pernyataan data di laporan: sumber, tanggal ambil (27 Okt 2026), jumlah (139 dokumen), bahasa Inggris, topik ML/AI/HCI, 100% kata kunci diturunkan otomatis (TF-IDF), bukan dari penulis.
- [ ] Pernyataan keterbatasan Adaptive QE: heuristik berbasis kualitas hasil awal dengan ambang yang dikalibrasi, bukan deteksi ambiguitas sempurna.
- [ ] Pernyataan bahwa tes otomatis memakai model palsu; evaluasi sungguhan memakai model e5 asli dan label manusia.
- [ ] Jangan klaim "dokumen akademik Indonesia" bila data masih bahasa Inggris.
- [ ] Tidak ada rahasia/API key dalam repositori.
