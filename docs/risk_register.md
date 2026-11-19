# Risk Register

| Risiko | Dampak | Probabilitas | Mitigasi | Penanggung jawab |
|---|---|---|---|---|
| Data tanpa abstrak | Dokumen terbuang, koleksi menyusut | Sedang | Filter otomatis; cek jumlah dokumen valid setelah pembersihan | Anggota data |
| Duplikasi data | Hasil ganda, metrik bias | Sedang | Dedup DOI lalu judul ternormalisasi | Anggota data |
| Colab berhenti/sesi habis | Embedding tidak selesai | Sedang | Simpan ke Drive; jalankan ulang; dataset ≤1.000 dokumen cepat; bisa pakai CPU | Anggota embedding |
| Artifact embedding hilang | Pencarian semantic tidak jalan | Rendah | Cadangkan `artifacts/` (Drive + flashdisk); BM25 tetap bisa dipakai | Ketua tim |
| Model belum terunduh saat demo | Semantic gagal dimuat | Sedang | Jalankan aplikasi dan satu pencarian semantic sebelum demo (cache); siapkan koneksi cadangan | Anggota aplikasi |
| Query expansion memperburuk hasil | Relevansi turun | Sedang | Evaluasi offline; tampilkan detail proses; bandingkan dengan tanpa QE | Anggota evaluasi |
| Aplikasi tidak berjalan saat demo | Demo gagal | Rendah–Sedang | Uji di laptop demo; siapkan screenshot/video cadangan; checklist pra-demo | Ketua tim |

Dampak/probabilitas: Rendah, Sedang, Tinggi. Kolom penanggung jawab diisi nama anggota tim.
