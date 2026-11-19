# Sumber Data & Ketentuan Penggunaan

> Status dokumen: disusun pada Tahap 4 (27 Okt 2026). Semua isian di bawah berasal dari pencarian web pada tanggal itu
> dan **harus diverifikasi ulang oleh Anda** sebelum data dipublikasikan. Saya bukan penasihat hukum.

## Prinsip
1. **Hanya metadata** (judul, abstrak, kata kunci, penulis, tahun, DOI, nama jurnal, URL). Tidak ada teks penuh/PDF.
2. **Data nyata dan sintetis dipisah total**: folder `data/real/` vs `data/synthetic/`, artifacts `artifacts_real/` vs `artifacts/`,
   kolom `data_source`. `scripts.build_dataset` menolak baris yang tidak sesuai jenis build-nya.
3. Setiap baris nyata menyimpan asal (`Source_Note`: endpoint OAI, lisensi jurnal, tanggal ambil) dan tautan asli (URL/DOI).
4. Hasil pencarian selalu menautkan ke artikel asli; abstrak yang ditampilkan dipotong dengan tombol "baca lengkap" dan atribusi jurnal.

## Sumber awal (config/journals.csv)
| source_id | Jurnal | Lisensi (bukti) | Status endpoint OAI-PMH |
|---|---|---|---|
| `resti` | Jurnal RESTI (IAII) | CC BY 4.0 (situs jurnal; DOAJ mencatat CC BY, dicek 7 Nov 2026) | Situs & ±1.100 artikel terkonfirmasi (DOAJ/GARUDA); URL OAI pola standar OJS, **belum diuji langsung** |
| `ijccs` | IJCCS (UGM / IndoCEISS) | CC BY-SA 4.0 (halaman artikel) | URL **ditebak** dari pola OJS; belum diuji; artikel berbahasa Inggris sejak 2018 |
| `jnteti` | JNTETI (UGM) | **CC BY-SA 4.0** (footer situs journal.ugm.ac.id/v3/JNTETI, dicek 7 Nov 2026) | Situs terkonfirmasi (terbit id & en); URL OAI pola OJS, **belum diuji**; cakupan juga teknik elektro (filter topik wajib). CC BY-SA: turunan dibagikan dengan lisensi serupa |
| `jlk` | Jurnal Linguistik Komputasional (INACL) | CC BY-NC-SA (DOAJ) | **Situs resmi belum ditemukan.** Jangan memakai `jlk.fib.unand.ac.id` (itu jurnal *Linguistika Kultura*) |

Status `verified` hanya diberikan oleh `python -m scripts.check_sources --update-config` setelah Identify dan ListRecords berhasil.
Lisensi NC (JLK): pemakaian nonkomersial saja; aplikasi capstone ini harus tetap nonkomersial bila JLK dipakai.

## Ekspor Publish or Perish (`data/real/raw/data_pop.csv`), ditambahkan 7 Nov 2026
Sumber Crossref lewat Publish or Perish (QueryDate 2026-10-27), 1000 baris, kueri umum ML/AI/HCI, **berbahasa Inggris**.
Hanya 146 baris punya abstrak; setelah pembersihan menjadi 139 dokumen (`artifacts_real`). Kata kunci kosong di sumber, lalu diisi otomatis
dengan `scripts.derive_keywords` (ditandai di `Source_Note`). Penerbit beragam (IEEE, Springer, IGI Global, dll.); **lisensi per artikel belum diverifikasi**.
Berkas mentah di-gitignore. Tidak cocok sebagai dataset utama untuk topik "akademik Indonesia".

## GARUDA
Pencarian pada 27 Okt 2026 **tidak menemukan API atau OAI-PMH resmi GARUDA**; yang ditemukan hanya keterangan bahwa GARUDA adalah
indeks nasional di bawah Kemdiktisaintek (domain berpindah dari kemdikbud.go.id). Karena itu:
- GARUDA dipakai **manual** sebagai alat penemuan jurnal (cari jurnal bidang informatika/AI/NLP → catat situs OJS-nya → tambahkan ke `config/journals.csv`).
- Tidak ada scraping halaman GARUDA. Bila Anda menemukan API/izin resmi, tambahkan kolektor baru di `src/harvest.py`.

## Crossref
Dipakai untuk verifikasi dan pelengkap berdasarkan DOI (`scripts.crossref_enrich`), dengan `mailto` agar masuk "polite pool".
Hanya mengisi kolom kosong; judul yang tidak cocok ditandai untuk ditinjau manual. Periksa ketentuan penggunaan data Crossref
di dokumentasi resmi mereka sebelum mendistribusikan ulang hasil pengayaan.

## Sebelum memublikasikan dataset
- [ ] Lisensi tiap jurnal di `config/journals.csv` terisi dan sudah Anda baca di situs jurnal.
- [ ] Atribusi (nama jurnal + tautan asli) tampil di hasil pencarian (sudah).
- [ ] Putuskan apakah berkas CSV mentah boleh ikut repositori publik (default: **tidak**, lihat `.gitignore`).
- [ ] Cantumkan pernyataan data di README/laporan capstone (sumber, tanggal ambil, jumlah, batasan).
