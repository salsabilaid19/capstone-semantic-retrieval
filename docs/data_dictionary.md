# Kamus Data

## Berkas masukan (CSV mentah)
Wajib: **Title**, **Abstract**. Opsional (nama kolom tidak peka huruf besar/kecil, alias Indonesia didukung):
Authors, Year (atau Date; tahun 4 digit diambil otomatis), DOI, URL, Keywords, Journal/Publication/Source,
Type, Language, **Data_Source** (`real` | `synthetic`), Source_Note.

## Berkas keluaran `artifacts/documents_processed.csv`
| Kolom | Isi |
|---|---|
| `doc_id` | ID **stabil** `DOC-<10 hex>`: hash DOI ternormalisasi, atau judul ternormalisasi bila tanpa DOI. Tidak bergantung urutan baris, jadi label relevansi tetap valid saat dataset dibuat ulang. Bila DOI ditambahkan belakangan pada artikel yang tadinya tanpa DOI, `doc_id` ikut berubah, sehingga label perlu dipetakan ulang. |
| `title`, `abstract`, `keywords`, `author` | Teks bersih (spasi dirapikan) |
| `year` | Bilangan bulat; kosong bila tidak diketahui |
| `journal` | Nama jurnal/sumber (terpisah dari `document_type`) |
| `doi` | DOI ternormalisasi huruf kecil, tanpa awalan URL |
| `language` | `id` / `en` / `unknown`. Dari kolom Language bila ada; selain itu **heuristik kata fungsi** (bukan pendeteksi bahasa penuh) |
| `document_type` | Tipe dari kolom Type, default `article` |
| `source_url` | URL asli; bila kosong dibentuk dari DOI (`https://doi.org/...`) |
| `data_source` | `real` / `synthetic` / `unspecified`. Nilai `real` hanya diterima bila tertulis di CSV atau diberikan secara eksplisit; sistem tidak memverifikasinya (verifikasi: Tahap 4) |
| `source_note` | Catatan asal/ketentuan penggunaan |
| `document_text` | `Judul: ... Abstrak: ... Kata kunci: ...` (masukan untuk embedding) |

## Aturan deduplikasi
DOI sama = duplikat. Judul ternormalisasi sama = duplikat bila salah satu tanpa DOI. Dua DOI berbeda dengan judul sama dipertahankan. Baris ber-DOI diprioritaskan.

## `artifacts/model_config.json`
`model_name`, prefix, `embedding_dim`, `num_documents`, `doc_ids_sha1` (hash urutan doc_id). Aplikasi menolak embedding yang jumlahnya, modelnya, norma vektornya, atau urutan dokumennya tidak cocok.
