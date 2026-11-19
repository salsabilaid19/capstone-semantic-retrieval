# Panduan Anotator

Anda menilai seberapa relevan sebuah artikel terhadap sebuah kueri pencarian. Anda **tidak** diberi tahu sistem mana yang menemukan artikel itu;
urutan baris diacak. Nilai hanya berdasarkan **judul, abstrak, dan kata kunci** yang tertera.

## Skala (kolom `relevance`)
| Nilai | Arti | Pertanyaan bantu |
|---|---|---|
| **2** | Sangat relevan | Apakah artikel ini inti pembahasannya sama dengan kebutuhan informasi di kueri? Pencari pasti ingin membacanya. |
| **1** | Sebagian relevan | Artikel membahas aspek, metode, atau konteks yang berkaitan, mungkin berguna, tetapi bukan inti kueri. |
| **0** | Tidak relevan | Topiknya berbeda, atau kata yang sama dipakai dalam arti lain. |

Kolom kosong = **belum dinilai** (bukan 0). Jangan mengosongkan baris yang sebenarnya sudah Anda nilai 0.

## Aturan
1. Bayangkan seorang mahasiswa/peneliti yang mengetik kueri itu; nilai berdasarkan kebutuhan informasinya, bukan kecocokan kata persis.
2. Jangan menilai kualitas artikel, tahun terbit, atau jurnalnya.
3. Jangan mencari artikel di internet atau melihat hasil sistem lain. Bila abstrak kurang jelas, nilai apa adanya dan tulis keraguan di `notes`.
4. Isi kolom `annotator` dengan nama/inisial Anda (atau tentukan saat konversi).
5. Kerjakan secara independen dari anotator lain; perbedaan diselesaikan belakangan lewat adjudikasi.
6. Jangan mengubah kolom `query_id`, `doc_id`, `query`.

## Ilustrasi format (BUKAN data)
```
query_id,query,doc_id,title,...,relevance,annotator,notes
Q001,<kueri>,DOC-xxxxxxxxxx,<judul artikel>,...,2,AB,
Q001,<kueri>,DOC-yyyyyyyyyy,<judul artikel>,...,0,AB,kata sama tapi konteks berbeda
```
