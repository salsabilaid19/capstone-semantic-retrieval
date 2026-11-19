"""Tes validasi dataset: pemisahan real/sintetis dan keterverifikasian."""
import pandas as pd

from src.preprocessing import clean_dataset
from src.validation import report_markdown, validate_dataset


def make(rows, source="real", note="OAI-PMH x"):
    raw = pd.DataFrame(rows, columns=["Title", "Abstract", "Year", "DOI", "URL", "Keywords", "Publication"])
    raw["Data_Source"], raw["Source_Note"] = source, note
    return clean_dataset(raw)


PAD = " Metode, data, dan hasil dijelaskan secara rinci agar abstrak memenuhi panjang minimal dua ratus karakter yang ditetapkan."
GOOD = [["Deteksi Hoaks", "Penelitian ini mendeteksi hoaks menggunakan klasifikasi teks pada berita daring Indonesia." + PAD, 2022,
         "10.5555/a1", "https://j.id/1", "hoaks; nlp", "Jurnal A"],
        ["Sistem Rekomendasi", "Studi ini membangun sistem rekomendasi untuk perpustakaan dengan filtering kolaboratif." + PAD, 2021,
         "10.5555/a2", "https://j.id/2", "rekomendasi", "Jurnal B"]]


def test_dataset_real_baik_lolos_dengan_peringatan_jumlah():
    rep = validate_dataset(make(GOOD), expected_kind="real")
    assert rep["errors"] == []
    assert any("target awal" in w for w in rep["warnings"])
    assert rep["stats"]["persen_doi"] == 100.0 and rep["stats"]["rentang_tahun"] == [2021, 2022]


def test_real_dengan_doi_contoh_atau_fixture_ditolak():
    rows = [GOOD[0][:3] + ["10.1000/sample.001"] + GOOD[0][4:], GOOD[1]]
    assert any("sintetis" in e for e in validate_dataset(make(rows))["errors"])
    rows = [GOOD[0][:3] + ["10.99999/fixture.3"] + GOOD[0][4:], GOOD[1]]
    assert any("sintetis" in e for e in validate_dataset(make(rows))["errors"])


def test_real_tanpa_doi_dan_url_tidak_terverifikasi():
    rows = [["Tanpa Sumber", "Abstrak yang cukup panjang untuk lolos ambang pendek karakter di validasi dataset ini.",
             2020, "", "", "k", "Jurnal A"], GOOD[1]]
    assert any("tidak dapat diverifikasi" in e for e in validate_dataset(make(rows))["errors"])


def test_real_tanpa_source_note_ditolak():
    assert any("source_note" in e for e in validate_dataset(make(GOOD, note=""))["errors"])


def test_campuran_real_dan_sintetis_ditolak():
    df = pd.concat([make(GOOD[:1], "real"), make(GOOD[1:], "synthetic")], ignore_index=True)
    assert any("mencampur" in e for e in validate_dataset(df)["errors"])


def test_jenis_tidak_sesuai_harapan():
    assert any("diharapkan 'real'" in e for e in validate_dataset(make(GOOD, "synthetic"), expected_kind="real")["errors"])


def test_sintetis_hanya_peringatan():
    rep = validate_dataset(make(GOOD, "synthetic", ""), expected_kind="synthetic")
    assert rep["errors"] == [] and any("SINTETIS" in w for w in rep["warnings"])


def test_peringatan_kualitas():
    rows = [["Judul A", "pendek", 1950, "10.5555/z1", "", "", ""], ["Judul B", "lagi pendek", 2022, "doi-ngawur", "u", "", "J"]]
    w = " ".join(validate_dataset(make(rows, "synthetic"))["warnings"])
    assert "kata kunci" in w and "tanpa nama jurnal" in w and "abstrak sangat pendek" in w
    assert "di luar 1990" in w and "tidak lazim" in w


def test_duplikat_dan_dataset_kosong():
    df = make(GOOD)
    dup = pd.concat([df, df.iloc[:1]], ignore_index=True)
    errs = " ".join(validate_dataset(dup)["errors"])
    assert "doc_id duplikat" in errs and "DOI duplikat" in errs
    assert validate_dataset(df.iloc[0:0])["errors"] == ["Dataset kosong."]


def test_report_markdown():
    md = report_markdown("x", validate_dataset(make(GOOD, note="")))
    assert "GAGAL" in md and "## Error" in md and "jumlah_dokumen" in md


def test_abstrak_kurang_dari_200_karakter_error_pada_data_real_dan_peringatan_pada_sintetis():
    pendek = [["Halaman Depan", "Halaman Depan JNTETI", 2020, "10.5555/z1", "https://j.id/9", "", "Jurnal C"]] + GOOD
    rep = validate_dataset(make(pendek), expected_kind="real")
    assert any("< 200 karakter" in e for e in rep["errors"]) and rep["stats"]["abstrak_kurang_dari_200"] == 1
    assert len(rep["stats"]["doc_id_abstrak_kurang_dari_200"]) == 1
    sintetis = validate_dataset(make(pendek, source="synthetic"), expected_kind="synthetic")
    assert not any("< 200 karakter" in e for e in sintetis["errors"])
    assert any("< 200 karakter" in w for w in sintetis["warnings"])
