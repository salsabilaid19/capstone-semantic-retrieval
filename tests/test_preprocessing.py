"""Tes preprocessing: skema, doc_id stabil, tahun, dedup, penanda data sintetis."""
import pandas as pd
import pytest

from src.preprocessing import (
    STANDARD_COLUMNS, clean_dataset, detect_language, format_year, make_doc_id,
    normalize_doi, parse_year,
)


def _raw(rows):
    return pd.DataFrame(rows, columns=["Title", "Abstract", "Year", "DOI", "Publication", "Keywords"])


ROWS = [
    ["Deteksi Banjir dengan IoT", "Penelitian ini menggunakan sensor untuk banjir.", 2022, "10.1/a", "Jurnal A", "banjir; iot"],
    ["Klasifikasi Teks Berita", "Studi ini menerapkan klasifikasi pada berita.", "", "10.1/b", "Jurnal B", ""],
    ["Sistem Rekomendasi Buku", "Rekomendasi buku untuk perpustakaan.", "2020-05-03", "", "Jurnal C", "rekomendasi"],
]


def test_skema_dan_journal_terpisah_dari_document_type():
    df = clean_dataset(_raw(ROWS))
    assert list(df.columns) == STANDARD_COLUMNS + ["document_text"]
    assert df.loc[0, "journal"] == "Jurnal A"
    assert df.loc[0, "document_type"] == "article"
    assert df.loc[0, "doi"] == "10.1/a"


def test_doc_id_stabil_tidak_bergantung_urutan_atau_baris_lain():
    a = clean_dataset(_raw(ROWS))
    b = clean_dataset(_raw(ROWS[1:] + ROWS[:1]))        # urutan berubah
    c = clean_dataset(_raw(ROWS[2:]))                   # sebagian baris tak ada
    id_a = dict(zip(a.title, a.doc_id))
    id_b = dict(zip(b.title, b.doc_id))
    assert id_a == id_b
    assert id_a["Sistem Rekomendasi Buku"] == dict(zip(c.title, c.doc_id))["Sistem Rekomendasi Buku"]
    assert len(set(a.doc_id)) == 3


def test_doc_id_dari_doi_ternormalisasi():
    assert make_doc_id("https://doi.org/10.1/ABC", "x") == make_doc_id("10.1/abc", "judul lain")
    assert normalize_doi("doi: 10.1/X") == "10.1/x"


def test_tahun_tidak_pernah_float():
    df = clean_dataset(_raw(ROWS))
    assert df["year"].isna().sum() == 1                          # baris tanpa tahun
    assert int(df.loc[2, "year"]) == 2020                        # dari "2020-05-03"
    assert format_year(2021.0) == "2021" and format_year("2021.0") == "2021"
    assert format_year("") == "" and format_year(None) == ""
    assert parse_year("abc") is pd.NA


def test_dedup_doi_sama_dan_judul_sama_tanpa_doi():
    rows = ROWS + [
        ["Deteksi Banjir dengan IoT!", "Versi lain tanpa DOI.", 2022, "", "Jurnal A", ""],   # judul sama, tanpa DOI
        ["Judul Lain", "Abstrak lain.", 2021, "10.1/b", "Jurnal Z", ""],                    # DOI sama
    ]
    df = clean_dataset(_raw(rows))
    assert len(df) == 3


def test_dua_doi_berbeda_judul_sama_dipertahankan():
    rows = [["Judul Sama", "Abstrak satu.", 2020, "10.1/x", "J", ""],
            ["Judul Sama", "Abstrak dua.", 2021, "10.1/y", "J", ""]]
    assert len(clean_dataset(_raw(rows))) == 2


def test_baris_tanpa_judul_atau_abstrak_dibuang():
    rows = ROWS + [["", "Abstrak ada.", 2020, "10.1/q", "J", ""], ["Judul ada", "", 2020, "10.1/r", "J", ""]]
    assert len(clean_dataset(_raw(rows))) == 3


def test_data_source_default_dan_nilai_tidak_valid():
    df = clean_dataset(_raw(ROWS))
    assert set(df.data_source) == {"unspecified"}
    df2 = clean_dataset(_raw(ROWS), default_data_source="synthetic")
    assert set(df2.data_source) == {"synthetic"}
    raw = _raw(ROWS); raw["Data_Source"] = ["real", "synthetic", "asal-asalan"]
    assert clean_dataset(raw).data_source.tolist() == ["real", "synthetic", "unspecified"]
    with pytest.raises(ValueError):
        clean_dataset(_raw(ROWS), default_data_source="palsu")


def test_kolom_wajib_hilang():
    with pytest.raises(ValueError, match="Kolom wajib"):
        clean_dataset(pd.DataFrame({"Title": ["a"]}))


def test_deteksi_bahasa_heuristik():
    assert detect_language("Penelitian ini menggunakan metode untuk sistem") == "id"
    assert detect_language("This study is based on the results of the paper") == "en"
    assert detect_language("xyz") == "unknown"


# --- pembersihan markup (ditemukan pada data Publish or Perish / Crossref) ---
from src.preprocessing import strip_markup


def test_strip_markup_tag_entitas_dan_jats():
    assert strip_markup("<p>Halo   dunia</p>") == "Halo dunia"
    assert strip_markup("<jats:p>Isi <jats:italic>penting</jats:italic>.</jats:p>") == "Isi penting."
    assert strip_markup("<jats:title>Abstract</jats:title><jats:p>Isi.</jats:p>") == "Isi."
    assert strip_markup("Journal of AI &amp; Cloud") == "Journal of AI & Cloud"
    assert strip_markup("A &amp;amp; B") == "A & B"
    assert strip_markup(None) == "" and strip_markup(float("nan")) == ""


def test_strip_markup_tidak_merusak_teks_biasa():
    assert strip_markup("Nilai p < 0.05 dan x > 3") == "Nilai p < 0.05 dan x > 3"
    assert strip_markup("Deteksi banjir (IoT): studi kasus") == "Deteksi banjir (IoT): studi kasus"


def test_clean_dataset_membersihkan_markup_di_abstrak_dan_jurnal():
    df = clean_dataset(_raw([["Judul A &amp; B", "<p>Penelitian ini menggunakan metode.</p>", 2022, "10.1/z",
                              "Jurnal AI &amp; Cloud", ""]]))
    assert df.loc[0, "title"] == "Judul A & B"
    assert df.loc[0, "abstract"] == "Penelitian ini menggunakan metode."
    assert df.loc[0, "journal"] == "Jurnal AI & Cloud"
