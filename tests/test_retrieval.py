"""Tes retrieval: tokenizer BM25, BM25 mandiri, validasi embedding, pipeline 4 metode (model PALSU)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import clean_dataset
from fakes import FakeModel, make_artifacts
from src.retrieval import (
    ALL_METHODS, MODEL_NAME, bm25_search, build_bm25_index, load_artifacts, load_documents,
    load_embeddings, search_with_method, tokenize,
)

RAW = pd.DataFrame({
    "Title": ["Sistem Peringatan Dini Banjir", "Prediksi Curah Hujan", "Analisis Sentimen Ulasan", "Deteksi Hoaks Berita"],
    "Abstract": ["Sensor air untuk banjir.", "Jaringan saraf untuk hujan, banjir.", "Naive Bayes untuk sentimen.", "Klasifikasi berita hoaks."],
    "Year": [2022, "", 2020, 2021],
    "DOI": ["10.1/a", "10.1/b", "10.1/c", ""],
    "Publication": ["J1", "J2", "J3", "J4"],
    "Keywords": ["banjir; sensor; iot", "hujan; jaringan saraf", "sentimen; naive bayes", "hoaks; berita"],
})


@pytest.fixture
def art(tmp_path):
    df = make_artifacts(tmp_path)
    emb = np.load(tmp_path / "embeddings.npy")
    return tmp_path, df, emb


# ---- B1 tokenizer ----
def test_tokenizer_membuang_tanda_baca_dan_stopword():
    assert tokenize("Banjir, sensor. IoT dan yang!") == ["banjir", "sensor", "iot"]
    assert tokenize("dan yang") == ["dan", "yang"]       # kueri semua stopword tidak boleh kosong
    assert tokenize(None) == []


def test_bm25_mencocokkan_kata_bertanda_baca():
    # Korpus 8 dokumen: BM25Okapi memberi IDF=0 pada istilah yang muncul di tepat separuh dokumen
    # (artefak korpus mini), jadi 'banjir' dibuat muncul di 2 dari 8 dokumen.
    extra = pd.DataFrame({
        "Title": [f"Topik Lain {i}" for i in range(4)],
        "Abstract": [f"Pembahasan {w}." for w in ("robot", "citra", "suara", "grafik")],
        "Year": 2020, "DOI": [f"10.1/x{i}" for i in range(4)], "Publication": "J", "Keywords": "x"})
    df = clean_dataset(pd.concat([RAW, extra], ignore_index=True))
    r = bm25_search("banjir", df, build_bm25_index(df), 10)
    assert set(r[r.score > 0].title) == {"Sistem Peringatan Dini Banjir", "Prediksi Curah Hujan"}


def test_label_templat_tidak_masuk_indeks_bm25(art):
    _, df, _ = art
    r = bm25_search("judul abstrak", df, build_bm25_index(df), 10)
    assert (r.score == 0).all()


# ---- B2 BM25 mandiri, B3 tahun, B5 kolom ----
def test_load_documents_tanpa_embeddings(art):
    path, df0, _ = art
    (path / "embeddings.npy").unlink()
    docs = load_documents(str(path))
    assert len(docs) == len(df0)
    assert docs.loc[docs.title == "Prediksi Curah Hujan", "year"].iloc[0] == ""      # bukan "nan"/float
    assert docs.loc[0, "year"] == "2022"
    r = bm25_search("hoaks", docs, build_bm25_index(docs), 3)
    assert r.iloc[0].title == "Deteksi Hoaks Berita"
    assert {"journal", "doi", "data_source"} <= set(r.columns)
    with pytest.raises(FileNotFoundError):
        load_artifacts(str(path))


def test_file_lama_tanpa_kolom_baru_tetap_terbaca(tmp_path):
    pd.DataFrame({"doc_id": ["D1"], "title": ["t"], "abstract": ["a"], "keywords": ["k"],
                  "author": [""], "year": [2021.0], "document_type": ["x"], "source_url": [""],
                  "document_text": ["Judul: t."]}).to_csv(tmp_path / "documents_processed.csv", index=False)
    d = load_documents(str(tmp_path))
    assert d.loc[0, "year"] == "2021" and d.loc[0, "journal"] == "" and d.loc[0, "data_source"] == ""


# ---- B8 validasi embedding ----
def test_validasi_lolos_pada_artifacts_benar(art):
    path, df, _ = art
    docs, emb = load_artifacts(str(path))
    assert emb.shape == (len(docs), 64) and len(docs) == len(df)


def test_jumlah_tidak_sama_ditolak(art):
    path, df, emb = art
    np.save(path / "embeddings.npy", emb[:3])  # jumlah != jumlah dokumen
    with pytest.raises(ValueError, match="Jumlah dokumen"):
        load_artifacts(str(path))


def test_urutan_dokumen_berubah_ditolak_lewat_hash(art):
    path, df, _ = art
    df.iloc[::-1].to_csv(path / "documents_processed.csv", index=False)       # jumlah sama, urutan beda
    with pytest.raises(ValueError, match="hash doc_id"):
        load_artifacts(str(path))


def test_model_berbeda_ditolak(art):
    path, df, emb = art
    cfg = json.loads((path / "model_config.json").read_text())
    cfg["model_name"] = "model/lain"
    (path / "model_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="model"):
        load_artifacts(str(path))


def test_embedding_tidak_ternormalisasi_atau_nan_ditolak(art):
    path, df, emb = art
    np.save(path / "embeddings.npy", emb * 3)
    with pytest.raises(ValueError, match="ternormalisasi"):
        load_artifacts(str(path))
    bad = emb.copy(); bad[0, 0] = np.nan
    np.save(path / "embeddings.npy", bad)
    with pytest.raises(ValueError, match="NaN"):
        load_artifacts(str(path))


def test_config_lama_tanpa_hash_hanya_peringatan(art):
    path, df, emb = art
    cfg = json.loads((path / "model_config.json").read_text()); cfg.pop("doc_ids_sha1")
    (path / "model_config.json").write_text(json.dumps(cfg))
    with pytest.warns(UserWarning, match="doc_ids_sha1"):
        load_embeddings(str(path), load_documents(str(path)))


# ---- pipeline 4 metode (alur kode, model palsu) ----
def test_empat_metode_berjalan_dan_kontrak_hasil(art):
    path, _, _ = art
    docs, emb = load_artifacts(str(path))
    bm = build_bm25_index(docs)
    for m in ALL_METHODS:
        res, info = search_with_method(m, "banjir", docs, emb, FakeModel(), bm, top_k=3)
        assert len(res) == 3 and res["rank"].tolist() == [1, 2, 3]
        assert info["method"] == m and info["original_query"] == "banjir"
    _, adaptif = search_with_method(ALL_METHODS[3], "sistem deteksi banjir dengan sensor", docs, emb, FakeModel(), bm, 3)
    assert adaptif["is_expanded"] is False
    _, statik = search_with_method(ALL_METHODS[2], "banjir", docs, emb, FakeModel(), bm, 3)
    assert statik["is_expanded"] is True and statik["final_query"].startswith("banjir ")


def test_kueri_kosong_dan_top_k_melebihi_jumlah_dokumen(art):
    path, _, _ = art
    docs, emb = load_artifacts(str(path)); bm = build_bm25_index(docs)
    res, _ = search_with_method(ALL_METHODS[0], "   ", docs, emb, None, bm, 5)
    assert res.empty
    res, _ = search_with_method(ALL_METHODS[1], "banjir", docs, emb, FakeModel(), bm, 50)
    assert len(res) == len(docs)
