"""Tes fallback BM25, perbandingan 4 metode, dan utilitas UI (model PALSU)."""
import numpy as np
import pytest

from fakes import BrokenModel, FakeModel, make_artifacts
from src.retrieval import (
    ALL_METHODS, METHOD_ADAPTIVE, METHOD_BM25, METHOD_SEMANTIC, METHOD_STATIC, build_bm25_index,
    load_documents, load_embeddings,
)
from src.service import (
    ADAPTIVE_QE_DISCLAIMER, DEFAULT_PUBLIC_METHOD, SemanticUnavailable, article_link, compare_methods,
    describe_failure, keyword_coverage, overlap_matrix, runs_to_frame, search_with_fallback, try_load,
)


@pytest.fixture
def ctx(tmp_path):
    make_artifacts(tmp_path)
    docs = load_documents(str(tmp_path))
    emb = load_embeddings(str(tmp_path), docs)
    return docs, emb, build_bm25_index(docs)


def _boom(exc):
    def f():
        raise exc
    return f


def test_default_publik_adalah_semantic_adaptive():
    assert DEFAULT_PUBLIC_METHOD == METHOD_ADAPTIVE


def test_tanpa_fallback_bila_semantic_tersedia(ctx):
    docs, emb, bm = ctx
    out = search_with_fallback("banjir", 3, docs, bm, lambda: emb, lambda: FakeModel())
    assert not out.fell_back and out.used_method == METHOD_ADAPTIVE and out.notice == ""
    assert out.info["is_expanded"] is True and len(out.results) == 3


def test_fallback_bila_embedding_tidak_ada(ctx):
    docs, _, bm = ctx
    out = search_with_fallback("banjir", 3, docs, bm, _boom(FileNotFoundError("embeddings.npy belum ada")),
                               lambda: FakeModel())
    assert out.fell_back and out.used_method == METHOD_BM25 and out.requested_method == METHOD_ADAPTIVE
    assert "BM25" in out.notice and "embedding" in out.notice.lower() and "tidak tersedia" in out.notice
    assert len(out.results) == 3


def test_fallback_bila_embedding_tidak_cocok(ctx):
    docs, _, bm = ctx
    out = search_with_fallback("banjir", 3, docs, bm, _boom(ValueError("hash doc_id tidak cocok")), lambda: FakeModel())
    assert out.fell_back and "tidak cocok" in out.fallback_reason


def test_fallback_bila_model_gagal_dimuat_dan_model_tidak_dimuat_jika_embedding_gagal(ctx):
    docs, emb, bm = ctx
    out = search_with_fallback("banjir", 3, docs, bm, lambda: emb, _boom(OSError("tidak ada internet")))
    assert out.fell_back and "Model semantic" in out.fallback_reason
    dipanggil = []
    search_with_fallback("banjir", 3, docs, bm, _boom(FileNotFoundError("x")), lambda: dipanggil.append(1))
    assert dipanggil == []   # tidak membuang waktu memuat model bila embedding sudah gagal


def test_fallback_bila_encode_gagal_saat_kueri(ctx):
    docs, emb, bm = ctx
    out = search_with_fallback("banjir", 3, docs, bm, lambda: emb, lambda: BrokenModel())
    assert out.fell_back and out.used_method == METHOD_BM25


def test_fallback_import_error_dan_pesan_ramah(ctx):
    docs, emb, bm = ctx
    out = search_with_fallback("banjir", 3, docs, bm, lambda: emb, _boom(ImportError("no module")))
    assert "sentence-transformers" in out.fallback_reason
    assert "Traceback" not in out.notice


def test_bm25_diminta_langsung_tidak_memanggil_loader(ctx):
    docs, _, bm = ctx
    out = search_with_fallback("banjir", 3, docs, bm, _boom(AssertionError), _boom(AssertionError), method=METHOD_BM25)
    assert not out.fell_back and out.used_method == METHOD_BM25


def test_hasil_fallback_punya_kolom_lengkap(ctx):
    docs, _, bm = ctx
    out = search_with_fallback("banjir", 3, docs, bm, _boom(FileNotFoundError("x")), lambda: FakeModel())
    assert {"title", "abstract", "year", "journal", "source_url", "doi", "data_source"} <= set(out.results.columns)


def test_try_load_dan_describe_failure():
    assert try_load(lambda: 5, "model") == (5, "")
    val, reason = try_load(_boom(RuntimeError("x")), "model")
    assert val is None and "RuntimeError" in reason
    assert describe_failure(SemanticUnavailable("pesan sendiri")) == "pesan sendiri"
    assert "belum tersedia" in describe_failure(FileNotFoundError("a"), "embedding")
    assert "Memori" in describe_failure(MemoryError(), "model")


# ---- Mode eksperimen ----
def test_compare_methods_empat_metode(ctx):
    docs, emb, bm = ctx
    runs = compare_methods("banjir", 3, docs, bm, lambda: emb, lambda: FakeModel())
    assert [r.method for r in runs] == ALL_METHODS and all(r.available for r in runs)


def test_compare_tanpa_substitusi_diam_diam(ctx):
    docs, _, bm = ctx
    runs = compare_methods("banjir", 3, docs, bm, _boom(FileNotFoundError("x")), lambda: FakeModel())
    by = {r.method: r for r in runs}
    assert by[METHOD_BM25].available
    for m in (METHOD_SEMANTIC, METHOD_STATIC, METHOD_ADAPTIVE):
        assert not by[m].available and by[m].results is None and by[m].error
    assert len(overlap_matrix(runs)) == 1 and not runs_to_frame("banjir", runs).empty


def test_static_dan_adaptive_identik_untuk_kueri_pendek_dan_beda_untuk_kueri_panjang(ctx):
    docs, emb, bm = ctx
    pendek = {r.method: r for r in compare_methods("banjir", 4, docs, bm, lambda: emb, lambda: FakeModel())}
    assert pendek[METHOD_STATIC].info["final_query"] == pendek[METHOD_ADAPTIVE].info["final_query"]
    assert pendek[METHOD_STATIC].results["doc_id"].tolist() == pendek[METHOD_ADAPTIVE].results["doc_id"].tolist()
    panjang = {r.method: r for r in compare_methods("sistem deteksi banjir dengan sensor air", 4, docs, bm,
                                                    lambda: emb, lambda: FakeModel())}
    assert panjang[METHOD_STATIC].info["is_expanded"] is True
    assert panjang[METHOD_ADAPTIVE].info["is_expanded"] is False


def test_overlap_matrix_diagonal_satu_dan_simetris(ctx):
    docs, emb, bm = ctx
    m = overlap_matrix(compare_methods("banjir", 3, docs, bm, lambda: emb, lambda: FakeModel()))
    assert np.allclose(np.diag(m.values), 1.0) and np.allclose(m.values, m.values.T)


def test_runs_to_frame_kolom_dan_isi(ctx):
    docs, emb, bm = ctx
    f = runs_to_frame("banjir", compare_methods("banjir", 3, docs, bm, lambda: emb, lambda: FakeModel()))
    assert len(f) == 12 and set(f.method) == set(ALL_METHODS) and (f["query"] == "banjir").all()


# ---- utilitas ----
def test_article_link_dan_keyword_coverage(ctx):
    docs, _, _ = ctx
    assert article_link({"source_url": "https://x.org/a", "doi": "10.1/a"}) == "https://x.org/a"
    assert article_link({"source_url": "", "doi": "10.1/a"}) == "https://doi.org/10.1/a"
    assert article_link({"source_url": "", "doi": ""}) == ""
    assert keyword_coverage(docs) == 1.0
    docs2 = docs.copy(); docs2.loc[:2, "keywords"] = ""
    assert keyword_coverage(docs2) == 0.5


def test_disclaimer_adaptive_transparan():
    t = ADAPTIVE_QE_DISCLAIMER.lower()
    assert "z-score" in t and "tidak** mendeteksi ambiguitas" in t and "heuristik" in t
    assert all(x in t for x in ("no expansion", "light expansion", "strong expansion"))
