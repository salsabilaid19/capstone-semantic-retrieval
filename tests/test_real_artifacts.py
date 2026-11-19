"""Integritas artefak PENELITIAN (artifacts_real) terhadap ketentuan proyek + uji alur pada data nyata.

Dilewati bila artifacts_real tidak ada. Uji alur memakai ProxyModel (BUKAN e5): hanya memastikan pipeline bekerja pada
900 dokumen/384 dimensi nyata, bukan menilai kualitas retrieval."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from src import retrieval as R  # noqa: E402
from src.artifacts import doc_ids_hash  # noqa: E402
from src.validation import MIN_ABSTRACT_CHARS, validate_dataset  # noqa: E402

REAL = ROOT / "artifacts_real"
pytestmark = pytest.mark.skipif(not (REAL / "embeddings.npy").exists(), reason="artifacts_real tidak ada")

# Dokumen yang DIKETAHUI melanggar aturan abstrak >= 200 karakter (sampul jurnal, bukan artikel). Menunggu keputusan
# peneliti. Tes gagal bila muncul pelanggaran BARU di luar daftar ini.
PELANGGAR_DIKETAHUI = {"DOC-26c0fe0352", "DOC-35a5ce65cf"}


@pytest.fixture(scope="module")
def docs():
    return pd.read_csv(REAL / "documents_processed.csv", dtype=str, keep_default_na=False)


def test_empat_artefak_utama_ada():
    for f in ("dataset_manifest.json", "documents_processed.csv", "embeddings.npy", "model_config.json"):
        assert (REAL / f).exists(), f


def test_komposisi_dataset_sesuai_ketentuan_penelitian(docs):
    assert len(docs) == 900 and docs["doc_id"].is_unique
    jr = docs["journal"].str.split(" \\(").str[0].value_counts().to_dict()
    assert jr == {"IJCCS": 300, "JNTETI": 300, "Jurnal RESTI": 300}
    assert docs["language"].value_counts().to_dict() == {"en": 616, "id": 284}
    assert set(docs["data_source"]) == {"real"}                    # bukan data sintetis


def test_model_dan_embedding_sesuai_ketentuan(docs):
    cfg = json.loads((REAL / "model_config.json").read_text(encoding="utf-8"))
    assert cfg["model_name"] == "intfloat/multilingual-e5-small" and cfg["embedding_dim"] == 384
    assert cfg["document_prefix"] == "passage: " and cfg["query_prefix"] == "query: "
    emb = np.load(REAL / "embeddings.npy")
    assert emb.shape == (900, 384) and np.isfinite(emb).all()
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)
    assert doc_ids_hash(docs["doc_id"]) == cfg["doc_ids_sha1"]       # urutan & ID dokumen cocok dengan embedding
    assert R.load_embeddings(str(REAL), R.load_documents(str(REAL))).shape == (900, 384)


def test_manifest_konsisten_dengan_data(docs):
    m = json.loads((REAL / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert m.get("num_documents", len(docs)) == len(docs)


def test_aturan_abstrak_minimal_200_karakter_tidak_ada_pelanggaran_baru(docs):
    pendek = set(docs.loc[docs["abstract"].str.len() < MIN_ABSTRACT_CHARS, "doc_id"])
    assert pendek <= PELANGGAR_DIKETAHUI, f"pelanggaran baru: {sorted(pendek - PELANGGAR_DIKETAHUI)}"
    rep = validate_dataset(R.load_documents(str(REAL)), expected_kind="real")
    assert set(rep["stats"]["doc_id_abstrak_kurang_dari_200"]) == pendek      # validator melaporkan persis yang sama
    assert (not pendek) or any("< 200 karakter" in e for e in rep["errors"])  # dan tidak disembunyikan


def test_pelanggar_diketahui_adalah_sampul_jurnal_bukan_artikel(docs):
    x = docs[docs["doc_id"].isin(PELANGGAR_DIKETAHUI) & (docs["abstract"].str.len() < MIN_ABSTRACT_CHARS)]
    assert x["title"].str.contains("Halaman", case=False).all()


def test_alur_empat_metode_pada_data_nyata_dengan_model_proksi():
    from proxy_model import ProxyModel
    d = R.load_documents(str(REAL)); e = R.load_embeddings(str(REAL), d)
    pm, bm = ProxyModel(d, e), R.build_bm25_index(d)
    for q in ("analisis sentimen", "deteksi penyakit tanaman dengan convolutional neural network", "forecasting"):
        for m in R.ALL_METHODS:
            res, info = R.search_with_method(m, q, d, e, pm, bm, 10)
            assert len(res) == 10 and res["doc_id"].is_unique and set(res["doc_id"]) <= set(d["doc_id"])
            assert info["original_query"] == q
            if m == R.METHOD_ADAPTIVE:
                assert info["decision_basis"] == "quality"                        # 900 dok -> indikator kualitas dipakai
                assert info["expansion_level"] in ("none", "light", "strong")
                assert info["is_expanded"] == (info["final_query"] != q) == bool(info["expansion_terms"])
                assert len(info["expansion_terms"]) <= {"none": 0, "light": 1, "strong": 3}[info["expansion_level"]]
                low = [t.lower() for t in info["expansion_terms"]]
                assert len(low) == len(set(low)) and not any(t in q.lower().split() for t in low)
                assert all(k in info["indicators"] for k in ("top1_z", "top5_mean_z", "gap_z", "spread_z"))
