"""Tes pembuatan & pemeriksaan kompatibilitas embedding (model PALSU; membuktikan alur, bukan kualitas model asli)."""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from fakes import FakeModel, make_artifacts
from scripts import check_embeddings, generate_embeddings
from src import retrieval as R
from src.artifacts import DOC_PREFIX, QUERY_PREFIX, doc_ids_hash
from src.embedding import encode_documents, projected_size_mb, write_embedding_artifacts
from src.retrieval import MODEL_NAME, load_artifacts, load_documents

ROOT = Path(__file__).resolve().parents[1]


class Spy(FakeModel):
    def __init__(self): self.texts, self.kw = [], {}
    def encode(self, texts, **kw):
        self.texts, self.kw = list(texts), kw
        return super().encode(texts, **kw)


@pytest.fixture
def art(tmp_path):
    df = make_artifacts(tmp_path, data_source="real", with_embeddings=False)
    return tmp_path, load_documents(str(tmp_path))


# ---------- encode & tulis ----------
def test_encode_documents_kontrak(art):
    _, docs = art
    spy = Spy()
    emb = encode_documents(spy, docs, batch_size=4)
    assert emb.dtype == np.float32 and emb.shape == (len(docs), 64)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert all(t.startswith(DOC_PREFIX) for t in spy.texts) and spy.kw["normalize_embeddings"] is True and spy.kw["batch_size"] == 4
    assert spy.texts[0] == DOC_PREFIX + docs["document_text"].iloc[0]


def test_encode_documents_menolak_bentuk_salah_dan_kosong(art):
    _, docs = art

    class Bad:
        def encode(self, texts, **kw): return np.zeros((len(texts) - 1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Bentuk"):
        encode_documents(Bad(), docs)
    with pytest.raises(ValueError, match="Tidak ada dokumen"):
        encode_documents(FakeModel(), docs.iloc[0:0])


def test_prefix_kueri_konsisten_dengan_retrieval(art):
    _, docs = art

    class QSpy(FakeModel):
        seen = None
        def encode(self, texts, **kw):
            QSpy.seen = list(texts)
            return super().encode(texts, **kw)
    R.encode_query("banjir", QSpy())
    assert QSpy.seen == [QUERY_PREFIX + "banjir"]


def test_write_roundtrip_dan_hash(art):
    path, docs = art
    emb = encode_documents(FakeModel(), docs)
    cfg = write_embedding_artifacts(path, docs, emb, MODEL_NAME)
    loaded_docs, loaded_emb = load_artifacts(str(path))
    assert np.array_equal(loaded_emb, emb) and cfg["doc_ids_sha1"] == doc_ids_hash(loaded_docs["doc_id"])
    assert not list(path.glob("*.tmp"))


def test_write_gagal_tidak_mengubah_berkas_lama(art):
    path, docs = art
    emb = encode_documents(FakeModel(), docs)
    write_embedding_artifacts(path, docs, emb, MODEL_NAME)
    before = ((path / "embeddings.npy").read_bytes(), (path / "model_config.json").read_text())
    bad = emb.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        write_embedding_artifacts(path, docs, bad, MODEL_NAME)
    with pytest.raises(ValueError):
        write_embedding_artifacts(path, docs, emb * 2, MODEL_NAME)                      # tidak ternormalisasi
    assert before == ((path / "embeddings.npy").read_bytes(), (path / "model_config.json").read_text())
    assert not list(path.glob("*.tmp"))


def test_catatan_versi_library_masuk_config(art):
    from src.embedding import library_versions
    path, docs = art
    v = library_versions()
    assert set(v) == {"sentence-transformers", "torch", "transformers", "numpy"} and v["numpy"] != "tidak terpasang"
    write_embedding_artifacts(path, docs, encode_documents(FakeModel(), docs), MODEL_NAME, extra={"library_versions": v})
    assert json.loads((path / "model_config.json").read_text())["library_versions"] == v
    load_artifacts(str(path))                                   # kunci tambahan tidak mengganggu validasi


def test_cli_generate_mencatat_versi_dan_check_menampilkannya(art, capsys):
    path, _ = art
    assert generate_embeddings.main(["--artifacts-dir", str(path)], model=FakeModel()) == 0
    assert "library_versions" in json.loads((path / "model_config.json").read_text())
    capsys.readouterr()
    assert check_embeddings.main(["--artifacts-dir", str(path), "--expected-dim", "64"]) == 0
    assert "Dibuat dengan:" in capsys.readouterr().out


def test_perkiraan_ukuran():
    assert projected_size_mb(500) == pytest.approx(500 * 384 * 4 / 1024 / 1024)
    assert projected_size_mb(500) < 1.0          # 200-500 dokumen: kurang dari 1 MB


# ---------- generate_embeddings ----------
def test_cli_generate_sukses_dan_hasilnya_lolos_pemeriksa(art, capsys):
    path, _ = art
    assert generate_embeddings.main(["--artifacts-dir", str(path)], model=FakeModel()) == 0
    out = capsys.readouterr().out
    assert "embeddings.npy" in out and "perkiraan 500 dok" in out
    assert check_embeddings.main(["--artifacts-dir", str(path), "--expected-dim", "64", "--sanity"], model=FakeModel()) == 0


def test_cli_generate_tanpa_dokumen_dan_model_gagal(tmp_path, art, monkeypatch, capsys):
    assert generate_embeddings.main(["--artifacts-dir", str(tmp_path / "kosong")], model=FakeModel()) == 2
    path, _ = art

    def no_lib(*a, **k): raise ImportError("no module")
    monkeypatch.setattr("src.retrieval.load_embedding_model", no_lib)
    assert generate_embeddings.main(["--artifacts-dir", str(path)]) == 2 and "pip install sentence-transformers" in capsys.readouterr().err

    def offline(*a, **k): raise OSError("offline")
    monkeypatch.setattr("src.retrieval.load_embedding_model", offline)
    assert generate_embeddings.main(["--artifacts-dir", str(path)]) == 2 and "koneksi internet" in capsys.readouterr().err
    assert not (path / "embeddings.npy").exists()


# ---------- check_embeddings ----------
@pytest.fixture
def ready(art):
    path, docs = art
    write_embedding_artifacts(path, docs, encode_documents(FakeModel(), docs), MODEL_NAME)
    return path, docs


def chk(path, *extra, model=None):
    return check_embeddings.main(["--artifacts-dir", str(path), "--expected-dim", "64", *extra], model=model)


def test_check_struktural_dan_sanity_lolos(ready, capsys):
    path, _ = ready
    assert chk(path) == 0 and "struktural saja" in capsys.readouterr().out
    assert chk(path, "--sanity", model=FakeModel()) == 0 and "Sanity 1" in capsys.readouterr().out


def test_check_dimensi_default_untuk_e5_adalah_384(ready, capsys):
    path, _ = ready
    assert check_embeddings.main(["--artifacts-dir", str(path)]) == 1 and "384" in capsys.readouterr().out


def test_check_urutan_dokumen_berubah_gagal(ready, capsys):
    path, docs = ready
    docs.iloc[::-1].to_csv(path / "documents_processed.csv", index=False)
    assert chk(path) == 1 and "hash doc_id" in capsys.readouterr().out


def test_check_sanity_menangkap_baris_tertukar_walau_hash_cocok(ready, capsys):
    path, _ = ready
    emb = np.load(path / "embeddings.npy")
    np.save(path / "embeddings.npy", emb[::-1].copy())                    # jumlah, norma, hash config tetap valid
    assert chk(path) == 0                                                  # struktural TIDAK cukup untuk menangkapnya
    assert chk(path, "--sanity", model=FakeModel()) == 1
    assert "TIDAK cocok" in capsys.readouterr().out


def test_check_sanity_menangkap_prefix_salah(art, capsys):
    path, docs = art

    class WrongPrefix(FakeModel):
        def encode(self, texts, **kw):
            return super().encode([t.replace(DOC_PREFIX, QUERY_PREFIX, 1) for t in texts], **kw)
    write_embedding_artifacts(path, docs, encode_documents(WrongPrefix(), docs), MODEL_NAME)
    assert chk(path, "--sanity", model=FakeModel()) == 1 and "prefix" in capsys.readouterr().out.lower()


def test_check_config_prefix_berbeda_dan_tanpa_embeddings(ready, tmp_path, capsys):
    path, _ = ready
    cfg = json.loads((path / "model_config.json").read_text())
    cfg["document_prefix"] = ""
    (path / "model_config.json").write_text(json.dumps(cfg))
    assert chk(path) == 1 and "Prefix" in capsys.readouterr().out
    assert check_embeddings.main(["--artifacts-dir", str(tmp_path / "x")]) == 2
    (tmp_path / "y").mkdir(); make_artifacts(tmp_path / "y", with_embeddings=False)
    assert check_embeddings.main(["--artifacts-dir", str(tmp_path / "y")]) == 2


def test_check_sanity_model_tidak_dapat_dimuat(ready, monkeypatch, capsys):
    path, _ = ready

    def offline(*a, **k): raise OSError("offline")
    monkeypatch.setattr("src.retrieval.load_embedding_model", offline)
    assert chk(path, "--sanity") == 2


# ---------- tanpa sentence-transformers ----------
def test_load_embedding_model_tanpa_library_melempar_importerror(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError):
        R.load_embedding_model()


def test_aplikasi_publik_fallback_saat_library_tidak_terpasang(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    make_artifacts(tmp_path)                                    # embedding ada, tetapi library TIDAK ada
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.chdir(ROOT)
    st.cache_data.clear(); st.cache_resource.clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at = at.switch_page("views/search.py").run()
    at.text_input[0].set_value("banjir")
    next(b for b in at.button if b.label == "Cari").click()
    at.run()
    warn = " ".join(w.value for w in at.warning)
    assert not at.exception and "sentence-transformers" in warn and "BM25" in warn
    st.cache_data.clear(); st.cache_resource.clear()


# ---------- konsistensi requirements ----------
IMPORT_TO_PACKAGE = {"rank_bm25": "rank-bm25", "sentence_transformers": "sentence-transformers", "sklearn": "scikit-learn"}


def third_party_imports():
    own = {"src", "scripts", "tests", "app", "fakes", "oai_helpers"}
    found = set()
    for f in [ROOT / "app.py", *ROOT.glob("src/*.py"), *ROOT.glob("pages/*.py"), *ROOT.glob("scripts/*.py")]:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            for n in names:
                top = n.split(".")[0]
                if top not in sys.stdlib_module_names and top not in own:
                    found.add(IMPORT_TO_PACKAGE.get(top, top))
    return found


def declared(path):
    out = set()
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            out.add(re.split(r"[<>=!~ ]", line)[0].lower())
    return out


def test_semua_import_pihak_ketiga_terdaftar_di_requirements():
    assert third_party_imports() <= declared("requirements.txt")


def test_requirements_tidak_memuat_paket_tak_terpakai_atau_alat_dev():
    runtime = declared("requirements.txt")
    assert runtime - third_party_imports() == set(), f"tidak dipakai: {runtime - third_party_imports()}"
    assert "pytest" not in runtime and "pytest" in declared("requirements-dev.txt")
