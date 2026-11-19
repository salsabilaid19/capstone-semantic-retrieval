"""Uji UI headless (streamlit.testing) untuk aplikasi multi-halaman: Dashboard, Document Search, About.
Jalur semantic memakai model PALSU (alur kode saja). Cache Streamlit dikosongkan antar tes."""
import re
from pathlib import Path

import pytest

from fakes import FakeModel, make_artifacts

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
DASH, SEARCH, ABOUT = "views/dashboard.py", "views/search.py", "views/about.py"
METODE_DEFAULT = "Semantic Retrieval + Adaptive Query Expansion"
BANDINGKAN = "Bandingkan 4 metode"


@pytest.fixture(autouse=True)
def bersih(monkeypatch):
    monkeypatch.chdir(ROOT)
    st.cache_data.clear(); st.cache_resource.clear()
    yield
    st.cache_data.clear(); st.cache_resource.clear()


@pytest.fixture
def semantic_ok(tmp_path, monkeypatch):
    make_artifacts(tmp_path)
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr("src.retrieval.load_embedding_model", lambda *a, **k: FakeModel())
    return tmp_path


@pytest.fixture
def tanpa_embedding(tmp_path, monkeypatch):
    make_artifacts(tmp_path, with_embeddings=False)
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr("src.retrieval.load_embedding_model", lambda *a, **k: FakeModel())
    return tmp_path


def buka(page=DASH):
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, at.exception
    if page != DASH:
        at = at.switch_page(page).run()
    assert not at.exception, at.exception
    return at


def cari(kueri, mode=None, metode=None):
    at = buka(SEARCH)
    if mode:
        at.radio[0].set_value(mode)
    if metode:
        at.selectbox[0].set_value(metode)
    at.text_input[0].set_value(kueri)
    next(b for b in at.button if b.label == "Cari").click()
    return at.run()


def teks(at, elemen):
    return " ".join(e.value for e in getattr(at, elemen))


def kartu(at):
    return [m.value for m in at.markdown if 'class="res"' in m.value]


# ---------------- Document Search: satu metode ----------------
def test_cari_default_semantic_adaptive_tanpa_fallback(semantic_ok):
    at = cari("banjir")
    assert not at.exception and not at.error
    assert not any("tidak tersedia" in w.value for w in at.warning)
    assert METODE_DEFAULT in teks(at, "caption")
    assert "Alasan keputusan" in teks(at, "markdown") and "Kueri asli" in teks(at, "markdown")
    assert len(kartu(at)) == 6          # fixture hanya 6 dokumen (< slider 10)


def test_cari_hasil_menampilkan_tahun_jurnal_abstrak_skor_tautan(semantic_ok):
    c = " ".join(kartu(cari("banjir")))
    assert re.search(r"20\d\d", c) and "Buka artikel asli" in c and "Sensor air" in c and "Skor relevansi" in c
    assert not re.search(r"\bnan\b", c)


def test_cari_keempat_metode_dapat_dipilih(semantic_ok):
    for nama, penuh in (("BM25", "BM25"), ("Semantic", "Semantic Retrieval"),
                        ("Semantic + Static QE", "Semantic Retrieval + Static Query Expansion"),
                        ("Semantic + Adaptive QE", METODE_DEFAULT)):
        at = cari("banjir", metode=nama)
        assert not at.exception and not at.error
        assert f"Metode pencarian yang berjalan: **{penuh}**" in teks(at, "caption"), nama
        assert kartu(at)


def test_cari_fallback_bm25_saat_embedding_tidak_ada_dengan_pemberitahuan_jelas(tanpa_embedding):
    at = cari("banjir")
    assert not at.exception and not at.error
    w = teks(at, "warning")
    assert "tidak tersedia" in w and "BM25" in w and METODE_DEFAULT in w
    assert "Metode pencarian yang berjalan: **BM25**" in teks(at, "caption")
    assert kartu(at)


def test_cari_fallback_saat_model_gagal_dimuat_dan_kegagalan_tidak_diulang(semantic_ok, monkeypatch):
    panggilan = []

    def gagal(*a, **k):
        panggilan.append(1)
        raise OSError("tidak ada internet (simulasi)")
    monkeypatch.setattr("src.retrieval.load_embedding_model", gagal)
    at = cari("banjir")
    assert not at.exception
    assert "Model semantic gagal" in teks(at, "warning") and "BM25" in teks(at, "warning")
    at.text_input[0].set_value("hoaks")
    next(b for b in at.button if b.label == "Cari").click()
    at.run()
    assert len(panggilan) == 1          # kegagalan di-cache; tidak mengunduh ulang tiap pencarian


def test_cari_kueri_kosong_dan_tanpa_hasil(semantic_ok):
    assert "kosong" in teks(cari("   "), "warning")
    assert not cari("xyzxyz").exception


def test_cari_menandai_data_sintetis(semantic_ok):
    at = cari("banjir")
    assert "data contoh sintetis" in teks(at, "markdown") and "DATA SINTETIS" in " ".join(kartu(at))


def test_cari_data_asli_tidak_diberi_label_sintetis(tmp_path, monkeypatch):
    make_artifacts(tmp_path, data_source="real", with_embeddings=False)
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    at = cari("banjir")
    assert "sintetis" not in teks(at, "markdown").lower()


def test_tanpa_data_pesan_ramah(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and "Data belum tersedia" in teks(at, "error")


def test_default_artifacts_memilih_dataset_penelitian_bila_ada(tmp_path, monkeypatch):
    from src import ui_common as ui
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts").mkdir()
    assert ui.artifacts_dir() == "artifacts"                       # hanya contoh sintetis
    (tmp_path / "artifacts_real").mkdir()
    (tmp_path / "artifacts_real" / "documents_processed.csv").write_text("doc_id\n")
    assert ui.artifacts_dir() == "artifacts_real"                  # dataset penelitian diutamakan
    monkeypatch.setenv("ARTIFACTS_DIR", "lain")
    assert ui.artifacts_dir() == "lain"                            # variabel lingkungan menang


def test_artifacts_contoh_bawaan_repo_berjalan_dengan_fallback(monkeypatch, tmp_path):
    """Folder `artifacts` bawaan (sintetis, tanpa embeddings.npy) -> BM25 + pemberitahuan, tanpa crash."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(ROOT / "artifacts"))
    if (ROOT / "artifacts" / "embeddings.npy").exists():
        pytest.skip("embeddings.npy ada; skenario fallback tidak berlaku")
    at = cari("banjir")
    assert not at.exception and "BM25" in teks(at, "warning")


# ---------------- Document Search: bandingkan 4 metode ----------------
def test_bandingkan_empat_metode_berdampingan(semantic_ok):
    at = cari("banjir", mode=BANDINGKAN)
    assert not at.exception and not at.error
    judul = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert judul == ["#### BM25", "#### Semantic Retrieval", "#### Semantic Retrieval + Static Query Expansion", "#### " + METODE_DEFAULT]
    assert len(at.columns) >= 4 and at.dataframe                      # matriks irisan
    assert "hanya ilustrasi" in teks(at, "caption")
    assert at.get("download_button")


def test_bandingkan_menjelaskan_dasar_keputusan_adaptive_secara_jujur(semantic_ok):
    md = teks(cari("banjir", mode=BANDINGKAN), "markdown")
    assert "tidak** mendeteksi ambiguitas" in md and "z-score" in md and "Light Expansion" in md


def test_bandingkan_kueri_panjang_menjelaskan_alasan(semantic_ok):
    md = teks(cari("sistem deteksi banjir dengan sensor air", mode=BANDINGKAN), "markdown")
    assert "Alasan keputusan" in md and "lebih dari 3 kata" in md     # korpus kecil -> aturan panjang kueri dinyatakan


def test_bandingkan_tanpa_substitusi_bm25_saat_semantic_tidak_ada(tanpa_embedding):
    at = cari("banjir", mode=BANDINGKAN)
    assert not at.exception
    err = [e.value for e in at.error]
    assert len(err) == 3 and all(e.startswith("Tidak tersedia:") for e in err)   # 3 metode semantic ditandai
    assert len([m for m in at.markdown if m.value.startswith("#### ")]) == 4      # BM25 tetap tampil


def test_bandingkan_peringatan_cakupan_kata_kunci_rendah(tmp_path, monkeypatch):
    from fakes import RAW
    raw = RAW.copy(); raw["Keywords"] = ""
    make_artifacts(tmp_path, with_embeddings=True, raw=raw)
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr("src.retrieval.load_embedding_model", lambda *a, **k: FakeModel())
    assert "memiliki kata kunci" in teks(cari("banjir", mode=BANDINGKAN), "warning")


# ---------------- Dashboard ----------------
def test_dashboard_data_contoh(semantic_ok):
    at = buka(DASH)
    md = teks(at, "markdown")
    assert "Adaptive Query Expansion" in md and "data contoh sintetis" in md
    assert "Embedding valid" in md and "Empat metode pencarian" in md


@pytest.mark.skipif(not (ROOT / "artifacts_real" / "embeddings.npy").exists(), reason="artifacts_real tidak ada")
def test_dashboard_dataset_penelitian_menampilkan_angka_sesuai_spesifikasi(monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(ROOT / "artifacts_real"))
    md = teks(buka(DASH), "markdown")
    assert ">900<" in md and "616 EN · 284 ID" in md and "384" in md and "e5-small" in md
    for j in ("IJCCS", "JNTETI", "Jurnal RESTI"):
        assert j in md
    assert "sintetis" not in md.lower()
    assert "berabstrak &lt; 200 karakter" in md           # pelanggaran aturan dilaporkan, bukan disembunyikan


def test_dashboard_tidak_error_dengan_kode_bahasa_tak_dikenal(tmp_path, monkeypatch):
    make_artifacts(tmp_path, with_embeddings=False)
    import pandas as pd
    p = tmp_path / "documents_processed.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False); df["language"] = "unknown"; df.to_csv(p, index=False)
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    assert not buka(DASH).exception


# ---------------- About ----------------
def test_about_menjelaskan_metode_dan_batasan(semantic_ok):
    at = buka(ABOUT)
    md = teks(at, "markdown") + " " + " ".join(str(c.value) for c in at.table)
    for kata in ("BM25", "Semantic Retrieval", "Static QE", "Adaptive QE", "No Expansion", "Light Expansion", "Strong Expansion",
                 "Batasan jujur", "Alur kerja sistem"):
        assert kata in md, kata


def test_navigasi_antar_halaman_tanpa_error(semantic_ok):
    at = buka(DASH)
    for page in (SEARCH, ABOUT, DASH):
        at = at.switch_page(page).run()
        assert not at.exception, page


def test_corpus_is_indonesian_mengikuti_bahasa_dataset(tmp_path):
    import pandas as pd
    from src import ui_common as ui
    pd.DataFrame({"language": ["en"] * 8 + ["id"] * 2}).to_csv(tmp_path / "documents_processed.csv", index=False)
    assert ui.corpus_is_indonesian(str(tmp_path)) is False
    pd.DataFrame({"language": ["id"] * 6 + ["en"] * 4}).to_csv(tmp_path / "documents_processed.csv", index=False)
    assert ui.corpus_is_indonesian(str(tmp_path)) is True
    assert ui.corpus_is_indonesian(str(tmp_path / "tidak_ada")) is True  # gagal baca -> teks lama


def test_aplikasi_berjalan_dari_folder_kerja_lain(semantic_ok, tmp_path, monkeypatch):
    """Regresi: aset (logo) dirujuk absolut sehingga tidak crash bila cwd bukan folder proyek."""
    lain = tmp_path / "lain"; lain.mkdir(); monkeypatch.chdir(lain)
    assert not buka(DASH).exception
