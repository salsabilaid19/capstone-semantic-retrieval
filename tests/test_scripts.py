"""Tes integrasi skrip CLI dengan fixture rekaan; semua berjalan di folder sementara."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fakes import FakeModel
from oai_helpers import FIX, oai_fetch
from scripts import build_dataset, check_sources, collect_oai, crossref_enrich, import_csv, validate_dataset
from src.artifacts import build_model_config
from src.harvest import HarvestError, SOURCE_COLUMNS, load_sources, save_sources
from src.retrieval import MODEL_NAME, load_artifacts

ROOT = FIX.parents[1]


def source(**kw):
    s = dict.fromkeys(SOURCE_COLUMNS, "")
    s.update(source_id="fx", name="Jurnal Fixture", license="CC BY 4.0", oai_base_url="http://fixture.invalid/oai",
             enabled=True, verification="registry_reported")
    s.update(kw)
    return s


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Ruang kerja sementara dengan config + topic terms."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    save_sources([source()], tmp_path / "config" / "journals.csv")
    shutil.copy(ROOT / "config" / "topic_terms.txt", tmp_path / "config" / "topic_terms.txt")
    return tmp_path


def harvest(ws, **extra):
    return collect_oai.main(["--delay", "0", "--out-dir", "data/real/harvest"], fetch=oai_fetch, sleep=lambda s: None)


# ---------- check_sources ----------
def test_check_sources_ok_dan_update_config(ws):
    assert check_sources.main(["--update-config", "--delay", "0"], fetch=oai_fetch) == 0
    s = load_sources("config/journals.csv")[0]
    assert s["verification"] == "verified" and s["verified_on"] and s["enabled"]


def test_check_sources_fallback_ke_url_alternatif(ws):
    save_sources([source(oai_base_url="http://salah.invalid/oai", oai_alt_urls="http://fixture.invalid/oai", enabled=False)],
                 "config/journals.csv")

    def fetch(url):
        if "salah.invalid" in url:
            raise HarvestError("HTTP 404")
        return oai_fetch(url)
    r = check_sources.check_source(load_sources("config/journals.csv")[0], fetch)
    assert r["ok"] and r["url"] == "http://fixture.invalid/oai"
    assert r["pct_abstract"] == 100 and r["pct_keywords"] == 50 and r["pct_doi"] == 50     # 2 record valid pada halaman 1


def test_check_sources_gagal_tidak_mengubah_config(ws):
    def fetch(url):
        raise HarvestError("HTTP 404")
    before = Path("config/journals.csv").read_text(encoding="utf-8")
    assert check_sources.main(["--update-config", "--delay", "0"], fetch=fetch) == 1
    assert Path("config/journals.csv").read_text(encoding="utf-8") == before


def test_check_sources_butuh_email_tanpa_fetch(ws, capsys, monkeypatch):
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    assert check_sources.main([]) == 2 and "contact-email" in capsys.readouterr().err


def test_check_sources_url_kosong():
    r = check_sources.check_source(source(oai_base_url=""), oai_fetch)
    assert not r["ok"] and "kosong" in r["detail"]


# ---------- collect_oai ----------
def test_collect_menulis_csv_real(ws):
    assert harvest(ws) == 0
    df = pd.read_csv("data/real/harvest/fx.csv", dtype=str, keep_default_na=False)
    assert len(df) == 4 and set(df.Data_Source) == {"real"} and set(df.Publication) == {"Jurnal Fixture"}


def test_collect_endpoint_mati_dan_tanpa_sumber_aktif(ws, capsys):
    def mati(url):
        raise HarvestError("HTTP 404")
    assert collect_oai.main(["--delay", "0"], fetch=mati, sleep=lambda s: None) == 1
    assert not Path("data/real/harvest/fx.csv").exists()
    save_sources([source(enabled=False)], "config/journals.csv")
    assert collect_oai.main(["--delay", "0"], fetch=oai_fetch) == 2
    assert collect_oai.main(["--delay", "0", "--sources", "fx"], fetch=oai_fetch, sleep=lambda s: None) == 0   # eksplisit boleh


def test_collect_gagal_parsial_menyimpan_yang_didapat(ws):
    def fetch(url):
        if "resumptionToken" in url:
            raise HarvestError("putus")
        return oai_fetch(url)
    assert collect_oai.main(["--delay", "0"], fetch=fetch, sleep=lambda s: None) == 1
    assert len(pd.read_csv("data/real/harvest/fx.csv")) == 2


def test_collect_tanpa_email_ditolak(ws, monkeypatch):
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    assert collect_oai.main([]) == 2


# ---------- import_csv ----------
def _csv(tmp, rows, name="in.csv"):
    p = Path(tmp) / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_import_real_wajib_source_note_dan_menolak_klaim_bertentangan(ws, capsys):
    src = _csv(ws, [{"Judul": "Paper A", "Abstrak": "Isi abstrak A.", "Tahun": "2021.0", "DOI": "10.5555/x1"}])
    assert import_csv.main(["--input", src, "--name", "m", "--data-source", "real"]) == 2
    assert import_csv.main(["--input", src, "--name", "m", "--data-source", "real", "--source-note", "Manual dari situs X"]) == 0
    out = pd.read_csv("data/real/imports/m.csv", dtype=str, keep_default_na=False)
    assert out.Year.iloc[0] == "2021" and out.Data_Source.iloc[0] == "real" and "Manual" in out.Source_Note.iloc[0]
    bentrok = _csv(ws, [{"Title": "T", "Abstract": "A", "Data_Source": "synthetic"}], "b.csv")
    assert import_csv.main(["--input", bentrok, "--name", "b", "--data-source", "real", "--source-note", "n"]) == 2


def test_import_membuang_baris_kosong_dan_kolom_wajib(ws, capsys):
    src = _csv(ws, [{"Title": "Ada", "Abstract": "Isi"}, {"Title": "Tanpa abstrak", "Abstract": ""}])
    assert import_csv.main(["--input", src, "--name", "s", "--data-source", "synthetic"]) == 0
    assert len(pd.read_csv("data/synthetic/imports/s.csv")) == 1
    assert import_csv.main(["--input", _csv(ws, [{"Foo": "x"}], "x.csv"), "--name", "z", "--data-source", "synthetic"]) == 2


# ---------- crossref_enrich ----------
def test_crossref_enrich_cli(ws):
    from test_crossref import WORK, FakeCrossref
    pd.DataFrame({"Title": ["Klasifikasi Sentimen Ulasan Aplikasi Menggunakan Naive Bayes"], "Abstract": [""],
                  "DOI": ["10.99999/fixture.1"], "Year": [""], "Publication": [""], "Authors": [""]}).to_csv("a.csv", index=False)
    assert crossref_enrich.main(["--input", "a.csv", "--output", "b.csv", "--cache", "c.json", "--delay", "0"],
                                fetch=FakeCrossref({"fixture.1": WORK})) == 0
    out = pd.read_csv("b.csv", dtype=str, keep_default_na=False)
    assert out.Year.iloc[0] == "2024" and out.Crossref_Status.iloc[0] == "ok"
    pd.DataFrame({"Title": ["x"]}).to_csv("n.csv", index=False)
    assert crossref_enrich.main(["--input", "n.csv", "--output", "o.csv"], fetch=FakeCrossref({})) == 2


# ---------- build_dataset: alur lengkap ----------
def test_alur_lengkap_harvest_build_embedding_muat(ws):
    harvest(ws)
    # Fixture ber-DOI 10.99999/fixture.* sengaja ditolak validator bila diklaim 'real' (penjaga data sintetis)
    assert build_dataset.main(["--kind", "real"]) == 1
    assert not Path("artifacts_real/documents_processed.csv").exists()
    assert build_dataset.main(["--kind", "real", "--force"]) == 0       # --force hanya untuk pengujian alur ini
    docs = pd.read_csv("artifacts_real/documents_processed.csv", dtype=str, keep_default_na=False)
    # 4 baris panen -> duplikat DOI (beda huruf besar) & tanpa abstrak dibuang -> 2 dokumen
    assert len(docs) == 2 and set(docs.data_source) == {"real"}
    man = json.loads(Path("artifacts_real/dataset_manifest.json").read_text(encoding="utf-8"))
    assert man["kind"] == "real" and man["sources"][0]["license"] == "CC BY 4.0" and man["num_documents"] == 2
    assert Path("data/real/documents_raw.csv").exists()
    # embedding (model PALSU) + muat dengan validasi hash -> kompatibel dengan aplikasi
    emb = FakeModel().encode(["passage: " + t for t in docs.document_text])
    np.save("artifacts_real/embeddings.npy", emb)
    Path("artifacts_real/model_config.json").write_text(
        json.dumps(build_model_config(MODEL_NAME, docs.doc_id.tolist(), emb.shape[1])), encoding="utf-8")
    loaded, e = load_artifacts("artifacts_real")
    assert len(loaded) == 2 and e.shape == (2, 64)
    assert man["doc_ids_sha1"] == json.loads(Path("artifacts_real/model_config.json").read_text())["doc_ids_sha1"]


def test_build_real_menolak_baris_sintetis(ws, capsys):
    harvest(ws)
    Path("data/real/imports").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Title": "S", "Abstract": "A", "Data_Source": "synthetic"}]).to_csv("data/real/imports/s.csv", index=False)
    assert build_dataset.main(["--kind", "real"]) == 2
    assert not Path("artifacts_real/documents_processed.csv").exists()
    assert "dipisah" in capsys.readouterr().err


def test_build_real_gagal_validasi_tidak_menulis_kecuali_force(ws):
    Path("data/real/imports").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Title": "Tanpa Sumber", "Abstract": "Abstrak cukup panjang untuk contoh pengujian validasi dataset real.",
                   "Data_Source": "real", "Source_Note": "manual"}]).to_csv("data/real/imports/x.csv", index=False)
    assert build_dataset.main(["--kind", "real"]) == 1
    assert not Path("artifacts_real/documents_processed.csv").exists()
    assert build_dataset.main(["--kind", "real", "--force"]) == 0           # --force menulis walau ada error validasi
    assert Path("artifacts_real/documents_processed.csv").exists()


def test_build_tanpa_input(ws, capsys):
    assert build_dataset.main(["--kind", "real"]) == 2


def test_build_synthetic_dari_data_contoh_bawaan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "synthetic").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "synthetic" / "sample_synthetic.csv", tmp_path / "data" / "synthetic")
    assert build_dataset.main(["--kind", "synthetic"]) == 0
    df = pd.read_csv("artifacts/documents_processed.csv", dtype=str, keep_default_na=False)
    assert len(df) == 13 and set(df.data_source) == {"synthetic"}
    assert df.equals(pd.read_csv(ROOT / "artifacts" / "documents_processed.csv", dtype=str, keep_default_na=False))


# ---------- validate_dataset ----------
def test_validate_script_artifacts_bawaan(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    assert validate_dataset.main(["--artifacts-dir", "artifacts", "--kind", "synthetic", "--results-dir", str(tmp_path)]) == 0
    assert validate_dataset.main(["--artifacts-dir", "artifacts", "--kind", "real", "--results-dir", str(tmp_path)]) == 1
    assert validate_dataset.main(["--artifacts-dir", str(tmp_path / "kosong"), "--results-dir", str(tmp_path)]) == 2


# ---------- exclude-types & derive_keywords ----------
def test_import_csv_exclude_types(ws):
    import pandas as pd
    rows = [{"Title": "A", "Abstract": "isi satu", "Type": "journal-article", "DOI": "10.1/1"},
            {"Title": "B", "Abstract": "isi dua", "Type": "edited-book", "DOI": "10.1/2"}]
    src = _csv(ws, rows, "t.csv")
    out_dir = ws / "o"
    assert import_csv.main(["--input", src, "--name", "t", "--data-source", "real", "--source-note", "n",
                            "--exclude-types", "edited-book, monograph", "--out-dir", str(out_dir)]) == 0
    assert list(pd.read_csv(out_dir / "t.csv")["Title"]) == ["A"]


def test_derive_keywords_hanya_mengisi_yang_kosong_dan_menandai_catatan():
    import pandas as pd
    from scripts import derive_keywords as dk
    df = pd.DataFrame({
        "Title": ["Deteksi banjir sensor", "Klasifikasi teks berita", "Sensor banjir pintar"],
        "Abstract": ["sensor air banjir sungai", "klasifikasi teks berita online", "banjir sensor sungai kota"],
        "Keywords": ["asli; penulis", "", ""], "Source_Note": ["catatan", "", ""]})
    out = dk.derive(df, top=3, min_df=1)
    assert out.loc[0, "Keywords"] == "asli; penulis" and out.loc[0, "Source_Note"] == "catatan"
    assert out.loc[1, "Keywords"] != "" and dk.NOTE in out.loc[1, "Source_Note"]
    assert dk.derive(out, top=3, min_df=1).loc[1, "Source_Note"].count(dk.NOTE) == 1
