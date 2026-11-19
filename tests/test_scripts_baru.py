"""Uji skrip kalibrasi Adaptive QE dan penanganan abstrak pendek."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from scripts import calibrate_adaptive as cal, fix_short_abstracts as fix  # noqa: E402
from src import retrieval as R  # noqa: E402
from src.artifacts import doc_ids_hash  # noqa: E402

REAL = ROOT / "artifacts_real"
needs_real = pytest.mark.skipif(not (REAL / "embeddings.npy").exists(), reason="artifacts_real tidak ada")


def test_summarize_sebaran_tingkat():
    df = pd.DataFrame({"level": ["none", "none", "light", "strong"], "top1_z": [4, 5, 3.8, 2.5], "top5_mean_z": [3, 4, 3, 2],
                       "gap_z": [.1, .2, .1, .0], "spread_z": [1, 1, 1, .3]})
    s = cal.summarize(df)
    assert s["share"].to_dict() == {"none": .5, "light": .25, "strong": .25} and s["n"] == 4


@needs_real
def test_kalibrasi_pada_data_nyata_dengan_model_proksi(tmp_path):
    from proxy_model import ProxyModel
    d = R.load_documents(str(REAL)); e = R.load_embeddings(str(REAL), d)
    df = cal.calibrate(["analisis sentimen", "forecasting", "klasifikasi citra medis dengan deep learning pada rumah sakit"], d, e, ProxyModel(d, e))
    assert len(df) == 3 and set(df["level"]) <= {"none", "light", "strong"} and (df["basis"] == "quality").all()
    assert df[cal.INDICATORS].notna().all().all()


def test_kalibrasi_tanpa_berkas_kueri_gagal_dengan_pesan(tmp_path, capsys):
    assert cal.main(["--queries", str(tmp_path / "tidak_ada.csv")]) == 1
    assert "belum ada" in capsys.readouterr().out


def _mini(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    docs = pd.DataFrame({"doc_id": ["A", "B", "C"], "journal": ["J"] * 3, "year": ["2020"] * 3, "title": ["t1", "Halaman Depan", "t3"],
                         "abstract": ["x" * 250, "pendek", "y" * 300]})
    docs.to_csv(src / "documents_processed.csv", index=False)
    np.save(src / "embeddings.npy", np.arange(12, dtype=np.float32).reshape(3, 4))
    (src / "model_config.json").write_text(json.dumps({"num_documents": 3, "doc_ids_sha1": doc_ids_hash(docs["doc_id"])}))
    (src / "dataset_manifest.json").write_text("{}")
    return src


def test_fix_dry_run_tidak_mengubah_apa_pun(tmp_path, capsys):
    src = _mini(tmp_path); before = (src / "embeddings.npy").read_bytes()
    assert fix.main(["--artifacts-dir", str(src)]) == 0
    assert "Dry-run" in capsys.readouterr().out and (src / "embeddings.npy").read_bytes() == before
    assert not (tmp_path / "artifacts_real_filtered").exists()


def test_fix_apply_menulis_salinan_baru_dengan_hash_dan_baris_embedding_benar(tmp_path):
    src = _mini(tmp_path); out = tmp_path / "out"
    assert fix.main(["--artifacts-dir", str(src), "--apply", "--out", str(out)]) == 0
    nd = pd.read_csv(out / "documents_processed.csv", dtype=str)
    assert nd["doc_id"].tolist() == ["A", "C"]
    ne = np.load(out / "embeddings.npy"); old = np.load(src / "embeddings.npy")
    assert ne.shape == (2, 4) and np.array_equal(ne, old[[0, 2]])
    cfg = json.loads((out / "model_config.json").read_text())
    assert cfg["num_documents"] == 2 and cfg["doc_ids_sha1"] == doc_ids_hash(nd["doc_id"])
    assert len(pd.read_csv(src / "documents_processed.csv")) == 3          # sumber utuh


def test_fix_menolak_menimpa_folder_sumber(tmp_path):
    src = _mini(tmp_path)
    assert fix.main(["--artifacts-dir", str(src), "--apply", "--out", str(src)]) == 2
