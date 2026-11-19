"""Tes evaluasi (metrik, validasi label, statistik, pooling, CLI).
SEMUA kueri/label di sini adalah FIXTURE REKAAN yang dibuat otomatis dari aturan sederhana pada data rekaan;
bukan label penelitian dan tidak mewakili hasil evaluasi sistem."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fakes import RAW, FakeModel, make_artifacts
from src import evaluation as ev
from src.pooling import build_pool, pool_to_labels
from src.retrieval import (ALL_METHODS, METHOD_ADAPTIVE, METHOD_BM25, METHOD_SEMANTIC, METHOD_STATIC, build_bm25_index,
                           load_artifacts)
from scripts import agreement, make_pool, pool_to_labels as pool_to_labels_cli

QUERIES = [("Q1", "banjir"), ("Q2", "hoaks"), ("Q3", "sentimen"), ("Q4", "padi"), ("Q5", "hujan banjir"),
           ("Q6", "sistem deteksi banjir dengan sensor air"), ("Q7", "klasifikasi berita hoaks pada media daring"),
           ("Q8", "analisis sentimen ulasan aplikasi seluler"), ("Q9", "jaringan saraf untuk prediksi hujan harian"),
           ("Q10", "kriptografi")]       # Q10: tak ada dokumen relevan -> harus dikecualikan


def rule_labels(queries, docs):
    """Label REKAAN: relevan bila kata (>3 huruf) kueri ada di judul (2) atau abstrak (1); Q10 hanya berlabel 0."""
    rows = []
    for qid, q in queries:
        words = [w for w in q.lower().split() if len(w) > 3]
        for _, d in docs.iterrows():
            t, a = d["title"].lower(), d["abstract"].lower()
            rel = 2 if any(w in t for w in words) else 1 if any(w in a for w in words) else 0
            if qid == "Q10" or rel > 0 or d["doc_id"] == docs["doc_id"].iloc[0]:
                rows.append({"query_id": qid, "doc_id": d["doc_id"], "relevance": str(0 if qid == "Q10" else rel)})
    return pd.DataFrame(rows)


@pytest.fixture
def env(tmp_path):
    adir = tmp_path / "artifacts_real"
    adir.mkdir()
    make_artifacts(adir, data_source="real")
    docs, emb = load_artifacts(str(adir))
    queries = pd.DataFrame(QUERIES, columns=["query_id", "query"])
    labels = rule_labels(QUERIES, docs)
    queries.to_csv(tmp_path / "q.csv", index=False)
    labels.to_csv(tmp_path / "l.csv", index=False)
    return {"dir": adir, "docs": docs, "emb": emb, "queries": queries, "labels": labels, "tmp": tmp_path,
            "bm25": build_bm25_index(docs)}


# ---------- metrik ----------
def test_metrik_nilai_diketahui():
    rel = {"a": 2, "b": 0, "c": 1}
    assert ev.precision_at_k(["a", "b", "c"], rel, 3) == pytest.approx(2 / 3)
    assert ev.mrr_at_k(["a", "b", "c"], rel, 3) == 1.0
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert ev.ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx((2 + 1 / math.log2(4)) / ideal)
    assert ev.ndcg_at_k(["a", "c", "b"], rel, 3) == pytest.approx((2 + 1 / math.log2(3)) / ideal) == pytest.approx(1.0)
    assert ev.mrr_at_k(["x", "y", "c"], rel, 3) == pytest.approx(1 / 3)
    assert ev.mrr_at_k(["x", "y"], rel, 3) == 0.0 and ev.ndcg_at_k(["a"], {}, 3) == 0.0


def test_unjudged_dan_segmen():
    assert ev.unjudged_at_k(["a", "x", "y", "b"], {"a": 1, "b": 0}, 4) == 0.5
    assert ev.unjudged_at_k([], {}, 3) == 0.0
    assert ev.segment_of("banjir") == ev.SHORT_SEGMENT and ev.segment_of("a b c") == ev.SHORT_SEGMENT
    assert ev.segment_of("a b c d") == ev.LONG_SEGMENT


# ---------- validasi berkas ----------
def test_validasi_lolos_dan_peringatan_jumlah(env):
    rep = ev.validate_evaluation_files(env["queries"], env["labels"], env["docs"])
    assert rep["errors"] == []
    w = " ".join(rep["warnings"])
    assert "tanpa dokumen relevan" in w and "< 30" in w
    assert rep["stats"]["kueri_dapat_dievaluasi"] == 9


def test_validasi_error_label(env):
    q, d = env["queries"], env["docs"]
    L = lambda rows: pd.DataFrame(rows, columns=["query_id", "doc_id", "relevance"])        # noqa: E731
    did = d["doc_id"].iloc[0]
    assert any("di luar" in e for e in ev.validate_evaluation_files(q, L([["Q1", did, "3"]]), d)["errors"])
    assert any("di luar" in e for e in ev.validate_evaluation_files(q, L([["Q1", did, ""]]), d)["errors"])
    assert any("di luar" in e for e in ev.validate_evaluation_files(q, L([["Q1", did, "abc"]]), d)["errors"])
    assert any("TIDAK ada di dataset" in e for e in ev.validate_evaluation_files(q, L([["Q1", "DOC-hilang", "1"]]), d)["errors"])
    assert any("tidak ada di berkas kueri" in e for e in ev.validate_evaluation_files(q, L([["QX", did, "1"]]), d)["errors"])
    conflict = ev.validate_evaluation_files(q, L([["Q1", did, "1"], ["Q1", did, "2"]]), d)
    assert any("BERBEDA" in e for e in conflict["errors"])
    same = ev.validate_evaluation_files(q, L([["Q1", did, "1"], ["Q1", did, "1"]]), d)
    assert same["errors"] == [] and any("label ganda" in w for w in same["warnings"])


def test_validasi_error_kueri(env):
    d = env["docs"]
    q = pd.DataFrame([["Q1", "a"], ["Q1", "b"], ["Q2", ""]], columns=["query_id", "query"])
    errs = " ".join(ev.validate_evaluation_files(q, env["labels"].iloc[0:0], d)["errors"])
    assert "duplikat" in errs and "kosong" in errs


def test_validasi_peringatan_segmen_kecil(env):
    rep = ev.validate_evaluation_files(env["queries"], env["labels"], env["docs"])
    assert any("Segmen kueri" in w for w in rep["warnings"])


def test_load_files_hilang_menyebut_templat(tmp_path):
    with pytest.raises(FileNotFoundError, match="templat"):
        ev.load_evaluation_files(tmp_path / "a.csv", tmp_path / "b.csv")
    (tmp_path / "a.csv").write_text("id,teks\n1,x\n"); (tmp_path / "b.csv").write_text("query_id,doc_id,relevance\n")
    with pytest.raises(ValueError, match="Kolom"):
        ev.load_evaluation_files(tmp_path / "a.csv", tmp_path / "b.csv")


# ---------- run_evaluation ----------
def run(env, **kw):
    return ev.run_evaluation(env["docs"], env["emb"], FakeModel(), env["bm25"], env["queries"], env["labels"], k=3, **kw)


def test_run_evaluation_struktur_dan_pengecualian(env):
    pq, summary = run(env)
    assert len(pq) == 10 * 4 and {"segment", "n_words", "unjudged@3", "n_relevant_known"} <= set(pq.columns)
    assert summary["n_queries"].tolist() == [9] * 4 and summary["method"].tolist() == ALL_METHODS     # Q10 dikecualikan
    manual = pq[(pq.method == METHOD_BM25) & (pq.n_relevant_known > 0)]["precision@3"].mean()
    assert summary.loc[summary.method == METHOD_BM25, "precision@3"].iloc[0] == pytest.approx(manual)
    _, with_all = run(env, exclude_no_relevant=False)
    assert with_all["n_queries"].tolist() == [10] * 4
    assert (with_all["precision@3"].values <= summary["precision@3"].values + 1e-12).all()    # menyertakan nol menurunkan rata-rata


def test_per_segmen_dan_expanded_rate(env):
    pq, _ = run(env)
    seg = ev.summarize_by_segment(pq, 3)
    s = seg.set_index(["segment", "method"])
    assert s.loc[(ev.LONG_SEGMENT, METHOD_ADAPTIVE), "expanded_rate"] == 0.0           # >3 kata: Adaptive tak pernah memperluas
    assert s.loc[(ev.SHORT_SEGMENT, METHOD_BM25), "expanded_rate"] == 0.0
    assert s.loc[(ev.SHORT_SEGMENT, METHOD_STATIC), "expanded_rate"] == s.loc[(ev.SHORT_SEGMENT, METHOD_ADAPTIVE), "expanded_rate"]


def test_pooling_invarian_unjudged_nol_saat_depth_sama_dengan_k(env):
    pool, prov = build_pool(env["queries"], env["docs"], env["emb"], FakeModel(), env["bm25"], depth=3)
    assert "method" not in " ".join(pool.columns).lower() and (pool["relevance"] == "").all()
    assert {f"rank[{m}]" for m in ALL_METHODS} <= set(prov.columns)
    # isi pool dengan label rekaan untuk SEMUA pasangan, lalu evaluasi: unjudged harus 0 untuk keempat metode
    lab = rule_labels(QUERIES, env["docs"]).set_index(["query_id", "doc_id"])["relevance"]
    pool["relevance"] = [lab.get((q, d), "0") for q, d in zip(pool.query_id, pool.doc_id)]
    labels, problems = pool_to_labels(pool, annotator="fixture")
    assert problems == [] and set(labels.label_source) == {"pooled_manual"} and set(labels.annotator) == {"fixture"}
    pq, summary = ev.run_evaluation(env["docs"], env["emb"], FakeModel(), env["bm25"], env["queries"], labels, k=3)
    assert (pq["unjudged@3"] == 0).all() and (summary["unjudged@3"] == 0).all()


def test_pool_acak_deterministik_dan_metode_tersembunyi(env):
    a, _ = build_pool(env["queries"], env["docs"], env["emb"], FakeModel(), env["bm25"], depth=6, seed=1)
    b, _ = build_pool(env["queries"], env["docs"], env["emb"], FakeModel(), env["bm25"], depth=6, seed=1)
    c, _ = build_pool(env["queries"], env["docs"], env["emb"], FakeModel(), env["bm25"], depth=6, seed=2)
    assert a.equals(b) and sorted(a.doc_id) == sorted(c.doc_id)
    q1 = a[a.query_id == "Q1"]
    assert len(q1) == len(set(q1.doc_id)) == 6                                        # union, tanpa duplikat


def test_pool_to_labels_blank_invalid_dan_allow_blank():
    f = pd.DataFrame({"query_id": ["Q1"] * 3, "doc_id": ["a", "b", "c"], "relevance": ["2", "", "0"],
                      "annotator": ["", "", "X"], "notes": ""})
    out, problems = pool_to_labels(f)
    assert out.empty and "belum dinilai" in problems[0]
    out, problems = pool_to_labels(f, annotator="AB", allow_blank=True)
    assert problems == [] and out["relevance"].tolist() == [2, 0] and out["annotator"].tolist() == ["AB", "X"]
    f.loc[0, "relevance"] = "5"
    assert "selain 0/1/2" in pool_to_labels(f, allow_blank=True)[1][0]


# ---------- statistik ----------
def test_pvalue_eksak_dan_monte_carlo():
    assert ev.paired_randomization_pvalue([0, 0, 0]) == 1.0
    assert ev.paired_randomization_pvalue([0.1] * 10) == pytest.approx(2 / 1024)
    assert ev.paired_randomization_pvalue([0.1, -0.1, 0.1, -0.1]) == 1.0
    big = [0.1] * 20
    p1, p2 = ev.paired_randomization_pvalue(big, seed=3), ev.paired_randomization_pvalue(big, seed=3)
    assert p1 == p2 and p1 < 0.001


def test_bootstrap_ci_dan_holm():
    lo, hi = ev.bootstrap_ci([0.1, 0.2, 0.3, 0.4], seed=1)
    assert lo <= 0.25 <= hi and ev.bootstrap_ci([0.1, 0.2], seed=1) == ev.bootstrap_ci([0.1, 0.2], seed=1)
    assert all(math.isnan(x) for x in ev.bootstrap_ci([]))
    assert ev.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ev.holm_adjust([0.9, 0.9]) == [1.0, 1.0]                      # 2*0.9 dibatasi 1.0
    assert ev.holm_adjust([0.001]) == [0.001]


def test_tabel_signifikansi(env):
    pq, _ = run(env)
    sig = ev.significance_table(pq, 3, n_resamples=500)
    assert len(sig) == 3 * 4 and set(sig.method_b) == {METHOD_SEMANTIC, METHOD_STATIC, METHOD_ADAPTIVE}
    assert (sig["p_holm"] >= sig["p_value"] - 1e-12).all() and (sig["p_holm"] <= 1).all()
    sa = sig[(sig.method_a == METHOD_STATIC) & (sig.method_b == METHOD_ADAPTIVE)]
    n_long_expanded = int(((pq.method == METHOD_STATIC) & (pq.segment == ev.LONG_SEGMENT) & pq.is_expanded & (pq.n_relevant_known > 0)).sum())
    assert (sa["n_changed"] == n_long_expanded).all()


def test_static_sama_adaptive_pada_kueri_pendek_p_satu(env):
    short = env["queries"][env["queries"]["query"].map(ev.segment_of) == ev.SHORT_SEGMENT]
    pq, _ = ev.run_evaluation(env["docs"], env["emb"], FakeModel(), env["bm25"], short, env["labels"], k=3)
    sa = ev.significance_table(pq, 3, n_resamples=200)
    sa = sa[(sa.method_a == METHOD_STATIC) & (sa.method_b == METHOD_ADAPTIVE)]
    assert (sa["n_changed"] == 0).all() and (sa["mean_diff_b_minus_a"] == 0).all() and (sa["p_value"] == 1.0).all()


def test_kappa():
    assert ev.cohens_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1.0)
    assert ev.cohens_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)
    assert math.isnan(ev.cohens_kappa([1, 1], [1, 1]))                   # tidak terdefinisi: tanpa variasi kategori
    assert math.isnan(ev.cohens_kappa([], []))


# ---------- CLI ----------
def cli(env, *extra):
    return ev.main(["--k", "3", "--artifacts", str(env["dir"]), "--queries", str(env["tmp"] / "q.csv"),
                    "--labels", str(env["tmp"] / "l.csv"), "--results-dir", str(env["tmp"] / "res"), *extra], model=FakeModel())


def test_cli_sukses_menulis_hasil_dan_metadata(env, capsys):
    assert cli(env) == 0
    res = env["tmp"] / "res"
    for name in ("per_query", "summary", "by_segment", "significance"):
        assert (res / f"evaluation_{name}.csv").exists()
    meta = json.loads((res / "evaluation_run.json").read_text(encoding="utf-8"))
    assert meta["dataset_kind"] == ["real"] and len(meta["queries_sha256"]) == 64 and len(meta["labels_sha256"]) == 64
    assert meta["kueri_dikecualikan_tanpa_relevan"] == 1 and meta["k"] == 3
    out = capsys.readouterr().out
    assert "BUKAN temuan penelitian" not in out and "[PERINGATAN]" in out                  # peringatan jumlah kueri


def test_cli_menolak_dataset_sintetis_kecuali_diizinkan(env, tmp_path, capsys):
    sdir = tmp_path / "syn"
    sdir.mkdir()
    make_artifacts(sdir, data_source="synthetic")
    docs, _ = load_artifacts(str(sdir))
    rule_labels(QUERIES, docs).to_csv(env["tmp"] / "l.csv", index=False)       # doc_id sama (stabil), jadi label tetap cocok
    args = ["--k", "3", "--artifacts", str(sdir), "--queries", str(env["tmp"] / "q.csv"),
            "--labels", str(env["tmp"] / "l.csv"), "--results-dir", str(env["tmp"] / "res2")]
    assert ev.main(args, model=FakeModel()) == 2 and not (env["tmp"] / "res2").exists()
    assert "DITOLAK" in capsys.readouterr().out
    assert ev.main(args + ["--allow-synthetic"], model=FakeModel()) == 0
    assert "BUKAN temuan penelitian" in capsys.readouterr().out
    assert json.loads((env["tmp"] / "res2" / "evaluation_run.json").read_text())["dataset_sintetis_diizinkan"] is True


def test_cli_label_salah_membatalkan_tanpa_hasil(env, capsys):
    bad = env["labels"].copy(); bad.loc[0, "doc_id"] = "DOC-hilang"
    bad.to_csv(env["tmp"] / "l.csv", index=False)
    assert cli(env) == 1 and not (env["tmp"] / "res").exists()
    assert "[ERROR]" in capsys.readouterr().out


def test_cli_berkas_hilang(env, capsys):
    (env["tmp"] / "l.csv").unlink()
    assert cli(env) == 2 and "templat" in capsys.readouterr().out


# ---------- skrip pooling/anotasi ----------
def test_skrip_make_pool_dan_pool_to_labels(env, capsys):
    out = env["tmp"] / "ann"
    assert make_pool.main(["--queries", str(env["tmp"] / "q.csv"), "--artifacts-dir", str(env["dir"]), "--depth", "3",
                           "--out-dir", str(out)], model=FakeModel()) == 0
    pool = pd.read_csv(out / "pool.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert (pool["relevance"] == "").all() and (out / "pool_provenance.csv").exists()
    inp, lab = env["tmp"] / "terisi.csv", env["tmp"] / "labels_out.csv"
    pool.to_csv(inp, index=False)
    assert pool_to_labels_cli.main(["--input", str(inp), "--output", str(lab)]) == 1          # masih kosong -> ditolak
    assert not lab.exists()
    pool["relevance"] = "1"
    pool.to_csv(inp, index=False)
    assert pool_to_labels_cli.main(["--input", str(inp), "--output", str(lab), "--annotator", "Fx"]) == 0
    assert pool_to_labels_cli.main(["--input", str(inp), "--output", str(lab)]) == 2          # tidak menimpa tanpa --overwrite
    assert set(pd.read_csv(lab).annotator) == {"Fx"}


def test_make_pool_menolak_sintetis_dan_tanpa_kueri(env, tmp_path, capsys):
    assert make_pool.main(["--queries", str(tmp_path / "tidak_ada.csv")]) == 2
    sdir = tmp_path / "syn"; sdir.mkdir(); make_artifacts(sdir, data_source="synthetic")
    assert make_pool.main(["--queries", str(env["tmp"] / "q.csv"), "--artifacts-dir", str(sdir), "--out-dir", str(tmp_path / "o")],
                          model=FakeModel()) == 2


def test_skrip_agreement(tmp_path, capsys):
    a = pd.DataFrame({"query_id": ["Q1"] * 4, "doc_id": list("abcd"), "relevance": ["2", "1", "0", "0"], "title": "t"})
    b = a.copy(); b.loc[1, "relevance"] = "0"
    a.to_csv(tmp_path / "a.csv", index=False); b.to_csv(tmp_path / "b.csv", index=False)
    assert agreement.main(["--a", str(tmp_path / "a.csv"), "--b", str(tmp_path / "b.csv"), "--out", str(tmp_path / "d.csv")]) == 0
    out = capsys.readouterr().out
    assert "75.0%" in out and "Cohen's kappa" in out
    assert len(pd.read_csv(tmp_path / "d.csv")) == 1
