"""Kalibrasi ambang Adaptive QE TANPA label: lihat sebaran indikator & keputusan pada kueri nyata.

    python -m scripts.calibrate_adaptive --artifacts-dir artifacts_real --queries data/evaluation_queries.csv

Keluaran: results/adaptive_calibration.csv (per kueri) + ringkasan (sebaran tingkat, kuantil indikator).
Skrip ini TIDAK memakai penilaian relevansi dan tidak menilai kemanfaatan ekspansi; ia hanya memeriksa apakah ambang
menghasilkan sebaran keputusan yang masuk akal (mis. bukan 100% satu tingkat). Bila ambang diubah SETELAH pool
anotasi dibuat, buat ulang pool (Unjudged@K akan naik). Jangan menyetel ambang memakai label evaluasi yang sama
dengan yang dipakai untuk melapor (kebocoran)."""
import argparse
from pathlib import Path

import pandas as pd

from src import retrieval as R
from src.query_expansion import DEFAULT_CONFIG, AdaptiveConfig, adaptive_expand_query, score_diagnostics

INDICATORS = ["top1_z", "top5_mean_z", "gap_z", "spread_z"]


def calibrate(queries, docs, embeddings, model, config: AdaptiveConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    rows = []
    for q in queries:
        scores = R.semantic_scores(q, embeddings, model)
        initial = R._format_results(docs, scores, R.INITIAL_DEPTH)
        info = adaptive_expand_query(q, initial, score_diagnostics(scores), config)
        ind = info.get("indicators") or {}
        rows.append({"query": q, "n_words": info["word_count"], "level": info["expansion_level"],
                     "basis": info["decision_basis"], "terms": "; ".join(info["expansion_terms"]),
                     **{k: ind.get(k) for k in INDICATORS}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    share = df["level"].value_counts(normalize=True).reindex(["none", "light", "strong"]).fillna(0.0)
    q = df[INDICATORS].quantile([.1, .25, .5, .75, .9]).round(3) if len(df) else pd.DataFrame()
    return {"share": share, "quantiles": q, "n": len(df)}


def main(argv=None, model=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts-dir", default="artifacts_real")
    ap.add_argument("--queries", default="data/evaluation_queries.csv")
    ap.add_argument("--out", default="results/adaptive_calibration.csv")
    a = ap.parse_args(argv)
    if not Path(a.queries).exists():
        print(f"Berkas kueri {a.queries} belum ada. Contoh draf: data/templates/evaluation_queries_DRAFT_corpus900.csv")
        return 1
    qs = pd.read_csv(a.queries, encoding="utf-8-sig", dtype=str, keep_default_na=False)["query"].str.strip().tolist()
    docs = R.load_documents(a.artifacts_dir)
    emb = R.load_embeddings(a.artifacts_dir, docs)
    model = model or R.load_embedding_model()
    df = calibrate(qs, docs, emb, model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8")
    s = summarize(df)
    print(f"{s['n']} kueri. Sebaran keputusan:\n{(s['share'] * 100).round(1).to_string()}\n\nKuantil indikator:\n{s['quantiles'].to_string()}")
    if s["share"].max() > 0.8:
        print("\nPERHATIAN: >80% kueri jatuh ke satu tingkat; ambang kemungkinan kurang cocok untuk kueri ini. "
              "Pertimbangkan menggeser ambang mendekati kuantil indikator di atas, lalu dokumentasikan alasannya.")
    print(f"\nTersimpan: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
