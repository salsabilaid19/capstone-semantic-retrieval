"""Evaluasi offline: Precision@k, nDCG@k, MRR@k untuk 4 metode, dengan validasi label, pelaporan
dokumen belum dinilai (unjudged), segmentasi panjang kueri, dan uji signifikansi berpasangan.

Jalankan dari root proyek:  python -m src.evaluation --artifacts artifacts_real
Kode ini terpisah dari UI agar eksperimen dapat diulang (reproducible).

Asumsi metrik (harus ditulis di laporan):
- Dokumen tanpa label dianggap tidak relevan (praktik pooling standar). Karena itu `unjudged@k` dilaporkan.
- Precision@k membagi dengan k (bukan dengan jumlah hasil). Dokumen relevan = relevance >= 1.
- nDCG memakai gain = nilai relevance (0/1/2); DCG ideal dihitung dari dokumen BERLABEL saja.
- Kueri tanpa satu pun dokumen relevan dikecualikan dari rata-rata (metrik tak terdefinisi) dan dilaporkan.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.artifacts import doc_ids_hash
from src.query_expansion import count_query_words
from src.retrieval import (
    ALL_METHODS, METHOD_ADAPTIVE, METHOD_BM25, METHOD_SEMANTIC, METHOD_STATIC, MODEL_NAME,
    build_bm25_index, load_artifacts, load_embedding_model, search_with_method,
)

SHORT_SEGMENT, LONG_SEGMENT = "1-3 kata", ">3 kata"
PRIMARY_PAIRS = [(METHOD_BM25, METHOD_SEMANTIC), (METHOD_SEMANTIC, METHOD_STATIC),
                 (METHOD_SEMANTIC, METHOD_ADAPTIVE), (METHOD_STATIC, METHOD_ADAPTIVE)]
MIN_QUERIES = 30
UNJUDGED_WARN = 0.20

QUERIES_PATH = Path("data") / "evaluation_queries.csv"
LABELS_PATH = Path("data") / "relevance_labels.csv"
RESULTS_DIR = Path("results")
PER_QUERY_PATH = RESULTS_DIR / "evaluation_per_query.csv"
SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.csv"
VALID_RELEVANCE = (0, 1, 2)


# ---------------- Metrik ----------------
def precision_at_k(ranked_doc_ids: List[str], relevance: Dict[str, int], k: int = 10) -> float:
    """Proporsi dokumen relevan (relevance >= 1) pada k hasil teratas."""
    if k <= 0:
        return 0.0
    top = ranked_doc_ids[:k]
    hits = sum(1 for d in top if relevance.get(d, 0) >= 1)
    return hits / k


def dcg_at_k(gains: List[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_doc_ids: List[str], relevance: Dict[str, int], k: int = 10) -> float:
    """nDCG dengan gain = nilai relevance (0/1/2). Dokumen tanpa label dianggap 0."""
    actual = dcg_at_k([relevance.get(d, 0) for d in ranked_doc_ids[:k]], k)
    ideal = dcg_at_k(sorted(relevance.values(), reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked_doc_ids: List[str], relevance: Dict[str, int], k: int = 10) -> float:
    """Reciprocal rank dari dokumen relevan pertama dalam k teratas (0 bila tidak ada)."""
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        if relevance.get(d, 0) >= 1:
            return 1.0 / i
    return 0.0


def unjudged_at_k(ranked_doc_ids: List[str], relevance: Dict[str, int], k: int = 10) -> float:
    """Proporsi dokumen pada k teratas yang TIDAK punya label (dianggap tidak relevan oleh metrik)."""
    top = ranked_doc_ids[:k]
    return sum(1 for d in top if d not in relevance) / len(top) if top else 0.0


def segment_of(query: str) -> str:
    """Segmen panjang kueri sesuai aturan Adaptive QE (1-3 kata diperluas)."""
    return SHORT_SEGMENT if count_query_words(query) <= 3 else LONG_SEGMENT


# ---------------- Berkas evaluasi ----------------
def load_evaluation_files(queries_path: Path = QUERIES_PATH, labels_path: Path = LABELS_PATH):
    missing = [str(p) for p in (queries_path, labels_path) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "File evaluasi belum tersedia: " + ", ".join(missing) + ". "
            "Gunakan templat di data/templates/ dan baca docs/panduan_evaluasi.md. "
            "Format: evaluation_queries.csv -> query_id,query[,query_type,notes] ; "
            "relevance_labels.csv -> query_id,doc_id,relevance(0/1/2)[,annotator,label_source,notes]."
        )
    queries = pd.read_csv(queries_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    labels = pd.read_csv(labels_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for df, cols, name in ((queries, {"query_id", "query"}, "evaluation_queries.csv"),
                           (labels, {"query_id", "doc_id", "relevance"}, "relevance_labels.csv")):
        if not cols.issubset(df.columns):
            raise ValueError(f"Kolom {name} harus berisi: {sorted(cols)}")
    for df in (queries, labels):
        df["query_id"] = df["query_id"].str.strip()
    queries["query"] = queries["query"].str.strip()
    labels["doc_id"] = labels["doc_id"].str.strip()
    return queries, labels


def validate_evaluation_files(queries: pd.DataFrame, labels: pd.DataFrame, documents_df: Optional[pd.DataFrame] = None,
                              min_queries: int = MIN_QUERIES) -> dict:
    """Periksa berkas kueri/label. errors = harus diperbaiki; warnings = perlu diperhatikan."""
    errors: List[str] = []
    warnings: List[str] = []
    if (queries["query_id"] == "").any() or (queries["query"] == "").any():
        errors.append("Ada kueri dengan query_id atau teks kueri kosong.")
    dup_ids = queries["query_id"][queries["query_id"].duplicated()].unique().tolist()
    if dup_ids:
        errors.append(f"query_id duplikat: {dup_ids[:5]}")
    if queries["query"].str.lower().duplicated().any():
        warnings.append("Ada teks kueri yang sama dengan query_id berbeda.")

    rel = pd.to_numeric(labels["relevance"], errors="coerce")
    bad = labels[~rel.isin(VALID_RELEVANCE)]
    if len(bad):
        errors.append(f"{len(bad)} label relevance di luar {{0,1,2}} atau kosong/bukan angka (contoh baris: {bad.index[0] + 2}).")
    labels = labels.assign(relevance=rel)
    labels = labels[rel.isin(VALID_RELEVANCE)]
    labels = labels.assign(relevance=labels["relevance"].astype(int))

    unknown_q = sorted(set(labels["query_id"]) - set(queries["query_id"]))
    if unknown_q:
        errors.append(f"Label untuk query_id yang tidak ada di berkas kueri: {unknown_q[:5]}")
    if documents_df is not None:
        unknown_d = labels[~labels["doc_id"].isin(set(documents_df["doc_id"]))]
        if len(unknown_d):
            errors.append(f"{len(unknown_d)} label menunjuk doc_id yang TIDAK ada di dataset (contoh: {unknown_d['doc_id'].iloc[0]}). "
                          "Dataset mungkin dibuat ulang setelah anotasi atau salah dataset.")
    dups = labels[labels.duplicated(["query_id", "doc_id"], keep=False)]
    if len(dups):
        conflict = dups.groupby(["query_id", "doc_id"])["relevance"].nunique()
        if (conflict > 1).any():
            errors.append(f"{int((conflict > 1).sum())} pasangan (query_id, doc_id) berlabel ganda dengan nilai BERBEDA; "
                          "selesaikan (adjudikasi) sebelum evaluasi.")
        else:
            warnings.append("Ada label ganda dengan nilai sama (diabaikan duplikatnya).")

    labeled_q = set(labels["query_id"])
    has_rel = set(labels.loc[labels["relevance"] >= 1, "query_id"])
    no_label = [q for q in queries["query_id"] if q not in labeled_q]
    no_rel = [q for q in queries["query_id"] if q in labeled_q and q not in has_rel]
    if no_label:
        warnings.append(f"{len(no_label)} kueri belum punya label sama sekali (tidak dievaluasi): {no_label[:5]}")
    if no_rel:
        warnings.append(f"{len(no_rel)} kueri tanpa dokumen relevan (dikecualikan dari rata-rata): {no_rel[:5]}")
    evaluable = queries[queries["query_id"].isin(has_rel)]
    seg = evaluable["query"].map(segment_of).value_counts().to_dict()
    if len(evaluable) < min_queries:
        warnings.append(f"Hanya {len(evaluable)} kueri dapat dievaluasi (< {min_queries}). Uji signifikansi berdaya rendah; "
                        "hasil bersifat indikatif. Praktik umum evaluasi IR memakai puluhan kueri (sekitar 25-50).")
    for name in (SHORT_SEGMENT, LONG_SEGMENT):
        if seg.get(name, 0) < 10:
            warnings.append(f"Segmen kueri '{name}' hanya {seg.get(name, 0)} kueri (<10): analisis per panjang kueri "
                            "tidak dapat diandalkan; sebaiknya seimbangkan kueri pendek (1-3 kata) dan panjang (>3 kata).")
    stats = {"jumlah_kueri": len(queries), "kueri_berlabel": len(labeled_q), "kueri_dapat_dievaluasi": len(evaluable),
             "segmen_kueri_dapat_dievaluasi": seg, "jumlah_label": len(labels),
             "sebaran_relevance": labels["relevance"].value_counts().sort_index().to_dict()}
    return {"errors": errors, "warnings": warnings, "stats": stats}


# ---------------- Eksperimen ----------------
def run_evaluation(documents_df, embeddings, model, bm25_index,
                   queries: pd.DataFrame, labels: pd.DataFrame, k: int = 10, exclude_no_relevant: bool = True):
    """Hitung metrik per kueri dan per metode. Mengembalikan (per_query_df, summary_df).
    Kueri tanpa dokumen relevan dikecualikan dari rata-rata bila exclude_no_relevant (default)."""
    labels = labels.assign(relevance=pd.to_numeric(labels["relevance"]).astype(int)).drop_duplicates(["query_id", "doc_id"])
    rel_by_query = {qid: dict(zip(g["doc_id"].astype(str), g["relevance"])) for qid, g in labels.groupby("query_id")}
    rows = []
    for _, q in queries.iterrows():
        relevance = rel_by_query.get(q["query_id"], {})
        n_rel = sum(1 for v in relevance.values() if v >= 1)
        for method in ALL_METHODS:
            results, info = search_with_method(method, q["query"], documents_df, embeddings, model, bm25_index, top_k=k)
            ranked = results["doc_id"].astype(str).tolist()
            rows.append({
                "query_id": q["query_id"], "query": q["query"], "n_words": count_query_words(q["query"]),
                "segment": segment_of(q["query"]), "method": method, "final_query": info["final_query"],
                "is_expanded": info["is_expanded"], "n_relevant_known": n_rel,
                "expansion_level": info.get("expansion_level", ""),
                "decision_basis": info.get("decision_basis", ""),
                "top1_z": (info.get("indicators") or {}).get("top1_z"),
                "top5_mean_z": (info.get("indicators") or {}).get("top5_mean_z"),
                f"precision@{k}": precision_at_k(ranked, relevance, k),
                f"ndcg@{k}": ndcg_at_k(ranked, relevance, k),
                f"mrr@{k}": mrr_at_k(ranked, relevance, k),
                f"unjudged@{k}": unjudged_at_k(ranked, relevance, k),
            })
    per_query = pd.DataFrame(rows)
    return per_query, summarize(per_query, k, exclude_no_relevant)


def _eligible(per_query: pd.DataFrame, exclude_no_relevant: bool) -> pd.DataFrame:
    return per_query[per_query["n_relevant_known"] > 0] if exclude_no_relevant else per_query


def metric_columns(k: int) -> List[str]:
    return [f"precision@{k}", f"ndcg@{k}", f"mrr@{k}"]


def summarize(per_query: pd.DataFrame, k: int, exclude_no_relevant: bool = True) -> pd.DataFrame:
    cols = metric_columns(k) + [f"unjudged@{k}"]
    el = _eligible(per_query, exclude_no_relevant)
    out = el.groupby("method")[cols].mean().reindex(ALL_METHODS)
    out.insert(0, "n_queries", el.groupby("method")["query_id"].nunique().reindex(ALL_METHODS).fillna(0).astype(int))
    return out.reset_index()


def summarize_by_segment(per_query: pd.DataFrame, k: int, exclude_no_relevant: bool = True) -> pd.DataFrame:
    """Rata-rata per segmen panjang kueri + seberapa sering QE benar-benar memperluas kueri."""
    cols = metric_columns(k) + [f"unjudged@{k}"]
    el = _eligible(per_query, exclude_no_relevant)
    g = el.groupby(["segment", "method"])
    out = g[cols].mean()
    out.insert(0, "expanded_rate", g["is_expanded"].mean())
    out.insert(0, "n_queries", g["query_id"].nunique())
    return out.reset_index()


def adaptive_levels(per_query: pd.DataFrame) -> pd.Series:
    """Tingkat keputusan Adaptive QE per query_id (none/light/strong)."""
    a = per_query[per_query["method"] == METHOD_ADAPTIVE]
    return a.drop_duplicates("query_id").set_index("query_id")["expansion_level"]


def summarize_by_decision(per_query: pd.DataFrame, k: int, exclude_no_relevant: bool = True) -> pd.DataFrame:
    """Rata-rata metrik tiap metode, dikelompokkan menurut KEPUTUSAN Adaptive QE pada kueri itu (No/Light/Strong).

    Menjawab: pada kueri yang diputuskan 'No/Light/Strong Expansion', bagaimana Adaptive dibanding Semantic dan Static?
    `delta_vs_semantic` = rata-rata (metode - Semantic) pada kueri yang sama. Kelompok kecil (< 10 kueri) hanya indikatif."""
    el = _eligible(per_query, exclude_no_relevant).copy()
    el["adaptive_level"] = el["query_id"].map(adaptive_levels(per_query))
    el = el[el["adaptive_level"].isin(["none", "light", "strong"])]
    if el.empty:
        return pd.DataFrame()
    cols = metric_columns(k) + [f"unjudged@{k}"]
    g = el.groupby(["adaptive_level", "method"])
    out = g[cols].mean()
    out.insert(0, "n_queries", g["query_id"].nunique())
    base = el[el["method"] == METHOD_SEMANTIC].set_index(["adaptive_level", "query_id"])[metric_columns(k)]
    for m in metric_columns(k):
        diffs = []
        for (lvl, meth), sub in g:
            b = base.loc[lvl][m] if lvl in base.index.get_level_values(0) else pd.Series(dtype=float)
            diffs.append((sub.set_index("query_id")[m] - b.reindex(sub["query_id"]).values).mean())
        out[f"delta_vs_semantic_{m}"] = diffs
    out["small_group"] = out["n_queries"] < 10
    out = out.reset_index()
    out["adaptive_level"] = pd.Categorical(out["adaptive_level"], ["none", "light", "strong"], ordered=True)
    out["method"] = pd.Categorical(out["method"], ALL_METHODS, ordered=True)
    return out.sort_values(["adaptive_level", "method"]).reset_index(drop=True).astype({"adaptive_level": str, "method": str})


def decision_distribution(per_query: pd.DataFrame) -> pd.DataFrame:
    """Berapa kueri jatuh ke No/Light/Strong dan seberapa sering istilah benar-benar ditambahkan."""
    a = per_query[per_query["method"] == METHOD_ADAPTIVE].drop_duplicates("query_id")
    out = a.groupby("expansion_level").agg(n_queries=("query_id", "nunique"), expanded_rate=("is_expanded", "mean"),
                                           mean_top1_z=("top1_z", "mean"), mean_top5_z=("top5_mean_z", "mean"))
    return out.reindex(["none", "light", "strong"]).fillna({"n_queries": 0}).reset_index()


# ---------------- Statistik ----------------
def paired_randomization_pvalue(diffs: Sequence[float], n_resamples: int = 10000, seed: int = 0) -> float:
    """Uji randomisasi berpasangan (sign-flip), dua sisi. Eksak bila selisih tak-nol <= 15, selain itu Monte Carlo."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    observed = abs(d.sum())
    if n <= 15:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        return float(np.mean(np.abs(signs @ d) >= observed - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
    return float((1 + np.sum(np.abs(signs @ d) >= observed - 1e-12)) / (n_resamples + 1))


def bootstrap_ci(diffs: Sequence[float], n_resamples: int = 10000, seed: int = 0, alpha: float = 0.05):
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_resamples, len(d)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Koreksi Holm-Bonferroni (mengendalikan family-wise error)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adj[idx] = running
    return adj


def significance_table(per_query: pd.DataFrame, k: int, pairs=PRIMARY_PAIRS, exclude_no_relevant: bool = True,
                       seed: int = 0, n_resamples: int = 10000) -> pd.DataFrame:
    """Perbandingan berpasangan antar metode per metrik. Holm diterapkan di dalam tiap metrik.
    `n_changed` = jumlah kueri tempat kedua metode memakai kueri akhir berbeda (satu-satunya tempat QE bisa berpengaruh)."""
    el = _eligible(per_query, exclude_no_relevant)
    rows = []
    for metric in metric_columns(k):
        wide = el.pivot(index="query_id", columns="method", values=metric)
        fq = el.pivot(index="query_id", columns="method", values="final_query")
        block = []
        for a, b in pairs:
            diffs = (wide[b] - wide[a]).dropna()
            lo, hi = bootstrap_ci(diffs.values, n_resamples, seed)
            block.append({"metric": metric, "method_a": a, "method_b": b, "n_queries": len(diffs),
                          "mean_a": wide[a].mean(), "mean_b": wide[b].mean(), "mean_diff_b_minus_a": diffs.mean(),
                          "ci95_low": lo, "ci95_high": hi, "n_changed": int((fq[a] != fq[b]).sum()),
                          "n_metric_differs": int((diffs.abs() > 1e-12).sum()),
                          "p_value": paired_randomization_pvalue(diffs.values, n_resamples, seed)})
        for row, adj in zip(block, holm_adjust([r["p_value"] for r in block])):
            row["p_holm"] = adj
        rows.extend(block)
    return pd.DataFrame(rows)


# ---------------- Kesepakatan anotator ----------------
def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa dua anotator pada label kategorik. nan bila tidak terdefinisi."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    return float("nan") if pe == 1.0 else (po - pe) / (1 - pe)


# ---------------- CLI ----------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None, model=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluasi offline 4 metode pencarian.")
    parser.add_argument("--k", type=int, default=10, help="Nilai k (default 10)")
    parser.add_argument("--artifacts", default="artifacts_real" if Path("artifacts_real/documents_processed.csv").exists() else "artifacts",
                        help="Folder artifacts (default: artifacts_real bila ada)")
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--include-no-relevant", action="store_true", help="jangan kecualikan kueri tanpa dokumen relevan")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="izinkan dataset sintetis (hanya uji alur; hasil BUKAN temuan penelitian)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        queries, labels = load_evaluation_files(Path(args.queries), Path(args.labels))
        documents_df, embeddings = load_artifacts(args.artifacts)
    except (FileNotFoundError, ValueError) as err:
        print(f"[PESAN] {err}")
        return 2
    kinds = sorted(set(documents_df["data_source"]))
    if kinds != ["real"] and not args.allow_synthetic:
        print(f"[DITOLAK] Dataset berjenis {kinds}, bukan 'real'. Evaluasi pada data sintetis tidak boleh dilaporkan sebagai hasil "
              "penelitian. Gunakan dataset nyata, atau --allow-synthetic hanya untuk menguji alur.")
        return 2
    rep = validate_evaluation_files(queries, labels, documents_df)
    for w in rep["warnings"]:
        print(f"[PERINGATAN] {w}")
    if rep["errors"]:
        for e in rep["errors"]:
            print(f"[ERROR] {e}")
        print("Evaluasi dibatalkan: perbaiki berkas label/kueri.")
        return 1

    if model is None:
        print("Memuat model embedding (pertama kali butuh internet untuk mengunduh)...")
        model = load_embedding_model()
    bm25_index = build_bm25_index(documents_df)
    exclude = not args.include_no_relevant
    per_query, summary = run_evaluation(documents_df, embeddings, model, bm25_index, queries, labels, args.k, exclude)
    by_segment = summarize_by_segment(per_query, args.k, exclude)
    sig = significance_table(per_query, args.k, exclude_no_relevant=exclude, seed=args.seed)
    by_decision = summarize_by_decision(per_query, args.k, exclude)
    dist = decision_distribution(per_query)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(out / "evaluation_per_query.csv", index=False, encoding="utf-8")
    summary.to_csv(out / "evaluation_summary.csv", index=False, encoding="utf-8")
    by_segment.to_csv(out / "evaluation_by_segment.csv", index=False, encoding="utf-8")
    sig.to_csv(out / "evaluation_significance.csv", index=False, encoding="utf-8")
    by_decision.to_csv(out / "evaluation_by_decision.csv", index=False, encoding="utf-8")
    dist.to_csv(out / "evaluation_decision_distribution.csv", index=False, encoding="utf-8")
    meta = {"dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"), "k": args.k, "model": MODEL_NAME,
            "dataset_kind": kinds, "num_documents": len(documents_df), "doc_ids_sha1": doc_ids_hash(documents_df["doc_id"]),
            "queries_sha256": _sha256(Path(args.queries)), "labels_sha256": _sha256(Path(args.labels)),
            "kueri_dievaluasi": int(summary["n_queries"].max()), "kueri_dikecualikan_tanpa_relevan": 0 if not exclude else
            int(queries["query_id"].nunique() - per_query[per_query["n_relevant_known"] > 0]["query_id"].nunique()),
            "uji_signifikansi": "randomisasi berpasangan (sign-flip) dua sisi, CI bootstrap 95%, koreksi Holm per metrik",
            "seed": args.seed, "validasi": rep, "dataset_sintetis_diizinkan": bool(args.allow_synthetic and kinds != ["real"])}
    (out / "evaluation_run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\nRingkasan rata-rata (k={args.k}):")
    print(summary.round(4).to_string(index=False))
    high = summary[summary[f"unjudged@{args.k}"] > UNJUDGED_WARN]
    if len(high):
        print(f"\n[PERINGATAN] unjudged@{args.k} > {UNJUDGED_WARN:.0%} pada: {high['method'].tolist()}. Dokumen tanpa label dianggap tidak "
              "relevan sehingga metrik metode tersebut bisa meremehkan; lengkapi label lewat pooling (scripts.make_pool).")
    print("\nPer segmen panjang kueri:")
    print(by_segment.round(4).to_string(index=False))
    print("\nKeputusan Adaptive QE (jumlah kueri per tingkat):")
    print(dist.round(3).to_string(index=False))
    print("\nMetrik menurut keputusan Adaptive QE (kelompok < 10 kueri hanya indikatif):")
    print(by_decision.round(4).to_string(index=False) if len(by_decision) else "(tidak ada)")
    print("\nUji berpasangan (Holm per metrik):")
    print(sig[["metric", "method_a", "method_b", "n_queries", "n_changed", "mean_diff_b_minus_a", "p_value", "p_holm"]]
          .round(4).to_string(index=False))
    print(f"\nTersimpan di {out}/ (evaluation_*.csv dan evaluation_run.json)")
    if args.allow_synthetic and kinds != ["real"]:
        print("\n*** DATASET SINTETIS: hasil ini hanya uji alur, BUKAN temuan penelitian. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
