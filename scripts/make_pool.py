"""Buat berkas anotasi (pool) dari top-k keempat metode untuk tiap kueri.

Contoh:  python -m scripts.make_pool --artifacts-dir artifacts_real --depth 10
Keluaran: data/annotation/pool.csv (BERIKAN ke anotator; tanpa info metode) dan data/annotation/pool_provenance.csv (JANGAN diberikan).
Butuh embedding dan model e5 asli (jalankan di komputer Anda; pertama kali mengunduh model).
Gunakan depth >= k evaluasi agar unjudged@k mendekati 0.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.evaluation import load_evaluation_files, validate_evaluation_files
from src.pooling import build_pool
from src.retrieval import build_bm25_index, load_artifacts, load_embedding_model


def main(argv=None, model=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default="data/evaluation_queries.csv")
    ap.add_argument("--artifacts-dir", default="artifacts_real")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default="data/annotation")
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args(argv)
    if not Path(args.queries).exists():
        print(f"Berkas kueri belum ada: {args.queries}. Salin data/templates/evaluation_queries_template.csv dan isi (lihat docs/panduan_evaluasi.md).",
              file=sys.stderr)
        return 2
    try:
        queries = pd.read_csv(args.queries, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        docs, emb = load_artifacts(args.artifacts_dir)
    except (FileNotFoundError, ValueError) as err:
        print(err, file=sys.stderr)
        return 2
    if set(docs["data_source"]) != {"real"} and not args.allow_synthetic:
        print("Dataset bukan 'real'; pooling untuk penelitian harus memakai dataset nyata (--allow-synthetic hanya uji alur).", file=sys.stderr)
        return 2
    queries["query_id"] = queries["query_id"].str.strip()
    queries["query"] = queries["query"].str.strip()
    rep = validate_evaluation_files(queries, pd.DataFrame({"query_id": [], "doc_id": [], "relevance": []}, dtype=str), docs)
    errs = [e for e in rep["errors"] if "query" in e.lower() and "label" not in e.lower()]
    if errs:
        print("\n".join(errs), file=sys.stderr)
        return 1
    if model is None:
        print("Memuat model embedding...")
        model = load_embedding_model()
    pool, prov = build_pool(queries, docs, emb, model, build_bm25_index(docs), depth=args.depth, seed=args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out / "pool.csv", index=False, encoding="utf-8-sig")
    prov.to_csv(out / "pool_provenance.csv", index=False, encoding="utf-8")
    per_q = pool.groupby("query_id").size()
    print(f"{len(pool)} pasangan (kueri, dokumen) dari {len(queries)} kueri; rata-rata {per_q.mean():.1f} dokumen per kueri untuk dinilai.")
    print(f"Berikan {out / 'pool.csv'} kepada anotator (isi kolom relevance 0/1/2). Simpan pool_provenance.csv sendiri.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
