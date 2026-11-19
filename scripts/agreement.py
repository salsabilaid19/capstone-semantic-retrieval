"""Kesepakatan dua anotator (Cohen's kappa) dan daftar perbedaan untuk adjudikasi.

Contoh:  python -m scripts.agreement --a pool_A.csv --b pool_B.csv --out data/annotation/perbedaan.csv
Hanya pasangan (query_id, doc_id) yang dinilai KEDUANYA yang dihitung. Label final tetap harus diputuskan manusia (adjudikasi).
"""
import argparse
import sys

import pandas as pd

from src.evaluation import cohens_kappa


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    cols = ["query_id", "doc_id", "relevance"]
    a = pd.read_csv(args.a, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    b = pd.read_csv(args.b, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for df in (a, b):
        df["relevance"] = pd.to_numeric(df["relevance"].replace("", pd.NA), errors="coerce")
    m = a[cols + [c for c in ("title",) if c in a.columns]].merge(b[cols], on=["query_id", "doc_id"], suffixes=("_a", "_b"))
    m = m.dropna(subset=["relevance_a", "relevance_b"])
    if m.empty:
        print("Tidak ada pasangan yang dinilai kedua anotator.", file=sys.stderr)
        return 2
    ra, rb = m["relevance_a"].astype(int).values, m["relevance_b"].astype(int).values
    print(f"Pasangan dinilai keduanya : {len(m)}")
    print(f"Persen sama persis        : {100 * (ra == rb).mean():.1f}%")
    print(f"Cohen's kappa (0/1/2)     : {cohens_kappa(ra, rb):.3f}")
    print(f"Cohen's kappa (relevan>=1): {cohens_kappa((ra >= 1).astype(int), (rb >= 1).astype(int)):.3f}")
    diff = m[ra != rb]
    if args.out:
        diff.assign(relevance_final="").to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"{len(diff)} perbedaan ditulis ke {args.out} (isi relevance_final setelah diskusi).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
