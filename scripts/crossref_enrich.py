"""Verifikasi/lengkapi metadata CSV mentah lewat Crossref (berdasarkan DOI).

Contoh:  python -m scripts.crossref_enrich --input data/real/harvest/resti.csv --output data/real/harvest/resti.csv --mailto anda@kampus.ac.id
Hanya mengisi kolom kosong (Year, Publication, Abstract, Authors); tidak menimpa data penerbit.
Menambah kolom Crossref_Status (ok/not_found/error/no_doi) dan Crossref_Title_Match (match/mismatch) untuk ditinjau manual.
"""
import argparse
import sys

import pandas as pd

from src.crossref import enrich_frame, make_fetcher


def main(argv=None, fetch=None, sleep=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mailto", default="", help="email untuk polite pool Crossref (disarankan)")
    ap.add_argument("--cache", default="data/real/cache/crossref.json")
    ap.add_argument("--delay", type=float, default=0.2)
    ap.add_argument("--no-fill", action="store_true", help="hanya verifikasi, jangan isi kolom kosong")
    args = ap.parse_args(argv)
    if fetch is None:
        if not args.mailto:
            print("Disarankan memberi --mailto; Crossref memprioritaskan klien yang mengidentifikasi diri.", file=sys.stderr)
        fetch = make_fetcher(args.mailto)
    df = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if "DOI" not in df.columns:
        print("Kolom DOI tidak ada di file masukan.", file=sys.stderr)
        return 2
    kw = {"sleep": sleep} if sleep else {}
    out = enrich_frame(df, fetch, args.mailto, args.cache, args.delay, fill_missing=not args.no_fill, **kw)
    out.to_csv(args.output, index=False, encoding="utf-8")
    print("Status Crossref:", out["Crossref_Status"].value_counts().to_dict())
    mism = int(out["Crossref_Title_Match"].str.startswith("mismatch").sum())
    print(f"Judul tidak cocok dengan Crossref (perlu tinjauan manual): {mism}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
