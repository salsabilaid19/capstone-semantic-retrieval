"""Ubah pool yang sudah diisi anotator menjadi data/relevance_labels.csv.

Contoh:  python -m scripts.pool_to_labels --input data/annotation/pool_terisi.csv --annotator "Nama Anotator"
Kolom relevance kosong berarti BELUM DINILAI (bukan 0): gagal kecuali --allow-blank (baris kosong dibuang dan dilaporkan).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.pooling import pool_to_labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/relevance_labels.csv")
    ap.add_argument("--annotator", default="")
    ap.add_argument("--allow-blank", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if Path(args.output).exists() and not args.overwrite:
        print(f"{args.output} sudah ada; gunakan --overwrite bila memang ingin menimpa.", file=sys.stderr)
        return 2
    filled = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    labels, problems = pool_to_labels(filled, args.annotator, args.allow_blank)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    labels.to_csv(args.output, index=False, encoding="utf-8")
    skipped = len(filled) - len(labels)
    print(f"{len(labels)} label ditulis ke {args.output}" + (f" ({skipped} baris belum dinilai dilewati)" if skipped else "")
          + f". Sebaran: {labels['relevance'].value_counts().sort_index().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
