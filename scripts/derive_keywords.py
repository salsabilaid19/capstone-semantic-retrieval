"""Isi kolom Keywords yang kosong dengan kata kunci OTOMATIS (TF-IDF sederhana dari judul + abstrak).

Mengapa ada: Static/Adaptive Query Expansion mengambil kata kunci dari dokumen teratas. Ekspor Publish or Perish
umumnya tidak punya kata kunci, sehingga ekspansi tidak pernah terjadi. Skrip ini BUKAN pengganti kata kunci penulis:
hasilnya ditandai di Source_Note dan harus dilaporkan di laporan ("kata kunci diturunkan otomatis, bukan dari penulis").
Hanya baris dengan Keywords kosong yang diisi; kata kunci asli tidak disentuh. Berkas keluaran = berkas baru.

Contoh:  python -m scripts.derive_keywords --input data/real/imports/pop.csv --output data/real/imports/pop_kw.csv --top 5
"""
import argparse
import math
import re
import sys
from collections import Counter

import pandas as pd

from src.retrieval import tokenize

NOTE = "Keywords diturunkan otomatis (TF-IDF judul+abstrak), bukan dari penulis."
_NUM = re.compile(r"^\d+$")
_STOP_EN = frozenset("""
the and for with this that these those from into over under between among about using use used based via
are was were been being have has had having not but can could may might will would should also such than then
their its our your his her they them which what when where while whose who whom how why more most less many much
very good better best new novel paper study studies research results result approach approaches method methods
proposed propose present presents presented show shows shown analysis review overview towards toward each other
both only well high low large small different various several however thus therefore including include includes
real time first second one two three key main important recent current future role need needs make makes making
two three provide provides providing achieve achieved improve improved improving
""".split())


def _terms(text: str) -> list:
    return [t for t in tokenize(text) if len(t) >= 3 and not _NUM.match(t) and t not in _STOP_EN]


def derive(df: pd.DataFrame, top: int = 5, min_df: int = 2) -> pd.DataFrame:
    out = df.copy()
    docs = [_terms(f"{t} {t} {a}") for t, a in zip(out["Title"].fillna(""), out["Abstract"].fillna(""))]  # judul 2x
    df_count = Counter(w for d in docs for w in set(d))
    n = len(docs)
    kw_col = out["Keywords"].fillna("").astype(str)
    note_col = out["Source_Note"].fillna("").astype(str) if "Source_Note" in out else pd.Series([""] * n, index=out.index)
    new_kw, new_note = [], []
    for i, d in enumerate(docs):
        if kw_col.iloc[i].strip():
            new_kw.append(kw_col.iloc[i]); new_note.append(note_col.iloc[i]); continue
        tf = Counter(d)
        scored = [(c * math.log((n + 1) / (df_count[w] + 1)) + 1e-9, w) for w, c in tf.items() if df_count[w] >= min_df]
        scored.sort(key=lambda x: (-x[0], x[1]))
        picked = [w for _, w in scored[:top]]
        new_kw.append("; ".join(picked))
        base = note_col.iloc[i].strip()
        new_note.append(base if NOTE in base or not picked else (base + " " + NOTE).strip())
    out["Keywords"] = new_kw
    out["Source_Note"] = new_note
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--min-df", type=int, default=2, help="istilah harus muncul di >= N dokumen (buang istilah unik)")
    args = ap.parse_args(argv)
    if args.input == args.output:
        print("Berkas keluaran harus berbeda dari masukan.", file=sys.stderr)
        return 2
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for c in ("Title", "Abstract", "Keywords"):
        if c not in df.columns:
            print(f"Kolom '{c}' tidak ada. Jalankan scripts.import_csv lebih dulu.", file=sys.stderr)
            return 2
    before = int((df["Keywords"].str.strip() != "").sum())
    out = derive(df, args.top, args.min_df)
    after = int((out["Keywords"].str.strip() != "").sum())
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Keywords terisi: {before} -> {after} dari {len(out)} baris. Ditulis ke {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
