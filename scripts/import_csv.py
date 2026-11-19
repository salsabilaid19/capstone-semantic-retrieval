"""Impor CSV buatan sendiri (jalur alternatif bila pengambilan otomatis tidak tersedia).

Contoh:  python -m scripts.import_csv --input paper_saya.csv --name manual_ui --data-source real \
             --source-note "Disalin manual dari situs jurnal X, lisensi CC BY 4.0, diambil 2026-10-27"
Kolom wajib: Title, Abstract. Dianjurkan: Authors, Year, DOI, URL, Keywords, Publication (nama jurnal).
Data 'real' wajib punya --source-note (asal & ketentuan penggunaan). Hasil: data/<real|synthetic>/imports/<name>.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.preprocessing import map_columns

OUT_COLUMNS = ["Title", "Abstract", "Authors", "Year", "DOI", "URL", "Keywords", "Publication", "Type", "Language",
               "Data_Source", "Source_Note"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", required=True, help="nama berkas keluaran (tanpa .csv)")
    ap.add_argument("--data-source", required=True, choices=["real", "synthetic"])
    ap.add_argument("--source-note", default="")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--exclude-types", default="", help="tipe dokumen yang dibuang, dipisah koma (mis. "
                    "'edited-book,monograph,book,proceedings'); berguna untuk ekspor Publish or Perish/Crossref "
                    "karena abstrak buku sering hanya kata pengantar/ucapan terima kasih")
    args = ap.parse_args(argv)
    if args.data_source == "real" and not args.source_note.strip():
        print("Data 'real' wajib disertai --source-note (asal data dan ketentuan penggunaan).", file=sys.stderr)
        return 2
    try:
        raw = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        raw = pd.read_csv(args.input, encoding="latin-1", dtype=str, keep_default_na=False)
    try:
        m = map_columns(raw, default_data_source=args.data_source)
    except ValueError as err:
        print(f"Gagal: {err}", file=sys.stderr)
        return 2
    claimed = m["data_source"][m["data_source"] != args.data_source]
    if len(claimed):
        print(f"Berkas memuat Data_Source '{claimed.iloc[0]}' yang bertentangan dengan --data-source {args.data_source}.",
              file=sys.stderr)
        return 2
    excluded_types = {t.strip().lower() for t in args.exclude_types.split(",") if t.strip()}
    type_drop = m["document_type"].str.lower().isin(excluded_types)
    keep = (m["title"] != "") & (m["abstract"] != "") & ~type_drop
    note = (m["source_note"].where(m["source_note"] != "", args.source_note.strip()))
    out = pd.DataFrame({
        "Title": m["title"], "Abstract": m["abstract"], "Authors": m["author"],
        "Year": m["year"].astype(str).replace("<NA>", ""), "DOI": m["doi"], "URL": m["source_url"],
        "Keywords": m["keywords"], "Publication": m["journal"], "Type": m["document_type"],
        "Language": m["language"], "Data_Source": args.data_source, "Source_Note": note})[keep][OUT_COLUMNS]
    out_dir = Path(args.out_dir or f"data/{args.data_source}/imports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.name}.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    empty = int(((m["title"] == "") | (m["abstract"] == "")).sum())
    print(f"{len(out)} baris ditulis ke {path} ({empty} baris dibuang karena judul/abstrak kosong, "
          f"{int((type_drop & ~((m['title'] == '') | (m['abstract'] == ''))).sum())} karena tipe dokumen dikecualikan).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
