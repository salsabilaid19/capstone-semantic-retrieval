"""Gabungkan hasil panen/impor menjadi dataset siap pakai (dedup, validasi, doc_id stabil).

Contoh:  python -m scripts.build_dataset --kind real        -> artifacts_real/documents_processed.csv
         python -m scripts.build_dataset --kind synthetic   -> artifacts/documents_processed.csv (data contoh)
Data real dan sintetis TIDAK PERNAH digabung: tiap build menolak baris dengan Data_Source yang tidak sesuai.
Dataset 'real' yang gagal validasi tidak ditulis kecuali memakai --force.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.artifacts import doc_ids_hash
from src.harvest import load_sources
from src.preprocessing import clean_dataset
from src.validation import report_markdown, validate_dataset

DEFAULTS = {"real": (["data/real/harvest", "data/real/imports"], "artifacts_real"),
            "synthetic": (["data/synthetic"], "artifacts")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["real", "synthetic"])
    ap.add_argument("--input-dirs", nargs="*", default=None)
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--config", default="config/journals.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--force", action="store_true", help="tulis hasil walau validasi menemukan error")
    args = ap.parse_args(argv)
    in_dirs, out_dir = DEFAULTS[args.kind]
    in_dirs, out_dir = args.input_dirs or in_dirs, Path(args.artifacts_dir or out_dir)

    files = sorted(p for d in in_dirs for p in Path(d).glob("*.csv") if p.name != "documents_raw.csv")
    if not files:
        print(f"Tidak ada CSV di {in_dirs}. Jalankan scripts.collect_oai atau scripts.import_csv dahulu.", file=sys.stderr)
        return 2
    frames = [pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False).assign(_file=p.name) for p in files]
    raw = pd.concat(frames, ignore_index=True)
    if "Data_Source" not in raw.columns:
        raw["Data_Source"] = ""
    bad = raw[raw["Data_Source"].str.lower() != args.kind]
    if len(bad):
        print(f"DITOLAK: {len(bad)} baris tidak bertanda Data_Source='{args.kind}' (mis. di {bad['_file'].iloc[0]}). "
              "Data real dan sintetis harus dipisah; perbaiki berkas atau pindahkan ke folder yang sesuai.", file=sys.stderr)
        return 2

    n_raw = len(raw)
    df = clean_dataset(raw.drop(columns=["_file"]), default_data_source=args.kind)
    rep = validate_dataset(df, expected_kind=args.kind)
    md = report_markdown(args.kind, rep)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f"validation_{args.kind}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Baris mentah {n_raw} -> {len(df)} dokumen setelah pembersihan & deduplikasi.")
    if rep["errors"] and not args.force:
        print("Dataset TIDAK ditulis karena ada error validasi (gunakan --force untuk menimpa).", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = Path(f"data/{args.kind}") / "documents_raw.csv"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw.drop(columns=["_file"]).to_csv(raw_out, index=False, encoding="utf-8")
    df.to_csv(out_dir / "documents_processed.csv", index=False, encoding="utf-8")
    cfg = {s["source_id"]: s for s in load_sources(args.config)} if Path(args.config).exists() else {}
    used = sorted({sid for sid in raw.get("source_id", pd.Series(dtype=str)) if sid})
    manifest = {
        "kind": args.kind, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_documents": len(df), "doc_ids_sha1": doc_ids_hash(df["doc_id"]),
        "input_files": [p.name for p in files],
        "sources": [{"source_id": sid, "name": cfg.get(sid, {}).get("name", ""), "license": cfg.get(sid, {}).get("license", ""),
                     "verification": cfg.get(sid, {}).get("verification", "")} for sid in used],
        "validation": {"errors": rep["errors"], "warnings": rep["warnings"], "stats": rep["stats"]},
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Ditulis: {out_dir / 'documents_processed.csv'} dan dataset_manifest.json")
    if (out_dir / "embeddings.npy").exists():
        print("PERHATIAN: embeddings.npy lama ada di folder ini dan tidak cocok dengan dataset baru; buat ulang embedding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
