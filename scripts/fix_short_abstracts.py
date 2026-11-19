"""Tangani dokumen berabstrak < 200 karakter TANPA mengubah artefak asli.

    python -m scripts.fix_short_abstracts --artifacts-dir artifacts_real            # dry-run: hanya melaporkan
    python -m scripts.fix_short_abstracts --artifacts-dir artifacts_real --apply --out artifacts_real_filtered

Mode --apply menulis salinan BARU (dokumen + embeddings + model_config yang hash doc_id-nya dihitung ulang). Baris embedding
dokumen lain tidak berubah (embedding dihitung per dokumen). Jumlah dokumen turun; lakukan HANYA setelah peneliti setuju,
karena komposisi 900/300/300/300 berubah, lalu perbarui dataset_manifest.json dan rujukan jumlah di laporan."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import doc_ids_hash
from src.validation import MIN_ABSTRACT_CHARS


def find_short(docs: pd.DataFrame, min_chars: int = MIN_ABSTRACT_CHARS) -> pd.DataFrame:
    return docs[docs["abstract"].astype(str).str.len() < min_chars]


def apply_filter(src: Path, out: Path, min_chars: int = MIN_ABSTRACT_CHARS) -> dict:
    docs = pd.read_csv(src / "documents_processed.csv", dtype=str, keep_default_na=False)
    emb = np.load(src / "embeddings.npy")
    if len(docs) != len(emb):
        raise ValueError("Jumlah dokumen dan embedding tidak sama; berhenti.")
    keep = (docs["abstract"].astype(str).str.len() >= min_chars).to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in ("documents_processed.csv", "embeddings.npy", "model_config.json"):
            shutil.copy2(f, out / f.name)
    new_docs = docs[keep].reset_index(drop=True)
    new_docs.to_csv(out / "documents_processed.csv", index=False, encoding="utf-8")
    np.save(out / "embeddings.npy", emb[keep])
    cfg = json.loads((src / "model_config.json").read_text(encoding="utf-8"))
    cfg.update({"num_documents": int(len(new_docs)), "doc_ids_sha1": doc_ids_hash(new_docs["doc_id"]),
                "catatan_filter": f"{int((~keep).sum())} dokumen berabstrak < {min_chars} karakter dikeluarkan dari {src.name}"})
    (out / "model_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"removed": docs.loc[~keep, "doc_id"].tolist(), "remaining": int(len(new_docs))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts-dir", default="artifacts_real")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", default="artifacts_real_filtered")
    a = ap.parse_args(argv)
    src = Path(a.artifacts_dir)
    docs = pd.read_csv(src / "documents_processed.csv", dtype=str, keep_default_na=False)
    short = find_short(docs)
    print(f"{len(short)} dokumen berabstrak < {MIN_ABSTRACT_CHARS} karakter:")
    for _, r in short.iterrows():
        print(f"  {r.doc_id} | {r.journal[:12]} | {r.year} | {len(r.abstract)} karakter | {r.title[:70]}")
    if not a.apply:
        print("\nDry-run: tidak ada berkas diubah. Gunakan --apply --out <folder baru> setelah ada persetujuan.")
        return 0
    if Path(a.out).resolve() == src.resolve():
        print("--out tidak boleh sama dengan folder sumber.")
        return 2
    res = apply_filter(src, Path(a.out))
    print(f"\nSalinan baru di {a.out}: {res['remaining']} dokumen tersisa; dikeluarkan {res['removed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
