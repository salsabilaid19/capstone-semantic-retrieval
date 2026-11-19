"""Periksa kompatibilitas embeddings.npy dengan dokumen, model_config.json, dan model.

Contoh:  python -m scripts.check_embeddings --artifacts-dir artifacts_real            (struktural, tanpa model)
         python -m scripts.check_embeddings --artifacts-dir artifacts_real --sanity   (dengan model asli)

Pemeriksaan struktural: jumlah baris, hash doc_id, nama model, dimensi, normalisasi, NaN, prefix, vektor kembar, ukuran berkas.
Uji --sanity (butuh model): (1) encode ulang sampel dokumen dan bandingkan dengan vektor tersimpan (mendeteksi teks/prefix/model
yang berbeda), (2) pencarian-diri: judul dokumen sebagai kueri harus menempatkan dokumennya di top-3. Ini uji SANITY, bukan evaluasi.
Kode keluar: 0 lolos, 1 gagal, 2 tidak dapat dijalankan.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from src import retrieval as R
from src.artifacts import DOC_PREFIX, QUERY_PREFIX, read_model_config, validate_embeddings
from src.embedding import EXPECTED_DIMS, encode_documents, projected_size_mb

COSINE_MIN = 0.999
SELF_HIT_MIN = 0.8


def sample_indices(n: int, size: int) -> np.ndarray:
    """Sampel deterministik tersebar merata."""
    return np.unique(np.linspace(0, n - 1, num=min(size, n)).astype(int))


def main(argv=None, model=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--sanity", action="store_true", help="uji dengan model (mengunduh model bila belum ada)")
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--expected-dim", type=int, default=None, help=f"default dari model: {EXPECTED_DIMS}")
    ap.add_argument("--model-name", default=R.MODEL_NAME)
    args = ap.parse_args(argv)

    fails, warns = [], []
    try:
        docs = R.load_documents(args.artifacts_dir)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 2
    emb_path = Path(args.artifacts_dir) / "embeddings.npy"
    if not emb_path.exists():
        print(f"{emb_path} belum ada. Jalankan scripts.generate_embeddings atau notebook Colab.", file=sys.stderr)
        return 2
    emb = np.load(emb_path, allow_pickle=False)
    cfg = read_model_config(args.artifacts_dir)
    print(f"Dokumen: {len(docs)} | embeddings: {emb.shape} {emb.dtype} | {emb_path.stat().st_size / 1024 / 1024:.2f} MB "
          f"(perkiraan 500 dok: {projected_size_mb(500, emb.shape[1] if emb.ndim == 2 else 384):.2f} MB)")

    try:
        warns += validate_embeddings(docs, emb, cfg, args.model_name)
    except ValueError as err:
        fails.append(str(err))
    if emb.ndim == 2:
        want = args.expected_dim or EXPECTED_DIMS.get(args.model_name)
        if want and emb.shape[1] != want:
            fails.append(f"Dimensi embedding {emb.shape[1]} != {want} yang diharapkan untuk {args.model_name}.")
        if emb.dtype != np.float32:
            warns.append(f"dtype {emb.dtype} (dibuat float32 oleh generator); hasil tetap bekerja.")
        if len(emb) > 1 and len(np.unique(np.round(emb, 6), axis=0)) < len(emb):
            warns.append("Ada vektor identik untuk dokumen berbeda (teks dokumen kembar atau embedding rusak).")
    if cfg and cfg.get("library_versions"):
        print("Dibuat dengan:", ", ".join(f"{k} {v}" for k, v in cfg["library_versions"].items()))
    if cfg:
        if cfg.get("document_prefix") != DOC_PREFIX or cfg.get("query_prefix") != QUERY_PREFIX:
            fails.append(f"Prefix di model_config.json ({cfg.get('document_prefix')!r}/{cfg.get('query_prefix')!r}) "
                         f"tidak sama dengan yang dipakai aplikasi ({DOC_PREFIX!r}/{QUERY_PREFIX!r}).")
        if not cfg.get("normalize_embeddings", True):
            fails.append("model_config.json menyatakan embedding tidak dinormalisasi; aplikasi mengasumsikan cosine via dot product.")

    if args.sanity and not fails:
        if model is None:
            try:
                model = R.load_embedding_model(args.model_name)
            except Exception as err:  # noqa: BLE001
                print(f"Model tidak dapat dimuat ({type(err).__name__}); uji --sanity tidak dapat dijalankan: {err}", file=sys.stderr)
                return 2
        idx = sample_indices(len(docs), args.sample)
        fresh = encode_documents(model, docs.iloc[idx])
        cos = (fresh * emb[idx]).sum(axis=1)
        print(f"Sanity 1 (encode ulang {len(idx)} dokumen): cosine min {cos.min():.5f}, rata-rata {cos.mean():.5f} (ambang {COSINE_MIN})")
        if cos.min() < COSINE_MIN:
            fails.append("Vektor tersimpan TIDAK cocok dengan hasil encode ulang: embedding dibuat dari teks/prefix/model yang berbeda "
                         "atau dari dokumen lain. Buat ulang embedding.")
        qv = np.asarray(model.encode([QUERY_PREFIX + t for t in docs.iloc[idx]["title"]], normalize_embeddings=True), dtype=np.float32)
        top3 = np.argsort(-(qv @ emb.T), axis=1)[:, :3]
        hit = float(np.mean([i in top3[r] for r, i in enumerate(idx)]))
        print(f"Sanity 2 (judul sebagai kueri -> dokumennya di top-3): {hit:.0%} (ambang {SELF_HIT_MIN:.0%})")
        if hit < SELF_HIT_MIN:
            fails.append(f"Hanya {hit:.0%} dokumen ditemukan lewat judulnya sendiri; embedding atau model mencurigakan.")
    elif args.sanity:
        print("Uji --sanity dilewati karena pemeriksaan struktural sudah gagal.")

    for w in warns:
        print(f"[PERINGATAN] {w}")
    for f in fails:
        print(f"[GAGAL] {f}")
    print("HASIL:", "GAGAL" if fails else ("LOLOS (struktural + sanity)" if args.sanity else "LOLOS (struktural saja; jalankan --sanity dengan model)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
