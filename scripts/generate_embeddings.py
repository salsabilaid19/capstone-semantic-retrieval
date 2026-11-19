"""Buat embeddings.npy + model_config.json dari documents_processed.csv (alternatif lokal untuk notebook Colab).

Contoh:  python -m scripts.generate_embeddings --artifacts-dir artifacts_real
Pertama kali mengunduh model intfloat/multilingual-e5-small dari Hugging Face (butuh internet; ukuran ratusan MB) dan
butuh `pip install sentence-transformers` (menarik PyTorch). Di CPU, ratusan dokumen biasanya selesai dalam hitungan menit.
"""
import argparse
import sys
import time
from pathlib import Path

from src import retrieval as R
from src.embedding import encode_documents, library_versions, projected_size_mb, write_embedding_artifacts


def main(argv=None, model=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--model-name", default=R.MODEL_NAME)
    args = ap.parse_args(argv)
    try:
        docs = R.load_documents(args.artifacts_dir)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 2
    kinds = sorted(set(docs["data_source"]) - {""})
    print(f"Dataset: {args.artifacts_dir} | {len(docs)} dokumen | data_source={kinds or ['(kosong)']}")
    if model is None:
        try:
            model = R.load_embedding_model(args.model_name)
        except ImportError:
            print("Library sentence-transformers belum terpasang. Jalankan: pip install sentence-transformers", file=sys.stderr)
            return 2
        except Exception as err:  # noqa: BLE001 - unduhan gagal, offline, disk penuh, dll.
            print(f"Model '{args.model_name}' tidak dapat dimuat ({type(err).__name__}: {err}). "
                  "Periksa koneksi internet/ruang disk atau gunakan notebook Colab.", file=sys.stderr)
            return 2
    t0 = time.time()
    emb = encode_documents(model, docs, batch_size=args.batch_size, show_progress=True)
    try:
        write_embedding_artifacts(args.artifacts_dir, docs, emb, args.model_name, extra={"library_versions": library_versions()})
    except ValueError as err:
        print(f"Embedding tidak valid, berkas lama tidak diubah: {err}", file=sys.stderr)
        return 1
    size = Path(args.artifacts_dir, "embeddings.npy").stat().st_size / 1024 / 1024
    print(f"Selesai dalam {time.time() - t0:.1f} detik. Shape {emb.shape}, embeddings.npy = {size:.2f} MB "
          f"(perkiraan 500 dok = {projected_size_mb(500, emb.shape[1]):.2f} MB, 10.000 dok = {projected_size_mb(10000, emb.shape[1]):.1f} MB).")
    print("Langkah berikutnya: python -m scripts.check_embeddings --sanity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
