"""Pembuatan embedding dokumen: satu-satunya sumber kebenaran untuk notebook, CLI lokal, dan pemeriksa.

Kontrak (harus sama dengan encode_query di src/retrieval.py):
- teks dokumen = "passage: " + document_text; teks kueri = "query: " + kueri (konvensi model E5)
- vektor float32, ternormalisasi (norma 1) sehingga dot product = cosine similarity
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.artifacts import CONFIG_NAME, DOC_PREFIX, build_model_config, validate_embeddings

EXPECTED_DIMS = {"intfloat/multilingual-e5-small": 384}


def encode_documents(model, documents_df: pd.DataFrame, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """Encode seluruh dokumen dengan prefix 'passage: '. Mengembalikan array float32 (n_dokumen, dim) ternormalisasi."""
    if len(documents_df) == 0:
        raise ValueError("Tidak ada dokumen untuk di-encode.")
    texts = [DOC_PREFIX + t for t in documents_df["document_text"].astype(str)]
    emb = np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                  show_progress_bar=show_progress), dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(texts):
        raise ValueError(f"Bentuk embedding tidak sesuai: {emb.shape} untuk {len(texts)} dokumen.")
    return emb


def _atomic_write(path: Path, writer) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def library_versions() -> dict:
    """Versi library yang memengaruhi embedding (untuk reprodusibilitas); 'tidak terpasang' bila tidak ada."""
    from importlib import metadata
    out = {}
    for pkg in ("sentence-transformers", "torch", "transformers", "numpy"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "tidak terpasang"
    return out


def write_embedding_artifacts(artifacts_dir, documents_df: pd.DataFrame, embeddings: np.ndarray, model_name: str,
                              extra: Optional[dict] = None) -> dict:
    """Validasi lalu tulis embeddings.npy + model_config.json. Bila validasi gagal, berkas lama TIDAK diubah.
    `extra` (mis. library_versions) digabung ke config sebagai catatan reprodusibilitas."""
    config = build_model_config(model_name, documents_df["doc_id"].tolist(), embeddings.shape[1])
    config.update(extra or {})
    validate_embeddings(documents_df, embeddings, config, model_name)      # melempar ValueError bila tidak valid
    d = Path(artifacts_dir)
    d.mkdir(parents=True, exist_ok=True)
    _atomic_write(d / "embeddings.npy", lambda p: _save_npy(p, embeddings))
    _atomic_write(d / CONFIG_NAME, lambda p: Path(p).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"))
    return config


def _save_npy(path: str, arr: np.ndarray) -> None:
    with open(path, "wb") as f:           # np.save pada nama tanpa .npy menambah ekstensi; pakai handle agar nama tetap
        np.save(f, arr, allow_pickle=False)


def projected_size_mb(n_docs: int, dim: int = 384, bytes_per_value: int = 4) -> float:
    """Perkiraan ukuran embeddings.npy (MB) tanpa header: n_docs x dim x 4 byte (float32)."""
    return n_docs * dim * bytes_per_value / 1024 / 1024
