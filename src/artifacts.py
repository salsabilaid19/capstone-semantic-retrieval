"""Konfigurasi & validasi artifacts (embedding) agar tidak salah pasang dengan dokumen/model."""
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

CONFIG_NAME = "model_config.json"
DOC_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def doc_ids_hash(doc_ids: Iterable[str]) -> str:
    """Hash berurutan dari daftar doc_id. Berubah bila dokumen/urutannya berubah."""
    return hashlib.sha1("\n".join(map(str, doc_ids)).encode("utf-8")).hexdigest()


def build_model_config(model_name: str, doc_ids: List[str], embedding_dim: int) -> dict:
    return {
        "model_name": model_name,
        "document_prefix": DOC_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "normalize_embeddings": True,
        "num_documents": len(doc_ids),
        "embedding_dim": int(embedding_dim),
        "doc_ids_sha1": doc_ids_hash(doc_ids),
    }


def read_model_config(artifacts_dir) -> Optional[dict]:
    path = Path(artifacts_dir) / CONFIG_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_embeddings(documents_df, embeddings: np.ndarray, config: Optional[dict],
                        expected_model: str) -> List[str]:
    """Periksa kecocokan embedding. Melempar ValueError bila pasti salah;
    mengembalikan daftar peringatan (str) untuk hal yang tidak bisa dipastikan."""
    warnings_out: List[str] = []
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings.npy harus berdimensi 2, ditemukan {embeddings.ndim}.")
    if len(documents_df) != embeddings.shape[0]:
        raise ValueError(
            f"Jumlah dokumen ({len(documents_df)}) tidak sama dengan jumlah embedding "
            f"({embeddings.shape[0]}). Buat ulang embedding dari dataset yang sama."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings.npy mengandung NaN/inf. Buat ulang embedding.")
    norms = np.linalg.norm(embeddings, axis=1)
    if len(norms) and float(np.abs(norms - 1.0).max()) > 0.01:
        raise ValueError(
            "Embedding tidak ternormalisasi (norma != 1). Skor memakai dot product sebagai cosine, "
            "jadi embedding harus dibuat dengan normalize_embeddings=True."
        )
    if config is None:
        warnings_out.append("model_config.json tidak ditemukan; kecocokan model dan dokumen tidak dapat dipastikan.")
        return warnings_out
    if config.get("model_name") != expected_model:
        raise ValueError(
            f"Embedding dibuat dengan model '{config.get('model_name')}', "
            f"tetapi aplikasi memakai '{expected_model}'. Buat ulang embedding."
        )
    if int(config.get("embedding_dim", embeddings.shape[1])) != embeddings.shape[1]:
        raise ValueError("embedding_dim di model_config.json tidak sama dengan bentuk embeddings.npy.")
    expected_hash = config.get("doc_ids_sha1")
    if expected_hash:
        if doc_ids_hash(documents_df["doc_id"]) != expected_hash:
            raise ValueError(
                "Daftar/urutan dokumen berbeda dari saat embedding dibuat (hash doc_id tidak cocok). "
                "Jalankan ulang notebooks/generate_embeddings.ipynb dengan dataset saat ini."
            )
    else:
        warnings_out.append("model_config.json tidak memuat doc_ids_sha1 (konfigurasi lama/template); "
                            "kesesuaian urutan dokumen hanya diperiksa lewat jumlah baris.")
    return warnings_out
