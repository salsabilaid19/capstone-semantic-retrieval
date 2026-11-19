"""Logika pencarian tingkat aplikasi: fallback otomatis ke BM25 dan perbandingan 4 metode.

Tanpa dependensi Streamlit agar dapat diuji. Prinsip:
- Pencarian PUBLIK boleh fallback ke BM25, tetapi fallback SELALU dilaporkan (alasan + metode yang berjalan).
- Mode EKSPERIMEN TIDAK boleh menukar metode secara diam-diam: metode semantic yang tidak tersedia
  ditandai tidak tersedia, bukan diganti BM25, agar perbandingan tidak menyesatkan.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd

from src.retrieval import (
    ALL_METHODS, METHOD_ADAPTIVE, METHOD_BM25, search_with_method,
)

DEFAULT_PUBLIC_METHOD = METHOD_ADAPTIVE

ADAPTIVE_QE_DISCLAIMER = (
    "**Cara kerja Adaptive Query Expansion (berbasis kualitas hasil awal).** Sistem mencari dulu dengan kueri asli, "
    "lalu menilai hasil awal dari skor kemiripan: seberapa jauh skor teratas di atas rata-rata seluruh korpus (z-score), "
    "apakah lima hasil teratas sama-sama menonjol, serta selisih skor teratas. Hasilnya tiga tingkat: **No Expansion**, "
    "**Light Expansion** (1 istilah tambahan), dan **Strong Expansion** (hingga 3 istilah). Istilah diambil dari kata kunci "
    "5 dokumen teratas dan hanya dipakai bila muncul pada minimal 2 dokumen teratas; bila kandidat tidak meyakinkan, "
    "tingkat diturunkan. Kueri panjang (>= 6 kata) diturunkan satu tingkat. "
    "Ini heuristik berbasis aturan dengan ambang awal yang perlu dikalibrasi: sistem **tidak** mendeteksi ambiguitas "
    "secara sempurna, dan kemanfaatan ekspansi baru dapat dinilai lewat evaluasi dengan penilaian relevansi manusia."
)


class SemanticUnavailable(RuntimeError):
    """Komponen semantic (embedding/model) tidak dapat dipakai; pesan sudah ramah pengguna."""


def _short(err: Exception, limit: int = 240) -> str:
    text = " ".join(str(err).split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def describe_failure(err: Exception, component: str = "model") -> str:
    """Ubah exception menjadi alasan singkat yang bisa dibaca pengguna (tanpa traceback)."""
    if isinstance(err, SemanticUnavailable):
        return str(err)
    if component == "embedding":
        if isinstance(err, FileNotFoundError):
            return f"Berkas embedding belum tersedia. {_short(err)}"
        if isinstance(err, ValueError):
            return f"Embedding tidak cocok dengan data. {_short(err)}"
        return f"Embedding gagal dimuat ({type(err).__name__})."
    if isinstance(err, ImportError):
        return "Library semantic (sentence-transformers) belum terpasang di server."
    if isinstance(err, MemoryError):
        return "Memori server tidak cukup untuk memuat model semantic."
    return (f"Model semantic gagal dimuat atau dijalankan ({type(err).__name__}). "
            "Kemungkinan model belum terunduh atau server tidak punya akses internet.")


def try_load(loader: Callable, component: str) -> Tuple[Optional[object], str]:
    """Jalankan loader; kembalikan (nilai, '') bila sukses atau (None, alasan) bila gagal."""
    try:
        return loader(), ""
    except Exception as err:  # noqa: BLE001 - sengaja luas: apa pun kegagalannya, aplikasi harus tetap hidup
        return None, describe_failure(err, component)


@dataclass
class SearchOutcome:
    results: pd.DataFrame
    info: dict
    requested_method: str
    used_method: str
    fallback_reason: str = ""

    @property
    def fell_back(self) -> bool:
        return self.used_method != self.requested_method

    @property
    def notice(self) -> str:
        if not self.fell_back:
            return ""
        return (f"Pencarian semantic sedang tidak tersedia, sehingga hasil di bawah memakai **{self.used_method}** "
                f"(pencocokan kata), bukan **{self.requested_method}**. Alasan: {self.fallback_reason} "
                "Hasil dapat berbeda dan kurang mampu menangkap makna kueri.")


def search_with_fallback(query: str, top_k: int, documents_df: pd.DataFrame, bm25_index,
                         get_embeddings: Callable, get_model: Callable,
                         method: str = DEFAULT_PUBLIC_METHOD) -> SearchOutcome:
    """Cari dengan `method`; bila komponen semantic gagal pada tahap mana pun, jatuh ke BM25 + alasan."""
    if method == METHOD_BM25:
        res, info = search_with_method(METHOD_BM25, query, documents_df, None, None, bm25_index, top_k)
        return SearchOutcome(res, info, method, METHOD_BM25)

    reason = ""
    try:
        try:
            embeddings = get_embeddings()
        except Exception as err:  # noqa: BLE001
            reason = describe_failure(err, "embedding")
            raise
        try:
            model = get_model()
        except Exception as err:  # noqa: BLE001
            reason = describe_failure(err, "model")
            raise
        try:
            res, info = search_with_method(method, query, documents_df, embeddings, model, bm25_index, top_k)
        except Exception as err:  # noqa: BLE001
            reason = describe_failure(err, "model")
            raise
        return SearchOutcome(res, info, method, method)
    except Exception:  # noqa: BLE001
        res, info = search_with_method(METHOD_BM25, query, documents_df, None, None, bm25_index, top_k)
        return SearchOutcome(res, info, method, METHOD_BM25, fallback_reason=reason)


# ---------------- Mode eksperimen ----------------
@dataclass
class MethodRun:
    method: str
    results: Optional[pd.DataFrame] = None
    info: Optional[dict] = None
    error: str = ""

    @property
    def available(self) -> bool:
        return self.results is not None


def compare_methods(query: str, top_k: int, documents_df: pd.DataFrame, bm25_index,
                    get_embeddings: Callable, get_model: Callable) -> List[MethodRun]:
    """Jalankan 4 metode. BM25 selalu jalan; metode semantic yang gagal ditandai error (TANPA substitusi)."""
    runs: List[MethodRun] = []
    emb, emb_reason = try_load(get_embeddings, "embedding")
    model, model_reason = (None, "") if emb is None else try_load(get_model, "model")
    unavailable = emb_reason or model_reason
    for method in ALL_METHODS:
        if method == METHOD_BM25:
            res, info = search_with_method(method, query, documents_df, None, None, bm25_index, top_k)
            runs.append(MethodRun(method, res, info))
        elif unavailable:
            runs.append(MethodRun(method, error=unavailable))
        else:
            try:
                res, info = search_with_method(method, query, documents_df, emb, model, bm25_index, top_k)
                runs.append(MethodRun(method, res, info))
            except Exception as err:  # noqa: BLE001
                runs.append(MethodRun(method, error=describe_failure(err, "model")))
    return runs


def overlap_matrix(runs: List[MethodRun]) -> pd.DataFrame:
    """Jaccard antar himpunan doc_id top-k tiap metode (hanya metode yang tersedia)."""
    avail = [r for r in runs if r.available]
    sets = {r.method: set(r.results["doc_id"]) for r in avail}
    names = [r.method for r in avail]
    data = [[(len(sets[a] & sets[b]) / len(sets[a] | sets[b])) if (sets[a] | sets[b]) else 0.0
             for b in names] for a in names]
    return pd.DataFrame(data, index=names, columns=names)


def runs_to_frame(query: str, runs: List[MethodRun]) -> pd.DataFrame:
    """Gabungkan hasil semua metode menjadi satu tabel (untuk diunduh sebagai CSV)."""
    frames = []
    for r in runs:
        if r.available and not r.results.empty:
            f = r.results[["rank", "doc_id", "title", "year", "journal", "score"]].copy()
            f.insert(0, "final_query", r.info["final_query"])
            f.insert(0, "method", r.method)
            f.insert(0, "query", query)
            frames.append(f)
    cols = ["query", "method", "final_query", "rank", "doc_id", "title", "year", "journal", "score"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def article_link(row) -> str:
    """URL asli artikel bila ada; selain itu tautan DOI; selain itu ''."""
    url = str(row.get("source_url", "") or "")
    if url.startswith("http"):
        return url
    doi = str(row.get("doi", "") or "")
    return f"https://doi.org/{doi}" if doi else ""


def keyword_coverage(documents_df: pd.DataFrame) -> float:
    """Persentase dokumen yang punya kata kunci (QE bergantung pada ini)."""
    if len(documents_df) == 0 or "keywords" not in documents_df:
        return 0.0
    return float((documents_df["keywords"].astype(str).str.strip() != "").mean())
