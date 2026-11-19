"""BM25 dan Semantic Retrieval (multilingual-e5-small)."""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.artifacts import QUERY_PREFIX, read_model_config, validate_embeddings
from src.preprocessing import STANDARD_COLUMNS, format_year

MODEL_NAME = "intfloat/multilingual-e5-small"
RESULT_COLUMNS = [
    "rank", "doc_id", "title", "abstract", "keywords", "author", "year", "journal",
    "doi", "document_type", "source_url", "data_source", "score",
]
_META_COLUMNS = RESULT_COLUMNS[1:-1]

# Stopword Indonesia ringkas (kata fungsi). Tanpa stemmer: imbuhan tidak dinormalisasi (keterbatasan).
STOPWORDS_ID = frozenset("""
yang dan di ke dari untuk pada dengan ini itu adalah dalam atau oleh sebagai juga serta dapat akan
terhadap antara tersebut agar karena bahwa sebuah para ada tidak lebih telah sudah bagi hingga
the of and for in on to a an is are with by
""".split())


def tokenize(text: str, remove_stopwords: bool = True) -> list:
    """Tokenisasi untuk BM25: huruf kecil, ambil token kata (tanda baca dibuang), buang stopword.
    Bila semua token kueri adalah stopword, token asli dipertahankan agar kueri tidak kosong."""
    if not isinstance(text, str):
        return []
    tokens = re.findall(r"\w+", text.lower())
    if not remove_stopwords:
        return tokens
    kept = [t for t in tokens if t not in STOPWORDS_ID]
    return kept if kept else tokens


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _format_results(documents_df: pd.DataFrame, scores: np.ndarray, top_k: int) -> pd.DataFrame:
    top_k = max(1, min(int(top_k), len(documents_df)))
    idx = np.argsort(-scores, kind="stable")[:top_k]
    out = documents_df.iloc[idx].reindex(columns=_META_COLUMNS).astype(object)
    out = out.where(out.notna(), "").reset_index(drop=True)
    out["year"] = out["year"].map(format_year)          # selalu teks: '2021' atau ''

    out["score"] = scores[idx].astype(float)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[RESULT_COLUMNS]


# ---------------- BM25 ----------------
def build_bm25_index(documents_df: pd.DataFrame) -> BM25Okapi:
    """Indeks dibangun dari judul + abstrak + kata kunci (bukan teks templat berlabel 'Judul:/Abstrak:')."""
    cols = [c for c in ("title", "abstract", "keywords") if c in documents_df.columns]
    if cols:
        texts = documents_df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    else:
        texts = documents_df["document_text"].fillna("")
    return BM25Okapi([tokenize(t) for t in texts])


def bm25_search(query: str, documents_df: pd.DataFrame, bm25_index, top_k: int = 10) -> pd.DataFrame:
    tokens = tokenize(query)
    if not tokens or len(documents_df) == 0:
        return _empty_result()
    scores = np.asarray(bm25_index.get_scores(tokens), dtype=float)
    return _format_results(documents_df, scores, top_k)


# ---------------- Semantic ----------------
def load_embedding_model(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def load_documents(artifacts_dir: str = "artifacts") -> pd.DataFrame:
    """Baca documents_processed.csv SAJA (cukup untuk BM25; tidak butuh embeddings.npy).
    Semua kolom dibaca sebagai teks; tahun dinormalisasi ('2021', bukan '2021.0')."""
    docs_path = Path(artifacts_dir) / "documents_processed.csv"
    if not docs_path.exists():
        raise FileNotFoundError(
            f"File {docs_path} belum ada. Jalankan notebook generate_embeddings.ipynb "
            "atau src.preprocessing.process_csv terlebih dahulu."
        )
    df = pd.read_csv(docs_path, encoding="utf-8", dtype=str, keep_default_na=False).fillna("")
    for col in STANDARD_COLUMNS + ["document_text"]:   # kompatibel dengan file lama
        if col not in df.columns:
            df[col] = ""
    df["year"] = df["year"].map(format_year)
    return df


def load_embeddings(artifacts_dir: str, documents_df: pd.DataFrame,
                    expected_model: str = MODEL_NAME) -> np.ndarray:
    """Baca embeddings.npy dan validasi terhadap dokumen & model_config.json."""
    emb_path = Path(artifacts_dir) / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"File {emb_path} belum ada. Buat embedding lewat notebooks/generate_embeddings.ipynb "
            "(Google Colab), lalu salin embeddings.npy ke folder artifacts/."
        )
    embeddings = np.load(emb_path)
    for msg in validate_embeddings(documents_df, embeddings, read_model_config(artifacts_dir), expected_model):
        warnings.warn(msg, stacklevel=2)
    return embeddings


def load_artifacts(artifacts_dir: str = "artifacts"):
    """Baca dokumen + embeddings (dengan validasi). Untuk BM25 saja gunakan load_documents()."""
    documents_df = load_documents(artifacts_dir)
    return documents_df, load_embeddings(artifacts_dir, documents_df)


def encode_query(query: str, model) -> np.ndarray:
    vec = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)
    return np.asarray(vec)[0]


def semantic_scores(query: str, document_embeddings: np.ndarray, model) -> np.ndarray:
    """Skor cosine kueri terhadap SEMUA dokumen (vektor panjang N). Dipakai juga oleh Adaptive QE."""
    q = encode_query(query.strip(), model)
    # Embedding sudah ternormalisasi -> dot product = cosine similarity
    return np.asarray(document_embeddings @ q, dtype=float)


def semantic_search(query: str, documents_df: pd.DataFrame, document_embeddings: np.ndarray,
                    model, top_k: int = 10) -> pd.DataFrame:
    if not isinstance(query, str) or not query.strip() or len(documents_df) == 0:
        return _empty_result()
    return _format_results(documents_df, semantic_scores(query, document_embeddings, model), top_k)


# ---------------- Pipeline 4 metode (dipakai app.py dan evaluation.py) ----------------
METHOD_BM25 = "BM25"
METHOD_SEMANTIC = "Semantic Retrieval"
METHOD_STATIC = "Semantic Retrieval + Static Query Expansion"
METHOD_ADAPTIVE = "Semantic Retrieval + Adaptive Query Expansion"
ALL_METHODS = [METHOD_BM25, METHOD_SEMANTIC, METHOD_STATIC, METHOD_ADAPTIVE]
INITIAL_DEPTH = 5   # kedalaman hasil awal untuk kandidat ekspansi (Adaptive memakai top-5, Static top-3)


def search_with_method(method: str, query: str, documents_df: pd.DataFrame,
                       document_embeddings: np.ndarray, model, bm25_index,
                       top_k: int = 10):
    """Jalankan salah satu dari 4 metode. Mengembalikan (hasil_df, info_dict)."""
    from src.query_expansion import adaptive_expand_query, score_diagnostics, static_expand_query

    q = (query or "").strip()
    info = {
        "method": method, "original_query": q, "final_query": q,
        "is_expanded": False, "expansion_terms": [], "decision_reason": "-",
    }
    if not q:
        info["decision_reason"] = "Kueri kosong."
        return _empty_result(), info

    if method == METHOD_BM25:
        info["decision_reason"] = "BM25 tidak menggunakan query expansion."
        return bm25_search(q, documents_df, bm25_index, top_k), info

    # Semua metode semantic memulai dengan pencarian awal (skor seluruh korpus dihitung sekali)
    scores = semantic_scores(q, document_embeddings, model)
    initial = _format_results(documents_df, scores, max(top_k, INITIAL_DEPTH))
    if method == METHOD_SEMANTIC:
        info["decision_reason"] = "Semantic Retrieval tanpa query expansion."
        return initial.head(top_k), info

    if method == METHOD_STATIC:
        decision = static_expand_query(q, initial)
    elif method == METHOD_ADAPTIVE:
        decision = adaptive_expand_query(q, initial, diagnostics=score_diagnostics(scores))
    else:
        raise ValueError(f"Metode tidak dikenal: {method}")

    info.update(decision)
    info["method"] = method
    if decision["is_expanded"]:
        results = semantic_search(decision["final_query"], documents_df,
                                  document_embeddings, model, top_k)
    else:
        results = initial.head(top_k)
    return results, info
