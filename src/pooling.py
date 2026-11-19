"""Pooling untuk anotasi relevansi: gabungan top-k keempat metode, diacak, tanpa menyebut metode.

Tujuan: label tidak bias terhadap satu metode (bila hanya hasil satu metode yang dinilai, metode lain dirugikan).
Pool = berkas untuk anotator (tanpa info metode). Provenance = berkas terpisah untuk peneliti (metode & peringkat).
Keterbatasan: metode BARU yang dievaluasi kemudian dan tidak ada di pool akan dirugikan (dokumen tak berlabel dianggap tak relevan).
"""
import random
from typing import List, Tuple

import pandas as pd

from src.retrieval import ALL_METHODS, search_with_method

POOL_COLUMNS = ["query_id", "query", "doc_id", "title", "abstract", "keywords", "year", "journal", "source_url",
                "relevance", "annotator", "notes"]
LABEL_COLUMNS = ["query_id", "doc_id", "relevance", "annotator", "label_source", "notes"]


def build_pool(queries: pd.DataFrame, documents_df: pd.DataFrame, embeddings, model, bm25_index,
               depth: int = 10, seed: int = 13) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Kembalikan (pool_untuk_anotator, provenance). Urutan dokumen per kueri diacak (deterministik oleh seed)."""
    docs = documents_df.set_index("doc_id")
    pool_rows, prov_rows = [], []
    for _, q in queries.iterrows():
        ranks = {}
        for method in ALL_METHODS:
            res, _ = search_with_method(method, q["query"], documents_df, embeddings, model, bm25_index, top_k=depth)
            for r, d in enumerate(res["doc_id"].astype(str), start=1):
                ranks.setdefault(d, {})[method] = r
        order = sorted(ranks)
        random.Random(f"{seed}-{q['query_id']}").shuffle(order)       # acak: peringkat dan metode tidak terbaca dari urutan
        for d in order:
            row = docs.loc[d]
            pool_rows.append({"query_id": q["query_id"], "query": q["query"], "doc_id": d, "title": row["title"],
                              "abstract": row["abstract"], "keywords": row["keywords"], "year": row["year"],
                              "journal": row["journal"], "source_url": row["source_url"],
                              "relevance": "", "annotator": "", "notes": ""})
            prov = {"query_id": q["query_id"], "doc_id": d, "n_methods": len(ranks[d])}
            prov.update({f"rank[{m}]": ranks[d].get(m, "") for m in ALL_METHODS})
            prov_rows.append(prov)
    return pd.DataFrame(pool_rows, columns=POOL_COLUMNS), pd.DataFrame(prov_rows)


def pool_to_labels(filled: pd.DataFrame, annotator: str = "", allow_blank: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Ubah pool yang sudah diisi menjadi relevance_labels.csv. Kolom relevance kosong = BELUM DINILAI, bukan 0.
    allow_blank=False: gagal bila ada yang kosong/tidak valid. allow_blank=True: baris kosong dibuang (tidak berlabel)."""
    problems: List[str] = []
    rel = pd.to_numeric(filled["relevance"].astype(str).str.strip().replace("", pd.NA), errors="coerce")
    blank = filled["relevance"].astype(str).str.strip() == ""
    invalid = ~blank & ~rel.isin((0, 1, 2))
    if invalid.any():
        problems.append(f"{int(invalid.sum())} baris berisi relevance selain 0/1/2 (baris {list((filled.index[invalid] + 2)[:5])}).")
    if blank.any() and not allow_blank:
        problems.append(f"{int(blank.sum())} baris belum dinilai (relevance kosong). Selesaikan anotasi atau gunakan --allow-blank.")
    if problems:
        return pd.DataFrame(columns=LABEL_COLUMNS), problems
    keep = ~blank
    out = filled[keep].copy()
    out["relevance"] = rel[keep].astype(int)
    ann = out["annotator"].astype(str).str.strip()
    out["annotator"] = ann.where(ann != "", annotator)
    out["label_source"] = "pooled_manual"
    return out[LABEL_COLUMNS].reset_index(drop=True), []
