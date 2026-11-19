"""MODEL PROKSI untuk uji alur/tampilan pada dataset nyata (900 dokumen, 384 dimensi) TANPA mengunduh e5.

BUKAN model e5 dan BUKAN representasi kualitas semantic: vektor kueri = rata-rata embedding NYATA dari dokumen
teratas hasil BM25 untuk kueri itu (ternormalisasi) ditambah derau kecil yang deterministik. Dipakai hanya untuk
memastikan bahwa pipeline, Adaptive QE, dan UI berjalan pada artefak nyata. Hasilnya TIDAK boleh dilaporkan
sebagai hasil penelitian."""
import zlib

import numpy as np

from src.retrieval import bm25_search, build_bm25_index


class ProxyModel:
    def __init__(self, documents_df, embeddings, top_n: int = 5, noise: float = 0.35):
        self.docs, self.emb, self.top_n, self.noise = documents_df, embeddings, top_n, noise
        self.bm25 = build_bm25_index(documents_df)
        self.pos = {d: i for i, d in enumerate(documents_df["doc_id"])}

    def encode(self, texts, normalize_embeddings=True, **kw):
        out = []
        for t in texts:
            q = str(t).removeprefix("query: ")
            res = bm25_search(q, self.docs, self.bm25, self.top_n)
            idx = [self.pos[d] for d in res["doc_id"]] if len(res) else []
            v = self.emb[idx].mean(axis=0) if idx else np.zeros(self.emb.shape[1], dtype=np.float32)
            rng = np.random.default_rng(zlib.crc32(q.encode()))
            v = v / (np.linalg.norm(v) or 1.0) + self.noise * rng.standard_normal(self.emb.shape[1]) / np.sqrt(self.emb.shape[1])
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.asarray(out, dtype=np.float32)
