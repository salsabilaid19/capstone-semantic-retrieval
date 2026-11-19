"""Model dan artifacts PALSU untuk menguji alur kode. Tidak mewakili kualitas model e5 asli."""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import build_model_config
from src.preprocessing import clean_dataset
from src.retrieval import MODEL_NAME


class FakeModel:
    dim = 64

    def encode(self, texts, normalize_embeddings=True, **kw):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in str(t).lower().replace(":", " ").split():
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return out / n


class BrokenModel:
    def encode(self, *a, **kw):
        raise RuntimeError("encode gagal (simulasi)")


RAW = pd.DataFrame({
    "Title": ["Sistem Peringatan Dini Banjir", "Prediksi Curah Hujan", "Analisis Sentimen Ulasan",
              "Deteksi Hoaks Berita", "Pemetaan Rawan Banjir", "Klasifikasi Penyakit Padi"],
    "Abstract": ["Sensor air untuk banjir.", "Jaringan saraf untuk hujan, banjir.", "Naive Bayes untuk sentimen.",
                 "Klasifikasi berita hoaks.", "SIG untuk banjir dan lahan.", "CNN untuk daun padi."],
    "Year": [2022, "", 2020, 2021, 2019, 2023],
    "DOI": ["10.1/a", "10.1/b", "10.1/c", "", "10.1/e", "10.1/f"],
    "Publication": ["J1", "J2", "J3", "J4", "J5", "J6"],
    "Keywords": ["banjir; sensor; iot", "hujan; jaringan saraf", "sentimen; naive bayes", "hoaks; berita",
                 "SIG; pemetaan", "padi; cnn"],
})


def make_artifacts(path, data_source="synthetic", with_embeddings=True, raw=None):
    """Tulis documents_processed.csv (+ embeddings.npy & model_config.json) ke `path`."""
    path = Path(path)
    df = clean_dataset(RAW if raw is None else raw, default_data_source=data_source)
    df.to_csv(path / "documents_processed.csv", index=False)
    if with_embeddings:
        emb = FakeModel().encode(["passage: " + t for t in df.document_text])
        np.save(path / "embeddings.npy", emb)
        (path / "model_config.json").write_text(
            json.dumps(build_model_config(MODEL_NAME, df.doc_id.tolist(), emb.shape[1])), encoding="utf-8")
    return df
