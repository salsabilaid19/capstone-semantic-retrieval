"""Komponen UI Streamlit bersama untuk halaman publik dan Mode Eksperimen."""
import os
import re

import pandas as pd
import streamlit as st

from src import retrieval as R
from src.artifacts import doc_ids_hash
from src.service import SemanticUnavailable, article_link, try_load


def artifacts_dir() -> str:
    """Folder artifacts. Urutan: variabel lingkungan ARTIFACTS_DIR; bila tidak ada, `artifacts_real` (dataset penelitian)
    jika berisi dokumen; terakhir `artifacts` (13 dokumen SINTETIS untuk uji coba)."""
    env = os.environ.get("ARTIFACTS_DIR")
    if env:
        return env
    return "artifacts_real" if os.path.exists(os.path.join("artifacts_real", "documents_processed.csv")) else "artifacts"


# ---------- Cache: dimuat sekali per proses server ----------
def corpus_is_indonesian(directory: str = None) -> bool:
    """True bila >= 50% dokumen berbahasa Indonesia (heuristik kolom `language`). Dibaca langsung dengan pandas
    (bukan lewat cache Streamlit) agar aman dipanggil sebelum st.set_page_config. Gagal baca -> True (teks lama)."""
    try:
        path = os.path.join(directory or artifacts_dir(), "documents_processed.csv")
        lang = pd.read_csv(path, usecols=["language"])["language"]
        return bool(len(lang)) and float((lang == "id").mean()) >= 0.5
    except Exception:
        return True


@st.cache_data(show_spinner=False)
def _load_docs(directory: str) -> pd.DataFrame:
    return R.load_documents(directory)


@st.cache_resource(show_spinner=False)
def _load_bm25(directory: str, key: str, _docs: pd.DataFrame):
    return R.build_bm25_index(_docs)


@st.cache_resource(show_spinner=False)
def _embeddings_cached(directory: str, key: str, _docs: pd.DataFrame):
    # Kegagalan ikut di-cache (tidak mengulang percobaan lambat pada tiap pencarian)
    return try_load(lambda: R.load_embeddings(directory, _docs), "embedding")


@st.cache_resource(show_spinner=False)
def _model_cached():
    return try_load(lambda: R.load_embedding_model(), "model")


def reset_semantic_cache() -> None:
    """Paksa percobaan ulang memuat embedding dan model (mis. setelah koneksi pulih)."""
    _embeddings_cached.clear()
    _model_cached.clear()


def load_documents_or_stop() -> pd.DataFrame:
    try:
        return _load_docs(artifacts_dir())
    except FileNotFoundError as err:
        st.error(f"Data belum tersedia. {err}")
        st.stop()


def make_search_context(docs: pd.DataFrame):
    """Kembalikan (bm25_index, get_embeddings, get_model) untuk src.service."""
    directory, key = artifacts_dir(), doc_ids_hash(docs["doc_id"])
    bm25 = _load_bm25(directory, key, docs)

    def get_embeddings():
        value, reason = _embeddings_cached(directory, key, docs)
        if value is None:
            raise SemanticUnavailable(reason)
        return value

    def get_model():
        value, reason = _model_cached()
        if value is None:
            raise SemanticUnavailable(reason)
        return value

    return bm25, get_embeddings, get_model


# ---------- Tampilan ----------
def truncate(text: str, limit: int = 380) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def data_notice(docs: pd.DataFrame) -> None:
    n = len(docs)
    n_syn = int((docs["data_source"] == "synthetic").sum())
    years = pd.to_numeric(docs["year"], errors="coerce").dropna()
    rentang = f", tahun {int(years.min())}–{int(years.max())}" if len(years) else ""
    if n_syn == n:
        st.warning(f"Seluruh {n} dokumen saat ini adalah **data contoh sintetis** untuk uji coba, "
                   "bukan artikel jurnal nyata.")
    elif n_syn:
        st.warning(f"{n_syn} dari {n} dokumen adalah **data contoh sintetis** (bukan artikel nyata) "
                   "dan diberi label pada hasil.")
    else:
        st.caption(f"Koleksi: {n} artikel{rentang}.")


def render_results(results: pd.DataFrame, show_score: bool = False) -> None:
    for _, row in results.iterrows():
        with st.container(border=True):
            st.markdown(f"**{int(row['rank'])}. {row['title']}**")
            meta = [str(row["year"]) if row["year"] else "",
                    str(row["journal"]) if row["journal"] else "",
                    str(row["author"]) if row["author"] else "",
                    "⚠️ DATA SINTETIS (bukan artikel nyata)" if row["data_source"] == "synthetic" else ""]
            st.caption(" · ".join(m for m in meta if m))
            abstract = str(row["abstract"] or "")
            st.write(truncate(abstract))
            if len(abstract) > 380:
                with st.expander("Baca abstrak lengkap"):
                    st.write(abstract)
            if row["keywords"]:
                st.markdown(f"*Kata kunci:* {row['keywords']}")
            if show_score:
                st.caption(f"Skor: {row['score']:.4f}")
            link = article_link(row)
            if link:
                st.markdown(f"[Buka artikel asli ↗]({link})")


def render_process_detail(info: dict, used_method: str) -> None:
    with st.expander("Detail proses pencarian", expanded=True):
        st.write(f"**Metode yang berjalan:** {used_method}")
        st.write(f"**Kueri asli:** {info['original_query']}")
        st.write(f"**Kueri akhir:** {info['final_query']}")
        st.write(f"**Expansion dipakai:** {'Ya' if info['is_expanded'] else 'Tidak'}")
        st.write(f"**Istilah expansion:** {', '.join(info['expansion_terms']) or '-'}")
        st.write(f"**Alasan keputusan:** {info['decision_reason']}")
        if "word_count" in info:
            st.write(f"**Jumlah kata pada kueri:** {info['word_count']}")


# ---------- Ringkasan dataset & status (Dashboard) ----------
def short_journal(name: str) -> str:
    """'IJCCS (Indonesian Journal ...)' -> 'IJCCS'; 'Jurnal RESTI (Rekayasa ...)' -> 'Jurnal RESTI'."""
    return str(name).split(" (")[0].strip() or "(tanpa jurnal)"


def dataset_overview(docs: pd.DataFrame) -> dict:
    """Statistik ringkas dataset untuk Dashboard (dihitung dari documents_processed.csv, bukan angka tetap)."""
    jr = docs["journal"].map(short_journal)
    lang = docs["language"].replace("", "?")
    years = pd.to_numeric(docs["year"], errors="coerce").dropna().astype(int)
    abs_len = docs["abstract"].astype(str).str.len()
    return {
        "n_docs": len(docs), "n_journals": int(jr.nunique()),
        "journals": jr.value_counts().rename_axis("jurnal").reset_index(name="n"),
        "languages": lang.value_counts().rename_axis("bahasa").reset_index(name="n"),
        "journal_language": pd.crosstab(jr, lang).stack().rename("n").reset_index().rename(columns={"journal": "jurnal", "language": "bahasa"}),
        "years": years.value_counts().sort_index().rename_axis("year").reset_index(name="n"),
        "year_range": (int(years.min()), int(years.max())) if len(years) else None,
        "keyword_coverage": float((docs["keywords"].astype(str).str.strip() != "").mean()),
        "short_abstract_ids": docs.loc[abs_len < 200, "doc_id"].tolist(),
        "median_abstract": int(abs_len.median()) if len(abs_len) else 0,
        "duplicate_titles": int(docs["title"].str.lower().str.strip().duplicated(keep=False).sum()),
        "synthetic": int((docs["data_source"] == "synthetic").sum()),
    }


def lang_names(code: str) -> str:
    return {"en": "Inggris", "id": "Indonesia"}.get(code, code)


def read_json(name: str):
    import json
    path = os.path.join(artifacts_dir(), name)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def embedding_status(docs: pd.DataFrame) -> dict:
    """Status embedding: dimuat & lolos validasi? (memakai cache yang sama dengan pencarian)."""
    value, reason = _embeddings_cached(artifacts_dir(), doc_ids_hash(docs["doc_id"]), docs)
    cfg = read_json("model_config.json")
    return {"ok": value is not None, "reason": reason, "shape": None if value is None else tuple(value.shape),
            "model_name": cfg.get("model_name", R.MODEL_NAME), "dim": cfg.get("embedding_dim"),
            "hash_checked": bool(cfg.get("doc_ids_sha1"))}


def semantic_library_installed() -> bool:
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None


def indicator_rows(info: dict) -> pd.DataFrame:
    """Tabel indikator keputusan Adaptive QE untuk ditampilkan ke pengguna."""
    ind = info.get("indicators") or {}
    if not ind:
        return pd.DataFrame(columns=["Indikator", "Nilai", "Arti"])
    th = info.get("thresholds", {})
    rows = [
        ("z skor top-1", f"{ind['top1_z']:.2f}", f"Seberapa jauh skor dokumen teratas di atas rata-rata korpus (kuat >= {th.get('top1_z_high')}, lemah < {th.get('top1_z_low')})"),
        ("z rata-rata top-5", f"{ind['top5_mean_z']:.2f}", f"Apakah lima hasil teratas sama-sama menonjol (kuat >= {th.get('top5_z_high')}, lemah < {th.get('top5_z_low')})"),
        ("Selisih top-1 vs top-2", f"{ind['gap_z']:.2f} SD", f">= {th.get('gap_z_clear')} SD berarti satu dokumen jelas menonjol"),
        ("Jarak top-1 ke top-10", f"{ind['spread_z']:.2f} SD", f"< {th.get('spread_z_flat')} SD berarti sepuluh teratas hampir sama (tak ada yang menonjol)"),
        ("Skor cosine top-1", f"{ind['top1_score']:.4f}", f"Rata-rata korpus {ind['mean_score']:.4f}, simpangan baku {ind['std_score']:.4f}"),
        ("Kandidat istilah", f"{info.get('n_supported_candidates', 0)} didukung dari {info.get('n_candidates', 0)}",
         f"Didukung = muncul pada >= {th.get('min_support')} dari {th.get('candidate_docs')} dokumen teratas"),
        ("Jumlah kata kueri", str(info.get("word_count", "")), f">= {th.get('specific_query_words')} kata dianggap spesifik (turun satu tingkat)"),
    ]
    return pd.DataFrame(rows, columns=["Indikator", "Nilai", "Arti"])


# ---------- Kartu hasil bergaya dashboard ----------
def render_results_cards(results: pd.DataFrame, docs: pd.DataFrame, show_score: bool = True) -> None:
    """Daftar hasil sebagai kartu: peringkat, judul, jurnal/tahun/bahasa, bilah skor, abstrak, kata kunci, tautan."""
    from src.ui_theme import esc
    lang = dict(zip(docs["doc_id"], docs["language"])) if "language" in docs else {}
    top = float(results["score"].max()) if len(results) and float(results["score"].max()) > 0 else 1.0
    for _, row in results.iterrows():
        chips = []
        if row["journal"]:
            chips.append(f'<span class="chip j">{esc(short_journal(row["journal"]))}</span>')
        if row["year"]:
            chips.append(f'<span class="chip y">{esc(row["year"])}</span>')
        if lang.get(row["doc_id"]):
            chips.append(f'<span class="chip y">{esc(lang_names(lang[row["doc_id"]]))}</span>')
        if row["data_source"] == "synthetic":
            chips.append('<span class="chip term">DATA SINTETIS (bukan artikel nyata)</span>')
        abstract = re.sub(r"^(Abstrak|Abstract)(?=[A-Z])", "", str(row["abstract"] or ""))   # awalan hasil scraping (tampilan saja)
        kws = [k.strip() for k in str(row["keywords"] or "").replace("|", ";").replace(",", ";").split(";") if k.strip()]
        seen, kw_html = set(), ""
        for k in kws:
            if k.lower() not in seen and len(seen) < 6:
                seen.add(k.lower())
                kw_html += f'<span class="chip k">{esc(k)}</span>'
        pct = max(2, min(100, int(100 * max(float(row["score"]), 0) / top)))
        score_html = (f'<div class="sc"><span>Skor relevansi</span><span class="bar"><i style="width:{pct}%"></i></span>'
                      f'<b>{float(row["score"]):.4f}</b></div>') if show_score else ""
        link = article_link(row)
        link_html = f'<a href="{esc(link)}" target="_blank" rel="noopener">Buka artikel asli &rarr;</a>' if link else ""
        st.markdown(
            f'<div class="res"><div><span class="rk">{int(row["rank"])}</span><span class="t">{esc(row["title"])}</span></div>'
            f'<div style="margin-top:6px">{"".join(chips)}</div>{score_html}'
            f'<div class="ab">{esc(truncate(abstract, 420))}</div>'
            f'<div>{kw_html}</div><div style="margin-top:6px">{link_html}</div></div>', unsafe_allow_html=True)
        if len(abstract) > 420:
            with st.expander("Baca abstrak lengkap"):
                st.write(abstract)
