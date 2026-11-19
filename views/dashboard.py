"""Dashboard: ringkasan dataset, model, empat metode, dan status sistem (semua angka dihitung dari artefak)."""
import streamlit as st

from src import charts, ui_common as ui
from src import ui_theme as T
from src.query_expansion import DEFAULT_CONFIG
from src.retrieval import ALL_METHODS

docs = ui.load_documents_or_stop()
ov = ui.dataset_overview(docs)
emb = ui.embedding_status(docs)
_names = [ui.lang_names(b) for b in ov["languages"]["bahasa"]]
_fixed = {"Inggris": T.BLUE, "Indonesia": T.VIOLET}
_spare = [c for c in (T.CYAN, T.ORANGE, T.SLATE) ]
LANG_COLORS = {n: _fixed.get(n) or _spare[i % len(_spare)] for i, n in enumerate(_names)}
lang = dict(zip(ov["languages"]["bahasa"], ov["languages"]["n"]))
lang_txt = " · ".join(f"{n} {c.upper()}" for c, n in lang.items())
yr = f"{ov['year_range'][0]}–{ov['year_range'][1]}" if ov["year_range"] else "-"

st.markdown(
    '<div class="hero"><span class="tag">Capstone Project · Information Retrieval</span>'
    '<h1>Sistem Temu Balik Dokumen Akademik Berbasis Semantic Retrieval dengan Adaptive Query Expansion</h1>'
    '<p>Cari artikel jurnal ilmiah berdasarkan makna kueri, bukan hanya kecocokan kata. Sistem membandingkan empat metode '
    'pencarian dan memilih tingkat perluasan kueri (No, Light, Strong) dari kualitas hasil pencarian awal.</p></div>',
    unsafe_allow_html=True)

if ov["synthetic"]:
    T.note("warn", f"<b>{ov['synthetic']} dari {ov['n_docs']} dokumen adalah data contoh sintetis</b> untuk uji coba, bukan artikel nyata. "
                   "Jangan dipakai untuk klaim hasil penelitian.")

c = st.columns(4)
cards = [("file", f"{ov['n_docs']:,}".replace(",", "."), f"Dokumen artikel akademik ({yr})", T.BLUE),
         ("book", ov["n_journals"], "Jurnal sumber: " + ", ".join(ov["journals"]["jurnal"]), T.VIOLET),
         ("globe", lang_txt, "Distribusi bahasa dokumen", T.CYAN),
         ("cpu", "e5-small", f"{emb['model_name']} · {emb['dim'] or '?'} dimensi", T.ORANGE)]
for col, (ic, v, label, color) in zip(c, cards):
    col.markdown(T.stat_card(ic, v, label, color), unsafe_allow_html=True)

st.markdown("&nbsp;", unsafe_allow_html=True)
g1, g2, g3 = st.columns([1.25, 1, 1.25])
with g1:
    st.markdown("##### Dokumen per jurnal")
    st.altair_chart(charts.bar_count(ov["journals"], "jurnal", "n", "Jumlah dokumen", T.BLUE), width="stretch")
with g2:
    st.markdown("##### Distribusi bahasa")
    d = ov["languages"].assign(bahasa=lambda x: x["bahasa"].map(ui.lang_names))
    st.altair_chart(charts.donut(d, "bahasa", "n", [LANG_COLORS[b] for b in d["bahasa"]]), width="stretch")
with g3:
    st.markdown("##### Dokumen per tahun terbit")
    st.altair_chart(charts.year_area(ov["years"]), width="stretch")

g4, g5 = st.columns([1.25, 1])
with g4:
    st.markdown("##### Komposisi bahasa per jurnal")
    jl = ov["journal_language"].assign(bahasa=lambda x: x["bahasa"].map(ui.lang_names))
    st.altair_chart(charts.stacked_bar(jl, "jurnal", "bahasa", "n", LANG_COLORS), width="stretch")
with g5:
    st.markdown("##### Kualitas & status data")
    items = []
    items.append(("ok" if emb["ok"] else "warn", "Embedding " + (f"valid: {emb['shape'][0]} × {emb['shape'][1]}, hash doc_id "
                  + ("cocok" if emb["hash_checked"] else "belum diperiksa") if emb["ok"] else "belum tersedia atau tidak cocok")))
    n_short = len(ov["short_abstract_ids"])
    items.append(("ok" if not n_short else "warn",
                  "Semua abstrak ≥ 200 karakter" if not n_short else f"{n_short} dokumen berabstrak < 200 karakter (melanggar ketentuan): "
                  + ", ".join(ov["short_abstract_ids"][:3])))
    items.append(("ok" if ov["keyword_coverage"] >= .5 else "warn", f"Kata kunci tersedia pada {ov['keyword_coverage']:.0%} dokumen (sumber istilah ekspansi)"))
    items.append(("info", f"Median panjang abstrak {ov['median_abstract']} karakter"))
    if ov["duplicate_titles"]:
        items.append(("warn", f"{ov['duplicate_titles']} dokumen berjudul identik (kemungkinan rekaman ganda)"))
    for kind, txt in items:
        T.note(kind, T.esc(txt))

st.markdown("### Empat metode pencarian")
desc = {
    ALL_METHODS[0]: ("Pencocokan kata (lexical) dengan skor BM25 atas judul, abstrak, dan kata kunci.", "Selalu tersedia"),
    ALL_METHODS[1]: ("Kueri dan dokumen diubah menjadi vektor makna (e5), diurutkan menurut cosine similarity.", "Butuh embedding + model"),
    ALL_METHODS[2]: ("Semantic + selalu menambah hingga 2 istilah dari kata kunci 3 dokumen teratas hasil awal.", "Butuh embedding + model"),
    ALL_METHODS[3]: ("Semantic + tingkat ekspansi (No / Light / Strong) dipilih dari kualitas hasil awal.", "Butuh embedding + model"),
}
cols = st.columns(4)
for col, m in zip(cols, ALL_METHODS):
    status = desc[m][1]
    if m != ALL_METHODS[0]:
        status = "Siap (embedding valid)" if emb["ok"] else "Embedding belum siap"
    col.markdown(f'<div class="mcard"><div class="top" style="background:{T.METHOD_COLORS[m]}">{T.esc(T.METHOD_SHORT[m])}</div>'
                 f'<div class="body">{T.esc(desc[m][0])}<br><span class="st">{T.esc(status)}</span></div></div>', unsafe_allow_html=True)

st.markdown("&nbsp;", unsafe_allow_html=True)
T.note("info", "<b>Model semantic</b> dimuat saat pencarian semantic pertama" +
       (" (library sentence-transformers terpasang)." if ui.semantic_library_installed() else
        ". <b>Library sentence-transformers belum terpasang</b> di lingkungan ini, sehingga pencarian semantic otomatis memakai BM25 dengan pemberitahuan."))
st.page_link("views/search.py", label="Mulai mencari", icon=":material/search:")
