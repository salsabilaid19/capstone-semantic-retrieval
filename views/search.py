"""Document Search: satu metode (dengan fallback BM25 yang dilaporkan) atau perbandingan 4 metode tanpa substitusi."""
import streamlit as st

from src import ui_common as ui
from src import ui_theme as T
from src.retrieval import ALL_METHODS, METHOD_ADAPTIVE, METHOD_STATIC
from src.service import (ADAPTIVE_QE_DISCLAIMER, compare_methods, keyword_coverage, overlap_matrix,
                         runs_to_frame, search_with_fallback)

docs = ui.load_documents_or_stop()
T.page_header("search", "Document Search", "Cari artikel akademik dengan salah satu dari empat metode, atau bandingkan keempatnya.")
if int((docs["data_source"] == "synthetic").sum()):
    T.note("warn", "Sebagian/seluruh dokumen adalah <b>data contoh sintetis</b> (bukan artikel nyata).")

NAMA = {T.METHOD_SHORT[m]: m for m in ALL_METHODS}
with st.form("form_cari"):
    kueri_input = st.text_input("Kata kunci atau topik", placeholder="contoh: analisis sentimen  |  deteksi penyakit daun dengan CNN")
    c1, c2, c3 = st.columns([2.2, 1.2, 1])
    metode_pilih = c1.selectbox("Metode pencarian", list(NAMA), index=3)
    top_k = c2.slider("Jumlah hasil", 3, 20, 10)
    c3.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    cari = c3.form_submit_button("Cari", type="primary", use_container_width=True)

tampil = st.radio("Tampilan", ["Hasil pencarian", "Bandingkan 4 metode"], horizontal=True, label_visibility="collapsed")
tampil_skor = st.checkbox("Tampilkan skor relevansi", value=True)

if cari:
    if not kueri_input.strip():
        st.warning("Kueri masih kosong. Silakan masukkan kata kunci atau topik.")
        st.session_state.pop("kueri_aktif", None)
    else:
        st.session_state["kueri_aktif"] = kueri_input.strip()
        st.session_state["metode_aktif"] = NAMA[metode_pilih]
        st.session_state["k_aktif"] = top_k

kueri = st.session_state.get("kueri_aktif")
metode = st.session_state.get("metode_aktif", METHOD_ADAPTIVE)
k = st.session_state.get("k_aktif", 10)


def panel_kueri(info: dict, used_method: str) -> None:
    """Kueri asli, kueri akhir, istilah tambahan, dan alasan keputusan."""
    terms = "".join(f'<span class="chip term">+ {T.esc(t)}</span>' for t in info.get("expansion_terms", []))
    lvl = info.get("expansion_level", "")
    badge = T.level_badge(lvl) if used_method == METHOD_ADAPTIVE and lvl in T.LEVEL_NAMES else ""
    if used_method == METHOD_STATIC:
        badge = '<span class="badge" style="background:#7C3AED">Static QE</span>'
    st.markdown(f'<div class="qbox"><div class="lab">Kueri asli</div><div class="q">{T.esc(info["original_query"])} {badge}</div>'
                f'<div class="lab">Kueri yang dipakai mencari</div><div class="q">{T.esc(info["final_query"])}</div>'
                f'<div class="lab">Istilah tambahan</div><div>{terms or "<span class=sub>Tidak ada (kueri asli dipakai)</span>"}</div></div>',
                unsafe_allow_html=True)
    if used_method in (METHOD_ADAPTIVE, METHOD_STATIC):
        st.markdown(f"**Alasan keputusan:** {info['decision_reason']}")
        if info.get("fallback_note"):
            st.caption(info["fallback_note"])
        if info.get("indicators"):
            with st.expander("Detail indikator keputusan Adaptive QE"):
                st.dataframe(ui.indicator_rows(info), hide_index=True, width="stretch")
                cands = info.get("candidates") or []
                if cands:
                    st.caption("Kandidat istilah (dukungan = jumlah dokumen dari top-5 yang memuat istilah)")
                    import pandas as pd
                    st.dataframe(pd.DataFrame(cands).rename(columns={"term": "Istilah", "support": "Dukungan", "first_doc_rank": "Peringkat dok. asal"}),
                                 hide_index=True, width="stretch")


if kueri and tampil == "Hasil pencarian":
    bm25, get_embeddings, get_model = ui.make_search_context(docs)
    with st.spinner("Mencari dokumen..."):
        outcome = search_with_fallback(kueri, k, docs, bm25, get_embeddings, get_model, method=metode)
    if outcome.fell_back:
        st.warning(outcome.notice)
    st.caption(f"Metode pencarian yang berjalan: **{outcome.used_method}**")
    panel_kueri(outcome.info, outcome.used_method)
    if outcome.results.empty:
        st.info("Tidak ada dokumen yang ditemukan untuk kueri tersebut.")
    else:
        st.markdown(f"##### {len(outcome.results)} dokumen teratas")
        ui.render_results_cards(outcome.results, docs, tampil_skor)

elif kueri:   # ---- perbandingan 4 metode (tanpa substitusi diam-diam) ----
    with st.expander("Bagaimana Adaptive QE memutuskan?"):
        st.markdown(ADAPTIVE_QE_DISCLAIMER)
    cakupan = keyword_coverage(docs)
    if cakupan < 0.5:
        st.warning(f"Hanya {cakupan:.0%} dokumen yang memiliki kata kunci; ekspansi sering tidak terjadi pada data seperti ini.")
    bm25, get_embeddings, get_model = ui.make_search_context(docs)
    with st.spinner("Menjalankan 4 metode..."):
        runs = compare_methods(kueri, min(k, 10), docs, bm25, get_embeddings, get_model)
    st.subheader(f"Kueri: “{kueri}”")
    cols = st.columns(len(runs))
    for col, run in zip(cols, runs):
        with col:
            st.markdown(f"#### {run.method}")
            st.markdown(f'<div style="height:4px;border-radius:4px;background:{T.METHOD_COLORS[run.method]};margin:-6px 0 10px 0"></div>', unsafe_allow_html=True)
            if not run.available:
                st.error(f"Tidak tersedia: {run.error}")
                continue
            if run.info["is_expanded"]:
                st.caption(f"Kueri akhir: *{run.info['final_query']}*")
            lvl = run.info.get("expansion_level", "")
            if run.method == METHOD_ADAPTIVE and lvl in T.LEVEL_NAMES:
                st.markdown(T.level_badge(lvl), unsafe_allow_html=True)
            if run.results.empty:
                st.info("Tidak ada hasil.")
            for _, row in run.results.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{int(row['rank'])}.** {row['title']}")
                    meta = " · ".join(x for x in (str(row["year"]), ui.short_journal(row["journal"])) if x)
                    st.caption(meta + (f"  |  skor {row['score']:.4f}" if tampil_skor else ""))
    st.subheader("Detail proses tiap metode")
    for run in runs:
        if run.available:
            with st.expander(run.method):
                st.write(f"**Kueri akhir:** {run.info['final_query']}")
                st.write(f"**Istilah expansion:** {', '.join(run.info['expansion_terms']) or '-'}")
                st.write(f"**Alasan keputusan:** {run.info['decision_reason']}")
    if sum(r.available for r in runs) > 1:
        st.subheader("Irisan hasil antar metode (Jaccard top-k)")
        st.caption("1.0 = himpunan dokumen top-k identik; 0.0 = tidak ada dokumen yang sama. Urutan tidak dihitung.")
        st.dataframe(overlap_matrix(runs).round(2), width="stretch")
    tabel = runs_to_frame(kueri, runs)
    if not tabel.empty:
        st.download_button("Unduh hasil (CSV)", tabel.to_csv(index=False).encode("utf-8"), file_name="hasil_perbandingan.csv", mime="text/csv")
    st.caption("Catatan: perbandingan satu kueri hanya ilustrasi untuk membandingkan keempat metode, bukan pengukuran kualitas resmi.")
else:
    T.note("info", "Masukkan kata kunci lalu tekan <b>Cari</b>. Metode bawaan: Semantic + Adaptive QE; bila embedding/model "
                   "semantic tidak tersedia, sistem memakai BM25 dan memberitahu Anda.")
