"""About / Methodology: penjelasan sistem, dataset, model, metode, QE, dan alur kerja (angka diambil dari artefak)."""
import streamlit as st

from src import ui_common as ui
from src import ui_theme as T
from src.query_expansion import DEFAULT_CONFIG as C
from src.retrieval import MODEL_NAME

docs = ui.load_documents_or_stop()
ov = ui.dataset_overview(docs)
cfg = ui.read_json("model_config.json")
T.page_header("layers", "About & Methodology", "Bagaimana sistem bekerja, data yang dipakai, dan batasan yang perlu diketahui.")

st.markdown("### Alur kerja sistem")
steps = [("Kueri", "Pengguna memasukkan kata kunci atau topik."),
         ("Pencarian awal", "Kueri diubah menjadi vektor (prefix <code>query: </code>) dan dibandingkan dengan seluruh embedding dokumen."),
         ("Penilaian kualitas", "Skor seluruh korpus dianalisis: z top-1, rata-rata top-5, selisih dan sebaran skor."),
         ("Keputusan QE", "No / Light / Strong Expansion; istilah diambil dari kata kunci 5 dokumen teratas."),
         ("Pencarian akhir", "Kueri akhir dicari ulang; hasil diurutkan menurut cosine similarity.")]
st.markdown('<div class="flow">' + "".join(f'<div class="s"><span class="n">{i}</span><b>{T.esc(t)}</b>{d}</div>' for i, (t, d) in enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)

a, b = st.columns(2)
with a:
    st.markdown("### Dataset")
    st.markdown(f"""
- **{ov['n_docs']} artikel akademik** dari {ov['n_journals']} jurnal: {', '.join(f"{r.jurnal} ({r.n})" for r in ov['journals'].itertuples())}.
- Bahasa: {', '.join(f"{ui.lang_names(r.bahasa)} {r.n}" for r in ov['languages'].itertuples())}.
- Tahun terbit {ov['year_range'][0] if ov['year_range'] else '-'}–{ov['year_range'][1] if ov['year_range'] else '-'}; data berupa **metadata** (judul, abstrak, kata kunci), bukan teks penuh.
- Ketentuan: abstrak wajib ≥ 200 karakter. Saat ini **{len(ov['short_abstract_ids'])} dokumen** belum memenuhinya ({', '.join(ov['short_abstract_ids']) or '-'}); lihat dokumentasi.
- Kata kunci tersedia pada {ov['keyword_coverage']:.0%} dokumen dan menjadi sumber istilah ekspansi.
""")
with b:
    st.markdown("### Model embedding")
    st.markdown(f"""
- Model: `{cfg.get('model_name', MODEL_NAME)}` (multilingual, mendukung Indonesia dan Inggris).
- Dimensi vektor: **{cfg.get('embedding_dim', 384)}**, dinormalisasi sehingga dot product = cosine similarity.
- Dokumen di-embed dengan prefix `passage: ` dan kueri dengan `query: ` (kontrak model e5).
- Embedding dibuat sekali dan divalidasi terhadap dokumen lewat hash `doc_id` (`model_config.json`).
""")

st.markdown("### Empat metode yang dibandingkan")
m1, m2 = st.columns(2)
m1.markdown("""
**1. BM25.** Pencocokan kata berbobot (frekuensi istilah dan kelangkaan istilah) atas judul + abstrak + kata kunci. Tokenisasi sederhana tanpa stemmer; kata asing dan imbuhan tidak dinormalisasi.

**2. Semantic Retrieval.** Dokumen dan kueri dipetakan ke ruang vektor yang sama; dokumen diurutkan menurut cosine similarity sehingga kueri dapat cocok dengan dokumen yang memakai kata berbeda tetapi bermakna dekat.
""")
m2.markdown("""
**3. Semantic + Static QE.** Selalu mencoba menambah hingga 2 istilah dari kata kunci 3 dokumen teratas hasil awal, lalu mencari ulang. Tidak mempertimbangkan kualitas hasil awal. Berfungsi sebagai pembanding.

**4. Semantic + Adaptive QE.** Menambah istilah **hanya sebanyak yang dibutuhkan** menurut kualitas hasil awal (rincian di bawah).
""")

st.markdown("### Adaptive Query Expansion: dasar keputusan yang diimplementasikan")
st.markdown(f"""
Skor cosine e5 sangat terkompresi (pada korpus ini skor antar-dokumen rata-rata ≈ 0,82 dengan simpangan baku ≈ 0,02), sehingga ambang skor mutlak
tidak bermakna. Karena itu indikator dihitung **relatif terhadap sebaran skor seluruh korpus untuk kueri tersebut** (z-score).
""")
st.table({
    "Tingkat": ["No Expansion", "Light Expansion", "Strong Expansion"],
    "Kapan dipilih": [f"Hasil awal kuat: z top-1 ≥ {C.top1_z_high} dan z rata-rata top-5 ≥ {C.top5_z_high}, atau satu dokumen unggul ≥ {C.gap_z_clear} SD dari berikutnya; atau kueri spesifik dengan hasil sedang",
                      "Hasil awal sedang (di antara kuat dan lemah); atau hasil lemah pada kueri spesifik",
                      f"Hasil awal lemah: z top-1 < {C.top1_z_low}, atau z rata-rata top-5 < {C.top5_z_low}, atau sepuluh teratas hampir sama skornya (< {C.spread_z_flat} SD)"],
    "Istilah tambahan": ["0", f"maks. {C.light_max_terms}", f"maks. {C.strong_max_terms}"],
})
st.markdown(f"""
**Pengaman:** istilah kandidat berasal dari kata kunci {C.candidate_docs} dokumen teratas dan hanya dipakai bila muncul pada ≥ {C.min_support} dokumen teratas
(kata kunci, judul, atau abstrak); istilah generik, duplikat, atau yang sudah ada di kueri dibuang. Bila kandidat tidak memadai, tingkat **diturunkan** (Strong → Light → No) untuk menghindari
penyimpangan topik. Kueri asli selalu disimpan dan alasan keputusan ditampilkan.

**Batasan jujur:** ini aturan heuristik, bukan pendeteksi ambiguitas yang sempurna. Ambang awal perlu dikalibrasi pada kueri nyata (`scripts/calibrate_adaptive.py`);
skor tinggi tidak menjamin relevan, dan skor rendah bisa berarti topiknya tidak ada di korpus.
""")

st.markdown("""
**Teknologi:** Python, Streamlit, sentence-transformers, rank-bm25, pandas, NumPy, Altair, pytest.
""")
