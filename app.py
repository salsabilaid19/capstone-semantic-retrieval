"""Semantic Retrieval Dokumen Akademik: aplikasi multi-halaman.   Jalankan:  streamlit run app.py
Halaman: Dashboard, Document Search, About & Methodology (folder views/).
Data: ARTIFACTS_DIR bila diatur; selain itu artifacts_real/ (dataset penelitian) bila ada; selain itu artifacts/ (sintetis)."""
from pathlib import Path

import streamlit as st

from src import ui_theme as T

ASSETS = Path(__file__).resolve().parent / "assets"   # absolut: aman dijalankan dari folder kerja mana pun

st.set_page_config(page_title="Semantic Retrieval Dokumen Akademik", page_icon=":material/travel_explore:", layout="wide")
T.inject_theme()
if (ASSETS / "logo.svg").exists():
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo_icon.svg"), size="large")

pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("views/search.py", title="Document Search", icon=":material/search:"),
    st.Page("views/about.py", title="About & Methodology", icon=":material/info:"),
]
st.navigation(pages).run()
st.sidebar.caption("Capstone: Semantic Retrieval + Adaptive Query Expansion")
