"""Tema visual & komponen HTML kecil untuk dashboard (CSS + helper). Semua teks dari data di-escape."""
import html

import streamlit as st

NAVY, BLUE, VIOLET, ORANGE, CYAN, SLATE = "#0B1F4B", "#2563EB", "#7C3AED", "#F97316", "#06B6D4", "#64748B"
METHOD_COLORS = {"BM25": SLATE, "Semantic Retrieval": BLUE,
                 "Semantic Retrieval + Static Query Expansion": VIOLET,
                 "Semantic Retrieval + Adaptive Query Expansion": ORANGE}
METHOD_SHORT = {"BM25": "BM25", "Semantic Retrieval": "Semantic",
                "Semantic Retrieval + Static Query Expansion": "Semantic + Static QE",
                "Semantic Retrieval + Adaptive Query Expansion": "Semantic + Adaptive QE"}
LEVEL_COLORS = {"none": "#10B981", "light": "#38BDF8", "strong": "#EC4899"}
LEVEL_NAMES = {"none": "No Expansion", "light": "Light Expansion", "strong": "Strong Expansion"}

_I = {  # ikon gaya Feather (stroke), tanpa emoji
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="16" y2="17"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15 15 0 0 1 0 20a15 15 0 0 1 0-20z"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "chart": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
}


def icon(name: str, size: int = 22, color: str = "currentColor") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_I[name]}</svg>')


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton {{ font-family: 'Inter','Segoe UI',system-ui,sans-serif; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 4.4rem !important; padding-bottom: 3rem; max-width: 1180px; }}
h1,h2,h3,h4 {{ color:{NAVY}; letter-spacing:-0.01em; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg,{NAVY} 0%,#13306f 60%,#2b2a86 100%); }}
[data-testid="stSidebar"] * {{ color:#E8EEFF; }}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {{ border-radius:10px; margin:2px 6px; }}
[data-testid="stSidebarNavLink"]:hover {{ background: rgba(255,255,255,.10); }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: linear-gradient(90deg,{BLUE},{VIOLET}); }}
[data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stCheckbox label {{ color:#E8EEFF !important; }}
.brand {{ display:flex; gap:10px; align-items:center; padding:6px 4px 10px 4px; }}
.brand .logo {{ width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,{BLUE},{VIOLET});
  display:flex;align-items:center;justify-content:center;color:#fff; }}
.brand b {{ color:#fff; font-size:1.02rem; line-height:1.1; }} .brand span {{ color:#B9C6F5; font-size:.74rem; }}
.hero {{ background: linear-gradient(120deg,{NAVY} 0%,{BLUE} 55%,{VIOLET} 100%); color:#fff; border-radius:20px;
  padding:34px 38px; margin-bottom:22px; box-shadow:0 14px 34px rgba(37,52,140,.28); position:relative; overflow:hidden; }}
.hero:after {{ content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
  background:radial-gradient(circle,rgba(255,255,255,.22),rgba(255,255,255,0)); }}
.hero h1 {{ color:#fff; font-size:1.95rem; margin:0 0 8px 0; font-weight:800; }}
.hero p {{ color:#DCE6FF; font-size:1.03rem; margin:0; max-width:760px; line-height:1.55; }}
.hero .tag {{ display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);
  padding:3px 12px; border-radius:99px; font-size:.78rem; margin-bottom:12px; }}
.page-title {{ display:flex; align-items:center; gap:12px; margin:0 0 4px 0; }}
.page-title .ic {{ width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,{BLUE},{VIOLET});color:#fff; }}
.page-title h2 {{ margin:0; font-size:1.55rem; font-weight:800; }}
.sub {{ color:#475569; margin:0 0 18px 0; }}
.stat {{ background:#fff; border-radius:16px; padding:18px 20px; box-shadow:0 2px 14px rgba(15,23,42,.07);
  border:1px solid #E7EBF5; display:flex; gap:14px; align-items:center; height:100%; }}
.stat .ic {{ width:48px;height:48px;border-radius:14px;display:flex;align-items:center;justify-content:center;color:#fff;flex:none; }}
.stat .v {{ font-size:1.65rem; font-weight:800; color:{NAVY}; line-height:1.1; }}
.stat .l {{ color:#64748B; font-size:.84rem; margin-top:2px; }}
.card {{ background:#fff; border-radius:16px; padding:18px 22px; box-shadow:0 2px 14px rgba(15,23,42,.06);
  border:1px solid #E7EBF5; margin-bottom:14px; }}
.mcard {{ background:#fff; border-radius:16px; border:1px solid #E7EBF5; box-shadow:0 2px 14px rgba(15,23,42,.06);
  overflow:hidden; height:100%; }}
.mcard .top {{ padding:12px 18px; color:#fff; font-weight:700; }}
.mcard .body {{ min-height:132px; }}
.mcard .st {{ display:inline-block; margin-top:10px; }}
.stAppDeployButton {{ display:none; }}
.mcard .body {{ padding:14px 18px 16px 18px; color:#334155; font-size:.92rem; line-height:1.5; }}
.mcard .st {{ font-size:.76rem; font-weight:600; background:#EEF2FF; color:#3730A3; padding:2px 10px; border-radius:99px; }}
.chip {{ display:inline-block; padding:2px 10px; border-radius:99px; font-size:.76rem; font-weight:600; margin:2px 6px 2px 0;
  background:#EEF2FF; color:#3730A3; border:1px solid #E0E7FF; }}
.chip.j {{ background:#EFF6FF; color:#1D4ED8; border-color:#DBEAFE; }}
.chip.y {{ background:#F1F5F9; color:#475569; border-color:#E2E8F0; }}
.chip.k {{ background:#F5F3FF; color:#6D28D9; border-color:#EDE9FE; font-weight:500; }}
.chip.term {{ background:#FFF7ED; color:#C2410C; border-color:#FED7AA; font-size:.86rem; }}
.badge {{ display:inline-block; padding:3px 12px; border-radius:99px; font-weight:700; font-size:.8rem; color:#fff; }}
.res {{ background:#fff; border:1px solid #E7EBF5; border-left:5px solid {BLUE}; border-radius:14px; padding:16px 20px;
  margin-bottom:12px; box-shadow:0 2px 12px rgba(15,23,42,.05); }}
.res:hover {{ box-shadow:0 6px 22px rgba(37,99,235,.15); }}
.res .rk {{ display:inline-flex; width:30px;height:30px;border-radius:9px;align-items:center;justify-content:center;
  background:linear-gradient(135deg,{BLUE},{VIOLET}); color:#fff; font-weight:700; font-size:.88rem; margin-right:10px; }}
.res .t {{ font-weight:700; color:{NAVY}; font-size:1.04rem; }}
.res .ab {{ color:#334155; font-size:.93rem; line-height:1.55; margin:8px 0 6px 0; }}
.res a {{ color:{BLUE}; font-weight:600; text-decoration:none; font-size:.88rem; }}
.sc {{ display:flex; align-items:center; gap:8px; margin:6px 0 4px 0; font-size:.78rem; color:#64748B; }}
.sc .bar {{ flex:0 0 150px; height:7px; background:#E8ECF7; border-radius:99px; overflow:hidden; }}
.sc .bar i {{ display:block; height:100%; background:linear-gradient(90deg,{BLUE},{VIOLET}); border-radius:99px; }}
.qbox {{ background:linear-gradient(135deg,#EEF2FF,#F5F3FF); border:1px solid #E0E7FF; border-radius:16px; padding:16px 20px; margin:10px 0 16px 0; }}
.qbox .lab {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:#6366F1; font-weight:700; }}
.qbox .q {{ font-size:1.08rem; font-weight:600; color:{NAVY}; margin-bottom:6px; }}
.note {{ border-radius:12px; padding:12px 16px; font-size:.92rem; margin:8px 0 14px 0; }}
.note.warn {{ background:#FFFBEB; border:1px solid #FDE68A; color:#92400E; }}
.note.info {{ background:#EFF6FF; border:1px solid #BFDBFE; color:#1E40AF; }}
.note.ok {{ background:#ECFDF5; border:1px solid #A7F3D0; color:#065F46; }}
.flow {{ display:flex; flex-wrap:wrap; gap:10px; align-items:stretch; margin:8px 0 14px 0; }}
.flow .s {{ flex:1 1 150px; background:#fff; border:1px solid #E7EBF5; border-radius:14px; padding:12px 14px; font-size:.88rem; position:relative; }}
.flow .s b {{ display:block; color:{NAVY}; margin-bottom:3px; }}
.flow .n {{ display:inline-flex;width:24px;height:24px;border-radius:50%;background:{BLUE};color:#fff;align-items:center;justify-content:center;
  font-size:.78rem;font-weight:700;margin-bottom:6px; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background: linear-gradient(90deg,{BLUE},{VIOLET}); border:0; color:#fff; font-weight:700; border-radius:12px; padding:.55rem 1.4rem; }}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{ filter:brightness(1.08); }}
.stButton > button[kind="secondary"], .stDownloadButton > button {{ border-radius:12px; border:1.5px solid #C7D2FE; color:#3730A3; background:#fff; font-weight:600; }}
.stTextInput input {{ border-radius:12px; font-size:1.02rem; padding:.7rem .9rem; }}
[data-testid="stForm"] {{ border:1px solid #E0E7FF; border-radius:16px; background:#fff; padding:18px 20px; box-shadow:0 2px 14px rgba(15,23,42,.05); }}
button[data-baseweb="tab"] {{ font-weight:600; }}
@media (max-width: 720px) {{ .hero {{ padding:22px 20px; }} .hero h1 {{ font-size:1.45rem; }} .sc .bar {{ flex-basis:90px; }} }}
</style>
"""


def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(f'<div class="brand"><div class="logo">{icon("search", 20, "#fff")}</div>'
                        '<div><b>Semantic Retrieval</b><br><span>Dokumen akademik</span></div></div>', unsafe_allow_html=True)


def page_header(ic: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-title"><div class="ic">{icon(ic, 22, "#fff")}</div><h2>{esc(title)}</h2></div>'
                f'<p class="sub">{esc(subtitle)}</p>', unsafe_allow_html=True)


def stat_card(ic: str, value, label: str, color: str = BLUE) -> str:
    return (f'<div class="stat"><div class="ic" style="background:linear-gradient(135deg,{color},{NAVY})">{icon(ic, 24, "#fff")}</div>'
            f'<div><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div></div>')


def note(kind: str, text_html: str) -> None:
    """kind: warn | info | ok. `text_html` harus sudah aman (gunakan esc untuk data)."""
    st.markdown(f'<div class="note {kind}">{text_html}</div>', unsafe_allow_html=True)


def level_badge(level: str) -> str:
    return f'<span class="badge" style="background:{LEVEL_COLORS.get(level, SLATE)}">{esc(LEVEL_NAMES.get(level, level))}</span>'
