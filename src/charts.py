"""Grafik Altair dengan palet konsisten (warna metode & tingkat ekspansi)."""
import altair as alt
import pandas as pd

from src.ui_theme import BLUE, LEVEL_COLORS, METHOD_COLORS, METHOD_SHORT, NAVY, VIOLET

CHART_LABEL = {"BM25": "BM25", "Semantic Retrieval": "Semantic", "Semantic Retrieval + Static Query Expansion": "Static QE",
               "Semantic Retrieval + Adaptive Query Expansion": "Adaptive QE"}
CHART_ORDER = list(CHART_LABEL.values())
_AX = dict(labelColor="#475569", titleColor="#334155", gridColor="#E8ECF7", domainColor="#CBD5E1")


def _style(ch):
    return ch.configure_view(strokeWidth=0).configure_axis(**_AX).configure_legend(labelColor="#334155", titleColor="#334155")


def bar_count(df: pd.DataFrame, cat: str, val: str, title: str = "", color: str = BLUE, horizontal: bool = True, height: int = 220):
    enc = dict(color=alt.value(color), tooltip=[cat, val])
    if horizontal:
        ch = alt.Chart(df).mark_bar(cornerRadiusEnd=6, size=26).encode(y=alt.Y(f"{cat}:N", sort="-x", title=None), x=alt.X(f"{val}:Q", title=title), **enc)
    else:
        ch = alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(x=alt.X(f"{cat}:O", title=None), y=alt.Y(f"{val}:Q", title=title), **enc)
    return _style(ch.properties(height=height))


def donut(df: pd.DataFrame, cat: str, val: str, colors, height: int = 220):
    ch = alt.Chart(df).mark_arc(innerRadius=58, outerRadius=92, cornerRadius=4).encode(
        theta=alt.Theta(f"{val}:Q"), color=alt.Color(f"{cat}:N", scale=alt.Scale(domain=list(df[cat]), range=colors), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[cat, val])
    return _style(ch.properties(height=height))


def stacked_bar(df: pd.DataFrame, cat: str, series: str, val: str, colors, height: int = 220):
    """colors: dict {nilai_series: warna} agar warna konsisten dengan grafik lain."""
    ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4, size=26).encode(
        y=alt.Y(f"{cat}:N", title=None, sort="-x"), x=alt.X(f"{val}:Q", title="Jumlah dokumen"),
        color=alt.Color(f"{series}:N", scale=alt.Scale(domain=[k for k in colors if k in set(df[series])], range=[v for k, v in colors.items() if k in set(df[series])]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[cat, series, val])
    return _style(ch.properties(height=height))


def year_area(df: pd.DataFrame, height: int = 220):
    ch = alt.Chart(df).mark_area(line={"color": VIOLET}, color=alt.Gradient(gradient="linear", stops=[
        alt.GradientStop(color="#C4B5FD", offset=0), alt.GradientStop(color="#EEF2FF", offset=1)], x1=1, x2=1, y1=1, y2=0), interpolate="monotone").encode(
        x=alt.X("year:O", title=None), y=alt.Y("n:Q", title="Dokumen"), tooltip=["year", "n"])
    return _style(ch.properties(height=height))


def metric_bars(df: pd.DataFrame, metric: str, title: str, height: int = 230):
    """df: kolom method + metric. Warna per metode (palet konsisten)."""
    d = df.assign(label=df["method"].map(CHART_LABEL))
    cmap = alt.Scale(domain=CHART_ORDER, range=list(METHOD_COLORS.values()))
    x = alt.X("label:N", sort=CHART_ORDER, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=140))
    ch = alt.Chart(d).mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7).encode(
        x=x, y=alt.Y(f"{metric}:Q", title=title, scale=alt.Scale(domain=[0, max(1.0, float(d[metric].max()) * 1.05)])),
        color=alt.Color("label:N", scale=cmap, legend=None), tooltip=["label", alt.Tooltip(f"{metric}:Q", format=".3f")])
    txt = alt.Chart(d).mark_text(dy=-7, color=NAVY, fontWeight="bold").encode(
        x=alt.X("label:N", sort=CHART_ORDER), y=f"{metric}:Q", text=alt.Text(f"{metric}:Q", format=".3f"))
    return _style((ch + txt).properties(height=height, title=title))


def grouped_metric(df: pd.DataFrame, group: str, metric: str, title: str, height: int = 230):
    """Batang per metode, dikelompokkan per `group` lewat facet (lebar tetap; tidak bergantung lebar kontainer/tab)."""
    d = df.assign(label=df["method"].map(CHART_LABEL))
    tips = [group, "label", alt.Tooltip(f"{metric}:Q", format=".3f")] + (["n_queries"] if "n_queries" in d else [])
    ch = alt.Chart(d).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("label:N", sort=CHART_ORDER, title=None, axis=None),
        y=alt.Y(f"{metric}:Q", title=title.split(" menurut")[0], scale=alt.Scale(domain=[0, max(1.0, float(d[metric].max()) * 1.1)])),
        color=alt.Color("label:N", scale=alt.Scale(domain=CHART_ORDER, range=list(METHOD_COLORS.values())), legend=alt.Legend(orient="bottom", title=None)),
        column=alt.Column(f"{group}:N", title=None, header=alt.Header(labelFontWeight="bold", labelColor=NAVY, labelFontSize=13)),
        tooltip=tips).properties(width=110, height=height)
    return _style(ch.properties(title=title))


def level_bar(df: pd.DataFrame, height: int = 200):
    names = {"none": "No Expansion", "light": "Light Expansion", "strong": "Strong Expansion"}
    d = df.assign(label=df["expansion_level"].map(names))
    ch = alt.Chart(d).mark_bar(cornerRadiusEnd=6, size=28).encode(
        y=alt.Y("label:N", sort=list(names.values()), title=None), x=alt.X("n_queries:Q", title="Jumlah kueri"),
        color=alt.Color("expansion_level:N", scale=alt.Scale(domain=list(LEVEL_COLORS), range=list(LEVEL_COLORS.values())), legend=None),
        tooltip=["label", "n_queries"])
    return _style(ch.properties(height=height))
