"""Pemetaan kolom, pembersihan data, doc_id stabil, dan pembuatan document_text.

Skema akhir (STANDARD_COLUMNS) memisahkan nama jurnal (`journal`) dari tipe dokumen
(`document_type`), menyimpan `doi`, dan menandai asal data (`data_source`:
real / synthetic / unspecified) agar data sintetis tidak tercampur dengan data nyata.
"""
import hashlib
import html
import re
from pathlib import Path

import pandas as pd

STANDARD_COLUMNS = [
    "doc_id", "title", "abstract", "keywords", "author", "year", "journal", "doi",
    "language", "document_type", "source_url", "data_source", "source_note",
]
DATA_SOURCES = ("real", "synthetic", "unspecified")

# Alias nama kolom (huruf kecil) -> nama internal
COLUMN_ALIASES = {
    "title": ["title", "judul"],
    "abstract": ["abstract", "abstrak"],
    "authors": ["authors", "author", "penulis"],
    "year": ["year", "tahun", "date", "publication year"],
    "doi": ["doi"],
    "url": ["url", "link", "articleurl", "article url"],
    "keywords": ["keywords", "keyword", "kata kunci"],
    "journal": ["journal", "publication", "source", "nama jurnal", "jurnal"],
    "type": ["type", "document type", "document_type"],
    "language": ["language", "bahasa"],
    "data_source": ["data_source", "sumber data"],
    "source_note": ["source_note", "catatan sumber"],
}

_LANG_MAP = {"id": "id", "ind": "id", "indonesia": "id", "indonesian": "id", "bahasa indonesia": "id",
             "en": "en", "eng": "en", "english": "en", "inggris": "en"}
_ID_HINTS = {"yang", "dan", "dengan", "untuk", "pada", "dalam", "adalah", "ini", "dari",
             "penelitian", "menggunakan", "sistem", "hasil", "oleh"}
_EN_HINTS = {"the", "and", "of", "for", "with", "this", "is", "in", "using", "study",
             "based", "results", "paper"}


def clean_text(value) -> str:
    """Ubah nilai apa pun menjadi string bersih; NaN/None -> ''."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


_JATS_ABSTRACT_TITLE = re.compile(r"<(?:jats:)?title[^>]*>\s*(?:abstract|abstrak|summary)\s*</(?:jats:)?title>", re.I)
_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9:]*)[^<>]*>")
_BLOCK_TAGS = {"p", "br", "div", "li", "ul", "ol", "title", "sec", "h1", "h2", "h3", "h4",
               "jats:p", "jats:title", "jats:sec", "jats:list", "jats:list-item"}


def strip_markup(value) -> str:
    """Bersihkan sisa markup dari metadata (Crossref/OAI sering membawa <p>, <jats:p>, &amp;).

    Urutan: buang judul 'Abstract' bergaya JATS, ubah penutup blok menjadi spasi, buang tag (tag inline dihapus tanpa spasi),
    decode entitas HTML (dua kali, karena ada yang ter-escape ganda seperti '&amp;amp;'), rapikan spasi.
    Teks biasa tanpa tag/entitas tidak berubah selain spasi berlebih.
    """
    text = clean_text(value)
    if not text:
        return ""
    text = _JATS_ABSTRACT_TITLE.sub(" ", text)
    for _ in range(2):
        if "&" not in text:
            break
        text = html.unescape(text)
    text = _TAG.sub(lambda m: " " if m.group(1).lower() in _BLOCK_TAGS else "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(title: str) -> str:
    """Normalisasi judul untuk deteksi duplikasi."""
    t = clean_text(title).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_doi(doi: str) -> str:
    d = clean_text(doi).lower()
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d)
    return d.strip()


def parse_year(value):
    """Ambil tahun 4 digit (1900-2099) dari nilai apa pun ('2021', '2021.0', '2021-05-03'). NA bila tidak ada."""
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", clean_text(value))
    return int(m.group(1)) if m else pd.NA


def format_year(value) -> str:
    """Tahun sebagai teks aman untuk tampilan ('' bila kosong, tidak pernah '2021.0')."""
    y = parse_year(value)
    return "" if y is pd.NA else str(y)


def detect_language(text: str) -> str:
    """Heuristik sederhana id/en/unknown berdasarkan kata fungsi. BUKAN pendeteksi bahasa penuh."""
    tokens = re.findall(r"\w+", clean_text(text).lower())
    n_id = sum(t in _ID_HINTS for t in tokens)
    n_en = sum(t in _EN_HINTS for t in tokens)
    if n_id == n_en:
        return "unknown"
    return "id" if n_id > n_en else "en"


def make_doc_id(doi: str, title: str) -> str:
    """doc_id stabil: dari DOI bila ada, selain itu dari judul ternormalisasi.
    Tidak bergantung pada urutan baris, sehingga label relevansi tetap valid
    walaupun dataset dibuat ulang."""
    key = f"doi:{normalize_doi(doi)}" if normalize_doi(doi) else f"title:{normalize_title(title)}"
    return "DOC-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def detect_columns(df: pd.DataFrame) -> dict:
    """Deteksi kolom input -> nama internal. Tidak sensitif huruf besar/kecil."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    mapping = {}
    for internal, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                mapping[internal] = lowered[alias]
                break
    return mapping


_MARKUP_FIELDS = {"title", "abstract", "authors", "keywords", "journal", "type"}


def _col(df, mapping, key):
    if key in mapping:
        return df[mapping[key]].map(strip_markup if key in _MARKUP_FIELDS else clean_text)
    return pd.Series([""] * len(df), index=df.index)


def _normalize_data_source(value: str, default: str) -> str:
    v = clean_text(value).lower()
    return v if v in DATA_SOURCES else default


def map_columns(df_raw: pd.DataFrame, default_data_source: str = "unspecified") -> pd.DataFrame:
    """Petakan kolom mentah ke kolom standar (belum ada doc_id/document_text)."""
    if default_data_source not in DATA_SOURCES:
        raise ValueError(f"default_data_source harus salah satu dari {DATA_SOURCES}")
    mapping = detect_columns(df_raw)
    missing = [k for k in ("title", "abstract") if k not in mapping]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {', '.join(missing)}. "
            f"Kolom yang ada di file: {list(df_raw.columns)}"
        )
    title = _col(df_raw, mapping, "title")
    abstract = _col(df_raw, mapping, "abstract")
    doi = _col(df_raw, mapping, "doi").map(normalize_doi)
    url = _col(df_raw, mapping, "url")

    lang_given = _col(df_raw, mapping, "language").str.lower().map(lambda v: _LANG_MAP.get(v, ""))
    lang_guess = (title + " " + abstract).map(detect_language)
    language = lang_given.where(lang_given != "", lang_guess)

    type_ = _col(df_raw, mapping, "type")
    return pd.DataFrame({
        "title": title,
        "abstract": abstract,
        "keywords": _col(df_raw, mapping, "keywords"),
        "author": _col(df_raw, mapping, "authors"),
        "year": pd.Series([parse_year(v) for v in _col(df_raw, mapping, "year")],
                          index=df_raw.index, dtype="Int64"),
        "journal": _col(df_raw, mapping, "journal"),
        "doi": doi,
        "language": language,
        "document_type": type_.where(type_ != "", "article"),
        "source_url": url.where(url != "", doi.map(lambda d: f"https://doi.org/{d}" if d else "")),
        "data_source": _col(df_raw, mapping, "data_source").map(
            lambda v: _normalize_data_source(v, default_data_source)),
        "source_note": _col(df_raw, mapping, "source_note"),
    })


def build_document_text(row) -> str:
    return (
        f"Judul: {row['title']}. Abstrak: {row['abstract']}. "
        f"Kata kunci: {row['keywords']}."
    )


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Buang duplikat. Aturan:
    1) DOI sama -> duplikat.
    2) Judul ternormalisasi sama -> duplikat bila salah satu tidak punya DOI
       (artikel yang sama dari dua sumber, satu tanpa DOI). Dua DOI berbeda dengan judul sama
       dianggap artikel berbeda (mis. versi berbeda) dan dipertahankan.
    Baris ber-DOI diprioritaskan sebagai wakil; urutan asli dijaga."""
    work = df.copy()
    work["_title_norm"] = work["title"].map(normalize_title)
    work["_order"] = range(len(work))
    work = work.sort_values(by="doi", key=lambda s: (s == ""), kind="stable")  # ber-DOI dulu

    seen_doi, seen_title, keep = set(), {}, []
    for idx, row in work.iterrows():
        doi, tn = row["doi"], row["_title_norm"]
        if doi and doi in seen_doi:
            continue
        if tn in seen_title and (not doi or not seen_title[tn]):
            continue
        keep.append(idx)
        if doi:
            seen_doi.add(doi)
        seen_title.setdefault(tn, doi)
    out = work.loc[keep].sort_values("_order")
    return out.drop(columns=["_title_norm", "_order"])


def clean_dataset(df_raw: pd.DataFrame, default_data_source: str = "unspecified") -> pd.DataFrame:
    """Pemetaan + validasi + deduplikasi + doc_id stabil + document_text."""
    df = map_columns(df_raw, default_data_source)
    df = df[(df["title"] != "") & (df["abstract"] != "")].copy()
    df = deduplicate(df).reset_index(drop=True)
    df.insert(0, "doc_id", [make_doc_id(d, t) for d, t in zip(df["doi"], df["title"])])
    if df["doc_id"].duplicated().any():  # sangat tidak mungkin; lebih baik gagal keras
        raise ValueError("Tabrakan doc_id terdeteksi setelah deduplikasi.")
    df["document_text"] = df.apply(build_document_text, axis=1)
    return df[STANDARD_COLUMNS + ["document_text"]]


def process_csv(raw_path, output_path, default_data_source: str = "unspecified") -> pd.DataFrame:
    """Baca CSV mentah, bersihkan, simpan ke output_path."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {raw_path}")
    try:
        df_raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_raw = pd.read_csv(raw_path, encoding="latin-1")
    df = clean_dataset(df_raw, default_data_source)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return df
