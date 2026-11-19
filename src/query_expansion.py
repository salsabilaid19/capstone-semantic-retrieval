"""Query expansion berbasis aturan: Static QE dan Adaptive QE (tiga tingkat).

Static QE   : selalu mencoba ekspansi (maks. 2 istilah dari kata kunci top-3 hasil semantic awal).
Adaptive QE : memilih TINGKAT ekspansi dari KUALITAS hasil pencarian semantic awal:
              No Expansion / Light Expansion / Strong Expansion.

Dasar keputusan Adaptive QE yang benar-benar diimplementasikan (lihat `AdaptiveConfig`, `decide_level`):
  1. Kekuatan hasil teratas  : z-score skor top-1 terhadap distribusi skor SELURUH korpus untuk kueri itu.
                               (Skor cosine e5 sangat terkompresi, mis. rata-rata ~0,82 dengan simpangan baku ~0,02,
                               sehingga ambang skor mutlak tidak bermakna; z-score bersifat relatif terhadap korpus.)
  2. Sebaran hasil teratas   : rata-rata z-score top-5, jarak skor top-1 ke top-10, dan selisih top-1 dengan top-2.
  3. Kandidat istilah        : jumlah dan "dukungan" tiap kandidat (berapa dokumen top-5 yang memuat istilah itu pada
                               kata kunci/judul/abstraknya). Kandidat tanpa dukungan lintas-dokumen tidak dipakai;
                               bila kandidat tidak memadai, tingkat ekspansi diturunkan.
  4. Spesifisitas kueri      : kueri panjang (>= 6 kata) diturunkan satu tingkat (sudah memuat banyak konteks).

BATASAN yang jujur: ini heuristik berbasis aturan dengan ambang awal yang HARUS dikalibrasi ulang pada kueri nyata
(`scripts/calibrate_adaptive.py`). Sistem tidak mendeteksi ambiguitas secara sempurna: skor tinggi tidak menjamin
dokumen relevan, dan skor rendah tidak menjamin kueri ambigu (bisa juga topiknya tidak ada di korpus).
Kemanfaatan ekspansi hanya dapat dinilai lewat evaluasi dengan penilaian relevansi manusia (src/evaluation.py).
"""
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

KEYWORD_SEPARATORS = r"[,;|]"
MAX_EXPANSION_TERMS = 2      # Static QE: jumlah istilah tambahan maksimal
TOP_N_DOCS_FOR_CANDIDATES = 3  # Static QE: kandidat diambil dari top-3 hasil awal
SHORT_QUERY_MAX_WORDS = 3    # kueri 1-3 kata dianggap pendek (segmen evaluasi & fallback berbasis panjang)

LEVEL_NONE, LEVEL_LIGHT, LEVEL_STRONG = "none", "light", "strong"
LEVEL_LABELS = {LEVEL_NONE: "No Expansion", LEVEL_LIGHT: "Light Expansion", LEVEL_STRONG: "Strong Expansion"}
LEVEL_ORDER = [LEVEL_NONE, LEVEL_LIGHT, LEVEL_STRONG]
MIN_DOCS_FOR_DIAGNOSTICS = 20   # z-score pada korpus lebih kecil dari ini tidak bermakna


@dataclass(frozen=True)
class AdaptiveConfig:
    """Ambang keputusan Adaptive QE. NILAI AWAL (heuristik): kalibrasi ulang pada kueri nyata sebelum dilaporkan.

    Nilai awal dipilih dari sebaran indikator pada embedding nyata 900 dokumen dengan embedding dokumen sebagai
    kueri-pengganti (leave-one-out); itu hanya untuk memastikan skala masuk akal, BUKAN kalibrasi kinerja."""
    top1_z_high: float = 4.2      # z top-1 >= ini (bersama z top-5 tinggi) -> hasil awal kuat
    top5_z_high: float = 3.3
    gap_z_clear: float = 1.5      # top-1 unggul >= 1,5 SD dari top-2 -> satu dokumen jelas menonjol
    top1_z_low: float = 3.4       # z top-1 < ini -> hasil awal lemah
    top5_z_low: float = 2.9
    spread_z_flat: float = 0.6    # (top-1 - top-10) < 0,6 SD dan top-1 tidak kuat -> tak ada dokumen menonjol
    specific_query_words: int = 6   # kueri >= ini dianggap spesifik -> turun satu tingkat
    light_max_terms: int = 1
    strong_max_terms: int = 3
    candidate_docs: int = 5       # kandidat & dukungan dihitung pada top-5 hasil awal
    min_support: int = 2          # kandidat harus termuat pada >= 2 dokumen teratas
    max_term_words: int = 4
    max_term_chars: int = 48


DEFAULT_CONFIG = AdaptiveConfig()


def count_query_words(query: str) -> int:
    """Hitung jumlah kata pada kueri (dipisahkan spasi)."""
    if not isinstance(query, str):
        return 0
    return len(query.strip().split())


def split_keywords(keywords: str) -> List[str]:
    """Pisahkan keywords dengan koma, titik koma, atau simbol pipe."""
    if not isinstance(keywords, str):
        return []
    return [k.strip() for k in re.split(KEYWORD_SEPARATORS, keywords)]


def _contains_term(query_lower: str, term_lower: str) -> bool:
    """Cek apakah istilah sudah muncul di kueri (cocok sebagai kata utuh)."""
    pattern = r"(?<!\w)" + re.escape(term_lower) + r"(?!\w)"
    return re.search(pattern, query_lower) is not None


def clean_candidates(candidates: List[str], query: str,
                     max_terms: int = MAX_EXPANSION_TERMS) -> List[str]:
    """Bersihkan kandidat: hapus kosong, duplikasi, istilah yang sudah ada di kueri,
    lalu batasi maksimal `max_terms` istilah (urutan kemunculan dijaga)."""
    query_lower = re.sub(r"\s+", " ", (query or "").lower()).strip()
    seen = set()
    result: List[str] = []
    for cand in candidates:
        term = re.sub(r"\s+", " ", str(cand)).strip()
        key = term.lower()
        if not term or key in seen or _contains_term(query_lower, key):
            continue
        seen.add(key)
        result.append(term)
        if len(result) >= max_terms:
            break
    return result


def get_expansion_candidates(initial_results: pd.DataFrame, query: str,
                             top_n_docs: int = TOP_N_DOCS_FOR_CANDIDATES) -> List[str]:
    """Ambil kandidat istilah dari kolom keywords pada top-N hasil semantic awal."""
    if initial_results is None or len(initial_results) == 0 or "keywords" not in initial_results:
        return []
    raw: List[str] = []
    for kw in initial_results.head(top_n_docs)["keywords"].fillna(""):
        raw.extend(split_keywords(kw))
    return clean_candidates(raw, query)


def _build_result(query: str, terms: List[str], reason: str) -> Dict:
    """Susun dict hasil keputusan expansion."""
    original = (query or "").strip()
    final = f"{original} {' '.join(terms)}".strip() if terms else original
    return {
        "original_query": original,
        "final_query": final,
        "is_expanded": bool(terms),
        "expansion_terms": terms,
        "decision_reason": reason,
    }


# ---------------------------------------------------------------- Static QE
def static_expand_query(query: str, initial_results: pd.DataFrame) -> Dict:
    """Static QE: selalu mencoba expansion untuk semua kueri."""
    terms = get_expansion_candidates(initial_results, query)
    if not terms:
        out = _build_result(
            query, [],
            "Tidak ditemukan kandidat istilah valid pada kata kunci top-3 hasil awal "
            "sehingga kueri asli digunakan.",
        )
    else:
        out = _build_result(
            query, terms,
            f"Static QE selalu mencoba ekspansi; {len(terms)} istilah diambil dari kata kunci "
            "top-3 hasil pencarian awal.",
        )
    out["expansion_level"] = "static" if terms else LEVEL_NONE
    return out


# ---------------------------------------------------------------- Indikator kualitas hasil awal
def score_diagnostics(scores) -> Dict:
    """Indikator kualitas hasil awal dari VEKTOR SKOR SELURUH KORPUS untuk satu kueri.

    z-score memakai rata-rata & simpangan baku skor seluruh dokumen. `reliable` False bila korpus terlalu kecil
    (< MIN_DOCS_FOR_DIAGNOSTICS) atau skor tak berubah (std = 0): pemanggil harus memakai fallback."""
    s = np.asarray(scores, dtype=float).ravel()
    n = int(s.size)
    out: Dict = {"n_docs": n, "reliable": False}
    if n < MIN_DOCS_FOR_DIAGNOSTICS or not np.isfinite(s).all():
        out["unreliable_reason"] = (f"korpus hanya {n} dokumen (< {MIN_DOCS_FOR_DIAGNOSTICS}) sehingga z-score tidak bermakna"
                                    if n < MIN_DOCS_FOR_DIAGNOSTICS else "skor mengandung nilai tak hingga/NaN")
        return out
    mean, std = float(s.mean()), float(s.std())
    if std < 1e-9:
        out["unreliable_reason"] = "seluruh skor hampir identik (simpangan baku ~0)"
        return out
    top = np.sort(s)[::-1][:10]
    out.update({
        "reliable": True,
        "mean_score": mean, "std_score": std,
        "top1_score": float(top[0]), "top2_score": float(top[1]), "top10_score": float(top[-1]),
        "top1_z": float((top[0] - mean) / std),
        "top5_mean_z": float((top[:5].mean() - mean) / std),
        "gap_z": float((top[0] - top[1]) / std),
        "spread_z": float((top[0] - top[-1]) / std),
    })
    return out


# ---------------------------------------------------------------- Kandidat istilah beserta dukungannya
_LEADING_CONJ = re.compile(r"^(and|dan|atau|or)\s+", re.IGNORECASE)
_STOP_TERMS = frozenset("dan atau yang di ke dari untuk the of and for in on to a an is are with by".split())
# Istilah terlalu umum (bukan penanda topik) yang sering muncul sebagai kata kunci; tidak dipakai sebagai kandidat
# Adaptive QE. Static QE sengaja TIDAK memakai penyaring ini agar tetap menjadi baseline yang tak berubah.
GENERIC_TERMS = frozenset("""
accuracy akurasi performance kinerja evaluation evaluasi analysis analisis method metode methods implementation
implementasi algorithm algoritma algoritme system sistem model design perancangan application aplikasi study
penelitian research data information informasi technology teknologi comparison perbandingan approach pendekatan
case study studi kasus optimization optimasi prediction prediksi
""".split())


def _normalize_term(raw: str) -> str:
    term = re.sub(r"\s+", " ", str(raw)).strip(" .:;,-\u2013\u2014\"'")
    return _LEADING_CONJ.sub("", term).strip()


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def score_candidates(initial_results: pd.DataFrame, query: str, config: AdaptiveConfig = DEFAULT_CONFIG) -> List[Dict]:
    """Kandidat istilah dari kata kunci top-N hasil awal + dukungan lintas-dokumen.

    support = jumlah dokumen (dari top-N) yang memuat istilah itu pada kata kunci, judul, atau abstrak.
    Dokumen asal kata kunci ikut dihitung (support minimal 1); support >= 2 berarti istilah juga muncul pada
    dokumen teratas lain, tanda istilah itu konsisten dengan konteks hasil awal, bukan kebetulan satu dokumen.
    Istilah dibuang bila: kosong/terlalu panjang, hanya angka/stopword/istilah generik, sudah ada di kueri, atau seluruh katanya
    sudah ada di kueri (tidak menambah apa pun). Urutan: support terbesar, lalu peringkat dokumen asal terbaik."""
    if initial_results is None or len(initial_results) == 0 or "keywords" not in initial_results:
        return []
    top = initial_results.head(config.candidate_docs)
    texts = []
    for _, row in top.iterrows():
        blob = " ".join(str(row.get(c, "") or "") for c in ("title", "keywords", "abstract"))
        texts.append(re.sub(r"\s+", " ", blob.lower()))
    query_tokens = set(_tokens(query))
    query_lower = re.sub(r"\s+", " ", (query or "").lower()).strip()

    found: Dict[str, Dict] = {}
    for doc_rank, kw in enumerate(top["keywords"].fillna(""), start=1):
        for raw in split_keywords(kw):
            term = _normalize_term(raw)
            key = term.lower()
            toks = _tokens(term)
            if (not term or key in found or not toks or key in _STOP_TERMS or key in GENERIC_TERMS
                    or len(term) > config.max_term_chars or len(toks) > config.max_term_words
                    or all(t.isdigit() for t in toks) or (len(toks) == 1 and len(toks[0]) < 3)
                    or _contains_term(query_lower, key) or set(toks) <= query_tokens):
                continue
            found[key] = {"term": term, "first_doc_rank": doc_rank, "order": len(found)}
    cands = []
    for key, c in found.items():
        c["support"] = sum(1 for t in texts if _contains_term(t, key))
        cands.append(c)
    cands.sort(key=lambda c: (-c["support"], c["first_doc_rank"], c["order"]))
    return [{"term": c["term"], "support": c["support"], "first_doc_rank": c["first_doc_rank"]} for c in cands]


# ---------------------------------------------------------------- Keputusan tingkat
def decide_level(diag: Dict, n_words: int, config: AdaptiveConfig = DEFAULT_CONFIG):
    """Tentukan tingkat dasar dari indikator. Mengembalikan (level, aturan_yang_dipakai, uraian_indikator)."""
    z1, z5, gap, spread = diag["top1_z"], diag["top5_mean_z"], diag["gap_z"], diag["spread_z"]
    ind = (f"z skor top-1 = {z1:.2f}, z rata-rata top-5 = {z5:.2f}, "
           f"selisih top-1 vs top-2 = {gap:.2f} SD, jarak top-1 ke top-10 = {spread:.2f} SD")
    clear_winner = z1 >= config.top1_z_high and gap >= config.gap_z_clear
    good = (z1 >= config.top1_z_high and z5 >= config.top5_z_high) or clear_winner
    flat = spread < config.spread_z_flat and z1 < config.top1_z_high
    poor = z1 < config.top1_z_low or z5 < config.top5_z_low or flat
    if good:
        level, rule = LEVEL_NONE, ("hasil awal kuat: " + ("satu dokumen jelas menonjol" if clear_winner and z5 < config.top5_z_high
                                                         else "skor top-1 dan rata-rata top-5 sama-sama tinggi"))
    elif poor:
        why = []
        if z1 < config.top1_z_low:
            why.append(f"skor top-1 rendah (z < {config.top1_z_low})")
        if z5 < config.top5_z_low:
            why.append(f"rata-rata top-5 rendah (z < {config.top5_z_low})")
        if flat and not why:
            why.append("sepuluh hasil teratas hampir sama skornya sehingga tidak ada dokumen yang menonjol")
        level, rule = LEVEL_STRONG, "hasil awal lemah: " + "; ".join(why)
    else:
        level, rule = LEVEL_LIGHT, "hasil awal sedang (tidak cukup kuat untuk dibiarkan, tidak cukup lemah untuk ekspansi luas)"
    if n_words >= config.specific_query_words and level != LEVEL_NONE:
        level = LEVEL_LIGHT if level == LEVEL_STRONG else LEVEL_NONE
        rule += f"; kueri cukup spesifik ({n_words} kata) sehingga tingkat diturunkan satu tingkat"
    return level, rule, ind


def _fmt_terms(cands: List[Dict], n: int) -> str:
    return ", ".join(f"{c['term']} (muncul di {c['support']} dari {n} dokumen teratas)" for c in cands)


def _length_only_fallback(query: str, initial_results: pd.DataFrame, why: str) -> Dict:
    """Aturan lama berbasis jumlah kata, dipakai HANYA bila skor seluruh korpus tak tersedia/tak andal."""
    normalized = re.sub(r"\s+", " ", (query or "")).strip()
    n_words = count_query_words(normalized)
    if n_words == 0:
        result = _build_result(normalized, [], "Kueri kosong sehingga tidak ada ekspansi.")
        level = LEVEL_NONE
    elif n_words > SHORT_QUERY_MAX_WORDS:
        result = _build_result(
            normalized, [],
            "Kueri terdiri dari lebih dari 3 kata sehingga dianggap cukup spesifik dan tidak diekspansi.",
        )
        level = LEVEL_NONE
    else:
        terms = get_expansion_candidates(initial_results, normalized)
        if terms:
            result = _build_result(
                normalized, terms,
                f"Kueri pendek ({n_words} kata) sehingga diekspansi dengan {len(terms)} istilah "
                "dari kata kunci top-3 hasil awal.",
            )
            level = LEVEL_LIGHT
        else:
            result = _build_result(
                normalized, [],
                f"Kueri pendek ({n_words} kata) tetapi tidak ada kandidat istilah valid "
                "sehingga kueri asli digunakan.",
            )
            level = LEVEL_NONE
    result.update({"word_count": n_words, "expansion_level": level, "expansion_label": LEVEL_LABELS[level],
                   "decision_basis": "length_only", "fallback_note": f"Indikator kualitas tidak dipakai: {why}.",
                   "indicators": {}, "candidates": []})
    return result


def adaptive_expand_query(query: str, initial_results: pd.DataFrame, diagnostics: Optional[Dict] = None,
                          config: AdaptiveConfig = DEFAULT_CONFIG) -> Dict:
    """Adaptive QE tiga tingkat. `diagnostics` = keluaran `score_diagnostics` (skor seluruh korpus).

    Tanpa `diagnostics` yang andal, jatuh ke aturan panjang kueri (1-3 kata) dan hal itu dinyatakan pada hasil
    (`decision_basis` = "length_only" dan `fallback_note`). Query asli selalu dipertahankan di `original_query`."""
    normalized = re.sub(r"\s+", " ", (query or "")).strip()
    n_words = count_query_words(normalized)
    if n_words == 0:
        return _length_only_fallback(normalized, initial_results, "kueri kosong")
    if not diagnostics or not diagnostics.get("reliable"):
        why = (diagnostics or {}).get("unreliable_reason", "skor seluruh korpus tidak diberikan")
        return _length_only_fallback(normalized, initial_results, why)

    level, rule, ind_text = decide_level(diagnostics, n_words, config)
    cands = score_candidates(initial_results, normalized, config)
    n_top = min(config.candidate_docs, len(initial_results))
    supported = [c for c in cands if c["support"] >= config.min_support]
    notes: List[str] = []

    if level == LEVEL_STRONG and len(supported) < 2:
        new = LEVEL_LIGHT if supported else LEVEL_NONE
        notes.append(f"Strong Expansion dibutuhkan, tetapi hanya {len(supported)} kandidat yang muncul pada "
                     f">= {config.min_support} dokumen teratas; diturunkan ke {LEVEL_LABELS[new]}.")
        level = new
    if level == LEVEL_LIGHT and not supported:
        notes.append(f"Light Expansion dibutuhkan, tetapi tidak ada kandidat istilah yang muncul pada "
                     f">= {config.min_support} dokumen teratas (risiko menyimpang dari topik); kueri asli dipertahankan.")
        level = LEVEL_NONE

    max_terms = {LEVEL_NONE: 0, LEVEL_LIGHT: config.light_max_terms, LEVEL_STRONG: config.strong_max_terms}[level]
    terms = [c["term"] for c in supported[:max_terms]]
    chosen = [c for c in supported[:max_terms]]
    if level == LEVEL_NONE:
        reason = f"{LEVEL_LABELS[level]}: {rule}. {ind_text}."
    else:
        reason = (f"{LEVEL_LABELS[level]}: {rule}. {ind_text}. "
                  f"Istilah tambahan: {_fmt_terms(chosen, n_top)}.")
    if notes:
        reason += " " + " ".join(notes)

    result = _build_result(normalized, terms, reason)
    result.update({
        "word_count": n_words,
        "expansion_level": level,
        "expansion_label": LEVEL_LABELS[level],
        "decision_basis": "quality",
        "indicators": {k: v for k, v in diagnostics.items() if k != "reliable"},
        "candidates": cands[:8],
        "n_candidates": len(cands),
        "n_supported_candidates": len(supported),
        "thresholds": asdict(config),
    })
    return result
