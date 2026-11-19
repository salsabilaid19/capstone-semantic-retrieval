"""Uji Adaptive QE tiga tingkat (No / Light / Strong).

Vektor skor dibuat TERKONTROL (bukan data relevansi): kepala skor ditentukan agar indikator kualitas jatuh di
zona yang diuji. Ini menguji LOGIKA keputusan, bukan kebermanfaatan ekspansi (itu hanya bisa dinilai lewat
evaluasi dengan label manusia)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import query_expansion as qe  # noqa: E402
from src.query_expansion import (  # noqa: E402
    LEVEL_LIGHT, LEVEL_NONE, LEVEL_STRONG, AdaptiveConfig, adaptive_expand_query, decide_level,
    score_candidates, score_diagnostics, static_expand_query,
)

N_DOCS = 300


def skor(kepala, n=N_DOCS, seed=0):
    """Vektor skor ala cosine e5 (rata-rata 0,82, SD 0,02) dengan z kepala skor PERSIS = `kepala`.

    Sisa dokumen diskalakan agar rata-rata dan SD keseluruhan tepat, sehingga z top-k yang dibaca
    `score_diagnostics` sama dengan yang dimaksud (tanpa pergeseran akibat menyisipkan kepala skor)."""
    h = np.asarray(kepala, dtype=float)
    k, m = len(h), n - len(h)
    u = np.clip(np.random.default_rng(seed).standard_normal(m), -2.0, 2.0)
    u = (u - u.mean()) / u.std()
    a = -h.sum() / m
    b = np.sqrt((n - (h ** 2).sum()) / m - a ** 2)
    z = np.concatenate([h, a + b * u])
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9
    return 0.82 + 0.02 * z


KUAT = [5.6, 5.2, 4.9, 4.7, 4.5, 4.2, 4.0, 3.9, 3.8, 3.7]        # hasil awal kuat
SEDANG = [3.9, 3.7, 3.6, 3.5, 3.3, 3.1, 3.0, 2.9, 2.8, 2.7]      # di antara kuat dan lemah
LEMAH = [2.6, 2.5, 2.4, 2.3, 2.3, 2.2, 2.2, 2.2, 2.1, 2.1]       # tak ada dokumen menonjol


def hasil_awal(rows=None):
    """Top-5 hasil awal rekaan: 'sentiment analysis' dan 'twitter' didukung >= 2 dokumen; sisanya hanya 1."""
    rows = rows or [
        ("Analisis sentimen twitter", "twitter; sentiment analysis; svm", "Penelitian sentiment analysis pada twitter."),
        ("Klasifikasi opini", "sentiment analysis; naive bayes; twitter", "Opini twitter diklasifikasi, sentiment analysis."),
        ("Ulasan produk", "sentiment analysis; lexicon", "Sentiment analysis pada ulasan."),
        ("Berita hoaks", "hoaks; text mining", "Deteksi hoaks."),
        ("Citra daun", "cnn; daun", "Klasifikasi citra daun."),
    ]
    return pd.DataFrame(rows, columns=["title", "keywords", "abstract"])


# ---------------- indikator ----------------
def test_diagnostics_z_score_dan_selisih():
    d = score_diagnostics(skor(KUAT))
    assert d["reliable"] and d["n_docs"] == N_DOCS
    assert d["top1_z"] > d["top5_mean_z"] > 3.0 and d["gap_z"] == pytest.approx((d["top1_score"] - d["top2_score"]) / d["std_score"])
    assert d["spread_z"] > d["gap_z"] > 0


def test_diagnostics_korpus_kecil_atau_skor_seragam_tidak_andal():
    assert not score_diagnostics(np.linspace(0.5, 0.9, 6))["reliable"]
    assert "dokumen" in score_diagnostics(np.linspace(0.5, 0.9, 6))["unreliable_reason"]
    assert not score_diagnostics(np.full(100, 0.8))["reliable"]
    assert not score_diagnostics(np.array([np.nan] * 50))["reliable"]


# ---------------- tiga tingkat ----------------
def test_no_expansion_bila_hasil_awal_kuat():
    out = adaptive_expand_query("sentiment", hasil_awal(), score_diagnostics(skor(KUAT)))
    assert out["expansion_level"] == LEVEL_NONE and out["expansion_label"] == "No Expansion"
    assert not out["is_expanded"] and out["expansion_terms"] == [] and out["final_query"] == "sentiment"
    assert out["decision_basis"] == "quality" and "hasil awal kuat" in out["decision_reason"]


def test_light_expansion_satu_istilah_bila_hasil_sedang():
    out = adaptive_expand_query("sentiment", hasil_awal(), score_diagnostics(skor(SEDANG)))
    assert out["expansion_level"] == LEVEL_LIGHT and out["expansion_label"] == "Light Expansion"
    assert out["is_expanded"] and len(out["expansion_terms"]) == 1
    assert out["expansion_terms"][0] == "sentiment analysis"          # dukungan terbesar (3 dari 5 dokumen)
    assert out["final_query"] == "sentiment sentiment analysis"
    assert "Light Expansion" in out["decision_reason"] and "3 dari 5" in out["decision_reason"]


def test_strong_expansion_sampai_tiga_istilah_bila_hasil_lemah():
    out = adaptive_expand_query("opini", hasil_awal(), score_diagnostics(skor(LEMAH)))
    assert out["expansion_level"] == LEVEL_STRONG and out["expansion_label"] == "Strong Expansion"
    assert 2 <= len(out["expansion_terms"]) <= 3
    assert set(out["expansion_terms"]) >= {"sentiment analysis", "twitter"}
    assert "hasil awal lemah" in out["decision_reason"]


def test_tingkat_ekspansi_naik_seiring_kualitas_hasil_menurun():
    urut = [adaptive_expand_query("opini", hasil_awal(), score_diagnostics(skor(h)))["expansion_level"]
            for h in (KUAT, SEDANG, LEMAH)]
    assert urut == [LEVEL_NONE, LEVEL_LIGHT, LEVEL_STRONG]


def test_hasil_ekspansi_tidak_hanya_ditentukan_panjang_kueri():
    """Kueri pendek yang hasil awalnya kuat TIDAK diperluas; kueri sama panjang dengan hasil lemah diperluas."""
    pendek_kuat = adaptive_expand_query("tf-idf", hasil_awal(), score_diagnostics(skor(KUAT)))
    pendek_lemah = adaptive_expand_query("tf-idf", hasil_awal(), score_diagnostics(skor(LEMAH)))
    assert pendek_kuat["word_count"] == pendek_lemah["word_count"] == 1
    assert (pendek_kuat["expansion_level"], pendek_lemah["expansion_level"]) == (LEVEL_NONE, LEVEL_STRONG)


def test_satu_dokumen_menonjol_jelas_dianggap_hasil_kuat():
    d = score_diagnostics(skor([6.5, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5]))
    assert d["gap_z"] >= AdaptiveConfig().gap_z_clear
    assert decide_level(d, 2)[0] == LEVEL_NONE


def test_distribusi_datar_tanpa_kepala_dianggap_lemah():
    d = score_diagnostics(skor([3.6, 3.55, 3.5, 3.45, 3.45, 3.4, 3.4, 3.4, 3.35, 3.35]))
    assert d["spread_z"] < AdaptiveConfig().spread_z_flat
    assert decide_level(d, 2)[0] == LEVEL_STRONG


def test_kueri_spesifik_diturunkan_satu_tingkat():
    kueri = "klasifikasi sentimen ulasan produk pada twitter indonesia"          # 7 kata
    assert qe.count_query_words(kueri) >= AdaptiveConfig().specific_query_words
    sedang = adaptive_expand_query(kueri, hasil_awal(), score_diagnostics(skor(SEDANG)))
    lemah = adaptive_expand_query(kueri, hasil_awal(), score_diagnostics(skor(LEMAH)))
    assert sedang["expansion_level"] == LEVEL_NONE and "spesifik" in sedang["decision_reason"]
    assert lemah["expansion_level"] == LEVEL_LIGHT and len(lemah["expansion_terms"]) == 1


# ---------------- kandidat: dukungan, relevansi, duplikat ----------------
def test_kandidat_diurutkan_menurut_dukungan_lintas_dokumen():
    c = score_candidates(hasil_awal(), "opini")
    assert c[0]["term"] == "sentiment analysis" and c[0]["support"] == 3
    assert {x["term"]: x["support"] for x in c}["twitter"] >= 2


def test_tanpa_kandidat_berdukungan_tingkat_diturunkan_dan_dijelaskan():
    sendiri = pd.DataFrame({"title": ["A", "B", "C", "D", "E"], "abstract": ["x"] * 5,
                            "keywords": ["alfa", "beta", "gamma", "delta", "epsilon"]})       # tak ada yang didukung >= 2 dokumen
    out = adaptive_expand_query("opini", sendiri, score_diagnostics(skor(LEMAH)))
    assert out["expansion_level"] == LEVEL_NONE and not out["is_expanded"] and out["final_query"] == "opini"
    assert "diturunkan" in out["decision_reason"] or "dipertahankan" in out["decision_reason"]
    assert out["n_candidates"] == 5 and out["n_supported_candidates"] == 0


def test_strong_dengan_hanya_satu_kandidat_berdukungan_diturunkan_ke_light():
    df = pd.DataFrame({"title": ["a", "b", "c", "d", "e"], "abstract": ["alfa"] * 3 + ["z", "z"],
                       "keywords": ["alfa", "beta", "gamma", "delta", "epsilon"]})
    out = adaptive_expand_query("opini", df, score_diagnostics(skor(LEMAH)))
    assert out["expansion_level"] == LEVEL_LIGHT and out["expansion_terms"] == ["alfa"]
    assert "diturunkan ke Light Expansion" in out["decision_reason"]


def test_istilah_generik_dan_yang_sudah_ada_di_kueri_tidak_dipakai():
    df = pd.DataFrame({"title": ["t"] * 5, "abstract": ["accuracy deteksi banjir sensor"] * 5,
                       "keywords": ["Accuracy; banjir; sensor air"] * 5})
    out = score_candidates(df, "deteksi banjir")
    # "Accuracy" generik dibuang, "banjir" sudah ada di kueri; hanya "sensor air" yang tersisa
    assert [(c["term"], c["support"]) for c in out] == [("sensor air", 5)]


def test_tidak_ada_istilah_duplikat_dan_seluruh_kata_sudah_di_kueri_dibuang():
    df = pd.DataFrame({"title": ["t"] * 5, "abstract": ["x"] * 5,
                       "keywords": ["IoT; iot; Sensor; sensor air; Banjir Bandang"] * 5})
    terms = [c["term"].lower() for c in score_candidates(df, "sensor air")]
    assert len(terms) == len(set(terms))
    assert "sensor" not in terms and "sensor air" not in terms      # kata-katanya sudah ada di kueri


def test_query_asli_disimpan_dan_ekspansi_tidak_berulang():
    cfg = AdaptiveConfig()
    pertama = adaptive_expand_query("  opini  ", hasil_awal(), score_diagnostics(skor(LEMAH)), cfg)
    assert pertama["original_query"] == "opini" and pertama["final_query"].startswith("opini ")
    # kueri yang sudah diperluas diberikan lagi: istilah yang sudah ada tidak ditambahkan ulang
    kedua = adaptive_expand_query(pertama["final_query"], hasil_awal(), score_diagnostics(skor(LEMAH)), cfg)
    token = kedua["final_query"].lower().split()
    for t in pertama["expansion_terms"]:
        assert kedua["final_query"].lower().count(t.lower()) == 1, kedua["final_query"]
    assert kedua["original_query"] == pertama["final_query"]
    assert len(token) >= len(pertama["final_query"].split())


def test_jumlah_istilah_dibatasi_per_tingkat():
    banyak = pd.DataFrame({"title": ["t"] * 5, "abstract": ["alfa beta gamma delta epsilon zeta"] * 5,
                           "keywords": ["alfa; beta; gamma; delta; epsilon; zeta"] * 5})
    light = adaptive_expand_query("opini", banyak, score_diagnostics(skor(SEDANG)))
    strong = adaptive_expand_query("opini", banyak, score_diagnostics(skor(LEMAH)))
    assert len(light["expansion_terms"]) == AdaptiveConfig().light_max_terms == 1
    assert len(strong["expansion_terms"]) == AdaptiveConfig().strong_max_terms == 3


def test_ambang_dapat_dikonfigurasi():
    ketat = AdaptiveConfig(top1_z_high=9.0, top1_z_low=8.0)          # hampir semua dianggap lemah
    out = adaptive_expand_query("opini", hasil_awal(), score_diagnostics(skor(KUAT)), ketat)
    assert out["expansion_level"] == LEVEL_STRONG and out["thresholds"]["top1_z_low"] == 8.0


# ---------------- fallback & kompatibilitas ----------------
def test_tanpa_indikator_kualitas_memakai_aturan_panjang_kueri_dan_menyatakannya():
    pendek = adaptive_expand_query("opini", hasil_awal(), diagnostics=None)
    assert pendek["decision_basis"] == "length_only" and "Indikator kualitas tidak dipakai" in pendek["fallback_note"]
    assert pendek["expansion_level"] == LEVEL_LIGHT and pendek["is_expanded"]
    panjang = adaptive_expand_query("klasifikasi opini twitter dengan svm", hasil_awal(), diagnostics=None)
    assert panjang["expansion_level"] == LEVEL_NONE and not panjang["is_expanded"]


def test_korpus_kecil_memakai_fallback_dengan_alasan_korpus_kecil():
    out = adaptive_expand_query("opini", hasil_awal(), score_diagnostics(np.linspace(0.5, 0.9, 8)))
    assert out["decision_basis"] == "length_only" and "korpus hanya 8 dokumen" in out["fallback_note"]


def test_kueri_kosong_aman():
    out = adaptive_expand_query("   ", hasil_awal(), score_diagnostics(skor(LEMAH)))
    assert out["expansion_level"] == LEVEL_NONE and out["final_query"] == "" and not out["is_expanded"]


def test_static_qe_tidak_berubah_dan_selalu_mencoba():
    out = static_expand_query("opini", hasil_awal())
    assert out["is_expanded"] and len(out["expansion_terms"]) <= 2 and out["expansion_level"] == "static"
    assert "Static QE selalu mencoba" in out["decision_reason"]
