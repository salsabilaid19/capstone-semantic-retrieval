"""Unit test sederhana untuk query expansion. Jalankan: pytest"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.query_expansion import (  # noqa: E402
    adaptive_expand_query, clean_candidates, count_query_words,
    get_expansion_candidates, static_expand_query,
)


def _hasil(keywords_list):
    return pd.DataFrame({"keywords": keywords_list})


def test_count_query_words():
    assert count_query_words("banjir") == 1
    assert count_query_words("  deteksi   banjir  ") == 2
    assert count_query_words("") == 0
    assert count_query_words(None) == 0


def test_query_pendek_boleh_diekspansi():
    hasil = _hasil(["sensor; peringatan dini", "iot", ""])
    out = adaptive_expand_query("banjir", hasil)
    assert out["word_count"] == 1
    assert out["is_expanded"] is True
    assert out["final_query"].startswith("banjir ")


def test_query_panjang_tidak_diekspansi():
    q = "sistem deteksi banjir menggunakan sensor IoT"
    out = adaptive_expand_query(q, _hasil(["sensor; iot"]))
    assert out["is_expanded"] is False
    assert out["final_query"] == q
    assert out["decision_reason"] == (
        "Kueri terdiri dari lebih dari 3 kata sehingga dianggap cukup spesifik dan tidak diekspansi."
    )


def test_kandidat_duplikat_hanya_sekali():
    assert clean_candidates(["IoT", "iot", "Sensor", "IOT"], "banjir") == ["IoT", "Sensor"]


def test_term_sudah_ada_di_query_tidak_ditambahkan():
    assert clean_candidates(["banjir", "sensor"], "deteksi banjir") == ["sensor"]


def test_maksimal_dua_term():
    hasil = _hasil(["a; b; c; d, e|f"])
    out = static_expand_query("banjir", hasil)
    assert len(out["expansion_terms"]) == 2


def test_keyword_kosong_tidak_error():
    assert get_expansion_candidates(_hasil([None, "", float("nan")]), "banjir") == []
    assert get_expansion_candidates(pd.DataFrame(), "banjir") == []
    assert get_expansion_candidates(None, "banjir") == []


def test_static_tanpa_kandidat_mempertahankan_query_asli():
    out = static_expand_query("banjir", _hasil(["", None]))
    assert out["is_expanded"] is False
    assert out["final_query"] == "banjir"
    assert out["expansion_terms"] == []
    assert out["decision_reason"]
