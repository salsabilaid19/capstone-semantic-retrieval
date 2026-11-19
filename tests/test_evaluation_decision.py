"""Uji agregasi analisis per keputusan Adaptive QE (tabel per-kueri BUATAN; hanya menguji hitungan)."""
import pandas as pd

from src import evaluation as ev
from src.retrieval import ALL_METHODS, METHOD_ADAPTIVE, METHOD_SEMANTIC

K = 3


def _pq():
    rows = []
    # q1: none, q2: light, q3: light, q4: strong
    spec = {"q1": "none", "q2": "light", "q3": "light", "q4": "strong"}
    sem = {"q1": 0.5, "q2": 0.4, "q3": 0.2, "q4": 0.1}
    adp = {"q1": 0.5, "q2": 0.6, "q3": 0.3, "q4": 0.4}
    for q, lvl in spec.items():
        for m in ALL_METHODS:
            v = adp[q] if m == METHOD_ADAPTIVE else sem[q]
            rows.append({"query_id": q, "query": q, "method": m, "n_relevant_known": 2, "segment": "1-3 kata",
                         "final_query": q, "is_expanded": lvl != "none" and m == METHOD_ADAPTIVE,
                         "expansion_level": lvl if m == METHOD_ADAPTIVE else "", "top1_z": 4.0, "top5_mean_z": 3.0,
                         f"precision@{K}": v, f"ndcg@{K}": v, f"mrr@{K}": v, f"unjudged@{K}": 0.0})
    return pd.DataFrame(rows)


def test_ringkasan_per_keputusan_menghitung_rata_rata_dan_selisih():
    out = ev.summarize_by_decision(_pq(), K)
    a = out[out["method"] == METHOD_ADAPTIVE].set_index("adaptive_level")
    assert a.loc["light", "n_queries"] == 2 and a.loc["none", "n_queries"] == 1 and a.loc["strong", "n_queries"] == 1
    assert abs(a.loc["light", f"ndcg@{K}"] - 0.45) < 1e-9
    assert abs(a.loc["light", f"delta_vs_semantic_ndcg@{K}"] - 0.15) < 1e-9     # ((0.6-0.4)+(0.3-0.2))/2
    assert abs(a.loc["none", f"delta_vs_semantic_ndcg@{K}"]) < 1e-9
    s = out[out["method"] == METHOD_SEMANTIC]
    assert (s[f"delta_vs_semantic_ndcg@{K}"].abs() < 1e-9).all()
    assert out["small_group"].all()                                              # semua < 10 kueri -> ditandai indikatif


def test_sebaran_keputusan_dan_kueri_tanpa_relevan_dikecualikan():
    pq = _pq()
    d = ev.decision_distribution(pq).set_index("expansion_level")
    assert d.loc["light", "n_queries"] == 2 and d.loc["light", "expanded_rate"] == 1.0 and d.loc["none", "expanded_rate"] == 0.0
    pq.loc[pq["query_id"] == "q4", "n_relevant_known"] = 0
    out = ev.summarize_by_decision(pq, K)
    assert "strong" not in set(out["adaptive_level"])


def test_tanpa_data_mengembalikan_tabel_kosong():
    assert ev.summarize_by_decision(_pq().iloc[0:0], K).empty
