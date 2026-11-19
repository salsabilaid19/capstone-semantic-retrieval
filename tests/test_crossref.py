"""Tes verifikasi/pelengkap Crossref dengan respons rekaan."""
import pandas as pd

from src.crossref import enrich_frame, lookup, parse_work, title_similarity

WORK = {"message": {"title": ["Klasifikasi Sentimen Ulasan Aplikasi Menggunakan Naive Bayes"],
                    "container-title": ["Jurnal Fixture"], "issued": {"date-parts": [[2024, 2, 15]]},
                    "abstract": "<jats:p>Abstrak dari <jats:italic>Crossref</jats:italic>.</jats:p>",
                    "author": [{"given": "Fulan", "family": "Contoh"}], "URL": "https://doi.org/10.99999/fixture.1",
                    "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}], "type": "journal-article"}}


class FakeCrossref:
    def __init__(self, works): self.works, self.calls = works, []
    def __call__(self, url):
        self.calls.append(url)
        for doi, payload in self.works.items():
            if doi in url.lower():
                if isinstance(payload, Exception):
                    raise payload
                return payload
        return None


def test_parse_work_dan_jats_dibersihkan():
    w = parse_work(WORK)
    assert w["year"] == 2024 and w["journal"] == "Jurnal Fixture" and w["authors"] == "Fulan Contoh"
    assert w["abstract"] == "Abstrak dari Crossref ." or w["abstract"].startswith("Abstrak dari")
    assert "<" not in w["abstract"] and w["licenses"] == ["https://creativecommons.org/licenses/by/4.0/"]


def test_lookup_status_cache_dan_error_diulang():
    f = FakeCrossref({"fixture.1": WORK, "fixture.err": RuntimeError("HTTP 500")})
    cache = {}
    assert lookup("10.99999/FIXTURE.1", f, cache, "a@b.id")["status"] == "ok"
    lookup("10.99999/fixture.1", f, cache)                              # dari cache
    assert len(f.calls) == 1 and "mailto=a%40b.id" in f.calls[0]
    assert lookup("10.99999/none", f, cache)["status"] == "not_found"
    assert lookup("10.99999/fixture.err", f, cache)["status"] == "error"
    lookup("10.99999/fixture.err", f, cache)                            # error tidak di-cache: dicoba lagi
    assert sum("fixture.err" in c for c in f.calls) == 2


def test_enrich_hanya_mengisi_yang_kosong_dan_menandai_mismatch(tmp_path):
    df = pd.DataFrame({
        "Title": ["Klasifikasi Sentimen Ulasan Aplikasi Menggunakan Naive Bayes", "Judul Lain Sama Sekali"],
        "Abstract": ["", "Abstrak penerbit tetap"], "Authors": ["", "Penulis Asli"], "Year": ["", "2020"],
        "DOI": ["10.99999/fixture.1", "10.99999/fixture.1"], "Publication": ["", "Jurnal Penerbit"]})
    sleeps, cache = [], tmp_path / "c.json"
    f = FakeCrossref({"fixture.1": WORK})
    out = enrich_frame(df, f, "a@b.id", cache, delay=0.2, sleep=sleeps.append)
    r0, r1 = out.iloc[0], out.iloc[1]
    assert (r0.Year, r0.Publication, r0.Authors) == ("2024", "Jurnal Fixture", "Fulan Contoh") and r0.Abstract.startswith("Abstrak dari")
    assert r0.Crossref_Title_Match == "match"
    assert (r1.Year, r1.Abstract, r1.Authors, r1.Publication) == ("2020", "Abstrak penerbit tetap", "Penulis Asli", "Jurnal Penerbit")
    assert r1.Crossref_Title_Match.startswith("mismatch")             # ditandai, TIDAK diubah
    assert sleeps == [0.2]                                            # jeda hanya untuk permintaan baru
    f2 = FakeCrossref({"fixture.1": WORK})
    enrich_frame(df, f2, "a@b.id", cache)                              # jalankan ulang: semua dari cache
    assert f2.calls == []


def test_enrich_tanpa_doi_dan_no_fill():
    df = pd.DataFrame({"Title": ["x", "y"], "Abstract": ["", ""], "Authors": ["", ""], "Year": ["", ""],
                       "DOI": ["", "10.99999/fixture.1"], "Publication": ["", ""]})
    out = enrich_frame(df, FakeCrossref({"fixture.1": WORK}), fill_missing=False, delay=0)
    assert out.Crossref_Status.tolist() == ["no_doi", "ok"] and out.Year.tolist() == ["", ""]


def test_title_similarity():
    assert title_similarity("Deteksi Hoaks!", "deteksi hoaks") == 1.0
    assert title_similarity("abc", "xyz") < 0.5
