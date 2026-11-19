"""Tes pengambilan OAI-PMH dengan FIXTURE rekaan (tanpa jaringan)."""
import urllib.error
from unittest import mock

import pytest

from oai_helpers import FIX, SOURCE, oai_fetch
from src.harvest import (HarvestError, OaiClient, candidate_urls, harvest_source, load_sources, load_topic_terms,
                         make_fetcher, matches_topic, parse_record, save_sources)

ROOT = FIX.parents[1]


def client(fetch=oai_fetch, sleeps=None):
    return OaiClient("http://fixture.invalid/oai", fetch, delay=1.0, sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


def test_identify_dan_format():
    c = client()
    assert c.identify()["repository_name"] == "Jurnal Fixture"
    assert "oai_dc" in c.metadata_formats()


def test_parse_record_lengkap():
    recs = list(client().list_records(max_pages=1))
    row = parse_record(recs[0])
    assert row["Title"].startswith("Klasifikasi Sentimen")
    assert row["Authors"] == "Fulan Contoh; Fulanah Contoh"
    assert row["Keywords"] == "analisis sentimen; naive bayes"
    assert row["DOI"] == "10.99999/fixture.1" and row["URL"].endswith("/article/view/1")
    assert row["Year"] == "2024-02-15" and row["Type"] == "article" and row["Language"] == "ind"
    assert row["rights"] == "CC BY 4.0" and row["oai_identifier"].endswith("article/1")


def test_record_dihapus_dilewati_dan_html_dibersihkan():
    recs = list(client().list_records(max_pages=1))
    assert parse_record(recs[1]) is None                       # status="deleted"
    row = parse_record(recs[2])
    assert "<" not in row["Abstract"] and row["Abstract"].startswith("This paper studies information retrieval")
    assert row["DOI"] == "" and row["Keywords"] == "" and row["Type"] == "Peer-reviewed Article"


def test_mengikuti_resumption_token_dan_jeda_sopan():
    sleeps = []
    recs = list(client(sleeps=sleeps).list_records())
    assert len(recs) == 6
    assert sleeps == [1.0]                      # 2 permintaan -> 1 jeda


def test_tanpa_record_bukan_error():
    c = client(lambda url: (FIX / "oai_norecords.xml").read_bytes())
    assert list(c.list_records()) == []


def test_respons_html_dan_error_protokol_jadi_harvest_error():
    with pytest.raises(HarvestError, match="bukan XML"):
        list(client(lambda u: b"<html><body>Cloudflare</body></html>x<").list_records())
    with pytest.raises(HarvestError, match="bukan OAI-PMH"):
        list(client(lambda u: b"<html><body>halaman</body></html>").list_records())
    bad = b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="badArgument">x</error></OAI-PMH>'
    with pytest.raises(HarvestError, match="badArgument"):
        list(client(lambda u: bad).list_records())


def test_harvest_source_filter_topik_statistik_dan_metadata():
    terms = load_topic_terms(ROOT / "config" / "topic_terms.txt")
    res = harvest_source(SOURCE, client(), terms=terms)
    assert not res.error
    assert res.stats == {"seen": 6, "deleted_or_empty": 1, "off_topic": 1, "kept": 4}
    assert "Pupuk" not in " ".join(r["Title"] for r in res.rows)
    r0 = res.rows[0]
    assert r0["Publication"] == "Jurnal Fixture" and r0["Data_Source"] == "real" and r0["source_id"] == "fx"
    assert "http://fixture.invalid/oai" in r0["Source_Note"] and "CC BY 4.0" in r0["Source_Note"]


def test_harvest_source_max_records_dan_tanpa_filter():
    res = harvest_source(SOURCE, client(), max_records=2)
    assert len(res.rows) == 2 and res.stats["kept"] == 2


def test_kegagalan_tengah_jalan_mempertahankan_hasil_parsial():
    def fetch(url):
        if "resumptionToken" in url:
            raise HarvestError("server mati")
        return oai_fetch(url)
    res = harvest_source(SOURCE, client(fetch))
    assert "server mati" in res.error and len(res.rows) == 2         # 2 record valid dari halaman 1


def test_matches_topic():
    assert matches_topic({"Title": "Deteksi Hoaks", "Keywords": "", "Abstract": "Machine Learning"}, ["machine learning"])
    assert not matches_topic({"Title": "Padi", "Keywords": "", "Abstract": "sawah"}, ["machine learning"])
    assert matches_topic({"Title": "apa saja"}, [])


# ---- konfigurasi sumber ----
def test_config_bawaan_valid_dan_jujur():
    src = load_sources(ROOT / "config" / "journals.csv")
    by = {s["source_id"]: s for s in src}
    assert {"resti", "ijccs", "jnteti", "jlk"} <= set(by)
    for s in src:
        assert not (s["verification"] == "verified" and not s["verified_on"])
        if s["enabled"]:
            assert s["oai_base_url"] and s["license"]                  # sumber aktif wajib punya URL & lisensi
    assert by["jlk"]["oai_base_url"] == "" and not by["jlk"]["enabled"]
    for s_ in src:   # jebakan: fib.unand.ac.id = jurnal Linguistika Kultura, bukan JLK; tidak boleh jadi URL sumber mana pun
        assert "fib.unand.ac.id" not in s_["site_url"] + s_["oai_base_url"] + s_["oai_alt_urls"]


def test_load_sources_validasi_dan_roundtrip(tmp_path):
    src = load_sources(ROOT / "config" / "journals.csv")
    path = tmp_path / "c.csv"
    save_sources(src, path)
    assert load_sources(path) == src
    path.write_text(path.read_text(encoding="utf-8").replace("http://jurnal.iaii", "ftp://jurnal.iaii", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="oai_base_url"):
        load_sources(path)
    dup = tmp_path / "d.csv"
    save_sources(src + [dict(src[0])], dup)
    with pytest.raises(ValueError, match="unik"):
        load_sources(dup)


def test_candidate_urls_tanpa_duplikat():
    s = {"oai_base_url": "http://a/oai", "oai_alt_urls": "https://a/oai|http://a/oai| "}
    assert candidate_urls(s) == ["http://a/oai", "https://a/oai"]


# ---- HTTP: retry/backoff ----
class _Resp:
    def __init__(self, data): self.data = data
    def read(self): return self.data
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _http_error(code, headers=None):
    import email.message
    h = email.message.Message()
    for k, v in (headers or {}).items():
        h[k] = v
    return urllib.error.HTTPError("http://x", code, "err", h, None)


def test_fetcher_retry_pada_5xx_lalu_berhasil():
    sleeps = []
    with mock.patch("urllib.request.urlopen", side_effect=[_http_error(503), _http_error(500), _Resp(b"ok")]):
        assert make_fetcher("ua", sleep=sleeps.append)("http://x") == b"ok"
    assert sleeps == [1, 2]


def test_fetcher_404_langsung_gagal_dan_429_menghormati_retry_after():
    with mock.patch("urllib.request.urlopen", side_effect=_http_error(404)) as m:
        with pytest.raises(HarvestError, match="HTTP 404"):
            make_fetcher("ua", sleep=lambda s: None)("http://x")
        assert m.call_count == 1
    sleeps = []
    with mock.patch("urllib.request.urlopen", side_effect=[_http_error(429, {"Retry-After": "7"}), _Resp(b"ok")]):
        make_fetcher("ua", sleep=sleeps.append)("http://x")
    assert sleeps == [7]


def test_fetcher_menyerah_setelah_retry_habis():
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("dns")):
        with pytest.raises(HarvestError, match="3 percobaan"):
            make_fetcher("ua", sleep=lambda s: None)("http://x")
