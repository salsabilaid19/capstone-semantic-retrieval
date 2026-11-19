"""Fungsi HTTP palsu berbasis fixture XML rekaan (tests/fixtures)."""
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"


def oai_fetch(url: str) -> bytes:
    if "verb=Identify" in url:
        return (FIX / "oai_identify.xml").read_bytes()
    if "verb=ListMetadataFormats" in url:
        return (FIX / "oai_formats.xml").read_bytes()
    if "resumptionToken=tok123" in url:
        return (FIX / "oai_page2.xml").read_bytes()
    if "verb=ListRecords" in url:
        return (FIX / "oai_page1.xml").read_bytes()
    raise AssertionError(f"URL tak terduga: {url}")


SOURCE = {"source_id": "fx", "name": "Jurnal Fixture", "license": "CC BY 4.0", "oai_set": "",
          "oai_base_url": "http://fixture.invalid/oai", "oai_alt_urls": "", "enabled": True}
