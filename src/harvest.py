"""Pengambilan metadata artikel jurnal via OAI-PMH (format oai_dc, lazim pada OJS).

Hanya metadata (judul, abstrak, kata kunci, DOI, dll.), bukan teks penuh. Hanya pustaka standar Python.
Fungsi HTTP (`fetch`) dapat disuntikkan sehingga seluruh logika dapat diuji tanpa jaringan.
"""
import csv
import html
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

OAI = "{http://www.openarchives.org/OAI/2.0/}"
OAI_DC = "{http://www.openarchives.org/OAI/2.0/oai_dc/}"
DC = "{http://purl.org/dc/elements/1.1/}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

HARVEST_COLUMNS = [
    "Title", "Abstract", "Authors", "Year", "DOI", "URL", "Keywords", "Publication", "Type", "Language",
    "Data_Source", "Source_Note", "source_id", "oai_identifier", "datestamp", "rights", "harvested_at",
]
SOURCE_COLUMNS = [
    "source_id", "name", "issn", "publisher", "site_url", "oai_base_url", "oai_alt_urls", "oai_set", "license",
    "license_evidence", "doi_prefix", "languages", "topic_focus", "enabled", "verification", "verified_on", "notes",
]
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)


class HarvestError(RuntimeError):
    """Kegagalan pengambilan (jaringan, respons bukan OAI-PMH, atau error protokol OAI)."""


# ---------------- HTTP ----------------
def make_fetcher(user_agent: str, timeout: int = 30, retries: int = 3,
                 sleep: Callable = time.sleep) -> Callable[[str], bytes]:
    """Fungsi HTTP GET dengan retry + backoff (5xx, timeout) dan penghormatan Retry-After (429)."""
    def fetch(url: str) -> bytes:
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/xml,text/xml"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as err:
                last = err
                if err.code == 429:
                    retry_after = err.headers.get("Retry-After", "")
                    sleep(min(int(retry_after), 60) if retry_after.isdigit() else 10 * (attempt + 1))
                elif 500 <= err.code < 600:
                    sleep(2 ** attempt)
                else:
                    raise HarvestError(f"HTTP {err.code} untuk {url}") from err
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last = err
                sleep(2 ** attempt)
        raise HarvestError(f"Gagal mengambil {url} setelah {retries} percobaan: {last}")
    return fetch


# ---------------- Klien OAI-PMH ----------------
class OaiClient:
    def __init__(self, base_url: str, fetch: Callable[[str], bytes], delay: float = 1.0,
                 sleep: Callable = time.sleep):
        self.base_url, self.fetch, self.delay, self.sleep = base_url, fetch, delay, sleep
        self._first = True

    def _get(self, params: Dict[str, str]) -> ET.Element:
        if not self._first:
            self.sleep(self.delay)          # sopan terhadap server penerbit
        self._first = False
        sep = "&" if "?" in self.base_url else "?"
        url = self.base_url + sep + urllib.parse.urlencode(params)
        raw = self.fetch(url)
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as err:
            raise HarvestError("Respons bukan XML valid (mungkin halaman HTML/penghalang bot/URL salah).") from err
        if root.tag != OAI + "OAI-PMH":
            raise HarvestError("Respons XML bukan OAI-PMH.")
        return root

    @staticmethod
    def _error(root: ET.Element) -> Optional[tuple]:
        err = root.find(OAI + "error")
        return (err.get("code", ""), (err.text or "").strip()) if err is not None else None

    def identify(self) -> dict:
        root = self._get({"verb": "Identify"})
        if self._error(root):
            raise HarvestError("Identify gagal: %s %s" % self._error(root))
        ident = root.find(OAI + "Identify")
        g = lambda tag: (ident.findtext(OAI + tag) or "").strip() if ident is not None else ""   # noqa: E731
        return {"repository_name": g("repositoryName"), "protocol_version": g("protocolVersion"),
                "earliest_datestamp": g("earliestDatestamp"), "granularity": g("granularity")}

    def metadata_formats(self) -> List[str]:
        root = self._get({"verb": "ListMetadataFormats"})
        if self._error(root):
            raise HarvestError("ListMetadataFormats gagal: %s %s" % self._error(root))
        return [e.text.strip() for e in root.iter(OAI + "metadataPrefix") if e.text]

    def list_records(self, from_date: Optional[str] = None, until: Optional[str] = None,
                     set_spec: Optional[str] = None, max_pages: Optional[int] = None) -> Iterator[ET.Element]:
        """Hasilkan elemen <record>, mengikuti resumptionToken."""
        params = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
        for key, val in (("from", from_date), ("until", until), ("set", set_spec)):
            if val:
                params[key] = val
        pages = 0
        while True:
            root = self._get(params)
            err = self._error(root)
            if err:
                if err[0] == "noRecordsMatch":
                    return
                raise HarvestError("ListRecords gagal: %s %s" % err)
            lr = root.find(OAI + "ListRecords")
            if lr is None:
                raise HarvestError("Respons ListRecords tidak memuat <ListRecords>.")
            for rec in lr.findall(OAI + "record"):
                yield rec
            pages += 1
            token_el = lr.find(OAI + "resumptionToken")
            token = (token_el.text or "").strip() if token_el is not None else ""
            if not token or (max_pages and pages >= max_pages):
                return
            params = {"verb": "ListRecords", "resumptionToken": token}


# ---------------- Parsing record oai_dc ----------------
def _clean(text: Optional[str]) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _vals(dc: ET.Element, tag: str) -> List[str]:
    return [v for v in (_clean(e.text) for e in dc.findall(DC + tag)) if v]


def parse_record(rec: ET.Element) -> Optional[dict]:
    """Ubah <record> menjadi dict baris. None bila record dihapus atau tanpa metadata."""
    header = rec.find(OAI + "header")
    if header is None or header.get("status") == "deleted":
        return None
    dc = rec.find(f"{OAI}metadata/{OAI_DC}dc")
    if dc is None:
        return None
    identifiers = _vals(dc, "identifier")
    doi, url = "", ""
    for ident in identifiers:
        m = DOI_RE.search(ident)
        if m and not doi:
            doi = m.group(0).rstrip(".,;)")
        elif ident.lower().startswith("http") and "doi.org" not in ident.lower() and not url:
            url = ident
    types = _vals(dc, "type")
    type_ = next((t for t in types if "article" in t.lower()), types[0] if types else "")
    dates = _vals(dc, "date")
    titles, abstracts = _vals(dc, "title"), _vals(dc, "description")
    return {
        "Title": titles[0] if titles else "",
        "Abstract": abstracts[0] if abstracts else "",
        "Authors": "; ".join(_vals(dc, "creator")),
        "Year": dates[0] if dates else "",
        "DOI": doi,
        "URL": url,
        "Keywords": "; ".join(_vals(dc, "subject")),
        "Type": type_.replace("info:eu-repo/semantics/", ""),
        "Language": (_vals(dc, "language") or [""])[0],
        "rights": " | ".join(_vals(dc, "rights")),
        "oai_identifier": _clean(header.findtext(OAI + "identifier")),
        "datestamp": _clean(header.findtext(OAI + "datestamp")),
    }


# ---------------- Konfigurasi sumber & filter topik ----------------
def load_sources(path) -> List[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in SOURCE_COLUMNS if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"Kolom konfigurasi hilang di {path}: {missing}")
    ids = [r["source_id"] for r in rows]
    if len(ids) != len(set(ids)) or "" in ids:
        raise ValueError("source_id harus unik dan tidak kosong.")
    for r in rows:
        r["enabled"] = r["enabled"].strip().lower() in ("true", "1", "yes", "ya")
        if r["oai_base_url"] and not r["oai_base_url"].startswith(("http://", "https://")):
            raise ValueError(f"oai_base_url tidak valid untuk {r['source_id']}: {r['oai_base_url']}")
    return rows


def save_sources(rows: List[dict], path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SOURCE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r); r["enabled"] = "true" if r["enabled"] else "false"
            w.writerow(r)


def candidate_urls(source: dict) -> List[str]:
    urls = [source["oai_base_url"]] + [u.strip() for u in source.get("oai_alt_urls", "").split("|")]
    seen, out = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u); out.append(u)
    return out


def load_topic_terms(path) -> List[str]:
    return [l.strip().lower() for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def matches_topic(row: dict, terms: List[str]) -> bool:
    if not terms:
        return True
    text = " ".join((row.get("Title", ""), row.get("Keywords", ""), row.get("Abstract", ""))).lower()
    return any(t in text for t in terms)


# ---------------- Panen satu sumber ----------------
@dataclass
class HarvestResult:
    source_id: str
    rows: List[dict] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    error: str = ""


def harvest_source(source: dict, client: OaiClient, from_date: Optional[str] = None,
                   max_records: Optional[int] = None, terms: Optional[List[str]] = None,
                   max_pages: Optional[int] = None, today: Optional[date] = None) -> HarvestResult:
    """Panen satu jurnal. Bila gagal di tengah jalan, hasil parsial tetap dikembalikan bersama pesan error."""
    res = HarvestResult(source["source_id"])
    st = dict.fromkeys(("seen", "deleted_or_empty", "off_topic", "kept"), 0)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    note = (f"OAI-PMH {client.base_url}; lisensi jurnal: {source.get('license') or 'tidak tercatat'}; "
            f"diambil {(today or date.today()).isoformat()}")
    try:
        for rec in client.list_records(from_date=from_date, set_spec=source.get("oai_set") or None,
                                       max_pages=max_pages):
            st["seen"] += 1
            row = parse_record(rec)
            if row is None:
                st["deleted_or_empty"] += 1
                continue
            if not matches_topic(row, terms or []):
                st["off_topic"] += 1
                continue
            row.update({"Publication": source["name"], "Data_Source": "real", "Source_Note": note,
                        "source_id": source["source_id"], "harvested_at": stamp})
            res.rows.append(row)
            st["kept"] += 1
            if max_records and st["kept"] >= max_records:
                break
    except HarvestError as err:
        res.error = str(err)
    res.stats = st
    return res


def write_harvest_csv(rows: List[dict], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HARVEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
