"""Verifikasi dan pelengkap metadata lewat REST API Crossref (https://api.crossref.org/works/{doi}).

Aturan: hanya MENGISI kolom yang kosong, tidak pernah menimpa metadata dari penerbit; ketidakcocokan judul
ditandai untuk ditinjau manual. Hasil di-cache ke disk agar dapat dilanjutkan dan tidak membebani API.
"""
import difflib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src.preprocessing import normalize_doi, normalize_title, parse_year

API = "https://api.crossref.org/works/"
TITLE_MATCH_THRESHOLD = 0.85


def make_fetcher(mailto: str, timeout: int = 30) -> Callable[[str], Optional[dict]]:
    """GET JSON. Mengembalikan None untuk 404 (DOI tidak ada); melempar RuntimeError untuk galat lain."""
    ua = f"CapstoneIR/1.0 (mailto:{mailto})" if mailto else "CapstoneIR/1.0"

    def fetch(url: str) -> Optional[dict]:
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            raise RuntimeError(f"HTTP {err.code}") from err
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as err:
            raise RuntimeError(str(err)) from err
    return fetch


def _strip_jats(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def parse_work(payload: dict) -> dict:
    """Ambil bidang yang relevan dari respons Crossref."""
    m = (payload or {}).get("message", {})
    issued = (m.get("issued", {}).get("date-parts") or [[None]])[0]
    authors = "; ".join(" ".join(x for x in (a.get("given", ""), a.get("family", "")) if x)
                        for a in m.get("author", []))
    return {
        "title": (m.get("title") or [""])[0],
        "journal": (m.get("container-title") or [""])[0],
        "year": issued[0] if issued and issued[0] else None,
        "abstract": _strip_jats(m.get("abstract", "")),
        "authors": authors,
        "url": m.get("URL", ""),
        "licenses": [l.get("URL", "") for l in m.get("license", [])],
        "type": m.get("type", ""),
    }


def title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def lookup(doi: str, fetch: Callable, cache: dict, mailto: str = "") -> dict:
    """Cari satu DOI (dengan cache). Hasil: {'status': ok|not_found|error, 'data': {...}}."""
    key = normalize_doi(doi)
    if key in cache and cache[key]["status"] != "error":
        return cache[key]
    q = f"?mailto={urllib.parse.quote(mailto)}" if mailto else ""
    try:
        payload = fetch(API + urllib.parse.quote(key, safe="/") + q)
        out = {"status": "not_found", "data": {}} if payload is None else {"status": "ok", "data": parse_work(payload)}
    except RuntimeError as err:
        out = {"status": "error", "data": {}, "error": str(err)}
    cache[key] = out
    return out


def enrich_frame(df: pd.DataFrame, fetch: Callable, mailto: str = "", cache_path=None,
                 delay: float = 0.2, sleep: Callable = time.sleep, fill_missing: bool = True) -> pd.DataFrame:
    """Tambah kolom Crossref_Status & Crossref_Title_Match; isi Year/Publication/Abstract/Authors yang kosong."""
    cache = {}
    if cache_path and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    out = df.copy()
    out["Crossref_Status"] = "no_doi"
    out["Crossref_Title_Match"] = ""
    for idx, row in out.iterrows():
        doi = normalize_doi(row.get("DOI", ""))
        if not doi:
            continue
        fresh = normalize_doi(doi) not in cache
        res = lookup(doi, fetch, cache, mailto)
        out.at[idx, "Crossref_Status"] = res["status"]
        if res["status"] == "ok":
            d = res["data"]
            sim = title_similarity(str(row.get("Title", "")), d["title"]) if d["title"] else 0.0
            out.at[idx, "Crossref_Title_Match"] = "match" if sim >= TITLE_MATCH_THRESHOLD else f"mismatch({sim:.2f})"
            if fill_missing:
                if (parse_year(row.get("Year", "")) is pd.NA) and d["year"]:
                    out.at[idx, "Year"] = str(d["year"])
                for col, key in (("Publication", "journal"), ("Abstract", "abstract"), ("Authors", "authors")):
                    if not str(row.get(col, "") or "").strip() and d[key]:
                        out.at[idx, col] = d[key]
        if fresh and delay:
            sleep(delay)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
