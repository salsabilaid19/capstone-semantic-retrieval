"""Periksa apakah endpoint OAI-PMH tiap jurnal di config/journals.csv benar-benar aktif dan layak dipanen.

Contoh:  python -m scripts.check_sources --contact-email anda@kampus.ac.id
         python -m scripts.check_sources --sources ijccs --update-config
Memeriksa: Identify, dukungan oai_dc, dan sampel satu halaman ListRecords (cakupan abstrak/kata kunci/DOI).
Dengan --update-config, sumber yang lolos ditandai verification=verified, enabled=true, dan URL yang bekerja disimpan.
"""
import argparse
import os
import sys
from datetime import date

from src.harvest import (HarvestError, OaiClient, candidate_urls, load_sources, make_fetcher, parse_record,
                         save_sources)


def check_source(source: dict, fetch, sample: int = 20, delay: float = 0.0, sleep=None) -> dict:
    result = {"source_id": source["source_id"], "ok": False, "url": "", "detail": "", "sample": 0,
              "pct_abstract": 0, "pct_keywords": 0, "pct_doi": 0}
    urls = candidate_urls(source)
    if not urls:
        result["detail"] = "oai_base_url kosong (isi dulu di config/journals.csv)"
        return result
    errors = []
    for url in urls:
        kw = {"sleep": sleep} if sleep else {}
        client = OaiClient(url, fetch, delay=delay, **kw)
        try:
            ident = client.identify()
            if "oai_dc" not in client.metadata_formats():
                errors.append(f"{url}: tidak mendukung oai_dc")
                continue
            rows = []
            for rec in client.list_records(max_pages=1):
                row = parse_record(rec)
                if row:
                    rows.append(row)
                if len(rows) >= sample:
                    break
        except HarvestError as err:
            errors.append(f"{url}: {err}")
            continue
        n = len(rows)
        pct = lambda key: round(100 * sum(bool(r[key]) for r in rows) / n) if n else 0   # noqa: E731
        result.update(ok=n > 0, url=url, sample=n, pct_abstract=pct("Abstract"), pct_keywords=pct("Keywords"),
                      pct_doi=pct("DOI"), detail=f"repositori: {ident['repository_name'] or '?'}")
        if n == 0:
            result["detail"] += "; tidak ada record pada halaman pertama"
        return result
    result["detail"] = " | ".join(errors)
    return result


def main(argv=None, fetch=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/journals.csv")
    ap.add_argument("--sources", default="", help="source_id dipisah koma (default: semua)")
    ap.add_argument("--contact-email", default=os.environ.get("CONTACT_EMAIL", ""))
    ap.add_argument("--update-config", action="store_true")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args(argv)
    if fetch is None:
        if not args.contact_email:
            print("Berikan --contact-email (atau env CONTACT_EMAIL) agar penerbit dapat menghubungi Anda.", file=sys.stderr)
            return 2
        fetch = make_fetcher(f"CapstoneIR/1.0 (mailto:{args.contact_email})")
    sources = load_sources(args.config)
    wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
    results = []
    for s in sources:
        if wanted and s["source_id"] not in wanted:
            continue
        r = check_source(s, fetch, delay=args.delay)
        results.append(r)
        flag = "OK  " if r["ok"] else "GAGAL"
        print(f"[{flag}] {r['source_id']:8s} sampel={r['sample']:<3d} abstrak={r['pct_abstract']}% "
              f"kata_kunci={r['pct_keywords']}% doi={r['pct_doi']}%  {r['url']}\n        {r['detail']}")
        if r["ok"] and args.update_config:
            s.update(oai_base_url=r["url"], verification="verified", verified_on=date.today().isoformat(), enabled=True)
    if args.update_config:
        save_sources(sources, args.config)
        print(f"Konfigurasi diperbarui: {args.config}")
    return 0 if results and all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
