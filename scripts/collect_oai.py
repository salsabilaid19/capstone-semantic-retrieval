"""Panen metadata artikel dari jurnal OJS via OAI-PMH sesuai config/journals.csv.

Contoh:  python -m scripts.collect_oai --contact-email anda@kampus.ac.id --max-per-source 150
Hasil: data/real/harvest/<source_id>.csv (data_source=real). Hanya sumber enabled=true yang dipanen
(kecuali --sources menyebutkannya). Hanya metadata; periksa lisensi tiap jurnal di config/ dan data/DATA_SOURCES.md.
"""
import argparse
import os
import sys
from pathlib import Path

from src.harvest import (HarvestError, OaiClient, candidate_urls, harvest_source, load_sources, load_topic_terms,
                         make_fetcher, write_harvest_csv)


def main(argv=None, fetch=None, sleep=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/journals.csv")
    ap.add_argument("--sources", default="", help="source_id dipisah koma (default: semua yang enabled)")
    ap.add_argument("--out-dir", default="data/real/harvest")
    ap.add_argument("--from-date", default=None, help="YYYY-MM-DD; hanya record yang diubah sejak tanggal ini")
    ap.add_argument("--max-per-source", type=int, default=300, help="batas artikel per jurnal setelah filter topik")
    ap.add_argument("--max-pages", type=int, default=None, help="batas halaman OAI per jurnal (uji coba)")
    ap.add_argument("--topic-terms", default="config/topic_terms.txt")
    ap.add_argument("--no-topic-filter", action="store_true")
    ap.add_argument("--delay", type=float, default=1.0, help="jeda detik antar permintaan")
    ap.add_argument("--contact-email", default=os.environ.get("CONTACT_EMAIL", ""))
    args = ap.parse_args(argv)
    if fetch is None:
        if not args.contact_email:
            print("Berikan --contact-email (atau env CONTACT_EMAIL) agar penerbit dapat menghubungi Anda.", file=sys.stderr)
            return 2
        fetch = make_fetcher(f"CapstoneIR/1.0 (mailto:{args.contact_email})")
    terms = [] if args.no_topic_filter else load_topic_terms(args.topic_terms)
    wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
    todo = [s for s in load_sources(args.config) if (s["source_id"] in wanted if wanted else s["enabled"])]
    if not todo:
        print("Tidak ada sumber yang dipanen (periksa kolom enabled di config atau opsi --sources).", file=sys.stderr)
        return 2
    failed = 0
    for s in todo:
        kw = {"sleep": sleep} if sleep else {}
        client, last_err = None, ""
        for url in candidate_urls(s):                     # coba URL utama lalu alternatif
            c = OaiClient(url, fetch, delay=args.delay, **kw)
            try:
                c.identify()
                client = c
                break
            except HarvestError as err:
                last_err = f"{url}: {err}"
        if client is None:
            print(f"[GAGAL] {s['source_id']}: endpoint tidak dapat diakses. {last_err or 'oai_base_url kosong'}")
            failed += 1
            continue
        res = harvest_source(s, client, from_date=args.from_date, max_records=args.max_per_source,
                             terms=terms, max_pages=args.max_pages)
        if res.rows:
            write_harvest_csv(res.rows, Path(args.out_dir) / f"{s['source_id']}.csv")
        st = res.stats
        status = "GAGAL-PARSIAL" if res.error else "OK"
        print(f"[{status}] {s['source_id']}: dilihat={st['seen']} dipakai={st['kept']} dihapus/kosong={st['deleted_or_empty']} "
              f"di-luar-topik={st['off_topic']}" + (f" | {res.error}" if res.error else ""))
        failed += bool(res.error)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
