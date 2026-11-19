"""Validasi dataset yang sudah diproses.  Contoh: python -m scripts.validate_dataset --artifacts-dir artifacts_real --kind real"""
import argparse
import sys
from pathlib import Path

from src.retrieval import load_documents
from src.validation import report_markdown, validate_dataset


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--kind", default="", choices=["", "real", "synthetic"])
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)
    try:
        df = load_documents(args.artifacts_dir)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 2
    rep = validate_dataset(df, expected_kind=args.kind)
    md = report_markdown(args.artifacts_dir, rep)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f"validation_{Path(args.artifacts_dir).name}.md").write_text(md, encoding="utf-8")
    print(md)
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
