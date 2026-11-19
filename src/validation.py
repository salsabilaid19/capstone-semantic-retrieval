"""Validasi dataset akhir (hasil clean_dataset): kualitas metadata, keterverifikasian, dan pemisahan real/sintetis."""
import re
from datetime import date
from typing import Dict, List

import pandas as pd

from src.preprocessing import normalize_title

DOI_VALID = re.compile(r"^10\.\d{4,9}/\S+$")
SYNTHETIC_DOI_PREFIXES = ("10.1000/sample", "10.99999/fixture")
TARGET_MIN, TARGET_MAX = 200, 1000   # rentang wajar; dataset penelitian ini 900 dokumen
MIN_ABSTRACT_CHARS = 200   # ketentuan penelitian: abstrak wajib minimal 200 karakter


def validate_dataset(df: pd.DataFrame, expected_kind: str = "") -> dict:
    """Kembalikan {'errors': [...], 'warnings': [...], 'stats': {...}}.
    errors = harus diperbaiki sebelum dipakai untuk evaluasi/publikasi; warnings = perlu diperhatikan."""
    errors: List[str] = []
    warnings: List[str] = []
    n = len(df)
    stats: Dict[str, object] = {"jumlah_dokumen": n}
    if n == 0:
        return {"errors": ["Dataset kosong."], "warnings": [], "stats": stats}

    src = df["data_source"].value_counts().to_dict()
    stats["data_source"] = src
    if len(src) > 1:
        errors.append(f"Dataset mencampur data_source berbeda {src}. Pisahkan data real dan sintetis.")
    if expected_kind and set(src) != {expected_kind}:
        errors.append(f"Dataset diharapkan '{expected_kind}' tetapi berisi {src}.")
    is_real = "real" in src

    for col, label in (("doi", "DOI"), ("source_url", "URL"), ("keywords", "kata kunci"),
                       ("journal", "nama jurnal"), ("author", "penulis"), ("year", "tahun")):
        has = (df[col].astype(str).str.strip() != "") & (df[col].astype(str) != "<NA>")
        stats[f"persen_{col}"] = round(float(has.mean()) * 100, 1)
    if stats["persen_keywords"] < 50:
        warnings.append(f"Hanya {stats['persen_keywords']}% dokumen punya kata kunci: Static/Adaptive QE akan jarang melakukan "
                        "ekspansi pada dataset ini.")
    if stats["persen_journal"] < 100:
        warnings.append("Sebagian dokumen tanpa nama jurnal (kolom yang ditampilkan di hasil pencarian).")

    for col, msg in (("doc_id", "doc_id duplikat"), ("title", "judul identik")):
        k = df[col].map(normalize_title) if col == "title" else df[col]
        d = int(k.duplicated().sum())
        stats[f"duplikat_{col}"] = d
        if d:
            (errors if col == "doc_id" else warnings).append(f"{msg}: {d} baris.")
    doi_non_empty = df[df["doi"].astype(str).str.strip() != ""]
    dup_doi = int(doi_non_empty["doi"].duplicated().sum())
    stats["duplikat_doi"] = dup_doi
    if dup_doi:
        errors.append(f"DOI duplikat: {dup_doi} baris (deduplikasi gagal).")
    bad_doi = doi_non_empty[~doi_non_empty["doi"].astype(str).str.match(DOI_VALID)]
    if len(bad_doi):
        warnings.append(f"{len(bad_doi)} DOI berformat tidak lazim, contoh: {bad_doi['doi'].iloc[0]}")

    years = pd.to_numeric(df["year"], errors="coerce")
    stats["rentang_tahun"] = [int(years.min()), int(years.max())] if years.notna().any() else []
    out_of_range = years[(years < 1990) | (years > date.today().year + 1)]
    if len(out_of_range):
        warnings.append(f"{len(out_of_range)} dokumen bertahun di luar 1990..{date.today().year + 1}.")

    alen = df["abstract"].astype(str).str.len()
    stats["panjang_abstrak_median"] = int(alen.median())
    short = int((alen < 80).sum())
    if short:
        warnings.append(f"{short} abstrak sangat pendek (<80 karakter); periksa apakah terpotong atau placeholder.")
    too_short = df.loc[alen < MIN_ABSTRACT_CHARS, "doc_id"].astype(str).tolist()
    stats["abstrak_kurang_dari_200"] = len(too_short)
    stats["doc_id_abstrak_kurang_dari_200"] = too_short
    if too_short:
        msg = (f"{len(too_short)} dokumen berabstrak < {MIN_ABSTRACT_CHARS} karakter (melanggar ketentuan penelitian): "
               f"{', '.join(too_short[:5])}{' ...' if len(too_short) > 5 else ''}. Periksa apakah bukan artikel "
               "(mis. sampul/daftar isi) atau abstrak terpotong.")
        (errors if is_real else warnings).append(msg)
    if int((df["title"].astype(str).str.strip() == df["abstract"].astype(str).str.strip()).sum()):
        warnings.append("Ada dokumen yang judulnya sama persis dengan abstraknya.")

    lang = df["language"].value_counts().to_dict()
    stats["bahasa"] = lang
    stats["per_jurnal"] = df["journal"].replace("", "(kosong)").value_counts().to_dict()
    if is_real and lang.get("id", 0) / n < 0.5:
        warnings.append("Kurang dari 50% dokumen terdeteksi berbahasa Indonesia (heuristik). Pastikan sesuai tujuan dataset Indonesia.")

    synth_doi = int(df["doi"].astype(str).str.startswith(SYNTHETIC_DOI_PREFIXES).sum())
    if is_real:
        if synth_doi:
            errors.append(f"{synth_doi} dokumen bertanda 'real' memakai DOI contoh/fixture (sintetis).")
        unverifiable = df[(df["doi"].astype(str).str.strip() == "") & (df["source_url"].astype(str).str.strip() == "")]
        if len(unverifiable):
            errors.append(f"{len(unverifiable)} dokumen 'real' tanpa DOI maupun URL, sehingga tidak dapat diverifikasi.")
        if (df["source_note"].astype(str).str.strip() == "").any():
            errors.append("Ada dokumen 'real' tanpa source_note (asal data/ketentuan penggunaan tidak tercatat).")
        if not (TARGET_MIN <= n <= TARGET_MAX):
            warnings.append(f"Jumlah dokumen {n} di luar target awal {TARGET_MIN}-{TARGET_MAX}.")
    elif "synthetic" in src:
        warnings.append("Dataset ini SINTETIS: hanya untuk uji coba aplikasi; jangan dipakai untuk klaim hasil penelitian.")
    if "unspecified" in src:
        warnings.append("Ada dokumen dengan data_source 'unspecified'; tandai sebagai real atau synthetic.")
    return {"errors": errors, "warnings": warnings, "stats": stats}


def report_markdown(name: str, rep: dict) -> str:
    lines = [f"# Laporan Validasi Dataset: {name}", "",
             f"Status: **{'GAGAL' if rep['errors'] else 'LOLOS'}** ({len(rep['errors'])} error, {len(rep['warnings'])} peringatan)", ""]
    for title, items in (("Error (harus diperbaiki)", rep["errors"]), ("Peringatan", rep["warnings"])):
        if items:
            lines += [f"## {title}"] + [f"- {i}" for i in items] + [""]
    lines += ["## Statistik"] + [f"- {k}: {v}" for k, v in rep["stats"].items()]
    return "\n".join(lines) + "\n"
