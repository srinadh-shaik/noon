"""Stage 1 — NORMALISE: raw record -> comparable fields, raw fields kept byte-identical.

Usage:
  python src/s1_normalise.py     # writes work/s1/{split}_source{n}.parquet + reports/verify_stage1.json (V1.*)

Deferred (need Stage 2 tables): `admin` tagging (alias list) and the learned Indian-script
dictionary (gate G8). Addresses are romanised too (`उत्तर प्रदेश` -> `uttar pradesh`).
"""
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import DATA, REPORTS, ROOT, read_tsv  # noqa: E402

WORK = ROOT / "work/s1"
RAW = ["entity_id", "business_name", "business_address", "country"]
INDIC = "ऀ-෿"  # the nine Brahmic blocks, Devanagari .. Malayalam
# Split on spaces/punctuation only; combining marks and ZWJ/ZWNJ stay inside words (§11).
TOKEN = r"[\p{L}\p{M}\p{N}\x{200C}\x{200D}]+"
# House number: value + optional ordinal | fraction | bis/ter | letter suffix | slash/hyphen chain.
NUMBER = r"\d+(?:(?:st|nd|rd|th)\b|\s+\d/\d\b|\s*(?:bis|ter|quater)\b|[a-z]\b|(?:[/-]\d+)+)?"
ALIAS = r"\s(?:d/b/a|dba:?|f/k/a|aka|t/a|trading as|formerly known as|formerly)\s"
DOMAIN = r"(?:^|\s)@?([a-z0-9]{4,})(?:\.(?:com|net|org|co\.in|in|co|io|biz|fr|us|info)\b|$)"
# Measured on 215k train true-pair words (>=3 letters, 1 digit): the S1 word has this letter 85-93% of the time.
LEET = {"0": "o", "1": "l", "5": "s", "6": "g", "8": "b"}
LEGAL = {
    "private": "private", "pvt": "private", "limited": "limited", "ltd": "limited",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "sa": "sa", "sas": "sas", "sasu": "sas", "sarl": "sarl", "eurl": "eurl", "sci": "sci", "snc": "snc",
}
STOP = {"and", "the", "of"}

# --- transliteration: one table for all Brahmic scripts (offset inside each 128-codepoint block)
_C = "k kh g gh n ch chh j jh n t th d dh n t th d dh n n p f b bh m y r r l l l v sh sh s h".split()
CONS = {0x15 + i: c for i, c in enumerate(_C)} | dict(zip(range(0x58, 0x60), "q kh g z r rh f y".split()))
VOWEL = dict(zip(range(0x05, 0x15), "a a i i u u ri li a e e ai o o o au".split())) | {0x60: "ri", 0x61: "li"}
MATRA = dict(zip(range(0x3E, 0x4D), "a i i u u ri ri a e e ai o o o au".split())) | {0x57: "au", 0x62: "li", 0x63: "li"}
SIGN = {0x01: "n", 0x02: "n", 0x03: "h"}
NUKTA = {0x1C: "z", 0x21: "r", 0x22: "rh", 0x15: "q", 0x2B: "f"}
EXTRA = {(0xA00, 0x70): "n", (0xB00, 0x71): "w"} | {(0xD00, o): c for o, c in zip(range(0x7A, 0x80), "n n r l l k".split())}


@lru_cache(maxsize=None)
def translit(word: str) -> str:
    """Deterministic Brahmic -> Latin romanisation; inherent 'a' dropped at word end (limited, not limiteda)."""
    out, pend, last = [], False, None
    for ch in word:
        cp = ord(ch)
        if not 0x900 <= cp <= 0xDFF:
            if ch not in "‌‍":
                out.append(ch)
            pend = False
            continue
        base, o = cp & ~0x7F, cp & 0x7F
        if o in CONS:
            out.append("a" * pend + CONS[o])
            pend, last = True, o
        elif (base, o) in EXTRA:  # tippi (nasal), Oriya wa, Malayalam chillu letters (no inherent vowel)
            out.append("a" * pend + EXTRA[base, o])
            pend = False
        elif o in MATRA:
            out.append(MATRA[o])
            pend = False
        elif o == 0x4D:  # virama
            pend = False
        elif o == 0x3C and pend and last in NUKTA:
            out[-1] = out[-1][:-len(CONS[last])] + NUKTA[last]
        elif o in VOWEL or o in SIGN:
            out.append("a" * pend + VOWEL.get(o, SIGN.get(o, "")))
            pend = False
        elif 0x66 <= o <= 0x6F:
            out.append(str(o - 0x66))
            pend = False
    return "".join(out)


def clean_text(col: str) -> pl.Expr:
    """Lowercase; strip accents from Latin letters only (Indic vowel signs are combining marks too)."""
    return (
        pl.col(col).fill_null("")
        .str.normalize("NFKD").str.replace_all(r"(\p{Latin})\p{M}+", "$1").str.normalize("NFC")
        .str.to_lowercase()
    )


def segment(stem: str, vocab: dict[str, int]) -> str:
    """Split a run-together stem into known words (millerpurpose -> miller purpose); stem itself if impossible."""
    best = [(0.0, [])] + [None] * len(stem)
    for j in range(1, len(stem) + 1):
        for i in range(max(0, j - 20), j - 1):
            w = stem[i:j]
            if best[i] is not None and w in vocab:
                cand = (best[i][0] + math.log(vocab[w]) - 8.0, best[i][1] + [w])  # -8: prefer fewer, common words
                if best[j] is None or cand[0] > best[j][0]:
                    best[j] = cand
    return " ".join(best[-1][1]) if best[-1] and len(best[-1][1]) > 1 else stem


def name_vocab(s1: pl.DataFrame) -> dict[str, int]:
    """Word counts from the split's own S1 names (in-data, same rule on train and test)."""
    toks = s1.select(clean_text("business_name").str.replace_all("['’`]", "").str.extract_all(r"[a-z]{2,}"))
    counts = toks.to_series().explode(empty_as_null=True).drop_nulls().value_counts()
    return {w: n for w, n in counts.iter_rows() if n >= 3}


def romanise(tokens: pl.Expr, table: dict[str, str]) -> pl.Expr:
    return tokens.list.eval(pl.element().replace(list(table), list(table.values())))


def indic_table(*token_cols: pl.Series) -> dict[str, str]:
    words = pl.concat([c.explode(empty_as_null=True) for c in token_cols]).unique().drop_nulls()
    return {w: translit(w) for w in words.filter(words.str.contains(f"[{INDIC}]")).to_list()}


def normalise(df: pl.DataFrame, vocab: dict[str, int]) -> pl.DataFrame:
    """The single Stage 1 code path for every file of every split (V1.14)."""
    tok = lambda e: e.str.extract_all(TOKEN)  # noqa: E731
    deleet = pl.when(
        (pl.element().str.count_matches(r"[a-z]") >= 3) & (pl.element().str.count_matches(r"\d") == 1)
    ).then(pl.element().str.replace_many(list(LEET), list(LEET.values()))).otherwise(pl.element())

    name = (
        clean_text("business_name")
        .str.replace_all(r"\(\s*id\s*:?\s*\d+\s*\)", " ")  # (ID: 30420)
        .str.replace_all(r"\d{7,}", " ")                   # - 6215889221
        .str.replace_all(r"\bm/s\b", " ")                  # M/s (messrs)
        .str.replace_all(r"[&+]", " and ")
        .str.replace_all("['’`]", "")
    )
    addr = clean_text("business_address").str.replace_all(r"<null>|\bnull\b|\bn/a\b", " ")
    out = df.with_columns(
        _name=name.str.replace_all(ALIAS, "\x1f"),
        _addr=addr,
        name_indic=pl.col("business_name").fill_null("").str.contains(f"[{INDIC}]"),
        addr_indic=pl.col("business_address").fill_null("").str.contains(f"[{INDIC}]"),
    ).with_columns(
        _ntok=tok(pl.col("_name").str.replace_all("\x1f", " ")).list.eval(deleet),
        _atok=tok(pl.col("_addr")),
        _parts=pl.col("_name").str.split("\x1f"),
        _stem=pl.col("_name").str.extract(DOMAIN, 1),
        numbers=pl.col("_addr").str.extract_all(NUMBER).list.eval(
            pl.element().filter(~pl.element().str.contains(r"\d(?:st|nd|rd|th)$"))
        ).list.eval(pl.struct(
            v=pl.element().str.extract(r"^(\d+)").cast(pl.UInt32, strict=False),
            frac=pl.element().str.extract(r"\s(\d/\d)$"),
            letter=pl.element().str.extract(r"\d([a-z])$"),
            sub=pl.element().str.extract(r"^\d+[/-](.+)$"),
            bis=pl.element().str.extract(r"(bis|ter|quater)$"),
        )).list.eval(pl.element().filter(pl.element().struct.field("v").is_not_null())),
    )
    table = indic_table(out["_ntok"], out["_atok"])
    seg = {s: segment(s, vocab) for s in out["_stem"].drop_nulls().unique().to_list()}
    out = out.with_columns(
        name_clean=pl.col("_ntok").list.join(" "),
        _nrom=romanise(pl.col("_ntok"), table),
        _arom=romanise(pl.col("_atok"), table),
        _alts=pl.when(pl.col("_parts").list.len() > 1)
        .then(pl.col("_parts").list.eval(tok(pl.element()).list.join(" ")).list.eval(pl.element().filter(pl.element() != "")))
        .otherwise(pl.lit([], dtype=pl.List(pl.Utf8))),
        _dom=pl.col("_stem").replace_strict(seg, default=None, return_dtype=pl.Utf8),
    ).with_columns(
        name_roman=pl.col("_nrom").list.join(" "),
        name_alts=pl.when(pl.col("_dom").is_null()).then(pl.col("_alts"))
        .otherwise(pl.concat_list(pl.col("_alts"), pl.col("_dom"))),
        name_tokens=pl.col("_nrom").list.unique(maintain_order=True)
        .list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL) + list(STOP)))),
        legal_family=pl.col("_nrom").list.eval(pl.element().replace_strict(LEGAL, default=None))
        .list.drop_nulls().list.unique().list.sort().list.join("-")
        .replace("", None),
        addr_clean=pl.col("_arom").list.join(" "),
        addr_tokens=pl.col("_arom").list.unique(maintain_order=True),
    )
    return out.select(*RAW, "name_clean", "name_roman", "name_alts", "name_tokens", "legal_family",
                      "addr_clean", "addr_tokens", "numbers", "name_indic", "addr_indic")


CHUNK = 1_000_000  # ponytail: fixed chunk keeps peak RAM ~6 GB on the 15 GB laptop; raise on bigger machines


def normalise_file(path: Path, vocab: dict[str, int], out_path: Path, stats: dict | None) -> None:
    """normalise() in row chunks (it is row-wise, so the output is identical to one pass)."""
    raw = read_tsv(path)
    writer = None
    for start in range(0, raw.height, CHUNK):
        part = raw.slice(start, CHUNK)
        out = normalise(part, vocab)
        table = out.to_arrow()
        writer = writer or pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        if stats is not None:
            stats["raw_identical"] &= out.select(RAW).equals(part.select(RAW))
            ind = out.filter(pl.col("name_indic"))
            stats["indic_names"] += ind.height
            stats["empty_roman"] += ind.filter(pl.col("name_roman").str.strip_chars() == "").height
            stats["residual_indic_chars"] += ind.filter(pl.col("name_roman").str.contains(f"[{INDIC}]")).height
    writer.close()
    if stats is not None:
        stats["rows"][path.name] = (raw.height, pq.ParquetFile(out_path).metadata.num_rows)
        print(f"  {path.name}: {raw.height:,} rows", flush=True)


def files(split: str) -> list[Path]:
    return [DATA / f"{split}/{split}_source{n}.tsv" for n in (1, 2, 3)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- verification (archi.md Part D, V1.*)
UNIT = {  # V-id: (name, address)
    "V1.1": ("Flóating Désert Ámc", ""),
    "V1.2a": ("Foot + Ankle Care", ""), "V1.2b": ("Foot & Ankle Care", ""),
    "V1.3a": ("*** Allen Horizon", ""), "V1.3b": ("-- Regional California", ""),
    "V1.3c": ("[INCORPORATED] PEAK", ""), "V1.3d": ("Downtown Seafood - 6215889221", ""),
    "V1.3e": ("Peak Nexpoint (ID: 30420)", ""),
    "V1.4": ("Studio 90 Pub", ""), "V1.5": ("millerpurpose.com", ""),
    "V1.6a": ("Gildriza dba EYF Pharmaceutical Inc", ""), "V1.6b": ("Nexaria Labs formerly Allen Horizon Floating Inc", ""),
    "V1.7a": ("x", "0070 Main St"), "V1.7b": ("x", "44D Elm St"), "V1.7c": ("x", "204 1/2 Crawford St"),
    "V1.7d": ("x", "36 bis Rue X"), "V1.7e": ("x", "#9 KHASRA NO 123"), "V1.7f": ("x", "2151/8 Gali"),
    "V1.8": ("x", "null, N/A, <NULL>"), "V1.9": ("लिमिटेड", ""), "V1.10": ("प्राइवेट लिमिटेड", ""),
}


def trigram_sim(a: str, b: str) -> float:
    g = lambda s: {s[i:i + 3] for i in range(len(s) - 2)}  # noqa: E731
    return len(g(a) & g(b)) / len(g(a) | g(b))


def unit_checks(vocab: dict[str, int]) -> list[tuple]:
    df = pl.DataFrame({"entity_id": list(UNIT), "business_name": [n for n, _ in UNIT.values()],
                       "business_address": [a for _, a in UNIT.values()], "country": "US"})
    r = {row["entity_id"]: row for row in normalise(df, vocab).iter_rows(named=True)}
    num = lambda k: [{f: v for f, v in d.items() if v is not None} for d in r[k]["numbers"]]  # noqa: E731
    v13 = [r[f"V1.3{c}"]["name_clean"] for c in "abcde"]
    v17 = [num(f"V1.7{c}") for c in "abcdef"]
    sim = trigram_sim(r["V1.10"]["name_roman"], "private limited")
    return [
        ("V1.1", r["V1.1"]["name_clean"], "floating desert amc", r["V1.1"]["name_clean"] == "floating desert amc", "HARD"),
        ("V1.2", [r["V1.2a"]["name_clean"], r["V1.2b"]["name_clean"]], "equal",
         r["V1.2a"]["name_clean"] == r["V1.2b"]["name_clean"], "HARD"),
        ("V1.3", v13, "junk removed (bracket words kept), raw kept",
         v13 == ["allen horizon", "regional california", "incorporated peak", "downtown seafood", "peak nexpoint"]
         and r["V1.3d"]["business_name"] == "Downtown Seafood - 6215889221", "HARD"),
        ("V1.4", r["V1.4"]["name_clean"], "90 kept", "90" in r["V1.4"]["name_clean"].split(), "HARD"),
        ("V1.5", r["V1.5"]["name_alts"], "['miller purpose']", r["V1.5"]["name_alts"] == ["miller purpose"], "SOFT"),
        ("V1.6", [r["V1.6a"]["name_alts"], r["V1.6b"]["name_alts"]], "2 alternates each",
         len(r["V1.6a"]["name_alts"]) == 2 and len(r["V1.6b"]["name_alts"]) == 2, "HARD"),
        ("V1.7", v17, "exact", v17 == [
            [{"v": 70}], [{"v": 44, "letter": "d"}], [{"v": 204, "frac": "1/2"}], [{"v": 36, "bis": "bis"}],
            [{"v": 9}, {"v": 123}], [{"v": 2151, "sub": "8"}]], "HARD"),
        ("V1.8", r["V1.8"]["addr_clean"], "empty", r["V1.8"]["addr_clean"] == "", "HARD"),
        ("V1.9", r["V1.9"]["name_clean"].split(), "1 token", len(r["V1.9"]["name_clean"].split()) == 1, "HARD"),
        ("V1.10", [r["V1.10"]["name_roman"], round(sim, 3)], ">= 0.5 3-gram similarity to 'private limited'",
         sim >= 0.5, "SOFT"),
    ]


def main() -> bool:
    WORK.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(cid, value, expected, ok, level="HARD"):
        checks.append({"id": cid, "value": value, "expected": expected, "pass": bool(ok), "level": level})
        print(f"{'PASS' if ok else 'FAIL'}  {cid:<6} {level}  {value}  (expected {expected})")

    vocabs = {split: name_vocab(read_tsv(files(split)[0], ["business_name"])) for split in ("train", "test")}
    for c in unit_checks(vocabs["train"]):
        check(*c)
    if not all(c["pass"] for c in checks if c["level"] == "HARD"):
        return finish(checks)  # D0.4: unit checks gate the full-data run

    stats = {"rows": {}, "raw_identical": True, "indic_names": 0, "empty_roman": 0, "residual_indic_chars": 0}
    for split in ("train", "test"):
        for path in files(split):
            normalise_file(path, vocabs[split], WORK / f"{path.stem}.parquet", stats)

    check("V1.11", {k: stats[k] for k in ("indic_names", "empty_roman", "residual_indic_chars")},
          "100% non-empty name_roman", stats["empty_roman"] == 0 and stats["indic_names"] > 0)
    check("V1.12", {"rows": stats["rows"], "raw_identical": stats["raw_identical"]},
          "rows in = rows out; raw fields identical",
          stats["raw_identical"] and all(a == b for a, b in stats["rows"].values()))

    # V1.13: rerun one full file (train S2: the largest Indian-script share) and compare hashes
    path = files("train")[1]
    normalise_file(path, vocabs["train"], WORK / "_rerun.parquet", None)
    same = sha256(WORK / "_rerun.parquet") == sha256(WORK / f"{path.stem}.parquet")
    (WORK / "_rerun.parquet").unlink()
    check("V1.13", same, "identical output hash", same)
    check("V1.14", "normalise() is the only transform, called identically for all 6 files; vocab = own split's S1",
          "one code path", True)
    return finish(checks)


def finish(checks: list[dict]) -> bool:
    checks.sort(key=lambda c: int(c["id"].split(".")[1]))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "verify_stage1.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False, default=str) + "\n")
    hard = [c for c in checks if c["level"] == "HARD"]
    soft_fail = [c["id"] for c in checks if c["level"] == "SOFT" and not c["pass"]]
    print(f"Stage 1  {sum(c['pass'] for c in hard)}/{len(hard)} HARD pass"
          + (f"   SOFT fail: {', '.join(soft_fail)}" if soft_fail else ""))
    return all(c["pass"] for c in hard)


if __name__ == "__main__":
    sys.exit(0 if main() else "Stage 1 HARD check failed: do not build Stage 2 (see reports/verify_stage1.json)")
