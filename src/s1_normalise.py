"""Stage 1 — NORMALISE: raw record -> comparable fields, raw fields kept byte-identical.

Usage:
  python src/s1_normalise.py     # writes work/s1/{split}_source{n}.parquet + reports/verify_stage1.json (V1.*)

General version: no per-language tables or word lists. Any non-Latin letter is romanised
from its Unicode character name; legal-form and stop-word weighting move to Stage 2 (learned
from pairs / per-split rarity), so this stage drops no words.

Deferred (need Stage 2 tables): `admin` tagging (alias list) and the learned script
dictionary (gate G8). Addresses are romanised too (`उत्तर प्रदेश` -> `uttar pradesh`).
"""
import hashlib
import json
import math
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import DATA, REPORTS, ROOT, read_tsv  # noqa: E402

WORK = ROOT / "work/s1"
RAW = ["entity_id", "business_name", "business_address", "country"]
NONLATIN = r"[\p{L}&&\P{Latin}]"  # any letter outside the Latin script (no script list)
# Split on spaces/punctuation only; combining marks and ZWJ/ZWNJ stay inside words (§11).
TOKEN = r"[\p{L}\p{M}\p{N}\x{200C}\x{200D}]+"
# House number: value + optional fraction | bis/ter | glued letters | slash/hyphen chain.
# Digits glued to 2-3 letters (1er, 2nd, 213th, 2eme, 40ft) are ordinals/units, not house numbers
# (bis/ter/quater excepted); >=4 glued letters is a missing space (74SECTOR, 905NEW), so the number is kept.
# Chosen on 130k train true pairs: ties the old rule on identical-number rate, keeps more numbers.
NUMBER = r"\d+(?:\s+\d/\d\b|\s*(?:bis|ter|quater)\b|[a-z]{1,3}\b|(?:[/-]\d+)+)?"
ORDINAL = r"^\d+[a-z]{2,3}$"
ALIAS = r"\s(?:d/b/a|dba:?|f/k/a|aka|t/a|trading as|formerly known as|formerly)\s"
# A web domain needs a real dot-TLD, or an explicit @handle; a bare last word is neither
# (it was: 56% of S1 names got junk alts).
DOMAIN = r"(?:^|\s)@?(?:www\.)?([a-z0-9]{4,})\.[a-z]{2,6}\b"
HANDLE = r"(?:^|\s)@([a-z0-9_]{4,})"
# Measured on 215k train true-pair words (>=3 letters, 1 digit): the S1 word has this letter 85-93% of the time.
LEET = {"0": "o", "1": "l", "5": "s", "6": "g", "8": "b"}
_VOWEL_NAMES = {"A", "AA", "I", "II", "U", "UU", "E", "EE", "AI", "O", "OO", "AU"}


def _piece(name: str, key: str) -> str:
    """'DEVANAGARI LETTER TTA' -> 'ta': last word of the name, repeated letters collapsed
    (Unicode spells retroflex/long sounds by doubling: TTA, NNA, AA, II)."""
    return re.sub(r"(.)\1+", r"\1", name.split(key, 1)[1].split()[-1].lower())


@lru_cache(maxsize=None)
def translit(word: str) -> str:
    """Generic romanisation of any non-Latin letters from Unicode character names only.

    A consonant letter carries an inherent 'a'; a following vowel sign replaces it, a virama
    removes it, and it is dropped at word end (limited, not limiteda).
    ponytail: abugida-shaped (every script in this data, DATA_NOTES §6). Alphabets such as
    Greek/Cyrillic would romanise from letter *names* (alpha, zhe); add a transliteration
    library if such scripts ever appear.
    """
    out, cons = [], False
    for ch in word:
        name = unicodedata.name(ch, "")
        if ch.isascii() or "LATIN" in name:
            out.append(ch)
            cons = False
        elif unicodedata.category(ch) == "Nd":
            out.append(str(unicodedata.digit(ch)))
            cons = False
        elif " LETTER " in name:
            p = _piece(name, " LETTER ")
            out.append(p)
            cons = len(p) > 1 and p.endswith("a") and name.split()[-1] not in _VOWEL_NAMES
        elif "VOWEL SIGN" in name:
            if cons:
                out[-1] = out[-1][:-1]
            out.append(_piece(name, "VOWEL SIGN"))
            cons = False
        elif "VIRAMA" in name:
            if cons:
                out[-1] = out[-1][:-1]
            cons = False
        elif any(k in name for k in ("ANUSVARA", "CANDRABINDU", "TIPPI", "BINDI")):
            out.append("n")
            cons = False
        elif "VISARGA" in name:
            out.append("h")
            cons = False
        # other marks (nukta, ZWJ/ZWNJ, gemination signs) carry no letter of their own
    if cons:
        out[-1] = out[-1][:-1]
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


def nonlatin_table(*token_cols: pl.Series) -> dict[str, str]:
    words = pl.concat([c.explode(empty_as_null=True) for c in token_cols]).unique().drop_nulls()
    return {w: translit(w) for w in words.filter(words.str.contains(NONLATIN)).to_list()}


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
        .str.replace_all(r"[&+]", " and ")
        .str.replace_all("['’`]", "")
    )
    addr = clean_text("business_address").str.replace_all(r"<null>|\bnull\b|\bn/a\b", " ")
    out = df.with_columns(
        _name=name.str.replace_all(ALIAS, "\x1f"),
        _addr=addr,
        name_nonlatin=pl.col("business_name").fill_null("").str.contains(NONLATIN),
        addr_nonlatin=pl.col("business_address").fill_null("").str.contains(NONLATIN),
    ).with_columns(
        _ntok=tok(pl.col("_name").str.replace_all("\x1f", " ")).list.eval(deleet),
        _atok=tok(pl.col("_addr")),
        _parts=pl.col("_name").str.split("\x1f"),
        _stem=pl.coalesce(
            pl.col("_name").str.extract(DOMAIN, 1),
            pl.col("_name").str.extract(HANDLE, 1).str.replace_all("_", " "),  # "_" is an explicit word break
        ),
        numbers=pl.col("_addr").str.extract_all(NUMBER).list.eval(
            pl.element().filter(
                ~pl.element().str.contains(ORDINAL) | pl.element().str.contains(r"(?:bis|ter|quater)$")
            )
        ).list.eval(pl.struct(
            v=pl.element().str.extract(r"^(\d+)").cast(pl.UInt32, strict=False),
            frac=pl.element().str.extract(r"\s(\d/\d)$"),
            letter=pl.element().str.extract(r"\d([a-z])$"),
            sub=pl.element().str.extract(r"^\d+[/-](.+)$"),
            bis=pl.element().str.extract(r"(bis|ter|quater)$"),
        )).list.eval(pl.element().filter(pl.element().struct.field("v").is_not_null())),
    )
    table = nonlatin_table(out["_ntok"], out["_atok"])
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
        # all words kept: legal/stop-word weighting is learned in Stage 2, not hand-listed here
        name_tokens=pl.col("_nrom").list.unique(maintain_order=True),
        addr_clean=pl.col("_arom").list.join(" "),
        addr_tokens=pl.col("_arom").list.unique(maintain_order=True),
    )
    return out.select(*RAW, "name_clean", "name_roman", "name_alts", "name_tokens",
                      "addr_clean", "addr_tokens", "numbers", "name_nonlatin", "addr_nonlatin")


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
            nl = out.filter(pl.col("name_nonlatin"))
            stats["nonlatin_names"] += nl.height
            stats["empty_roman"] += nl.filter(pl.col("name_roman").str.strip_chars() == "").height
            stats["residual_nonlatin_chars"] += nl.filter(pl.col("name_roman").str.contains(NONLATIN)).height
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
    # new: ordinals are not house numbers (French 1er was parsed as house number 1)
    "V1.15a": ("x", "78 BD Albert 1er, Bordeaux"), "V1.15b": ("x", "2Nd Floor 12 Park St, 213th Drive"),
    "V1.15c": ("x", "8BIS CLOS GUSTAVE"), "V1.15d": ("x", "74SECTOR-33 DWARKA"),
    # new: a bare last word is not a domain (junk alts on 56% of S1 names)
    "V1.16a": ("Hernandez Pipeline", ""), "V1.16b": ("Greensboro Scholarship Fund", ""),
    "V1.16c": ("@midwestinterstate", ""), "V1.16d": ("Allen Horizon @sarsa_consultants", ""),
    # new: generic romanisation across scripts (Devanagari, Tamil, Gujarati, Bengali, Gurmukhi)
    "V1.17a": ("लिमिटेड", ""), "V1.17b": ("லிமிடெட்", ""), "V1.17c": ("ટેક્નોલોજીસ", ""),
    "V1.17d": ("প্রাইভেট", ""), "V1.17e": ("ਪ੍ਰਾਈਵੇਟ", ""),
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
        ("V1.15", [num(f"V1.15{c}") for c in "abcd"], "ordinals dropped, bis kept, missing-space number kept",
         [num(f"V1.15{c}") for c in "abcd"] == [[{"v": 78}], [{"v": 12}], [{"v": 8, "bis": "bis"}],
                                                [{"v": 74}, {"v": 33}]], "HARD"),
        ("V1.16", [r[f"V1.16{c}"]["name_alts"] for c in "abcd"],
         "bare last word: no alternate; @handle: one segmented alternate",
         r["V1.16a"]["name_alts"] == [] and r["V1.16b"]["name_alts"] == []
         and len(r["V1.16c"]["name_alts"]) == 1 and " " in r["V1.16c"]["name_alts"][0]
         and len(r["V1.16d"]["name_alts"]) == 1, "HARD"),
        ("V1.17", [r[f"V1.17{c}"]["name_roman"] for c in "abcde"],
         "limited, limitet, teknolojis, praibhet, praivet",
         [r[f"V1.17{c}"]["name_roman"] for c in "abcde"] == ["limited", "limitet", "teknolojis", "praibhet", "praivet"],
         "SOFT"),
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

    stats = {"rows": {}, "raw_identical": True, "nonlatin_names": 0, "empty_roman": 0, "residual_nonlatin_chars": 0}
    for split in ("train", "test"):
        for path in files(split):
            normalise_file(path, vocabs[split], WORK / f"{path.stem}.parquet", stats)

    check("V1.11", {k: stats[k] for k in ("nonlatin_names", "empty_roman", "residual_nonlatin_chars")},
          "100% non-empty name_roman, no non-Latin letters left",
          stats["empty_roman"] == 0 and stats["residual_nonlatin_chars"] == 0 and stats["nonlatin_names"] > 0)
    check("V1.12", {"rows": stats["rows"], "raw_identical": stats["raw_identical"]},
          "rows in = rows out; raw fields identical",
          stats["raw_identical"] and all(a == b for a, b in stats["rows"].values()))

    # V1.13: rerun one full file (train S2: the largest Indian-script share) and compare hashes
    path = files("train")[1]
    normalise_file(path, vocabs["train"], WORK / "_rerun.parquet", None)
    same = sha256(WORK / "_rerun.parquet") == sha256(WORK / f"{path.stem}.parquet")
    (WORK / "_rerun.parquet").unlink()
    check("V1.13", same, "identical output hash", same)
    # V1.14: behavioural, not a claim. The same rows normalised with train's vs test's vocabulary must
    # agree on every column except name_alts (domain segmentation is the only vocab-dependent step).
    sample = read_tsv(files("test")[1]).slice(0, 200_000)
    a, b = normalise(sample, vocabs["train"]), normalise(sample, vocabs["test"])
    other = [c for c in a.columns if c != "name_alts"]
    same_rest = a.select(other).equals(b.select(other))
    alts_diff = int((a["name_alts"] != b["name_alts"]).sum())
    check("V1.14", {"rows": sample.height, "non_alt_columns_identical": same_rest, "rows_alts_differ": alts_diff},
          "one code path: only name_alts may depend on the split", same_rest)
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
