"""Stage 0 — HARNESS: exact macro F0.5 scorer, entity-grouped folds, validation worlds.

Worlds (archi.md Stage 0), all relabelings of full train:
  A   all S1 are queries, all S2/S3 records are indexed.
  B   ~20% of S1 hidden from the queries; their copies stay indexed as decoys.
  Bp  ~34% of S1 hidden; ONE random copy of each stays as a one-off decoy, the rest are dropped.

Usage:
  python src/s0_harness.py build        # writes work/s0/{s1,rec}.parquet + reports/verify_stage0.json (V0.*)
  python src/s0_harness.py score matching_results.tsv [A|B|Bp]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/6ab10eb3b23ba_student_resource/student_resource/dataset"
WORK = ROOT / "work/s0"
REPORTS = ROOT / "reports"
N_FOLDS = 5
HIDE = {"B": 0.20, "Bp": 0.34}
SALT = {"fold": 1, "B": 2, "Bp": 3, "keep": 4}
WORLDS = ("A", "B", "Bp")


def uniform(ids: pl.Series, salt: int) -> np.ndarray:
    """Stable U[0,1) per id: splitmix64 over (numeric part, source digit, salt)."""
    num = ids.str.slice(3).cast(pl.UInt64).to_numpy()
    src = ids.str.slice(1, 1).cast(pl.UInt64).to_numpy()
    x = num * np.uint64(8) + src + np.uint64(salt * 0x9E3779B97F4A7C15 % 2**64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def build_worlds(s1: pl.DataFrame, rec: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """s1: [s1, country]; rec: [rec, country, owner (null = decoy)] -> tables with folds and world flags."""
    assert rec["rec"].is_unique().all(), "a record appears twice (one owner per record is assumed)"
    k = rec.drop_nulls("owner").group_by("owner").len("k").rename({"owner": "s1"})
    s1 = (
        s1.join(k, on="s1", how="left")
        .with_columns(
            pl.col("k").fill_null(0).cast(pl.UInt32),
            fold=pl.Series((uniform(s1["s1"], SALT["fold"]) * N_FOLDS).astype(np.int8)),
            hide_B=pl.Series(uniform(s1["s1"], SALT["B"]) < HIDE["B"]),
            hide_Bp=pl.Series(uniform(s1["s1"], SALT["Bp"]) < HIDE["Bp"]),
        )
    )
    rec = (
        rec.with_columns(
            src=pl.col("rec").str.slice(0, 2),
            u_fold=pl.Series((uniform(rec["rec"], SALT["fold"]) * N_FOLDS).astype(np.int8)),
            u_keep=pl.Series(uniform(rec["rec"], SALT["keep"])),
        )
        .join(s1.select(pl.col("s1").alias("owner"), "fold", "hide_Bp"), on="owner", how="left")
        .with_columns(
            # a record follows its owner's fold; decoys are spread by hash
            fold=pl.coalesce("fold", "u_fold"),
            # B': a hidden owner keeps exactly one copy (lowest u_keep) as a one-off decoy
            in_Bp=~pl.col("hide_Bp").fill_null(False)
            | (pl.col("u_keep") == pl.col("u_keep").min().over("owner")),
        )
        .drop("u_fold", "u_keep", "hide_Bp")
    )
    return s1, rec


def queries(s1: pl.DataFrame, world: str) -> pl.DataFrame:
    return s1 if world == "A" else s1.filter(~pl.col(f"hide_{world}"))


def indexed(rec: pl.DataFrame, world: str) -> pl.DataFrame:
    return rec.filter(pl.col("in_Bp")) if world == "Bp" else rec


def f05(c, k, w):
    """Per-S1 F0.5 = 1.25c / (c + 0.25k + w); an empty answer on a singleton scores 1."""
    den = c + 0.25 * k + w
    return pl.when(den == 0).then(1.0).otherwise(1.25 * c / den)


def score(pred: pl.DataFrame, s1: pl.DataFrame, rec: pl.DataFrame, world: str = "A") -> pl.DataFrame:
    """pred: [s1, rec] pairs. Returns per-query-S1 [s1, country, k, n_pred, c, f05].

    Truth for a query S1 = records it owns. Copies of hidden S1 (Worlds B/Bp) belong to
    no query, so predicting them is a wrong ID, exactly like a real decoy.
    """
    q = queries(s1, world)
    hits = (
        pred.select("s1", "rec").unique()
        .join(q.select("s1"), on="s1", how="semi")
        .join(rec.select("rec", "owner"), on="rec", how="left")
        .group_by("s1")
        .agg(n_pred=pl.len(), c=(pl.col("owner") == pl.col("s1")).sum())
    )
    return (
        q.select("s1", "country", "k")
        .join(hits, on="s1", how="left")
        .with_columns(pl.col("n_pred", "c").fill_null(0).cast(pl.Float64))
        .with_columns(f05=f05(pl.col("c"), pl.col("k"), pl.col("n_pred") - pl.col("c")))
    )


def report(per_s1: pl.DataFrame) -> str:
    k_bucket = pl.when(pl.col("k") >= 5).then(pl.lit("5+")).otherwise(pl.col("k").cast(pl.Utf8))
    lines = [f"macro F0.5 = {per_s1['f05'].mean():.5f}   (n S1 = {per_s1.height:,})"]
    for name, key in (("country", pl.col("country")), ("true copies k", k_bucket)):
        t = per_s1.group_by(key.alias(name)).agg(n=pl.len(), f05=pl.col("f05").mean()).sort(name)
        lines += [f"  by {name}:"] + [
            f"    {r[name]:>8}  n={r['n']:>9,}  F0.5={r['f05']:.5f}" for r in t.iter_rows(named=True)
        ]
    return "\n".join(lines)


def read_tsv(path: Path, cols: list[str] | None = None) -> pl.DataFrame:
    # quote_char=None: business names contain raw quotes
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, columns=cols)


def read_id_lists(path: Path) -> pl.DataFrame:
    """A matching_results / ground-truth style TSV -> [s1, rec] pairs."""
    df = read_tsv(path)
    df.columns = ["s1", "rec"]
    return (
        df.with_columns(pl.col("rec").str.split(","))
        .explode("rec", empty_as_null=True)
        .filter(pl.col("rec").is_not_null() & (pl.col("rec") != ""))
    )


ROW_COUNTS = {  # archi.md V0.5
    "train/train_source1.tsv": 2_206_821, "train/train_source2.tsv": 5_034_616,
    "train/train_source3.tsv": 5_285_603, "train/train_ground_truth.tsv": 2_206_821,
    "test/test_source1.tsv": 1_732_544, "test/test_source2.tsv": 4_887_273, "test/test_source3.tsv": 5_082_316,
}


def norm_name(col: str) -> pl.Expr:
    """DATA_NOTES normalisation: casefold, NFKD, strip combining marks, \\w+ tokens."""
    return pl.col(col).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()


def sibling_share(query: pl.DataFrame, pool: pl.DataFrame) -> float:
    """Share of `query` records with another `pool` record of the same normalised name and
    address-token Jaccard >= 0.5 (DATA_NOTES §9c "sibling")."""
    pairs = query.join(pool, on=["country", "name_n"], suffix="_o").filter(pl.col("rec") != pl.col("rec_o"))
    union = pl.col("addr_t").list.set_union("addr_t_o").list.len()
    inter = pl.col("addr_t").list.set_intersection("addr_t_o").list.len()
    hits = pairs.filter((union > 0) & (inter / union >= 0.5))["rec"].n_unique()
    return hits / query.height


def build() -> bool:
    """Build folds + worlds, run archi.md Part D V0.* checks, write reports/verify_stage0.json."""
    checks = []

    def check(cid: str, value, expected: str, ok) -> None:
        checks.append({"id": cid, "value": value, "expected": expected, "pass": bool(ok), "level": "HARD"})
        print(f"{'PASS' if ok else 'FAIL'}  {cid:<6} {value}  (expected {expected})")

    full_text = ("train/train_source2.tsv", "train/train_source3.tsv")  # names/addresses needed for V0.10
    raw = {f: read_tsv(DATA / f, None if f in full_text else [0]) for f in ROW_COUNTS}
    counts = {f: df.height for f, df in raw.items()}
    check("V0.5", counts, "exact row counts", counts == ROW_COUNTS)
    dup = {f: df.height - df[:, 0].n_unique() for f, df in raw.items()}
    strings = all(df.schema[df.columns[0]] == pl.Utf8 for df in raw.values())
    check("V0.6", {"all_str": strings, "dups": sum(dup.values())}, "ids are strings, 0 duplicates", strings and not any(dup.values()))

    s1 = read_tsv(DATA / "train/train_source1.tsv", ["entity_id", "country"]).rename({"entity_id": "s1"})
    rec_raw = pl.concat([raw.pop(f"train/train_source{i}.tsv") for i in (2, 3)]).rename({"entity_id": "rec"})
    del raw
    gt = read_id_lists(DATA / "train/train_ground_truth.tsv").rename({"s1": "owner"})
    max_owners = gt.group_by("rec").len()["len"].max()
    unknown = gt.join(rec_raw, on="rec", how="anti").height
    check("V0.7", {"ids": gt.height, "max_owners": max_owners, "unknown": unknown},
          "7,638,365 ids, max owners 1, 0 unknown", gt.height == 7_638_365 and max_owners == 1 and unknown == 0)
    rec = rec_raw.select("rec", "country").join(gt, on="rec", how="left")
    crossed = rec.drop_nulls("owner").join(s1, left_on=["owner", "country"], right_on=["s1", "country"], how="anti")
    assert crossed.is_empty(), "a true pair crosses countries (or names an unknown S1)"

    s1_in, rec_in = s1, rec
    s1, rec = build_worlds(s1_in, rec_in)
    WORK.mkdir(parents=True, exist_ok=True)
    s1.write_parquet(WORK / "s1.parquet")
    rec.write_parquet(WORK / "rec.parquet")

    truth = rec.drop_nulls("owner").select(pl.col("owner").alias("s1"), "rec")
    ex_s1 = pl.DataFrame({"s1": ["S1-00001", "S1-00002", "S1-00003"], "country": "US", "k": [2, 0, 1]})
    ex_rec = pl.DataFrame({"rec": ["S2-00047", "S3-00812", "S2-00193", "S2-9"],
                           "owner": ["S1-00001", "S1-00001", None, "S1-00003"]})
    ex_pred = pl.DataFrame({"s1": ["S1-00001"] * 3 + ["S1-00002"], "rec": ["S2-00047", "S2-00193", "S3-00812", "S2-00193"]})
    f = dict(score(ex_pred, ex_s1, ex_rec).select("s1", "f05").iter_rows())
    f_empty = dict(score(ex_pred.clear(), ex_s1, ex_rec).select("s1", "f05").iter_rows())
    check("V0.1", round(f["S1-00001"], 4), "0.714 ± 0.001", abs(f["S1-00001"] - 0.714) <= 0.001)
    v04 = {"singleton+pred": f["S1-00002"], "singleton+empty": f_empty["S1-00002"], "nonsingleton+empty": f["S1-00003"]}
    check("V0.4", v04, "0 / 1 / 0", list(v04.values()) == [0.0, 1.0, 0.0])
    oracle = score(truth, s1, rec)["f05"].mean()
    check("V0.2", oracle, "1.000", oracle == 1.0)
    empty = score(truth.clear(), s1, rec)["f05"].mean()
    singleton_rate = (s1["k"] == 0).mean()
    check("V0.3", round(empty, 6), "0.0558 (= singleton rate)", empty == singleton_rate and round(empty, 4) == 0.0558)

    owned = rec.drop_nulls("owner").join(s1.select(pl.col("s1").alias("owner"), pl.col("fold").alias("of")), on="owner")
    s1_sizes = s1.group_by("fold").len()["len"]
    decoy_sizes = rec.filter(pl.col("owner").is_null()).group_by("fold").len()["len"]
    spread = max(max(abs(s / s.mean() - 1)) for s in (s1_sizes, decoy_sizes))
    v08 = {"s1_unique": s1["s1"].is_unique().all(), "fold_range_ok": s1["fold"].is_between(0, N_FOLDS - 1).all(),
           "owned_off_fold": int((owned["fold"] != owned["of"]).sum()), "max_fold_size_dev": round(spread, 5)}
    check("V0.8", v08, "each S1 one fold, owned records in owner's fold, sizes within ±1%",
          v08["s1_unique"] and v08["fold_range_ok"] and v08["owned_off_fold"] == 0 and spread <= 0.01)

    dens = {w: indexed(rec, w).height / queries(s1, w).height for w in WORLDS}
    check("V0.9", round(dens["B"], 3), "5.8 ± 0.1", abs(dens["B"] - 5.8) <= 0.1)

    # V0.10 sibling share, on a deterministic 1% sample of each query set
    text = rec_raw.select(
        "rec", "country",
        name_n=norm_name("business_name").str.extract_all(r"\w+").list.join(" "),
        addr_t=norm_name("business_address").str.extract_all(r"\w+").list.unique(),
    ).join(rec.select("rec", "owner", "in_Bp"), on="rec")
    del rec_raw
    sample = pl.Series(uniform(text["rec"], 99) < 0.01)
    bp_pool = text.filter(pl.col("in_Bp"))
    bp_decoy = bp_pool.join(queries(s1, "Bp").select(pl.col("s1").alias("owner")), on="owner", how="anti")
    anchors = {  # calibrates the sibling definition against DATA_NOTES §9c (18.5% / 0.24%)
        "owned_A": sibling_share(text.filter(sample & pl.col("owner").is_not_null()), text),
        "decoy_A": sibling_share(text.filter(sample & pl.col("owner").is_null()), text),
    }
    bp_share = sibling_share(bp_decoy.filter(pl.Series(uniform(bp_decoy["rec"], 99) < 0.01)), bp_pool)
    v010 = {"density": round(dens["Bp"], 3), "sibling_share": round(bp_share, 5),
            **{f"anchor_{k}": round(v, 5) for k, v in anchors.items()}}
    check("V0.10", v010, "5.8 ± 0.1; sibling share < 1%", abs(dens["Bp"] - 5.8) <= 0.1 and bp_share < 0.01)

    s1_2, rec_2 = build_worlds(s1_in, rec_in)
    s1_2.write_parquet(WORK / "_rebuild_s1.parquet")
    rec_2.write_parquet(WORK / "_rebuild_rec.parquet")
    same = all(sha256(WORK / f"{n}.parquet") == sha256(WORK / f"_rebuild_{n}.parquet") for n in ("s1", "rec"))
    for n in ("s1", "rec"):
        (WORK / f"_rebuild_{n}.parquet").unlink()
    check("V0.11", same, "identical file hashes", same)

    for w in WORLDS:
        q = queries(s1, w)
        print(f"World {w:>2}: {q.height:>9,} queries  {dens[w]:.2f} rec/S1  "
              f"decoy share {1 - q['k'].sum() / indexed(rec, w).height:.3f}")
    checks.sort(key=lambda c: int(c["id"].split(".")[1]))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "verify_stage0.json").write_text(json.dumps(checks, indent=1, default=str) + "\n")
    ok = all(c["pass"] for c in checks)
    print(f"Stage 0  {sum(c['pass'] for c in checks)}/{len(checks)} HARD pass")
    return ok


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.read_parquet(WORK / "s1.parquet"), pl.read_parquet(WORK / "rec.parquet")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        sys.exit(0 if build() else "Stage 0 HARD check failed: do not build Stage 1 (see reports/verify_stage0.json)")
    elif sys.argv[1:2] == ["score"] and len(sys.argv) > 2:
        s1, rec = load()
        world = sys.argv[3] if len(sys.argv) > 3 else "A"
        print(f"World {world}\n" + report(score(read_id_lists(Path(sys.argv[2])), s1, rec, world)))
    else:
        sys.exit(__doc__)
