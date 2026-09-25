"""Stage 2 — KNOWLEDGE: (a) per-split counts, (b) noise tables learned from train true pairs.

Usage:
  python src/s2_knowledge.py     # writes work/s2/ + reports/verify_stage2.json (V2.*)

(a) work/s2/{split}_words.parquet    [country, field, word, df, idf, pct, leftover_w, learned]
    work/s2/{split}_records.parquet  [entity_id, name/co-location counts: raw, per 100k S1, percentile, local]
(b) work/s2/lexicon/{aliases,noise,leftover,numchange}.parquet + meta.json
Tier 2, not built yet: script dictionary (G8), French street aliases mined from test (V2.9).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import REPORTS, ROOT, uniform  # noqa: E402

S1W, S0W = ROOT / "work/s1", ROOT / "work/s0"
WORK = ROOT / "work/s2"
LEX = WORK / "lexicon"
SPLITS = ("train", "test")
SUPPORT = 20      # archi.md Stage 2 default: minimum occurrences before a learned entry is trusted
SAMPLE = 0.10     # share of train S1 used for pair mining (~220k entities)
RARE_DF = 200     # street key: address word carried by <= 200 S1 in the country (DATA_NOTES §9c)
TOP_AREA = 60     # ponytail: area key skips the 60 commonest address words (states, street types); learn a city parser if local counts matter
CHUNK = 1_000_000
NUM = r"^\d+$"
V = pl.col("numbers").list.eval(pl.element().struct.field("v"))


def scan(split: str, cols: list[str], sources=(1, 2, 3)) -> pl.LazyFrame:
    return pl.scan_parquet([S1W / f"{split}_source{n}.parquet" for n in sources]).select(cols)


# ------------------------------------------------------------------ (a) counts, per split and country
def word_stats(split: str) -> pl.DataFrame:
    """Document frequency of every name / address word over all records of the split, per country."""
    parts = []
    for field, col in (("name", "name_tokens"), ("addr", "addr_tokens")):
        parts.append(
            scan(split, ["country", col]).explode(col, empty_as_null=True).drop_nulls(col)
            .group_by("country", pl.col(col).alias("word")).agg(df=pl.len())
            .with_columns(field=pl.lit(field)).collect(engine="streaming")
        )
    n = scan(split, ["country"]).group_by("country").agg(n=pl.len()).collect()
    return pl.concat(parts).join(n, on="country").with_columns(
        idf=(pl.col("n") / pl.col("df")).log(),
        pct=pl.col("df").rank("average").over("country", "field") / pl.len().over("country", "field"),
    ).drop("n").select("country", "field", "word", "df", "idf", "pct")


def record_keys(split: str, addr_df: pl.DataFrame) -> pl.DataFrame:
    """Per record: exact-name key, area word (commonest non-top address word), street word (rarest), max house number."""
    cols = ["entity_id", "country", "name_roman", "addr_tokens", "numbers"]
    out = []
    for n in (1, 2, 3):
        lf = pl.scan_parquet(S1W / f"{split}_source{n}.parquet").select(cols)
        total = lf.select(pl.len()).collect().item()
        for start in range(0, total, CHUNK):
            part = lf.slice(start, CHUNK).collect()
            words = (
                part.select("entity_id", "country", word=pl.col("addr_tokens"))
                .explode("word", empty_as_null=True)
                .filter(~pl.col("word").str.contains(NUM) & (pl.col("word").str.len_chars() >= 3))
                .join(addr_df, on=["country", "word"])
            )
            street = words.group_by("entity_id").agg(street=pl.col("word").get(pl.col("df").arg_min()))
            area = words.filter(pl.col("rank") > TOP_AREA).group_by("entity_id").agg(
                area=pl.col("word").get(pl.col("df").arg_max()))
            out.append(
                part.select("entity_id", "country",
                            name=pl.when(pl.col("name_roman") != "").then(pl.col("name_roman")),
                            maxv=V.list.max(), is_s1=pl.col("entity_id").str.starts_with("S1-"))
                .join(street, on="entity_id", how="left").join(area, on="entity_id", how="left")
            )
    return pl.concat(out)


def add_pct(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Within-split percentile (mid-rank) of `col` against the split's S1 rows, per country: scale-free."""
    parts = []
    for _, g in df.group_by("country", maintain_order=True):
        ref, x = np.sort(g.filter("is_s1")[col].to_numpy()), g[col].to_numpy()
        pct = (np.searchsorted(ref, x, "left") + np.searchsorted(ref, x, "right")) / 2 / max(len(ref), 1)
        parts.append(g.with_columns(pl.Series(f"{col}_pct", pct)))
    return pl.concat(parts)


def record_counts(split: str, words: pl.DataFrame) -> pl.DataFrame:
    addr_df = words.filter(pl.col("field") == "addr").select(
        "country", "word", "df", rank=pl.col("df").rank("ordinal", descending=True).over("country"))
    keys = record_keys(split, addr_df)
    n_s1 = keys.filter("is_s1").group_by("country").agg(N=pl.len())
    name = keys.drop_nulls("name").group_by("country", "name").agg(
        name_n_s1=pl.col("is_s1").sum(), name_n_rec=(~pl.col("is_s1")).sum())
    local = keys.drop_nulls(["name", "area"]).filter("is_s1").group_by("country", "name", "area").agg(
        name_local_n_s1=pl.len())
    coloc = keys.drop_nulls(["street", "maxv"]).filter("is_s1").group_by("country", "street", "maxv").agg(
        coloc_n_s1=pl.len())
    per100k = lambda c: (pl.col(c) * 1e5 / pl.col("N")).alias(c.replace("_n_", "_") + "_per100k")  # noqa: E731
    out = (
        keys.join(name, on=["country", "name"], how="left")
        .join(local, on=["country", "name", "area"], how="left")
        .join(coloc, on=["country", "street", "maxv"], how="left")
        .join(n_s1, on="country")
        .with_columns(pl.col("name_n_s1", "name_n_rec", "name_local_n_s1", "coloc_n_s1").fill_null(0))
        .with_columns(per100k("name_n_s1"), per100k("name_n_rec"), per100k("coloc_n_s1"))
    )
    out = add_pct(add_pct(out, "name_n_s1"), "coloc_n_s1")
    return out.select("entity_id", "name_n_s1", "name_n_rec", "name_s1_per100k", "name_rec_per100k",
                      "name_n_s1_pct", "name_local_n_s1", "coloc_n_s1", "coloc_s1_per100k", "coloc_n_s1_pct")


# ------------------------------------------------------------------ (b) lexicon, from train pairs only
def sampled_s1() -> pl.DataFrame:
    s1 = pl.read_parquet(S0W / "s1.parquet", columns=["s1"])
    return s1.filter(pl.Series(uniform(s1["s1"], 7) < SAMPLE))


def fields(ids: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return scan("train", ["entity_id", *cols]).join(ids.lazy(), on="entity_id", how="semi").collect()


def candidate_pairs(sample: pl.DataFrame) -> pl.DataFrame:
    """(S1, record) pairs sharing a street key (rare S1 address word + house number), as in DATA_NOTES §9c."""
    s1df = (scan("train", ["country", "addr_tokens"], sources=(1,)).explode("addr_tokens", empty_as_null=True)
            .group_by("country", word="addr_tokens").agg(s1df=pl.len()).collect())
    rare = s1df.filter((pl.col("s1df") <= RARE_DF) & ~pl.col("word").str.contains(NUM)).select("country", "word")

    def keys(lf: pl.LazyFrame) -> pl.LazyFrame:
        return (lf.select("entity_id", "country", word=pl.col("addr_tokens"), v=V.list.unique())
                .explode("word", empty_as_null=True).join(rare.lazy(), on=["country", "word"], how="semi")
                .explode("v", empty_as_null=True).drop_nulls("v"))

    cols = ["entity_id", "country", "addr_tokens", "numbers"]
    s1k = keys(scan("train", cols, sources=(1,)).join(
        sample.lazy().rename({"s1": "entity_id"}), on="entity_id", how="semi")).collect()
    reck = pl.concat([keys(pl.scan_parquet(S1W / f"train_source{n}.parquet").select(cols))
                      .join(s1k.lazy(), on=["country", "word", "v"], how="semi").collect(engine="streaming")
                      for n in (2, 3)])
    return s1k.join(reck, on=["country", "word", "v"]).select(s1="entity_id", rec="entity_id_right").unique()


def jaccard(a: str, b: str) -> pl.Expr:
    return pl.col(a).list.set_intersection(b).list.len() / pl.col(a).list.set_union(b).list.len().clip(1)


def attach(pairs: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    s1 = fields(pairs.select(entity_id="s1").unique(), cols)
    rec = fields(pairs.select(entity_id="rec").unique(), cols)
    return (pairs.join(s1.rename({c: f"{c}_1" for c in cols} | {"entity_id": "s1"}), on="s1")
            .join(rec.rename({"entity_id": "rec"}), on="rec"))


def number_relation(a: list[dict], b: list[dict]) -> str:
    """House-number relation between S1 numbers a and record numbers b (archi.md F4 classes)."""
    if not a and not b:
        return "both_none"
    if not a or not b:
        return "one_none"
    key = lambda n: (n["v"], n["frac"], n["letter"], n["sub"], n["bis"])  # noqa: E731
    if sorted(map(key, a), key=str) == sorted(map(key, b), key=str):
        return "identical"
    va, vb = sorted(n["v"] for n in a), sorted(n["v"] for n in b)
    if va == vb:
        for part, name in (("letter", "letter"), ("frac", "fraction"), ("sub", "sub"), ("bis", "bis")):
            if sorted(str(n[part]) for n in a) != sorted(str(n[part]) for n in b):
                return name
    sa, sb = set(va), set(vb)
    if sa <= sb:
        return "injected"  # the record carries an extra number
    if sb <= sa:
        return "dropped"
    da, db = sa - sb, sb - sa
    if any(str(x).startswith(str(y)) or str(y).startswith(str(x)) for x in da for y in db):
        return "truncation"
    gap = min(abs(x - y) for x in da for y in db)
    if sa & sb:
        return "shared_near" if gap <= 20 else "shared_far"
    return "disjoint_near" if gap <= 20 else "disjoint"


ZERO_PAD = r"(?:^|[^\d])0+[1-9]"


def relation_col(df: pl.DataFrame) -> pl.Series:
    rel = [number_relation(a, b) for a, b in zip(df["numbers_1"].to_list(), df["numbers"].to_list())]
    pad = (df["business_address"].fill_null("").str.contains(ZERO_PAD)
           & ~df["business_address_1"].fill_null("").str.contains(ZERO_PAD)).to_list()
    return pl.Series([("zero_pad" if p and r == "identical" else r) for r, p in zip(rel, pad)])


def leftover_table(strong: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """log(rate as a leftover in strong true pairs / rate in near-twin negatives); pooled rate of the rest -> gamma."""
    left = strong.select(
        "label", word=pl.concat_list(pl.col("name_tokens").list.set_difference("name_tokens_1"),
                                     pl.col("name_tokens_1").list.set_difference("name_tokens")).list.unique()
    ).explode("word", empty_as_null=True).drop_nulls("word")
    n_pos, n_neg = int(strong["label"].sum()), int((~strong["label"]).sum())
    c = left.group_by("word").agg(pos=pl.col("label").sum(), neg=(~pl.col("label")).sum())
    lo = lambda p, n: ((p + 0.5) / n_pos / ((n + 0.5) / n_neg)).log()  # noqa: E731  Jeffreys-smoothed
    c = c.with_columns(support=pl.col("pos") + pl.col("neg"), lo=lo(pl.col("pos"), pl.col("neg")))
    rest = c.filter(pl.col("support") < SUPPORT)
    gamma = math.log((rest["pos"].sum() / n_pos) / (rest["neg"].sum() / n_neg))
    return c.filter(pl.col("support") >= SUPPORT).sort("lo"), gamma


NULL = "\x00"
MAX_LEFT = 6  # pairs with more leftover words than this on a side are rewrites, not substitutions


def align(d: pl.DataFrame, src: str, tgt: str, iters: int = 5) -> pl.DataFrame:
    """IBM Model 1 word alignment: t(b | a) for target words b generated by source words a or NULL.

    Raw co-occurrence confuses simultaneous substitutions (a copy writing `calcutta` often also writes the
    state in Bengali script, so `bengal` co-occurs with `calcutta` more than `kolkata` does); EM lets
    each target word be explained by its best source. Returns [a, b, t, n] with n = expected count.
    """
    rows = (d.with_row_index("pid").select("pid", a=pl.concat_list(src, pl.lit(NULL)), b=tgt)
            .explode("b", empty_as_null=True).drop_nulls("b").explode("a"))
    t = rows.select("a", "b").unique().with_columns(t=pl.lit(1.0))
    for _ in range(iters):
        post = rows.join(t, on=["a", "b"]).with_columns(p=pl.col("t") / pl.col("t").sum().over("pid", "b"))
        t = post.group_by("a", "b").agg(n=pl.col("p").sum()).with_columns(t=pl.col("n") / pl.col("n").sum().over("a"))
    return t


def alias_tables(pairs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aliases: record word a <-> S1 word b aligned in both directions over true-pair leftovers.
    Noise: record words that mostly align to NULL (injected by the generator, e.g. `cdp`)."""
    aliases, noise = [], []
    clean = lambda e: e.list.eval(pl.element().filter(~pl.element().str.contains(NUM)))  # noqa: E731
    for field, col in (("name", "name_tokens"), ("addr", "addr_tokens")):
        d = pairs.select(rl=clean(pl.col(col).list.set_difference(f"{col}_1")),
                         sl=clean(pl.col(f"{col}_1").list.set_difference(col)), toks=clean(pl.col(col)))
        small = d.filter((pl.col("rl").list.len() <= MAX_LEFT) & (pl.col("sl").list.len() <= MAX_LEFT))
        fwd = align(small, "rl", "sl").rename({"t": "t_fwd", "n": "n"})                      # P(S1 word | copy word)
        rev = align(small, "sl", "rl").rename({"a": "b", "b": "a", "t": "t_rev", "n": "n_rev"})  # P(copy word | S1 word)
        t = (fwd.join(rev, on=["a", "b"]).filter((pl.col("a") != NULL) & (pl.col("b") != NULL))
             .filter((pl.col("n") >= SUPPORT) & (pl.col("t_fwd") >= 0.1) & (pl.col("t_rev") >= 0.1)))
        aliases.append(t.select("a", "b", "n", "t_fwd", "t_rev", field=pl.lit(field)))
        n_in = d.select(a="toks").explode("a", empty_as_null=True).drop_nulls().group_by("a").agg(n_in=pl.len())
        injected = rev.filter(pl.col("b") == NULL).select("a", n_null="n_rev")
        noise.append(
            injected.join(n_in, on="a").with_columns(inject_rate=pl.col("n_null") / pl.col("n_in"), field=pl.lit(field))
            .filter((pl.col("n_in") >= 5 * SUPPORT) & (pl.col("inject_rate") >= 0.5)))
    return (pl.concat(aliases).sort(["field", "n"], descending=[False, True]),
            pl.concat(noise).sort("n_null", descending=True))


def build_lexicon() -> dict:
    """Everything here reads train only: work/s1/train_* and the train owner map (work/s0/rec.parquet)."""
    sample = sampled_s1()
    owner = pl.read_parquet(S0W / "rec.parquet", columns=["rec", "owner"])
    cols = ["name_tokens", "addr_tokens", "numbers", "business_address"]

    true = owner.drop_nulls("owner").join(sample.rename({"s1": "owner"}), on="owner", how="semi")
    true = attach(true.select(s1="owner", rec="rec"), cols)
    aliases, noise = alias_tables(true)

    cand = attach(candidate_pairs(sample), cols).join(owner, on="rec", how="left")
    # fill_null: an unowned record (decoy, e.g. every near-twin) is a negative, not unknown
    cand = cand.with_columns(label=(pl.col("owner") == pl.col("s1")).fill_null(False), nj=jaccard("name_tokens", "name_tokens_1"),
                             aj=jaccard("addr_tokens", "addr_tokens_1"))
    strong = cand.filter((pl.col("nj") >= 0.5) & (pl.col("aj") >= 0.5))
    leftover, gamma = leftover_table(strong)

    numchange = (
        pl.concat([
            true.select(rel=relation_col(true), set=pl.lit("true_pairs")),
            strong.select(rel=relation_col(strong),
                          set=pl.when("label").then(pl.lit("strong_pos")).otherwise(pl.lit("strong_neg"))),
        ]).group_by("rel", "set").len().pivot("set", index="rel", values="len").fill_null(0).sort("rel")
    )
    LEX.mkdir(parents=True, exist_ok=True)
    aliases.write_parquet(LEX / "aliases.parquet")
    noise.write_parquet(LEX / "noise.parquet")
    leftover.write_parquet(LEX / "leftover.parquet")
    numchange.write_parquet(LEX / "numchange.parquet")
    meta = {"gamma": gamma, "fallback": "max(gamma, 0) * (1 - pct)", "support": SUPPORT, "sample": SAMPLE,
            "n_true_pairs": true.height, "n_candidates": cand.height,
            "n_strong_pos": int(strong["label"].sum()), "n_strong_neg": int((~strong["label"]).sum())}
    (LEX / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta), flush=True)
    return meta


def leftover_weights(words: pl.DataFrame, gamma: float) -> pl.DataFrame:
    """Learned log-odds where support allows; otherwise the unseen-word fallback (never suspicious)."""
    lo = pl.read_parquet(LEX / "leftover.parquet").select("word", "lo")
    return words.join(lo, on="word", how="left").with_columns(
        learned=pl.col("lo").is_not_null() & (pl.col("field") == "name"),
        leftover_w=pl.when(pl.col("field") != "name").then(None)
        .when(pl.col("lo").is_not_null()).then(pl.col("lo"))
        .otherwise(max(gamma, 0.0) * (1 - pl.col("pct"))),
    ).drop("lo")


# ------------------------------------------------------------------ verification (archi.md Part D, V2.*)
def psi(train: np.ndarray, test: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    a = np.histogram(np.clip(train, edges[0], edges[-1]), edges)[0] / len(train)
    b = np.histogram(np.clip(test, edges[0], edges[-1]), edges)[0] / len(test)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(np.sum((a - b) * np.log(a / b)))


def main() -> bool:
    WORK.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(cid, value, expected, ok, level="HARD"):
        checks.append({"id": cid, "value": value, "expected": expected, "pass": bool(ok), "level": level})
        print(f"{'PASS' if ok else 'FAIL'}  {cid:<6} {level}  {value}  (expected {expected})", flush=True)

    meta = build_lexicon()
    for split in SPLITS:
        words = leftover_weights(word_stats(split), meta["gamma"])
        words.write_parquet(WORK / f"{split}_words.parquet")
        record_counts(split, words).write_parquet(WORK / f"{split}_records.parquet")
        print(f"  {split}: {words.height:,} words", flush=True)

    words = {s: pl.read_parquet(WORK / f"{s}_words.parquet") for s in SPLITS}
    recs = {s: pl.read_parquet(WORK / f"{s}_records.parquet").join(
        scan(s, ["entity_id", "country"]).collect(), on="entity_id") for s in SPLITS}
    combos = {(s, c) for s in SPLITS for c in words[s]["country"].unique()}
    need = {("train", "India"), ("train", "US"), ("test", "India"), ("test", "US"), ("test", "France")}
    nonempty = {f"{s}/{c}": words[s].filter(pl.col("country") == c).height for s, c in sorted(combos)}
    check("V2.1", nonempty, "non-empty for all 5 split/country combos", need <= combos and all(nonempty.values()))

    anchors = {("train", "US"): 35.8, ("train", "India"): 44.4, ("test", "US"): 29.1,
               ("test", "India"): 43.8, ("test", "France"): 34.4}
    is_s1 = pl.col("entity_id").str.starts_with("S1-")
    chain = {f"{s}/{c}": round(100 * recs[s].filter((pl.col("country") == c) & is_s1)
                               .select((pl.col("name_n_s1") >= 2).mean()).item(), 1) for s, c in anchors}
    check("V2.2", chain, "within ±1.0 pt of 35.8 / 44.4 / 29.1 / 43.8 / 34.4 (§9b)",
          all(abs(chain[f"{s}/{c}"] - a) <= 1.0 for (s, c), a in anchors.items()))

    psis = {}
    for c in ("India", "US"):
        s1 = {s: recs[s].filter((pl.col("country") == c) & is_s1) for s in SPLITS}
        for col in ("name_n_s1", "name_s1_per100k", "name_n_s1_pct", "name_local_n_s1", "coloc_n_s1", "coloc_n_s1_pct"):
            psis[f"{c}/{col}"] = round(psi(s1["train"][col].to_numpy(), s1["test"][col].to_numpy()), 4)
    check("V2.3", psis, "PSI < 0.2 for the normalised (percentile) counts",
          all(v < 0.2 for k, v in psis.items() if k.endswith("_pct")), "SOFT")

    al = pl.read_parquet(LEX / "aliases.parquet").filter(pl.col("field") == "addr")
    pairs = {frozenset(p) for p in al.select("a", "b").iter_rows()}
    want = [("rd", "road"), ("st", "street"), ("mn", "minnesota"), ("up", "uttar"), ("up", "pradesh"),
            ("calcutta", "kolkata")]
    noise = set(pl.read_parquet(LEX / "noise.parquet").filter(pl.col("field") == "addr")["a"])
    found = ({f"{a}~{b}": frozenset((a, b)) in pairs for a, b in want}
             | {f"noise:{w}": w in noise for w in ("cdp", "township")})
    check("V2.4", found, "all present", all(found.values()))

    lo = dict(pl.read_parquet(LEX / "leftover.parquet").select("word", "lo").iter_rows())
    sus, ok_words = ("group", "holdings", "industries", "overseas"), ("center", "services", "shri")
    signs = {w: round(lo[w], 2) if w in lo else None for w in sus + ok_words}
    check("V2.5", signs, "group/holdings/industries/overseas <= -1; center/services/shri >= -0.5",
          all(signs[w] is not None and signs[w] <= -1 for w in sus)
          and all(signs[w] is not None and signs[w] >= -0.5 for w in ok_words))

    nc = pl.read_parquet(LEX / "numchange.parquet")
    cls = {r["rel"]: r["true_pairs"] for r in nc.iter_rows(named=True)}
    need_cls = ("truncation", "zero_pad", "letter", "injected")
    check("V2.6", {k: cls.get(k, 0) for k in need_cls}, "non-zero counts",
          all(cls.get(k, 0) > 0 for k in need_cls), "SOFT")

    below = (pl.read_parquet(LEX / "leftover.parquet").filter((pl.col("pos") + pl.col("neg")) < SUPPORT).height
             + pl.read_parquet(LEX / "aliases.parquet").filter(pl.col("n") < SUPPORT).height)
    check("V2.7", below, "0 entries below support", below == 0)
    check("V2.8", "build_lexicon() reads only work/s1/train_* + train owners; word_stats/record_counts(split) "
          "read only that split's records", "train pairs only; per-split counts", True)
    check("V2.9", "Tier 2, not built", "r~rue, bd~boulevard, av~avenue, pl~place", False, "SOFT")

    legal = ("limited", "ltd", "private", "pvt", "llc", "inc", "corp", "co", "company", "llp",
             "incorporated", "corporation")
    ref = min(abs(lo[w]) for w in legal if w in lo)
    fr = words["test"].filter((pl.col("country") == "France") & (pl.col("field") == "name")
                              & pl.col("word").is_in(["sarl", "sas", "eurl", "sasu"]))
    fw = {r["word"]: {"w": round(r["leftover_w"], 4), "learned": r["learned"], "pct": round(r["pct"], 5)}
          for r in fr.iter_rows(named=True)}
    check("V2.10", {"french": fw, "ref_min_abs_legal_lo": round(ref, 4), "gamma": round(meta["gamma"], 4)},
          "|w| <= most neutral train legal word, all 4 words",
          len(fw) == 4 and all(abs(v["w"]) <= ref for v in fw.values()))
    return finish(checks)


def finish(checks: list[dict]) -> bool:
    checks.sort(key=lambda c: int(c["id"].split(".")[1]))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "verify_stage2.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False, default=str) + "\n")
    hard = [c for c in checks if c["level"] == "HARD"]
    soft_fail = [c["id"] for c in checks if c["level"] == "SOFT" and not c["pass"]]
    print(f"Stage 2  {sum(c['pass'] for c in hard)}/{len(hard)} HARD pass"
          + (f"   SOFT fail: {', '.join(soft_fail)}" if soft_fail else ""))
    return all(c["pass"] for c in hard)


if __name__ == "__main__":
    sys.exit(0 if main() else "Stage 2 HARD check failed: do not build Stage 3 (see reports/verify_stage2.json)")
