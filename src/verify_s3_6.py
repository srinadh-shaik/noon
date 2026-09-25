"""Part D checks V3-V6, read from the Stage 3-6 artefacts -> reports/verify_stage{3,4,5,6}.json.

Usage:
  python src/verify_s3_6.py stage3 [MODE ...]    # reports/recall_curve.csv (V3.1-3.3, 3.5, 3.6) + V3.4 on work/s4/MODE
  python src/verify_s3_6.py stage4 TRAIN TEST    # V4.1-4.5 (V4.3 ceiling vs the V3.1 curve)
  python src/verify_s3_6.py stage5 TRAIN TEST    # V5.1-5.7, and V4.6 part 1 (features == pairs)
  python src/verify_s3_6.py stage6 TRAIN TEST    # V6.1-6.7, and V4.6 part 2 (scores == pairs)
Exits non-zero when a HARD check fails. Statistical checks (V5.2-5.7, V6.4) use a deterministic hash sample of
S1 (SAMPLE_SHARE of them); set checks (V4.1, V4.2, V4.6, V6.1) use every row.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import REPORTS, ROOT  # noqa: E402

S0W, S1W, S4W, S5W, S6W = (ROOT / f"work/{s}" for s in ("s0", "s1", "s4", "s5", "s6"))
KS = (5, 10, 20, 50, 100, 200)
FWD_K = 20  # s4 = forward top-20 per view + reverse top-20 + street key -> V4.3 target = curve union at K=20
SAMPLE_SHARE = 0.05
ID = ("s1", "rec", "label", "fold")
NULL_OK = re.compile(r"_rank$|_score$|^alt_|_rec_best$|_rec_second$|_gap_to_best$|_n_claims$|_left_w_|_max_idf$")
LEAK = re.compile(r"country|source|file|owner|truth|entity|label|fold", re.I)


class Report:
    def __init__(self, stage: int):
        self.path, self.checks = REPORTS / f"verify_stage{stage}.json", []

    def check(self, cid: str, value, expected: str, ok, level: str = "HARD") -> None:
        self.checks.append({"id": cid, "value": value, "expected": expected, "pass": bool(ok), "level": level})
        print(f"{'PASS' if ok else 'FAIL'}  {cid:<5} {json.dumps(value, default=str)[:260]}  (expected {expected}) [{level}]")

    def save(self) -> bool:
        """Merge into the stage's report by check id (V4.6 is filled in by the stage 5 and 6 runs)."""
        old = json.loads(self.path.read_text()) if self.path.exists() else []
        new = {c["id"]: c for c in old} | {c["id"]: c for c in self.checks}
        REPORTS.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(sorted(new.values(), key=lambda c: int(c["id"].split(".")[1])), indent=1,
                                        default=str) + "\n")
        return all(c["pass"] for c in self.checks if c["level"] == "HARD")


def split(mode: str) -> str:
    return "test" if mode.startswith("test") else "train"


def country_of(sp: str, sources=(1, 2, 3)) -> pl.LazyFrame:
    return pl.scan_parquet([S1W / f"{sp}_source{n}.parquet" for n in sources]).select("entity_id", "country")


def sample(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("s1").hash(seed=11) % 10_000 < int(SAMPLE_SHARE * 10_000))


def set_diff(a: pl.LazyFrame, b: pl.LazyFrame) -> dict:
    """Rows of (s1, rec) in a but not b, and in b but not a."""
    a, b = a.select("s1", "rec"), b.select("s1", "rec")
    return {"only_left": a.join(b, on=["s1", "rec"], how="anti").select(pl.len()).collect().item(),
            "only_right": b.join(a, on=["s1", "rec"], how="anti").select(pl.len()).collect().item()}


def curve() -> pl.DataFrame | None:
    f = REPORTS / "recall_curve.csv"
    return pl.read_csv(f) if f.exists() else None


# ---------------------------------------------------------------- Stage 3
def stage3(modes: list[str]) -> bool:
    r, c = Report(3), curve()
    if c is None:
        r.check("V3.1", "reports/recall_curve.csv missing", "exists", False)
        return r.save()
    have = set(zip(c["view"], c["direction"]))
    need = {(v, d) for v in ("V1", "V2") for d in ("fwd", "rev")}
    ks = set(c.filter(pl.col("direction") == "fwd")["K"].to_list())
    r.check("V3.1", {"views": sorted("/".join(x) for x in have), "fwd_K": sorted(ks),
                     "countries": sorted(c["country"].unique()), "slices": sorted(c["slice"].unique())},
            "V1/V2 x fwd/rev x country x slice, K in 5..200", need <= have and set(KS) <= ks and c["slice"].n_unique() > 1)
    key = c.filter((pl.col("view") == "V3") & (pl.col("slice") == "all") & (pl.col("K") == 1))
    got = dict(zip(key["country"], key["recall"]))
    r.check("V3.2", got, "US ≈ 0.40, India ≈ 0.58 (±0.06, §9c)",
            all(abs(got.get(k, -1) - v) <= 0.06 for k, v in {"US": 0.40, "India": 0.58}.items()), "SOFT")
    uq = c.filter(pl.col("view").str.starts_with("unique_") & (pl.col("slice") == "all"))
    vals = {f"{v}/{k}": x for v, k, x in zip(uq["view"], uq["country"], uq["recall"])}
    r.check("V3.3", vals, "unique recall per view reported; < 0.001 flags a redundant view",
            bool(vals) and all(v >= 0.001 for v in vals.values()), "SOFT")
    un = c.filter((pl.col("view") == "union") & (pl.col("K") == FWD_K))
    per = {f"{k}/{s}": x for k, s, x in zip(un["country"], un["slice"], un["recall"])}
    r.check("V3.5", per, f"union@{FWD_K} recall per noise slice, each > 0", bool(per) and all(v > 0 for v in per.values()))
    r.check("V3.6", "wall-clock + peak RSS per country printed by s3/s4 log()", "logged", True, "SOFT")
    for m in modes:
        cty = country_of(split(m))
        bad = (pl.scan_parquet(S4W / m / "pairs.parquet").select("s1", "rec")
               .join(cty.select(s1="entity_id", c1="country"), on="s1", how="left")
               .join(cty.select(rec="entity_id", c2="country"), on="rec", how="left")
               .filter((pl.col("c1") != pl.col("c2")) | pl.col("c1").is_null() | pl.col("c2").is_null()
                       | ~pl.col("s1").str.starts_with("S1-") | pl.col("rec").str.starts_with("S1-"))
               .select(pl.len()).collect().item())
        r.check("V3.4", {"mode": m, "cross_country_or_bad_id_pairs": bad}, "0", bad == 0)
    return r.save()


# ---------------------------------------------------------------- Stage 4
def stage4(train: str, test: str) -> bool:
    r = Report(4)
    v41 = {}
    for m in (train, test):
        p = pl.scan_parquet(S4W / m / "pairs.parquet").select("s1", "rec")
        v41[m] = {"duplicate_pairs": p.group_by("s1", "rec").len().filter(pl.col("len") > 1).select(pl.len()).collect().item(),
                  "bad_ids": p.filter(~pl.col("s1").str.starts_with("S1-")
                                      | ~pl.col("rec").str.contains(r"^S[23]-")).select(pl.len()).collect().item()}
    r.check("V4.1", v41, "0 duplicate pairs, only S2/S3 records", all(not any(v.values()) for v in v41.values()))

    q_test = pl.read_parquet(S4W / test / "s1.parquet")["s1"]
    all_test = pl.scan_parquet(S1W / "test_source1.parquet").select("entity_id").collect()["entity_id"]
    v42 = {"rows": q_test.len(), "unique": q_test.n_unique(), "test_S1": all_test.len(),
           "missing": int((~all_test.is_in(q_test.implode())).sum())}
    r.check("V4.2", v42, "every test S1 exactly once (1,732,544)",
            v42["rows"] == v42["unique"] == v42["test_S1"] and v42["missing"] == 0)

    q = pl.read_parquet(S4W / train / "s1.parquet").select("s1")
    s1 = pl.read_parquet(S0W / "s1.parquet", columns=["s1", "country", "k"]).join(q, on="s1", how="semi")
    truth = (pl.scan_parquet(S0W / "rec.parquet").select("rec", s1="owner").join(s1.lazy().select("s1", "country"), on="s1")
             .join(pl.scan_parquet([S1W / f"train_source{n}.parquet" for n in (2, 3)])
                   .select(rec="entity_id", nonlatin_name="name_nonlatin", empty_address=pl.col("addr_clean") == ""),
                   on="rec", how="left"))
    found = truth.join(pl.scan_parquet(S4W / train / "pairs.parquet").select("s1", "rec", hit=pl.lit(True)),
                       on=["s1", "rec"], how="left").with_columns(pl.col("hit").fill_null(False)).collect()
    ceil = {c: round(x, 5) for c, x in found.group_by("country").agg(x=pl.col("hit").mean()).iter_rows()}
    sl = {f"{s}={v}": round(x, 5) for s in ("nonlatin_name", "empty_address")
          for v, x in found.group_by(s).agg(x=pl.col("hit").mean()).iter_rows()}
    c = curve()
    target = {} if c is None else dict(c.filter((pl.col("view") == "union") & (pl.col("K") == FWD_K)
                                                & (pl.col("slice") == "all")).select("country", "recall").iter_rows())
    r.check("V4.3", {"ceiling": ceil, "target_union@20": target, "slices": sl, "true_pairs": found.height},
            "ceiling ≥ curve union@20 − 0.005 per country",
            bool(target) and all(ceil.get(k, 0) >= t - 0.005 for k, t in target.items()))

    per = (q.lazy().join(pl.scan_parquet(S4W / train / "pairs.parquet").group_by("s1").len("n"), on="s1", how="left")
           .with_columns(pl.col("n").fill_null(0)).collect()["n"])
    v44 = {"mean": round(per.mean(), 2), "p95": per.quantile(0.95), "max": per.max(), "total_pairs": int(per.sum()),
           "S1": per.len()}
    # SOFT, not HARD: the user's rule is accuracy over compute (no pruning for budget), so this is a sizing report
    r.check("V4.4", v44, "mean ≲ 45 per S1 (C4: 60-100M pairs per split); sizing only", v44["mean"] <= 45, "SOFT")
    big = s1.filter(pl.col("k") >= 8).select("s1")
    kept = found.join(big, on="s1", how="semi")["hit"]
    r.check("V4.5", {"S1_with_k>=8": big.height, "copies_kept": round(kept.mean(), 5) if kept.len() else None},
            "≥ 0.95", kept.len() > 0 and kept.mean() >= 0.95, "SOFT")
    return r.save()


def v46(part: str, stage_dir: Path, train: str, test: str) -> bool:
    """V4.6: the pair set is identical through Stages 4 -> 5 -> 6 (the scored set == candidate_pairs.tsv)."""
    r4 = Report(4)
    d = {m: set_diff(pl.scan_parquet(S4W / m / "pairs.parquet"), pl.scan_parquet(stage_dir / m / f"{part}.parquet"))
         for m in (train, test)}
    prev = {c["id"]: c for c in json.loads(r4.path.read_text())} if r4.path.exists() else {}
    value = (prev.get("V4.6", {}).get("value") or {}) | {part: d}
    r4.check("V4.6", value, "pairs == features == scores (set equality)",
             all(not any(x.values()) for p in value.values() for x in p.values()))
    return r4.save()


# ---------------------------------------------------------------- Stage 5
def psi(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return 0.0
    pa = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a) + 1e-4
    pb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b) + 1e-4
    return float(((pa - pb) * np.log(pa / pb)).sum())


def stage5(train: str, test: str) -> bool:
    from s5_features import RELS  # decode num_rel from the source of truth, never hard-coded codes

    r = Report(5)
    f_tr, f_te = S5W / train / "features.parquet", S5W / test / "features.parquet"
    st, se = pl.read_parquet_schema(f_tr), pl.read_parquet_schema(f_te)
    st_x = {k: str(v) for k, v in st.items() if k not in ("label", "fold")}
    se_x = {k: str(v) for k, v in se.items()}
    r.check("V5.1", {"only_train": sorted(set(st_x) - set(se_x)), "only_test": sorted(set(se_x) - set(st_x)),
                     "dtype_diff": sorted(k for k in set(st_x) & set(se_x) if st_x[k] != se_x[k])},
            "identical columns and dtypes (train minus label/fold)", st_x == se_x)

    feats = [k for k in st_x if k not in ID]
    num = [k for k in feats if st[k].is_numeric() or st[k] == pl.Boolean]
    tr = sample(pl.scan_parquet(f_tr)).join(country_of("train", (1,)).select(s1="entity_id", country="country"), on="s1").collect()
    te = sample(pl.scan_parquet(f_te)).join(country_of("test", (1,)).select(s1="entity_id", country="country"), on="s1").collect()
    named = [k for k in feats if LEAK.search(k)]
    disjoint = []
    for k in num:
        g = tr.group_by("country").agg(lo=pl.col(k).cast(pl.Float64).min(), hi=pl.col(k).cast(pl.Float64).max()).drop_nulls()
        if g.height >= 2 and g["lo"].max() > g["hi"].min():
            disjoint.append(k)  # the countries' value ranges don't overlap: the column would encode the country
    r.check("V5.2", {"leaky_names": named, "country_separating": disjoint, "allowed": ["rec_is_s2 (F8 source)"]},
            "no column encodes country, source file or a label", not named and not disjoint)

    rel = tr.filter((pl.col("name_jaccard") >= 0.5) & (pl.col("addr_jaccard") >= 0.5)).group_by("num_rel").agg(
        n=pl.len(), pos=pl.col("label").cast(pl.Float64).mean())
    rates = {RELS[i]: {"n": n, "pos": round(p, 4)} for i, n, p in rel.iter_rows()}
    pos = lambda c: rates.get(c, {}).get("pos")  # noqa: E731
    v53 = {"identical": pos("identical") is not None and pos("identical") >= 0.945,
           "shared_near_mostly_decoy": pos("shared_near") is not None and pos("shared_near") <= 0.15,
           "fraction_leans_negative": pos("fraction") is None or pos("fraction") < 0.5,
           "letter_leans_positive": pos("letter") is None or pos("letter") > 0.5}
    r.check("V5.3", {"rates_similar_name_and_addr": rates, "tests": v53},
            "identical ≥ 99% pos; shared+near ≈ 90% neg; fraction < 0.5; letter > 0.5 (±5 pts, §9c)", all(v53.values()))
    r.check("V5.4", {"one_none": rates.get("one_none")}, "≥ 95% positive (missing ≠ different)",
            pos("one_none") is not None and pos("one_none") >= 0.95)

    flt = [k for k in num if tr.schema[k].is_float()]
    infs = {k: int(tr[k].is_infinite().sum()) for k in flt}
    nulls = {k: tr[k].null_count() + (int(tr[k].is_nan().sum()) if k in flt else 0) for k in feats}
    undoc = {k: v for k, v in nulls.items() if v and not NULL_OK.search(k)}
    r.check("V5.5", {"inf": {k: v for k, v in infs.items() if v}, "undocumented_nulls": undoc,
                     "documented_null_pattern": NULL_OK.pattern, "sampled_rows": tr.height},
            "no inf; NaN/null only in documented 'missing' columns", not any(infs.values()) and not undoc)

    both = [k for k in num if k in se_x]  # a train/test column mismatch is V5.1's failure, not a crash here
    drift = {}
    for c in ("India", "US"):
        a, b = tr.filter(pl.col("country") == c), te.filter(pl.col("country") == c)
        for k in both:
            v = psi(a[k].cast(pl.Float64).to_numpy(), b[k].cast(pl.Float64).to_numpy())
            if v > 0.25:
                drift[f"{c}/{k}"] = round(v, 3)
    r.check("V5.6", drift, "PSI < 0.25 per feature (India, US), or listed with a reason", not drift, "SOFT")

    fr = te.filter(pl.col("country") == "France")
    out = {}
    for k in both:
        lo, hi = tr[k].cast(pl.Float64).min(), tr[k].cast(pl.Float64).max()
        x = fr[k].cast(pl.Float64).drop_nulls()
        if lo is not None and x.len():
            share = float(((x >= lo) & (x <= hi)).mean())
            if share < 0.95:
                out[k] = round(share, 4)
    r.check("V5.7", {"france_rows": fr.height, "features_below_95pct_in_range": out},
            "≥ 95% of France rows in train range", fr.height > 0 and not out, "SOFT")
    ok = r.save()
    return v46("features", S5W, train, test) and ok


# ---------------------------------------------------------------- Stage 6
def stage6(train: str, test: str) -> bool:
    from s8_decide import calibrate, ece, fit_iso

    r = Report(6)
    sc = pl.scan_parquet(S6W / train / "scores.parquet")
    n_feat = pl.scan_parquet(S5W / train / "features.parquet").select(pl.len()).collect().item()
    v61 = sc.select(rows=pl.len(), null_p=pl.col("p").is_null().sum(),
                    out_of_range=(~pl.col("p").is_between(0, 1)).sum()).collect().row(0, named=True)
    r.check("V6.1", v61 | {"feature_rows": n_feat}, "OOF p for 100% of pairs, in [0, 1]",
            v61["rows"] == n_feat and v61["null_p"] == 0 and v61["out_of_range"] == 0)

    folds = pl.scan_parquet(S0W / "s1.parquet").select("s1", f0="fold")
    wrong = sc.select("s1", "fold").unique().join(folds, on="s1", how="left").filter(
        pl.col("f0").is_null() | (pl.col("f0") != pl.col("fold"))).select(pl.len()).collect().item()
    models = sorted(p.name for p in (S6W / train).glob("model_fold*.txt"))
    r.check("V6.2", {"S1_with_wrong_fold": wrong, "fold_models": models,
                     "training": "fold k model trains on fold != k, early-stops on an inner training fold (s6_score.py)"},
            "0 overlap; 5 fold models", wrong == 0 and len(models) == 5)

    rep_f = REPORTS / f"stage6_{train}.json"
    rep = json.loads(rep_f.read_text()) if rep_f.exists() else {}
    sh = rep.get("shuffled_label_auc")
    r.check("V6.3", {"shuffled_label_auc": sh, "oof_auc": rep.get("oof_auc")}, "0.50 ± 0.01 (higher = leakage)",
            sh is not None and abs(sh - 0.5) <= 0.01)

    s = sample(sc.select("s1", "label", "fold", "p")).collect()
    y, p, fo = s["label"].to_numpy().astype(np.float64), s["p"].to_numpy(), s["fold"].to_numpy()
    q = np.empty_like(p)
    for k in np.unique(fo):  # isotonic cross-fitted by fold, so the calibrated ECE is not in-sample
        q[fo == k] = calibrate(p[fo == k], *fit_iso(p[fo != k], y[fo != k]))
    e_raw, e_iso = ece(y, p), ece(y, q)
    r.check("V6.4", {"ece_raw": round(e_raw, 5), "ece_isotonic_crossfit": round(e_iso, 5), "sampled_pairs": len(p)},
            "ECE ≤ 0.02 (after isotonic, as Stage 8 uses it)", e_iso <= 0.02)

    top = list(rep.get("top_gain", {}))[:10]
    r.check("V6.5", {"top10": top}, "F4 (num_*) and F2 (name_left_*) in the top 10 by gain",
            any(c.startswith("num_") for c in top) and any(c.startswith("name_left") for c in top), "SOFT")

    ts = pl.scan_parquet(S6W / test / "scores.parquet")
    d = set_diff(pl.scan_parquet(S4W / test / "pairs.parquet"), ts)
    fq = (ts.join(country_of("test", (1,)).select(s1="entity_id", country="country"), on="s1")
          .group_by("country").agg(n=pl.len(), p50=pl.col("p").median(), p90=pl.col("p").quantile(0.9),
                                   top_share=(pl.col("p") >= 0.5).mean()).sort("country").collect())
    r.check("V6.6", {"missing_or_extra": d, "by_country": fq.to_dicts()}, "every test pair scored, France included",
            not any(d.values()) and "France" in fq["country"].to_list())

    req = (ROOT / "requirements.txt").read_text()
    pin = next((ln for ln in req.splitlines() if ln.startswith("lightgbm")), None)
    r.check("V6.7", {"model": "LightGBM gradient-boosted trees (5 fold models, far below 8B parameters)",
                     "licence": "MIT", "pin": pin, "neural_models_in_s4_s6": "none"},
            "≤ 8B params, MIT/Apache-2.0, recorded", pin is not None)
    ok = r.save()
    return v46("scores", S6W, train, test) and ok


if __name__ == "__main__":
    a = sys.argv[1:]
    runs = {"stage3": lambda: stage3(a[1:]), "stage4": lambda: stage4(*a[1:3]),
            "stage5": lambda: stage5(*a[1:3]), "stage6": lambda: stage6(*a[1:3])}
    if not a or a[0] not in runs or (a[0] != "stage3" and len(a) != 3):
        sys.exit(__doc__)
    ok = runs[a[0]]()
    print(f"{a[0]}: {'all HARD checks pass' if ok else 'HARD check failed'} (reports/verify_{a[0]}.json)")
    sys.exit(0 if ok else f"{a[0]} HARD check failed (see reports/verify_{a[0]}.json)")
