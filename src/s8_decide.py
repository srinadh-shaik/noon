"""Stage 8 — DECIDE: the final list per S1, tuned on out-of-fold scores in Worlds A, B and B' (runs the V7 checks too).

Usage:
  python src/s8_decide.py tune NAME          # work/s6/NAME/scores.parquet (train OOF) -> work/s8/NAME/{thresholds.json,
                                             #   grid.parquet} + reports/verify_stage7.json, verify_stage8.json, gates.md
  python src/s8_decide.py apply NAME TUNED [--tau2 X] [--france-tau2 Y]
                                             # work/s6/NAME/scores.parquet + work/s8/TUNED/thresholds.json
                                             #   -> work/s7/NAME/owned.parquet, work/s8/NAME/final.parquet [s1, rec]
                                             #   --tau2 / --france-tau2: leaderboard probes P1-P3 (gates G7, G10) only
Rules, both after Stage 7 ownership (margin delta):
  two thresholds  keep the top candidate if p >= tau1, each further one if p >= tau2 (archi.md Stage 8 default);
  expected F0.5   gate G6: per S1, the prefix length with the highest expected F0.5 under isotonic-calibrated p.
Every setting is picked for the best worst-case macro F0.5 over A/B/B'; an upgrade (delta > 0 for G11, the
expected-F0.5 rule for G6) is kept only if it clears archi.md D0.6's epsilon. No country-specific setting (V8.4).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from gates import set_gate  # noqa: E402
from s0_harness import REPORTS, ROOT, WORLDS, indexed, load, queries  # noqa: E402
from s7_ownership import checks as own_checks, own  # noqa: E402

S1W, S4W, S5W, S6W, S7W, S8W = (ROOT / f"work/{s}" for s in ("s1", "s4", "s5", "s6", "s7", "s8"))
TAU1 = np.round(np.arange(0.02, 0.91, 0.02), 2)
TAU2 = np.round(np.arange(0.30, 0.99, 0.02), 2)
DELTA = (None, 0.0, 0.05, 0.1, 0.2)  # None = ownership off (the V7.3 control)
EPS_GAIN, EPS_LOSS = 0.002, 0.001  # archi.md D0.6
M = 16  # the expected-F0.5 rule looks at each S1's top-M owned candidates
# ponytail: candidates past rank M are ignored by the G6 rule (their calibrated p is ~0 after ownership); raise M if not


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def f05(c: np.ndarray, n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Per-S1 F0.5 from right picks c, list size n and true copies k (same formula as s0_harness.f05)."""
    den = c + 0.25 * k + (n - c)
    return np.where(den == 0, 1.0, 1.25 * c / np.maximum(den, 1e-12))


def ranked(owned: pl.DataFrame, s1: pl.Series) -> pl.DataFrame:
    """owned [s1, rec, p, ...] -> + i (row of s1 in `s1`) and r (0 = best) inside each S1, ties broken by rec id."""
    return (owned.join(pl.DataFrame({"s1": s1}).with_row_index("i"), on="s1")
            .sort("i", "p", "rec", descending=[False, True, False])
            .with_columns(r=pl.int_range(pl.len()).over("i")))


def two_threshold(pairs: pl.DataFrame, tau1: float, tau2: float) -> pl.DataFrame:
    return pairs.filter(((pl.col("r") == 0) & (pl.col("p") >= tau1)) | ((pl.col("r") > 0) & (pl.col("p") >= tau2)))


def calibrate(p: np.ndarray, x: list, y: list) -> np.ndarray:
    return np.interp(p, x, y)


def fit_iso(p: np.ndarray, y: np.ndarray, cap: int = 5_000_000) -> tuple[list, list]:
    idx = np.random.default_rng(0).choice(len(p), cap, replace=False) if len(p) > cap else slice(None)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[idx], y[idx])
    return iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()


def ece(y: np.ndarray, p: np.ndarray, bins: int = 20) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def expected_len(q: np.ndarray, miss: float, chunk: int = 20_000) -> np.ndarray:
    """Rows of q = one S1's calibrated candidate probabilities, best first, 0-padded. Returns the prefix length n
    with the highest expected F0.5 = E[1.25c / (n + 0.25k)], exact under independence:
    c ~ PoissonBinomial(q[:n]) right picks, r ~ PoissonBinomial(q[n:]) true copies left out, k = c + r + miss
    (miss = copies the candidates never held, added deterministically). Empty answer: P(r = 0) * exp(-miss)."""
    m = q.shape[1]
    n, c, r = np.ogrid[: m + 1, : m + 1, : m + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(n > 0, 1.25 * c / (n + 0.25 * (c + r + miss)), ((c == 0) & (r == 0)) * np.exp(-miss))
    g = np.nan_to_num(g)
    out = np.empty(len(q), np.int64)
    for s in range(0, len(q), chunk):
        x = q[s: s + chunk]
        b = len(x)
        pre = np.zeros((b, m + 1, m + 1))  # pre[:, j] = distribution of hits among the first j candidates
        pre[:, 0, 0] = 1
        suf = np.zeros((b, m + 1, m + 1))  # suf[:, j] = distribution of hits among candidates j..m-1
        suf[:, m, 0] = 1
        for j in range(m):
            pre[:, j + 1] = pre[:, j] * (1 - x[:, j: j + 1])
            pre[:, j + 1, 1:] += pre[:, j, :-1] * x[:, j: j + 1]
            jj = m - 1 - j
            suf[:, jj] = suf[:, jj + 1] * (1 - x[:, jj: jj + 1])
            suf[:, jj, 1:] += suf[:, jj + 1, :-1] * x[:, jj: jj + 1]
        e = ((pre[:, :, None, :] @ g)[:, :, 0, :] * suf).sum(-1)  # (b, n)
        out[s: s + b] = e.argmax(1)
    return out


def expected_rule(pairs: pl.DataFrame, n_s1: int, iso, miss: float) -> pl.DataFrame:
    """pairs from ranked() -> the pairs the expected-F0.5 rule keeps."""
    top = pairs.filter(pl.col("r") < M)
    q = np.zeros((n_s1, M))
    q[top["i"].to_numpy(), top["r"].to_numpy()] = calibrate(top["p"].to_numpy(), *iso)
    return pairs.filter(pl.col("r") < pl.Series(expected_len(q, miss)).gather(pairs["i"]))


class World:
    """One world's query S1 and owned pairs as flat arrays, so each threshold setting scores in milliseconds."""

    def __init__(self, qs: pl.DataFrame, owned: pl.DataFrame):
        self.s1, self.k, self.n = qs["s1"], qs["k"].to_numpy().astype(np.float64), qs.height
        self.pairs = ranked(owned.select("s1", "rec", "p", "label"), self.s1)
        a = {c: self.pairs[c].to_numpy() for c in ("i", "r", "p", "label")}
        top = a["r"] == 0
        self.top = (a["i"][top], a["p"][top], a["label"][top].astype(np.float64))
        self.ext = (a["i"][~top], a["p"][~top], a["label"][~top].astype(np.float64))

    def _counts(self, part, t: float):
        i, p, lab = part
        sel = p >= t
        return np.bincount(i[sel], weights=lab[sel], minlength=self.n), np.bincount(i[sel], minlength=self.n)

    def grid(self) -> dict:
        tops, exts = {t: self._counts(self.top, t) for t in TAU1}, {t: self._counts(self.ext, t) for t in TAU2}
        return {(t1, t2): float(f05(tops[t1][0] + exts[t2][0], tops[t1][1] + exts[t2][1], self.k).mean())
                for t1 in TAU1 for t2 in TAU2 if t2 >= t1}

    def evaluate(self, chosen: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """chosen: a subset of self.pairs -> per-S1 (f, c, n)."""
        i = chosen["i"].to_numpy()
        c = np.bincount(i, weights=chosen["label"].to_numpy().astype(np.float64), minlength=self.n)
        n = np.bincount(i, minlength=self.n)
        return f05(c, n, self.k), c, n


def better(new: dict, old: dict) -> bool:
    """archi.md D0.6: >= +EPS_GAIN in A and B', no loss worse than -EPS_LOSS in B."""
    return new["A"] - old["A"] >= EPS_GAIN and new["Bp"] - old["Bp"] >= EPS_GAIN and new["B"] - old["B"] >= -EPS_LOSS


def robust(grids: dict, keys) -> tuple:
    return max(keys, key=lambda key: (min(grids[w][key] for w in WORLDS), grids["A"][key]))


def universe(name: str, s1_all: pl.DataFrame, sc: pl.DataFrame) -> pl.DataFrame:
    """The query S1 for NAME: work/s4/NAME/s1.parquet if Stage 4 wrote it, all train S1 for `full`, else scored S1."""
    f = S4W / name / "s1.parquet"
    if f.exists():
        return s1_all.join(pl.read_parquet(f).select("s1"), on="s1", how="semi")
    if name == "full":
        return s1_all
    print(f"WARNING: {f} missing; query S1 = S1 with >= 1 scored pair (S1 without candidates are left out)")
    return s1_all.join(sc.select("s1").unique(), on="s1", how="semi")


def slices(u: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """Per query S1: the archi.md Stage 0 slices that the Stage 1 fields can define."""
    s1f = pl.read_parquet(S1W / "train_source1.parquet", columns=["entity_id", "addr_clean"])
    rf = pl.concat([pl.read_parquet(S1W / f"train_source{n}.parquet", columns=["entity_id", "name_nonlatin", "name_alts"])
                    for n in (2, 3)])
    copies = (rec.drop_nulls("owner").select("rec", "owner").join(rf, left_on="rec", right_on="entity_id")
              .group_by("owner").agg(nonlatin_copy=pl.col("name_nonlatin").any(),
                                     renamed_copy=(pl.col("name_alts").list.len() > 0).any()))
    return (u.select("s1", "country", "k")
            .join(s1f.select(s1="entity_id", s1_addr_empty=pl.col("addr_clean") == ""), on="s1", how="left")
            .join(copies.rename({"owner": "s1"}), on="s1", how="left")
            .with_columns(pl.col("nonlatin_copy", "renamed_copy").fill_null(False),
                          k_bucket=pl.when(pl.col("k") >= 5).then(pl.lit("5+")).otherwise(pl.col("k").cast(pl.Utf8))))


def slice_report(sl: pl.DataFrame, s1: pl.Series, f: np.ndarray) -> dict:
    d = pl.DataFrame({"s1": s1, "f05": f}).join(sl, on="s1", how="left")
    out = {}
    for col in ("country", "k_bucket", "s1_addr_empty", "nonlatin_copy", "renamed_copy"):
        t = d.group_by(col).agg(n=pl.len(), f05=pl.col("f05").mean()).sort(col)
        out[col] = {str(r[col]): {"n": r["n"], "f05": round(r["f05"], 5)} for r in t.iter_rows(named=True)}
    return out


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def tune(name: str) -> bool:
    t0 = time.time()
    s1_all, rec = load()
    sc = pl.read_parquet(S6W / name / "scores.parquet", columns=["s1", "rec", "p", "label"])
    u = universe(name, s1_all, sc)
    sc = sc.join(u.select("s1"), on="s1", how="semi")
    log(f"{sc.height:,} scored pairs, {u.height:,} query S1", t0)

    p, y = sc["p"].to_numpy(), sc["label"].to_numpy().astype(np.float64)
    iso = fit_iso(p, y)
    miss = float((u["k"].sum() - y.sum()) / u.height)  # true copies per S1 that no candidate holds (World A)
    calib = {"ece_raw": round(ece(y, p), 5), "ece_isotonic": round(ece(y, calibrate(p, *iso)), 5), "miss_per_s1": round(miss, 5)}
    log(f"calibration {calib}", t0)

    data, grids, dp = {}, {w: {} for w in WORLDS}, {w: {} for w in WORLDS}
    for w in WORLDS:
        qs = queries(u, w).select("s1", "k")
        data[w] = sc.join(qs.select("s1"), on="s1", how="semi").join(indexed(rec, w).select("rec"), on="rec", how="semi")
        for delta in DELTA:
            world = World(qs, own(data[w], delta))
            grids[w] |= {(delta, t1, t2): v for (t1, t2), v in world.grid().items()}
            if delta is not None:
                dp[w][delta] = float(world.evaluate(expected_rule(world.pairs, world.n, iso, miss))[0].mean())
            log(f"World {w} delta={delta}: grid best {max(v for k, v in grids[w].items() if k[0] == delta):.5f}"
                + (f"  expected-F0.5 rule {dp[w][delta]:.5f}" if delta is not None else ""), t0)

    keys = list(grids["A"])
    at = lambda key: {w: round(grids[w][key], 5) for w in WORLDS}  # noqa: E731
    best0 = robust(grids, [k for k in keys if k[0] == 0.0])
    best_any = robust(grids, [k for k in keys if k[0] is not None])
    keep_margin = best_any[0] != 0.0 and better(at(best_any), at(best0))
    tt = best_any if keep_margin else best0
    set_gate("G11", f"✅ keep δ={tt[0]}" if keep_margin else "❌ plain argmax (δ=0)",
             f"best worst-case: δ=0 {at(best0)} vs δ={best_any[0]} {at(best_any)} (s8 tune {name} @ {git_sha()})")
    dp_delta = max(dp["A"], key=lambda d: min(dp[w][d] for w in WORLDS))
    dp_at = {w: round(dp[w][dp_delta], 5) for w in WORLDS}
    keep_dp = better(dp_at, at(tt))
    set_gate("G6", "✅ keep expected-F0.5 rule" if keep_dp else "❌ two thresholds (simpler)",
             f"expected-F0.5 (δ={dp_delta}) {dp_at} vs two thresholds {at(tt)} (s8 tune {name} @ {git_sha()})")
    rule, delta = ("expected_f05", dp_delta) if keep_dp else ("two_threshold", tt[0])
    sf = (S6W / name / "scores.parquet").stat()
    th = {"rule": rule, "delta": delta, "tau1": float(tt[1]), "tau2": float(tt[2]), "M": M, "miss": miss,
          "iso_x": iso[0], "iso_y": iso[1], "tuned_on": name, "git": git_sha(),
          "scores_file": {"bytes": sf.st_size, "mtime": sf.st_mtime},
          "chosen_by": "best worst-case macro F0.5 over A/B/Bp on OOF; upgrades need archi.md D0.6 epsilon"}
    (S8W / name).mkdir(parents=True, exist_ok=True)
    (S8W / name / "thresholds.json").write_text(json.dumps(th) + "\n")
    pl.DataFrame([{"delta": -1.0 if k[0] is None else k[0], "tau1": k[1], "tau2": k[2], **at(k)} for k in keys]).write_parquet(
        S8W / name / "grid.parquet")  # delta -1 = ownership off
    log(f"chosen: rule={rule} delta={delta} tau1={tt[1]} tau2={tt[2]}", t0)

    # final evaluation of the chosen setting, the baselines and the V7/V8 checks
    sl = slices(u, rec)
    feats_f, cand_f = S5W / name / "features.parquet", S4W / name / "pairs.parquet"
    feats = pl.read_parquet(feats_f, columns=["s1", "rec", "name_jaccard", "addr_jaccard"]) if feats_f.exists() else None
    cands = pl.read_parquet(cand_f, columns=["s1", "rec"]) if cand_f.exists() else None
    off = robust(grids, [k for k in keys if k[0] is None])
    res, v7 = {}, {}
    for w in WORLDS:
        qs = queries(u, w).select("s1", "k")
        owned = own(data[w], delta)
        world = World(qs, owned)
        chosen = expected_rule(world.pairs, world.n, iso, miss) if keep_dp else two_threshold(world.pairs, tt[1], tt[2])
        f, c, n = world.evaluate(chosen)
        k = world.k
        first = chosen.filter(pl.col("r") == 0)
        r = {"macro_f05": round(float(f.mean()), 5),
             "empty_baseline": round(float((k == 0).mean()), 5),
             "oracle_candidates": round(float(world.evaluate(world.pairs.filter("label"))[0].mean()), 5),
             "ownership_off_best": round(grids[w][off], 5),
             "singleton_accuracy": round(float((n[k == 0] == 0).mean()), 5) if (k == 0).any() else None,
             "top1_precision": round(float(first["label"].mean()), 5) if first.height else None,
             "nonsingleton_empty_rate": round(float((n[k > 0] == 0).mean()), 5),
             "mean_predicted": round(float(n.mean()), 4),
             "slices": slice_report(sl, world.s1, f)}
        if feats is not None:  # B0: name J >= 0.5 and address J >= 0.5 among the candidates, then ownership (V8.3)
            b0 = (feats.join(data[w].select("s1", "rec", "label"), on=["s1", "rec"])
                  .filter((pl.col("name_jaccard") >= 0.5) & (pl.col("addr_jaccard") >= 0.5))
                  .with_columns(p=pl.col("name_jaccard") + pl.col("addr_jaccard")))
            bw = World(qs, own(b0, 0.0))
            r["B0_rule"] = round(float(bw.evaluate(bw.pairs)[0].mean()), 5)
        if cands is not None:
            r["final_not_in_candidates"] = chosen.select("s1", "rec").join(cands, on=["s1", "rec"], how="anti").height
        res[w] = r
        w0 = World(qs, own(data[w], 0.0))
        v7[w] = own_checks(data[w], owned) | {"f05_delta0": round(float(w0.evaluate(
            expected_rule(w0.pairs, w0.n, iso, miss) if keep_dp else two_threshold(w0.pairs, tt[1], tt[2]))[0].mean()), 5)}
        log(f"World {w}: macro F0.5 {r['macro_f05']}  B0 {r.get('B0_rule')}  oracle {r['oracle_candidates']}  "
            f"off {r['ownership_off_best']}  singleton acc {r['singleton_accuracy']}  top-1 prec {r['top1_precision']}", t0)

    checks = []

    def check(cid, value, expected, ok, level="HARD"):
        checks.append({"id": cid, "value": value, "expected": expected, "pass": bool(ok), "level": level})
        print(f"{'PASS' if ok else 'FAIL'}  {cid:<5} {json.dumps(value, default=str)[:300]}  (expected {expected})")

    check("V7.1", {w: v7[w]["max_owners"] for w in WORLDS}, "max owners per record = 1", all(v7[w]["max_owners"] <= 1 for w in WORLDS))
    check("V7.2", {w: {x: v7[w][x] for x in ("pairs_not_in_input", "q_above_p")} for w in WORLDS}, "0 added pairs, 0 raised q",
          all(v7[w]["pairs_not_in_input"] == 0 and v7[w]["q_above_p"] == 0 for w in WORLDS))
    v73 = {w: {"on": res[w]["macro_f05"], "off": res[w]["ownership_off_best"]} for w in WORLDS}
    check("V7.3", v73, "on >= off in A and B'", all(v73[w]["on"] >= v73[w]["off"] for w in ("A", "Bp")))
    check("V7.4", {w: {"delta": delta, "abstain_rate": v7[w]["abstain_rate"],
                       "f05_minus_delta0": round(res[w]["macro_f05"] - v7[w]["f05_delta0"], 5)} for w in WORLDS},
          "reported", True, "SOFT")
    s7, checks = checks, []
    check("V8.1", {"file": str((S8W / name / "thresholds.json").relative_to(ROOT)), "rule": rule, "delta": delta,
                   "tau1": th["tau1"], "tau2": th["tau2"], "tuned_on": f"OOF scores {name}", "git": th["git"]},
          "tuned on OOF only; saved with the model version", True)
    check("V8.2", {w: {"macro_f05": res[w]["macro_f05"], "slices": res[w]["slices"]} for w in WORLDS}, "reported", True)
    b0 = res["A"].get("B0_rule")
    check("V8.3", {"A": res["A"]["macro_f05"], "B0": b0, "empty": res["A"]["empty_baseline"]},
          ">= B0 + 0.05 in World A and > all-empty",
          b0 is not None and res["A"]["macro_f05"] >= b0 + 0.05 and res["A"]["macro_f05"] > res["A"]["empty_baseline"])
    check("V8.4", {"per_country_settings": 0}, "none", True)
    v85 = {w: res[w].get("final_not_in_candidates") for w in WORLDS}
    check("V8.5", v85, "0 final pairs outside candidate_pairs", all(v == 0 for v in v85.values()))
    check("V8.6", {w: {x: res[w][x] for x in ("singleton_accuracy", "top1_precision", "nonsingleton_empty_rate")} for w in WORLDS},
          "reported (45% of singletons have a look-alike)", True, "SOFT")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "verify_stage7.json").write_text(json.dumps(s7, indent=1, default=str) + "\n")
    (REPORTS / "verify_stage8.json").write_text(json.dumps(checks, indent=1, default=str) + "\n")
    (REPORTS / f"stage8_{name}.json").write_text(json.dumps(
        {"thresholds": {k: v for k, v in th.items() if not k.startswith("iso")}, "calibration": calib, "worlds": res},
        indent=1, default=str) + "\n")
    hard = [c for c in s7 + checks if c["level"] == "HARD"]
    print(f"Stages 7-8  {sum(c['pass'] for c in hard)}/{len(hard)} HARD pass")
    return all(c["pass"] for c in hard)


def apply(name: str, tuned: str, tau2: float | None = None, france_tau2: float | None = None) -> bool:
    """Tuned thresholds -> final pairs. tau2 / france_tau2 override only for leaderboard probes (G7 P2, G10 P3)."""
    t0 = time.time()
    th = json.loads((S8W / tuned / "thresholds.json").read_text())
    sc = pl.read_parquet(S6W / name / "scores.parquet", columns=["s1", "rec", "p"])
    owned = own(sc, th["delta"])
    (S7W / name).mkdir(parents=True, exist_ok=True)
    owned.write_parquet(S7W / name / "owned.parquet")
    split = "test" if name.startswith("test") else "train"
    country = pl.read_parquet(S1W / f"{split}_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    s1 = owned["s1"].unique().sort()
    pairs = ranked(owned, s1)
    if th["rule"] == "expected_f05":
        assert tau2 is None and france_tau2 is None, "probe overrides apply to the two-threshold rule only"
        final = expected_rule(pairs, len(s1), (th["iso_x"], th["iso_y"]), th["miss"])
    else:
        t2 = pl.lit(tau2 if tau2 is not None else th["tau2"])
        if france_tau2 is not None:  # G10 probe P3: the one country-specific setting archi.md allows, leaderboard-decided
            pairs = pairs.join(country, on="s1", how="left")
            t2 = pl.when(pl.col("country") == "France").then(pl.lit(france_tau2)).otherwise(t2)
        final = pairs.filter(((pl.col("r") == 0) & (pl.col("p") >= th["tau1"])) | ((pl.col("r") > 0) & (pl.col("p") >= t2)))
    final = final.select("s1", "rec").sort("s1", "rec")
    (S8W / name).mkdir(parents=True, exist_ok=True)
    final.write_parquet(S8W / name / "final.parquet")

    by_c = (country.join(final.group_by("s1").len("n"), on="s1", how="left").with_columns(pl.col("n").fill_null(0))
            .group_by("country").agg(s1=pl.len(), empty_share=(pl.col("n") == 0).mean(), mean_list=pl.col("n").mean())
            .sort("country"))
    quant = (sc.join(country, on="s1").group_by("country")
             .agg(**{f"p{int(x * 100)}": pl.col("p").quantile(x) for x in (0.5, 0.9, 0.99)}).sort("country"))
    cand_f = S4W / name / "pairs.parquet"
    outside = final.join(pl.read_parquet(cand_f, columns=["s1", "rec"]), on=["s1", "rec"], how="anti").height if cand_f.exists() else None
    max_owners = final.group_by("rec").len()["len"].max() or 0
    rep = {"thresholds_from": tuned, "rule": th["rule"], "delta": th["delta"], "tau1": th["tau1"],
           "tau2": tau2 if tau2 is not None else th["tau2"], "france_tau2": france_tau2,
           "scored_pairs": sc.height, "owned_pairs": owned.height, "final_pairs": final.height,
           "V8.5_final_not_in_candidates": outside, "V7.1_max_owners": max_owners,
           "by_country": by_c.to_dicts(), "score_quantiles_by_country": quant.to_dicts()}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"stage8_apply_{name}.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    log(json.dumps({k: v for k, v in rep.items() if not k.startswith(("by_", "score_"))}, default=str), t0)
    print(by_c)
    return outside in (0, None) and max_owners <= 1


def flag(args: list[str], name: str) -> float | None:
    return float(args[args.index(name) + 1]) if name in args else None


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["tune"] and len(a) == 2:
        sys.exit(0 if tune(a[1]) else "Stage 7/8 HARD check failed (see reports/verify_stage7.json, verify_stage8.json)")
    elif a[:1] == ["apply"] and len(a) >= 3:
        sys.exit(0 if apply(a[1], a[2], flag(a, "--tau2"), flag(a, "--france-tau2")) else "Stage 8 apply check failed")
    else:
        sys.exit(__doc__)
