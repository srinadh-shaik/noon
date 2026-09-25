"""Stage 7 upgrade (gate G5): each record chooses its owner among {its candidate S1s, none}.

Usage:
  python src/s7_owner_model.py fit TRAIN_NAME           # work/s6/TRAIN_NAME/scores.parquet (OOF) -> work/s7/TRAIN_NAME/
                                                        #   owner_q.parquet [s1, rec, q] (OOF, grouped by the S1 folds)
                                                        #   + owner_model.txt (fitted on every train claim)
  python src/s7_owner_model.py predict TEST_NAME TRAIN_NAME
                                                        # work/s6/TEST_NAME/scores.parquet + TRAIN_NAME's model
                                                        #   -> work/s7/TEST_NAME/owner_q.parquet
  python src/s7_owner_model.py gate NAME                # work/s8/NAME/grid.parquet (plain best-pick with margin) vs
                                                        #   work/s8/NAME_owner/grid.parquet (s8 tune NAME --owner-model)
                                                        #   -> G5 verdict in reports/gates.md (archi.md D0.6 epsilon)
A claim is one (s1, rec, p) row. LightGBM scores each claim from features relative to the record's other claims and to
the S1's other candidates (all country-neutral, archi.md Stage 5 rule); per record the scores are normalised so the
claims sum to <= 1 and the remainder is P(none). s8_decide.py --owner-model uses q in place of p, before ownership.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from gates import set_gate  # noqa: E402
from s0_harness import ROOT, WORLDS  # noqa: E402
from s7_ownership import prefilter  # noqa: E402  same pair set as s8_decide (p >= FLOOR + runner-ups)

S5W, S6W, S7W, S8W = (ROOT / f"work/{s}" for s in ("s5", "s6", "s7", "s8"))
S5_COLS = ["v1_n_claims", "v2_n_claims", "v1_gap_to_best", "v2_gap_to_best"]  # joined only if Stage 5 wrote them
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9,
              lambda_l2=1.0, verbose=-1, seed=0, num_threads=0)
ROUNDS = 300  # fixed, so no fold is used for early stopping (honest OOF) and the final model matches the OOF ones


def features(sc: pl.DataFrame, s5: Path | None = None) -> pl.DataFrame:
    """sc [s1, rec, p, ...] -> sc sorted by (rec, p desc, s1) + the claim features."""
    n = pl.len().over("rec")
    df = (sc.sort("rec", "p", "s1", descending=[False, True, False])
          .with_columns(claim_rank=pl.int_range(pl.len()).over("rec"), n_claims=n,
                        p_best=pl.col("p").max().over("rec"),
                        p_second=pl.when(n > 1).then(pl.col("p").top_k(2).min().over("rec")).otherwise(0.0),
                        sum_others=pl.col("p").sum().over("rec") - pl.col("p"),
                        s1_rank=pl.col("p").rank("ordinal", descending=True).over("s1") - 1,
                        s1_n=pl.len().over("s1"), s1_top=pl.col("p").max().over("s1"))
          .with_columns(gap_best=pl.col("p_best") - pl.col("p"), gap_second=pl.col("p") - pl.col("p_second"),
                        p_minus_mean_others=pl.col("p") - pl.col("sum_others") / (pl.col("n_claims") - 1).clip(1))
          .drop("sum_others"))
    if s5 is not None and s5.exists():
        have = [c for c in S5_COLS if c in pl.read_parquet_schema(s5)]
        if have:
            df = df.join(pl.read_parquet(s5, columns=["s1", "rec", *have]), on=["s1", "rec"], how="left")
    return df


def feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("s1", "rec", "label", "fold")]


def train(df: pl.DataFrame, cols: list[str]) -> lgb.Booster:
    x = df.select(cols).to_numpy().astype(np.float32)
    return lgb.train(PARAMS, lgb.Dataset(x, df["label"].to_numpy().astype(np.int8), feature_name=cols), ROUNDS)


def raw(m: lgb.Booster, df: pl.DataFrame) -> np.ndarray:
    return m.predict(df.select(m.feature_name()).to_numpy().astype(np.float32))


def normalise(df: pl.DataFrame, s: np.ndarray) -> pl.DataFrame:
    """Per-claim scores s -> q = s / max(1, sum of the record's s); 1 - sum(q) is P(none)."""
    # ponytail: independent per-claim scores rescaled, not a true softmax over {claims, none}; a listwise
    #   (multiclass/LambdaRank-style) model per record is the upgrade if G5 is close
    return (df.select("s1", "rec").with_columns(s=pl.Series(s))
            .with_columns(q=pl.col("s") / pl.max_horizontal(pl.lit(1.0), pl.col("s").sum().over("rec")))
            .select("s1", "rec", "q"))


def oof(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Train on folds != k, predict fold k (the Stage 0 S1 folds carried by scores.parquet)."""
    out, fold = np.full(df.height, np.nan), df["fold"].to_numpy()
    for k in np.unique(fold):
        out[fold == k] = raw(train(df.filter(pl.Series(fold != k)), cols), df.filter(pl.Series(fold == k)))
        print(f"fold {k}: {int((fold == k).sum()):,} claims", flush=True)
    return out


def fit(name: str) -> None:
    df = features(prefilter(pl.read_parquet(S6W / name / "scores.parquet", columns=["s1", "rec", "p", "label", "fold"])),
                  S5W / name / "features.parquet")
    cols = feature_cols(df)
    print(f"{df.height:,} claims, features {cols}", flush=True)
    q = normalise(df, oof(df, cols))
    (S7W / name).mkdir(parents=True, exist_ok=True)
    q.write_parquet(S7W / name / "owner_q.parquet")
    train(df, cols).save_model(str(S7W / name / "owner_model.txt"))
    lab = df.select("s1", "rec", "label").join(q, on=["s1", "rec"])
    print(f"wrote {S7W / name}: mean q on owners {lab.filter('label')['q'].mean():.4f}, "
          f"on non-owners {lab.filter(~pl.col('label'))['q'].mean():.4f}")


def predict(name: str, train_name: str) -> None:
    m = lgb.Booster(model_file=str(S7W / train_name / "owner_model.txt"))
    df = features(prefilter(pl.read_parquet(S6W / name / "scores.parquet", columns=["s1", "rec", "p"])), S5W / name / "features.parquet")
    miss = set(m.feature_name()) - set(df.columns)
    assert not miss, f"{name} lacks features the {train_name} model uses: {miss}"
    (S7W / name).mkdir(parents=True, exist_ok=True)
    normalise(df, raw(m, df)).write_parquet(S7W / name / "owner_q.parquet")
    print(f"wrote {S7W / name / 'owner_q.parquet'} ({df.height:,} claims)")


def best(grid: pl.DataFrame) -> dict:
    """Best worst-case over A/B/Bp among ownership-on settings (delta >= 0), as s8_decide.robust picks."""
    g = grid.filter(pl.col("delta") >= 0).with_columns(worst=pl.min_horizontal(*WORLDS)).sort("worst", "A", descending=True)
    return g.row(0, named=True)


def gate(name: str) -> bool:
    from s8_decide import better
    base, cand = (best(pl.read_parquet(S8W / n / "grid.parquet")) for n in (name, f"{name}_owner"))
    keep = better(cand, base)
    fmt = lambda r: {k: r[k] for k in ("delta", "tau1", "tau2", *WORLDS)}  # noqa: E731
    set_gate("G5", "✅ keep owner-or-none model" if keep else "❌ plain best-pick with margin",
             f"best worst-case (two thresholds): owner model {fmt(cand)} vs best-pick {fmt(base)} (s7_owner_model gate {name})")
    print(f"G5 {'keep' if keep else 'drop'}: owner {fmt(cand)} vs best-pick {fmt(base)}")
    return keep


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["fit"] and len(a) == 2:
        fit(a[1])
    elif a[:1] == ["predict"] and len(a) == 3:
        predict(a[1], a[2])
    elif a[:1] == ["gate"] and len(a) == 2:
        gate(a[1])  # a verdict, not a check: exits 0 either way
    else:
        sys.exit(__doc__)
