"""Stage 6 — SCORE: LightGBM P(match) per candidate pair, 5-fold out-of-fold grouped by S1 (Stage 0 folds).

Usage:
  python src/s6_score.py subset|full   # OOF [s1, rec, label, fold, p] + 5 fold models -> work/s6/MODE/
  python src/s6_score.py test          # [s1, rec, p]: mean of work/s6/full's fold models (OOF scale)
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import REPORTS, ROOT, uniform  # noqa: E402
from s3_retrieve import log  # noqa: E402
from s5_features import S5W  # noqa: E402

S6W = ROOT / "work/s6"
ID = ["s1", "rec", "label", "fold"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y.sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 20) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def matrix(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in df.columns if c not in ID]
    return df.select(pl.col(cols).cast(pl.Float32)).to_numpy(), cols


TRAIN_S1 = 500_000     # archi Stage 6: fold models train on a subsample of S1, each with all its candidates
PRED_ROWS = 5_000_000  # rows scored per chunk (OOF and test), so no step holds every pair


def stream_scores(name: str, out: Path, predict_rows) -> tuple[np.ndarray, np.ndarray]:
    """Score work/s5/NAME/features.parquet chunk by chunk into OUT; returns (labels or [], p) for the report."""
    lf = pl.scan_parquet(S5W / name / "features.parquet")
    n_rows, writer, ys, ps = lf.select(pl.len()).collect().item(), None, [], []
    for a in range(0, n_rows, PRED_ROWS):
        part = lf.slice(a, PRED_ROWS).collect()
        p = predict_rows(part)
        table = part.select([c for c in ID if c in part.columns]).with_columns(p=pl.Series(p)).to_arrow()
        writer = writer or pq.ParquetWriter(out, table.schema)
        writer.write_table(table)
        ps.append(p)
        if "label" in part.columns:
            ys.append(part["label"].to_numpy().astype(np.int8))
    writer.close()
    return (np.concatenate(ys) if ys else np.array([])), np.concatenate(ps)


def score(name: str) -> None:
    t0 = time.time()
    lf = pl.scan_parquet(S5W / name / "features.parquet")
    s1f = lf.select("s1", "fold").unique().collect()
    sub = s1f.filter(pl.Series(uniform(s1f["s1"], 31) < TRAIN_S1 / s1f.height)) if s1f.height > TRAIN_S1 else s1f
    df = lf.join(sub.lazy().select("s1"), on="s1", how="semi").collect()
    log(f"training sample: {sub.height:,} of {s1f.height:,} S1, {df.height:,} pairs", t0)
    x, cols = matrix(df)
    y, fold = df["label"].to_numpy().astype(np.int8), df["fold"].to_numpy()
    models, rounds, gain = {}, [], np.zeros(len(cols))
    folds = np.unique(fold)
    (S6W / name).mkdir(parents=True, exist_ok=True)
    for j, k in enumerate(folds):
        tr = fold != k
        # rounds come from an inner fold inside the training folds, so fold k's OOF never steers its own model
        inner = fold == folds[(j + 1) % len(folds)]
        es = lgb.train(PARAMS, lgb.Dataset(x[tr & ~inner], y[tr & ~inner], feature_name=cols, categorical_feature=["num_rel"]),
                       num_boost_round=3000, valid_sets=[lgb.Dataset(x[inner], y[inner], categorical_feature=["num_rel"])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        n = es.best_iteration
        del es
        m = lgb.train(PARAMS, lgb.Dataset(x[tr], y[tr], feature_name=cols, categorical_feature=["num_rel"]), num_boost_round=n)
        m.save_model(str(S6W / name / f"model_fold{k}.txt"))
        models[k] = m
        rounds.append(n)
        gain += m.feature_importance("gain")
        log(f"fold {k}: {n} rounds (inner early stop)", t0)
    # V6.3 label-shuffle control: a model trained on shuffled labels must not beat chance on held-out S1
    rng = np.random.default_rng(0)
    tr, va = fold != fold.min(), fold == fold.min()
    ms = lgb.train(PARAMS, lgb.Dataset(x[tr], rng.permutation(y[tr]), feature_name=cols), num_boost_round=100)
    shuffled_auc = auc(y[va], ms.predict(x[va]))
    del x, df

    def oof_rows(part: pl.DataFrame) -> np.ndarray:  # every pair scored by the model that never saw its fold
        xp, fp, p = part.select(pl.col(cols).cast(pl.Float32)).to_numpy(), part["fold"].to_numpy(), np.zeros(part.height)
        for k, m in models.items():
            if (mk := fp == k).any():
                p[mk] = m.predict(xp[mk])
        return p

    y, oof = stream_scores(name, S6W / name / "scores.parquet", oof_rows)
    top = sorted(zip(cols, gain / gain.sum()), key=lambda t: -t[1])[:15]
    rep = {"pairs": len(y), "positives": int(y.sum()), "oof_auc": auc(y, oof), "ece": ece(y, oof),
           "shuffled_label_auc": shuffled_auc, "rounds": rounds, "train_s1": sub.height,
           "top_gain": {c: round(g, 4) for c, g in top}}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"stage6_{name}.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(json.dumps({k: v for k, v in rep.items() if k != "top_gain"}), t0)
    log("top features: " + ", ".join(f"{c} {g:.3f}" for c, g in top), t0)


def predict(model_from: str, name: str) -> None:
    """Score unlabelled pairs with the mean of the 5 fold models: the same scale as the OOF p the thresholds saw."""
    t0 = time.time()
    models = [lgb.Booster(model_file=str(f)) for f in sorted((S6W / model_from).glob("model_fold*.txt"))]
    assert models, f"no fold models in work/s6/{model_from}"
    cols = models[0].feature_name()  # same columns, same order as training
    (S6W / name).mkdir(parents=True, exist_ok=True)
    _, p = stream_scores(name, S6W / name / "scores.parquet", lambda part: np.mean(
        [m.predict(part.select(pl.col(cols).cast(pl.Float32)).to_numpy()) for m in models], axis=0))
    log(f"{len(p):,} pairs scored with {len(models)} fold models from work/s6/{model_from}", t0)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode in ("subset", "full"):
        score(mode)
    elif mode == "test":
        predict("full", "test")
    else:
        sys.exit(__doc__)
