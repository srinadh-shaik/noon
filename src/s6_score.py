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

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import REPORTS, ROOT  # noqa: E402
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


def score(name: str) -> None:
    t0 = time.time()
    df = pl.read_parquet(S5W / name / "features.parquet")
    x, cols = matrix(df)
    y, fold = df["label"].to_numpy().astype(np.int8), df["fold"].to_numpy()
    oof, rounds, gain = np.zeros(len(y)), [], np.zeros(len(cols))
    folds = np.unique(fold)
    for j, k in enumerate(folds):
        tr, va = fold != k, fold == k
        # rounds come from an inner fold inside the training folds, so fold k's OOF never steers its own model
        inner = fold == folds[(j + 1) % len(folds)]
        es = lgb.train(PARAMS, lgb.Dataset(x[tr & ~inner], y[tr & ~inner], feature_name=cols, categorical_feature=["num_rel"]),
                       num_boost_round=3000, valid_sets=[lgb.Dataset(x[inner], y[inner], categorical_feature=["num_rel"])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        n = es.best_iteration
        del es
        m = lgb.train(PARAMS, lgb.Dataset(x[tr], y[tr], feature_name=cols, categorical_feature=["num_rel"]), num_boost_round=n)
        oof[va] = m.predict(x[va])
        (S6W / name).mkdir(parents=True, exist_ok=True)
        m.save_model(str(S6W / name / f"model_fold{k}.txt"))
        rounds.append(n)
        gain += m.feature_importance("gain")
        log(f"fold {k}: {n} rounds (inner early stop), AUC {auc(y[va], oof[va]):.5f}", t0)
    # V6.3 label-shuffle control: a model trained on shuffled labels must not beat chance on held-out S1
    rng = np.random.default_rng(0)
    tr, va = fold != fold.min(), fold == fold.min()
    ms = lgb.train(PARAMS, lgb.Dataset(x[tr], rng.permutation(y[tr]), feature_name=cols), num_boost_round=100)
    shuffled_auc = auc(y[va], ms.predict(x[va]))
    (S6W / name).mkdir(parents=True, exist_ok=True)
    df.select(ID).with_columns(p=pl.Series(oof)).write_parquet(S6W / name / "scores.parquet")
    top = sorted(zip(cols, gain / gain.sum()), key=lambda t: -t[1])[:15]
    rep = {"pairs": len(y), "positives": int(y.sum()), "oof_auc": auc(y, oof), "ece": ece(y, oof),
           "shuffled_label_auc": shuffled_auc, "rounds": rounds, "top_gain": {c: round(g, 4) for c, g in top}}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"stage6_{name}.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(json.dumps({k: v for k, v in rep.items() if k != "top_gain"}), t0)
    log("top features: " + ", ".join(f"{c} {g:.3f}" for c, g in top), t0)


def predict(model_from: str, name: str) -> None:
    """Score unlabelled pairs with the mean of the 5 fold models: the same scale as the OOF p the thresholds saw."""
    t0 = time.time()
    df = pl.read_parquet(S5W / name / "features.parquet")
    models = [lgb.Booster(model_file=str(f)) for f in sorted((S6W / model_from).glob("model_fold*.txt"))]
    assert models, f"no fold models in work/s6/{model_from}"
    cols = models[0].feature_name()
    x = df.select(pl.col(cols).cast(pl.Float32)).to_numpy()  # same columns, same order as training
    p = np.mean([m.predict(x) for m in models], axis=0)
    (S6W / name).mkdir(parents=True, exist_ok=True)
    df.select("s1", "rec").with_columns(p=pl.Series(p)).write_parquet(S6W / name / "scores.parquet")
    log(f"{len(p):,} pairs scored with {len(models)} fold models from work/s6/{model_from}", t0)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode in ("subset", "full"):
        score(mode)
    elif mode == "test":
        predict("full", "test")
    else:
        sys.exit(__doc__)
