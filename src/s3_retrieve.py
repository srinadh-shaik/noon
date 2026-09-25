"""Stage 3 — RETRIEVE: ranked candidate search per country, three views, two directions.

Usage:
  python src/s3_retrieve.py curve   # U1: recall@K on a 5k train-S1 sample vs the full train index
                                    #     -> reports/recall_curve.csv (V3.1-V3.3, V3.5, V3.6)
Views: V1 name char-3-gram TF-IDF, V2 address char-3-gram TF-IDF, V3 street key (rare word + number).
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import REPORTS, ROOT, uniform  # noqa: E402
import s2_knowledge  # noqa: E402

S1W, S0W = ROOT / "work/s1", ROOT / "work/s0"
KS = (5, 10, 20, 50, 100, 200)
REV_KS = (1, 2, 5, 10, 20)
COLS = ["entity_id", "country", "name_roman", "name_alts", "name_tokens", "addr_clean", "name_nonlatin"]


def view_text(df: pl.DataFrame, view: str) -> list[str]:
    if view == "V1":  # romanised name plus its dba/domain alternates
        return df.select(pl.concat_str("name_roman", pl.col("name_alts").list.join(" "), separator=" ")).to_series().to_list()
    return df["addr_clean"].to_list()


def vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32, min_df=2)


def topk(q, x, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k rows of x per row of q by cosine (TF-IDF rows are L2-normalised). Returns (qi, xi, rank)."""
    c = sp_matmul_topn(q, x.T.tocsr(), top_n=k, sort=True, n_threads=-1).tocsr()
    qi = np.repeat(np.arange(c.shape[0]), np.diff(c.indptr))
    rank = np.concatenate([np.arange(n) for n in np.diff(c.indptr)]) + 1 if c.nnz else np.array([], int)
    return qi, c.indices, rank


def log(msg: str, t0: float) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"  [{time.time() - t0:7.1f}s  peak {rss:5.1f} GB] {msg}", flush=True)


def recall_curve(n_sample: int = 5000) -> pl.DataFrame:
    t0 = time.time()
    s1_all = pl.read_parquet(S1W / "train_source1.parquet", columns=COLS)
    sample = s1_all.filter(pl.Series(uniform(s1_all["entity_id"], 11) < n_sample / s1_all.height))
    owner = pl.read_parquet(S0W / "rec.parquet", columns=["rec", "owner"]).drop_nulls()
    truth = owner.join(sample.select(owner="entity_id"), on="owner", how="semi").rename({"owner": "s1"})
    found = []  # rows: s1, rec, view, direction, rank
    for country in sorted(sample["country"].unique()):
        recs = pl.concat([pl.read_parquet(S1W / f"train_source{n}.parquet", columns=COLS) for n in (2, 3)]).filter(
            pl.col("country") == country)
        q = sample.filter(pl.col("country") == country)
        s1c = s1_all.filter(pl.col("country") == country)
        copies = recs.join(truth.filter(pl.col("s1").is_in(q["entity_id"].implode())), left_on="entity_id",
                           right_on="rec", how="semi")
        for view in ("V1", "V2"):
            vec = vectorizer()
            x = vec.fit_transform(view_text(recs, view))
            log(f"{country} {view}: index {x.shape[0]:,} x {x.shape[1]:,}, nnz {x.nnz:,}", t0)
            qi, xi, rank = topk(vec.transform(view_text(q, view)), x, max(KS))
            found.append(pl.DataFrame({"s1": q["entity_id"].gather(qi), "rec": recs["entity_id"].gather(xi),
                                       "rank": rank, "view": view, "direction": "fwd"}))
            log(f"{country} {view} forward top-{max(KS)} for {q.height:,} S1", t0)
            # reverse: each true copy searches the country's S1 index (fact 5: at most one owner)
            qi, xi, rank = topk(vec.transform(view_text(copies, view)), vec.transform(view_text(s1c, view)), max(REV_KS))
            found.append(pl.DataFrame({"s1": s1c["entity_id"].gather(xi), "rec": copies["entity_id"].gather(qi),
                                       "rank": rank, "view": view, "direction": "rev"}))
            log(f"{country} {view} reverse top-{max(REV_KS)} for {copies.height:,} copies", t0)
            del x, vec
        del recs
    key = s2_knowledge.candidate_pairs(sample.select(s1="entity_id"))
    found.append(key.with_columns(rank=pl.lit(1, pl.Int64), view=pl.lit("V3"), direction=pl.lit("key")))
    log("V3 street key", t0)
    found = pl.concat([f.with_columns(pl.col("rank").cast(pl.Int64)) for f in found])
    return score_curve(found, truth, sample, t0)


def slices(truth: pl.DataFrame, sample: pl.DataFrame) -> pl.DataFrame:
    """Noise slice of each true pair, from the copy's Stage 1 fields."""
    rec = pl.concat([pl.read_parquet(S1W / f"train_source{n}.parquet", columns=COLS) for n in (2, 3)]).join(
        truth, left_on="entity_id", right_on="rec", how="semi")
    t = truth.join(rec.rename({"entity_id": "rec"}), on="rec").join(
        sample.select(s1="entity_id", tok1="name_tokens"), on="s1")
    return t.select("s1", "rec", "country",
                    nonlatin_name="name_nonlatin",
                    empty_address=pl.col("addr_clean") == "",
                    dba=pl.col("name_alts").list.len() >= 2,
                    domain_or_handle=pl.col("name_alts").list.len() == 1,
                    no_shared_name_word=pl.col("name_tokens").list.set_intersection("tok1").list.len() == 0)


def score_curve(found: pl.DataFrame, truth: pl.DataFrame, sample: pl.DataFrame, t0: float) -> pl.DataFrame:
    sl = slices(truth, sample)
    names = ["all", "nonlatin_name", "empty_address", "dba", "domain_or_handle", "no_shared_name_word"]
    rows = []

    def add(view, direction, k, hit_pairs):
        h = sl.join(hit_pairs.unique(), on=["s1", "rec"], how="left", coalesce=True).with_columns(
            hit=pl.col("hit").fill_null(False))
        for name in names:
            part = h if name == "all" else h.filter(pl.col(name))
            for country, g in part.group_by("country"):
                rows.append({"view": view, "direction": direction, "country": country[0], "slice": name, "K": k,
                             "recall": round(g["hit"].mean(), 4), "n": g.height})

    for (view, direction), g in found.group_by("view", "direction"):
        for k in (REV_KS if direction == "rev" else KS if direction == "fwd" else (1,)):
            add(view, direction, k, g.filter(pl.col("rank") <= k).select("s1", "rec", hit=pl.lit(True)))
    for k in KS:  # union of every view and direction, forward at K, reverse at min(k, 5), key always
        u = found.filter(((pl.col("direction") == "fwd") & (pl.col("rank") <= k))
                         | ((pl.col("direction") == "rev") & (pl.col("rank") <= min(k, 5))) | (pl.col("view") == "V3"))
        add("union", "all", k, u.select("s1", "rec", hit=pl.lit(True)))
        if k == 50:  # V3.3: pairs found by one view only
            per = u.group_by("s1", "rec").agg(views=pl.col("view").unique())
            for view in ("V1", "V2", "V3"):
                only = per.filter((pl.col("views").list.len() == 1) & (pl.col("views").list.first() == view))
                add(f"unique_{view}", "all", k, only.select("s1", "rec", hit=pl.lit(True)))
    out = pl.DataFrame(rows).sort("view", "direction", "country", "slice", "K")
    REPORTS.mkdir(exist_ok=True)
    out.write_csv(REPORTS / "recall_curve.csv")
    log(f"recall curve: {out.height} rows -> reports/recall_curve.csv", t0)
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["curve"]:
        c = recall_curve()
        with pl.Config(tbl_rows=80, tbl_width_chars=160):
            print(c.filter(pl.col("slice") == "all").pivot("K", index=["view", "direction", "country"], values="recall"))
    else:
        sys.exit(__doc__)
