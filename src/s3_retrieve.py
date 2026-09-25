"""Stage 3 — RETRIEVE: ranked candidate search per country, three views, two directions.

Usage:
  python src/s3_retrieve.py curve              # U1: recall@K on a 5k train-S1 sample vs the full train index
                                               #     -> reports/recall_curve.csv (V3.1-V3.3, V3.5, V3.6)
  python src/s3_retrieve.py build SPLIT [N]    # full retrieval -> work/s3/SPLIT/COUNTRY/*.parquet (resumable);
                                               #     N = only the first N queries per direction (timing smoke test)
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


REV_K = 20        # reverse list per record (user decision 2026-09-25: 5 -> 20)


def vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32, min_df=2)


def search(q, xt, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Top-k index rows per query row by cosine (TF-IDF rows are L2-normalised); xt = index transposed, CSR.
    Returns (query row, index row, rank from 1, score)."""
    c = sp_matmul_topn(q, xt, top_n=k, sort=True, n_threads=-1).tocsr()
    counts = np.diff(c.indptr)
    rank = np.arange(c.nnz) - np.repeat(c.indptr[:-1], counts) + 1
    return np.repeat(np.arange(c.shape[0]), counts), c.indices, rank, c.data


def search_gpu(q, x, k: int, score_bytes: float = 1.5e9) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """search() on the GPU, same cosine: the index x stays sparse (CSR) on the device, queries are densified in
    chunks sized so the (index rows x chunk) score block fits in `score_bytes`. Takes x itself, not x.T."""
    import torch
    xg = torch.sparse_csr_tensor(torch.from_numpy(x.indptr).long(), torch.from_numpy(x.indices).long(),
                                 torch.from_numpy(x.data), size=x.shape, device="cuda")
    step = max(1, int(score_bytes / 4 / x.shape[0]))
    out_q, out_x, out_r, out_s = [], [], [], []
    for a in range(0, q.shape[0], step):
        qd = torch.from_numpy(q[a:a + step].toarray()).cuda()
        v, i = (xg @ qd.T).topk(min(k, x.shape[0]), dim=0)          # (k, chunk)
        v, i = v.T.cpu().numpy(), i.T.cpu().numpy()                 # (chunk, k), best first
        keep = v > 0                                                # the CPU search returns non-zero scores only
        out_q.append(np.repeat(np.arange(a, a + len(v)), keep.sum(1)))
        out_x.append(i[keep])
        out_r.append((np.cumsum(keep, 1))[keep])
        out_s.append(v[keep])
    return tuple(np.concatenate(o) for o in (out_q, out_x, out_r, out_s))


def topk(q, x, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return search(q, x.T.tocsr(), k)[:3]


def log(msg: str, t0: float) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"  [{time.time() - t0:7.1f}s  peak {rss:5.1f} GB] {msg}", flush=True)


CURVE_FOUND = ROOT / "work/s3/curve_found_lexical.parquet"


def curve_sample(s1_all: pl.DataFrame, n_sample: int = 5000) -> pl.DataFrame:
    """The fixed 5k train-S1 sample every recall curve is measured on (lexical and embedding alike)."""
    return s1_all.filter(pl.Series(uniform(s1_all["entity_id"], 11) < n_sample / s1_all.height))


def curve_truth(sample: pl.DataFrame) -> pl.DataFrame:
    owner = pl.read_parquet(S0W / "rec.parquet", columns=["rec", "owner"]).drop_nulls()
    return owner.join(sample.select(owner="entity_id"), on="owner", how="semi").rename({"owner": "s1"})


def recall_curve() -> pl.DataFrame:
    t0 = time.time()
    s1_all = pl.read_parquet(S1W / "train_source1.parquet", columns=COLS)
    sample = curve_sample(s1_all)
    truth = curve_truth(sample)
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
    CURVE_FOUND.parent.mkdir(parents=True, exist_ok=True)
    found.write_parquet(CURVE_FOUND)  # the embedding curve adds V4 to these and re-scores the union
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


def score_curve(found: pl.DataFrame, truth: pl.DataFrame, sample: pl.DataFrame, t0: float,
                name: str = "recall_curve.csv") -> pl.DataFrame:
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
    for k in KS:  # union of every view and direction: forward at K, reverse at REV_K, street key always
        u = found.filter(((pl.col("direction") == "fwd") & (pl.col("rank") <= k))
                         | ((pl.col("direction") == "rev") & (pl.col("rank") <= REV_K)) | (pl.col("view") == "V3"))
        add("union", "all", k, u.select("s1", "rec", hit=pl.lit(True)))
        if k == 50:  # V3.3: pairs found by one view only
            per = u.group_by("s1", "rec").agg(views=pl.col("view").unique())
            for view in sorted(found["view"].unique()):
                only = per.filter((pl.col("views").list.len() == 1) & (pl.col("views").list.first() == view))
                add(f"unique_{view}", "all", k, only.select("s1", "rec", hit=pl.lit(True)))
    out = pl.DataFrame(rows).sort("view", "direction", "country", "slice", "K")
    REPORTS.mkdir(exist_ok=True)
    out.write_csv(REPORTS / name)
    log(f"recall curve: {out.height} rows -> reports/{name}", t0)
    return out


# ------------------------------------------------------------------ full-scale build (resumable chunks)
S3W = ROOT / "work/s3"
FWD_K = 20        # forward list per S1 (user decision 2026-09-25)
CHUNK_Q = 20_000  # queries per written part; a rerun skips parts that already exist


def build(split: str, limit: int | None = None) -> None:
    t0 = time.time()
    s1 = pl.read_parquet(S1W / f"{split}_source1.parquet", columns=COLS)
    recs = pl.concat([pl.read_parquet(S1W / f"{split}_source{n}.parquet", columns=COLS) for n in (2, 3)])
    countries = sorted(s1["country"].unique())
    # V3.4: country is a clean, closed label in every file, so the per-country search loses no true pair
    bad = recs.filter(pl.col("country").is_null() | ~pl.col("country").is_in(countries))
    assert s1["country"].null_count() == 0 and bad.is_empty(), f"unexpected country labels: {bad.head(3)}"
    for country in countries:
        q, r = s1.filter(pl.col("country") == country), recs.filter(pl.col("country") == country)
        out_dir = S3W / (split if limit is None else f"{split}_limit{limit}") / country  # a smoke run never poisons the resumable full run
        out_dir.mkdir(parents=True, exist_ok=True)
        for view in ("V1", "V2"):
            vec = vectorizer()
            x = vec.fit_transform(view_text(r, view))  # IDF from this split's own records
            s = vec.transform(view_text(q, view))
            log(f"{split}/{country} {view}: {q.height:,} S1, index {x.shape[0]:,} x {x.shape[1]:,}", t0)
            for direction, qm, tm, qid, tid, k in (("fwd", s, x, q["entity_id"], r["entity_id"], FWD_K),
                                                   ("rev", x, s, r["entity_id"], q["entity_id"], REV_K)):
                tt, n, t1 = tm.T.tocsr(), min(limit or qm.shape[0], qm.shape[0]), time.time()
                for start in range(0, n, CHUNK_Q):
                    part = out_dir / f"{view}_{direction}_{start:09d}.parquet"
                    if part.exists():
                        continue
                    qi, ti, rank, score = search(qm[start:min(start + CHUNK_Q, n)], tt, k)
                    a, b = qid.gather(qi + start), tid.gather(ti)
                    pl.DataFrame({"s1": a if direction == "fwd" else b, "rec": b if direction == "fwd" else a,
                                  "view": view, "direction": direction, "rank": rank.astype(np.uint16),
                                  "score": score}).write_parquet(part)
                ms = 1000 * (time.time() - t1) / max(n, 1)
                log(f"{split}/{country} {view} {direction}: {n:,} queries, {ms:.1f} ms/query "
                    f"(full {qm.shape[0]:,} ~ {ms * qm.shape[0] / 3.6e6:.1f} h)", t0)
            del x, s, vec
        part = out_dir / "V3_key.parquet"
        if not part.exists():
            key = s2_knowledge.candidate_pairs(q.head(limit or q.height).select(s1="entity_id"), split)
            key.with_columns(view=pl.lit("V3"), direction=pl.lit("key"), rank=pl.lit(1, pl.UInt16),
                             score=pl.lit(1.0, pl.Float32)).write_parquet(part)
            log(f"{split}/{country} V3 street key: {key.height:,} pairs", t0)
    parts = pl.scan_parquet(out_dir.parent / "*" / "*.parquet")
    bad_ids = parts.filter(~pl.col("s1").str.starts_with("S1-") | pl.col("rec").str.starts_with("S1-"))
    assert bad_ids.select(pl.len()).collect().item() == 0, "V3.4: a candidate pair has the wrong id types"
    log(f"{split}: {parts.select(pl.len()).collect().item():,} retrieval rows", t0)


if __name__ == "__main__":
    if sys.argv[1:2] == ["curve"]:
        c = recall_curve()
        with pl.Config(tbl_rows=80, tbl_width_chars=160):
            print(c.filter(pl.col("slice") == "all").pivot("K", index=["view", "direction", "country"], values="recall"))
    elif sys.argv[1:2] == ["build"] and len(sys.argv) > 2:
        build(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        sys.exit(__doc__)
