"""Stage 3 — V4 embedding view: a multilingual text encoder fine-tuned on train true pairs, exact GPU search.

Usage:
  python src/s3_embed.py train     # contrastive fine-tune on fold-0 true pairs -> work/s3/embed/model
  python src/s3_embed.py encode SPLIT  # fp16 vectors for every record -> work/s3/embed/SPLIT_source{n}.npy
  python src/s3_embed.py curve     # recall@K of V4 on the lexical curve's 5k S1, alone and in the union
                                   #   -> reports/recall_curve_embed.csv (gate G9: unique recall)
Base model: intfloat/multilingual-e5-small (MIT, 118M params). Input is the raw record text; the
encoder learns the noise (scripts, legal forms, typos) from the pairs, so no hand rules are involved.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import ROOT, uniform  # noqa: E402
from s3_retrieve import (COLS, CURVE_FOUND, KS, REV_K, S0W, S1W, curve_sample, curve_truth, log,  # noqa: E402
                         score_curve)

BASE = "intfloat/multilingual-e5-small"
EMB = ROOT / "work/s3/embed"
MODEL = EMB / "model"
MAX_LEN = 96          # tokens; Indian-script names tokenise long
N_TRAIN = 500_000     # true pairs used for fine-tuning (from fold-0 S1 only)
TEXT = ["entity_id", "country", "business_name", "business_address"]


def text(df: pl.DataFrame) -> list[str]:
    return df.select(pl.concat_str(pl.lit("query: "), pl.col("business_name").fill_null(""), pl.lit(" | "),
                                   pl.col("business_address").fill_null(""))).to_series().to_list()


def train() -> None:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    t0 = time.time()
    s1_all = pl.read_parquet(S1W / "train_source1.parquet", columns=TEXT)
    held = curve_sample(s1_all)["entity_id"]  # never train on the S1 the recall curve measures
    folds = pl.read_parquet(S0W / "s1.parquet", columns=["s1", "fold"])
    pool = folds.filter((pl.col("fold") == 0) & ~pl.col("s1").is_in(held.implode()))
    owner = pl.read_parquet(S0W / "rec.parquet", columns=["rec", "owner"]).drop_nulls()
    pairs = owner.join(pool.select(owner="s1"), on="owner", how="semi")
    pairs = pairs.with_columns(u=pl.Series(uniform(pairs["rec"], 13))).sort("u").head(N_TRAIN)
    recs = pl.concat([pl.read_parquet(S1W / f"train_source{n}.parquet", columns=TEXT) for n in (2, 3)])
    pairs = (pairs.join(s1_all.rename({"entity_id": "owner"}), on="owner")
             .join(recs.select(rec="entity_id", rb="business_name", ra="business_address"), on="rec"))
    anchor = text(pairs.select("business_name", "business_address"))
    positive = text(pairs.select(business_name="rb", business_address="ra"))
    log(f"{len(anchor):,} training pairs from {pool.height:,} fold-0 S1", t0)

    model = SentenceTransformer(BASE, device="cuda")
    model.max_seq_length = MAX_LEN
    args = SentenceTransformerTrainingArguments(
        output_dir=str(EMB / "ckpt"), num_train_epochs=1, per_device_train_batch_size=128, learning_rate=5e-5,
        warmup_ratio=0.05, bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,  # two copies of one S1 never share a batch
        logging_steps=500, save_strategy="no", seed=0, report_to="none", dataloader_num_workers=2)
    SentenceTransformerTrainer(model=model, args=args, loss=losses.MultipleNegativesRankingLoss(model),
                               train_dataset=Dataset.from_dict({"anchor": anchor, "positive": positive})).train()
    model.save(str(MODEL))
    log(f"saved {MODEL}", t0)


def encode(model, df: pl.DataFrame, chunk: int = 200_000) -> np.ndarray:
    """fp16 unit vectors, encoded in chunks: encoding 4M+ records at once in fp32 runs out of RAM (15 GB)."""
    out, t0 = np.empty((df.height, model.get_sentence_embedding_dimension()), np.float16), time.time()
    for a in range(0, df.height, chunk):
        out[a:a + chunk] = model.encode(text(df.slice(a, chunk)), batch_size=512, convert_to_numpy=True,
                                        normalize_embeddings=True, show_progress_bar=False)
        if df.height > chunk:
            log(f"    encoded {min(a + chunk, df.height):,}/{df.height:,} ({(a + chunk) / (time.time() - t0):,.0f}/s)", t0)
    return out


def gpu_topk(q: np.ndarray, x: np.ndarray, k: int, qchunk: int = 2048, xblock: int = 500_000):
    """Exact top-k by cosine (rows are unit vectors). The index is streamed to the GPU in blocks (8 GB card).
    Returns (query row, index row, rank from 1, score)."""
    xs = [torch.from_numpy(x[i:i + xblock]) for i in range(0, len(x), xblock)]  # views, no copy (15 GB RAM)
    qi_all, xi_all, sc_all = [], [], []
    for a in range(0, len(q), qchunk):
        qg = torch.from_numpy(q[a:a + qchunk]).cuda()
        vals, idx = [], []
        for bi, blk in enumerate(xs):
            s = qg @ blk.cuda().T
            v, i = s.topk(min(k, s.shape[1]), dim=1)
            vals.append(v)
            idx.append(i + bi * xblock)
        v, j = torch.cat(vals, 1).topk(k, dim=1)
        xi_all.append(torch.cat(idx, 1).gather(1, j).cpu().numpy())
        sc_all.append(v.float().cpu().numpy())
        qi_all.append(np.repeat(np.arange(a, a + len(qg)), k))
    xi, sc = np.concatenate(xi_all).ravel(), np.concatenate(sc_all).ravel()
    return np.concatenate(qi_all), xi, np.tile(np.arange(1, k + 1), len(xi) // k), sc


def encode_split(split: str) -> None:
    """Encode every record of the split once (fp16, parquet row order) -> work/s3/embed/{split}_source{n}.npy."""
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(str(MODEL), device="cuda").half()  # fp16 inference: same vectors to ~1e-3, 2-3x faster
    model.max_seq_length = MAX_LEN
    for n in (1, 2, 3):
        path = EMB / f"{split}_source{n}.npy"
        if not path.exists():
            np.save(path, encode(model, pl.read_parquet(S1W / f"{split}_source{n}.parquet", columns=TEXT)))
            log(f"saved {path.name}", t0)


def vectors(split: str, n: int, rows: np.ndarray) -> np.ndarray:
    return np.asarray(np.load(EMB / f"{split}_source{n}.npy", mmap_mode="r")[rows])


def curve() -> None:
    t0 = time.time()
    encode_split("train")
    s1_all = pl.read_parquet(S1W / "train_source1.parquet", columns=TEXT).with_row_index("i")
    recs_all = pl.concat([pl.read_parquet(S1W / f"train_source{n}.parquet", columns=TEXT).with_row_index("i")
                          .with_columns(src=pl.lit(n)) for n in (2, 3)])
    sample = curve_sample(s1_all)
    truth = curve_truth(sample)
    found = []
    for country in sorted(sample["country"].unique()):
        recs = recs_all.filter(pl.col("country") == country)
        q, s1c = sample.filter(pl.col("country") == country), s1_all.filter(pl.col("country") == country)
        xr = np.concatenate([vectors("train", n, recs.filter(pl.col("src") == n)["i"].to_numpy()) for n in (2, 3)])
        recs = pl.concat([recs.filter(pl.col("src") == n) for n in (2, 3)])  # same row order as xr
        qi, xi, rank, _ = gpu_topk(vectors("train", 1, q["i"].to_numpy()), xr, max(KS))
        found.append(pl.DataFrame({"s1": q["entity_id"].gather(qi), "rec": recs["entity_id"].gather(xi),
                                   "rank": rank, "view": "V4", "direction": "fwd"}))
        copies = recs.with_row_index("j").join(truth.select(entity_id="rec"), on="entity_id", how="semi")
        qi, xi, rank, _ = gpu_topk(xr[copies["j"].to_numpy()], vectors("train", 1, s1c["i"].to_numpy()), REV_K)
        found.append(pl.DataFrame({"s1": s1c["entity_id"].gather(xi), "rec": copies["entity_id"].gather(qi),
                                   "rank": rank, "view": "V4", "direction": "rev"}))
        log(f"{country}: V4 forward {q.height:,} S1, reverse {copies.height:,} copies", t0)
        del xr
    v4 = pl.concat([f.with_columns(pl.col("rank").cast(pl.Int64)) for f in found])
    lexical = pl.read_parquet(CURVE_FOUND) if CURVE_FOUND.exists() else v4.clear()
    scored_sample = curve_sample(pl.read_parquet(S1W / "train_source1.parquet", columns=COLS))
    score_curve(pl.concat([lexical, v4]), truth, scored_sample, t0, name="recall_curve_embed.csv")


if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["encode"] and len(sys.argv) > 2:
        encode_split(sys.argv[2])
    else:
        {"train": train, "curve": curve}.get(cmd[0] if cmd else "", lambda: sys.exit(__doc__))()
