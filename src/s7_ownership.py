"""Stage 7 — OWNERSHIP: each S2/S3 record keeps at most one S1 owner (fact 5, gate G1).

Default rule (archi.md Stage 7): a record keeps its highest-scoring S1 only if that claim beats the
runner-up by a margin delta; nearly tied claims go to nobody. Every other claim is dropped, so the
output is always a subset of the input with q = p (it can only remove, never add or raise).

Used by src/s8_decide.py, which tunes delta jointly with tau1/tau2 in Worlds A/B/B' and runs the
V7.* checks (V7.3 needs the tuned decision rule on top). No CLI of its own.
"""
import polars as pl


def own(df: pl.DataFrame, delta: float | None) -> pl.DataFrame:
    """df [s1, rec, p, ...] -> kept rows plus q (= p). delta=None switches ownership off (V7.3 control).

    Ties on p are broken by s1 id so the result is deterministic; with delta > 0 a tie always abstains.
    """
    if delta is None:
        return df.with_columns(q=pl.col("p"))
    df = df.sort("rec", "p", "s1", descending=[False, True, False])
    second = pl.col("p").shift(-1).over("rec").fill_null(0.0)  # rows are p-sorted inside each rec
    return (df.with_columns(_lead=pl.col("p") - second)
            .filter(pl.col("rec").is_first_distinct() & (pl.col("_lead") >= delta))
            .drop("_lead").with_columns(q=pl.col("p")))


def checks(before: pl.DataFrame, after: pl.DataFrame) -> dict:
    """V7.1 (max owners per record), V7.2 (after ⊆ before, no q > p) and the V7.4 abstention rate."""
    max_owners = after.group_by("rec").len()["len"].max() or 0
    extra = after.join(before, on=["s1", "rec"], how="anti").height
    raised = after.join(before.select("s1", "rec", p0="p"), on=["s1", "rec"]).filter(pl.col("q") > pl.col("p0")).height
    n_rec, n_kept = before["rec"].n_unique(), after["rec"].n_unique()
    return {"max_owners": max_owners, "pairs_not_in_input": extra, "q_above_p": raised,
            "records": n_rec, "records_kept": n_kept, "abstain_rate": round(1 - n_kept / max(n_rec, 1), 5)}
