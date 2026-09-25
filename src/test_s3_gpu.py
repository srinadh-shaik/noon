"""GPU lexical search must return the CPU search's results. Run: .venv/bin/python src/test_s3_gpu.py"""
import time

import polars as pl

from s3_retrieve import COLS, S1W, search, search_gpu, vectorizer, view_text


def test(split="test", country="France", n=500, k=20):
    recs = pl.concat([pl.read_parquet(S1W / f"{split}_source{i}.parquet", columns=COLS) for i in (2, 3)]).filter(
        pl.col("country") == country)
    s1 = pl.read_parquet(S1W / f"{split}_source1.parquet", columns=COLS).filter(pl.col("country") == country).head(n)
    for view in ("V1", "V2"):
        vec = vectorizer()
        x = vec.fit_transform(view_text(recs, view))
        q = vec.transform(view_text(s1, view))
        t = time.time(); cpu = search(q, x.T.tocsr(), k); tc = time.time() - t
        t = time.time(); gpu = search_gpu(q, x, k); tg = time.time() - t
        c = {(a, b): s for a, b, s in zip(cpu[0], cpu[1], cpu[3])}
        g = {(a, b): s for a, b, s in zip(gpu[0], gpu[1], gpu[3])}
        # ties at the k-th score may be broken differently; compare everything strictly above the k-th score
        kth = {a: s for a, s in zip(cpu[0], cpu[3])}  # last write per query = its lowest kept score
        strict = {p for p, s in c.items() if s > kth[p[0]] + 1e-5}
        missing = [p for p in strict if p not in g]
        diff = max(abs(c[p] - g[p]) for p in c.keys() & g.keys())
        print(f"{view}: cpu {len(c)} pairs {tc:.2f}s, gpu {len(g)} pairs {tg:.2f}s, "
              f"above-kth missing on gpu {len(missing)}/{len(strict)}, max score diff {diff:.2e}")
        assert not missing and diff < 1e-4, view
    print("ok")


if __name__ == "__main__":
    test()
