"""Run: .venv/bin/python src/test_s9_output.py"""
import json
import tempfile
from pathlib import Path

import polars as pl

from s9_output import scan, write


def tsv(path: Path, ids: list[str]) -> None:
    pl.DataFrame({"entity_id": ids, "business_name": "x", "business_address": "y, z", "country": "US"}) \
        .write_csv(path, separator="\t")


FILE = {"s4": "pairs", "s6": "scores", "s8": "final"}


def put(work: Path, stage: str, pairs: list[tuple[str, str]]) -> None:
    (work / stage / "t").mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(pairs, schema=["s1", "rec"], orient="row").with_columns(p=pl.lit(0.5))
    df.write_parquet(work / stage / "t" / f"{FILE[stage]}.parquet")


def test():
    tmp = Path(tempfile.mkdtemp())
    data, work, out, rep = tmp / "test", tmp / "work", tmp / "out", tmp / "reports"
    data.mkdir()
    tsv(data / "test_source1.tsv", ["S1-4", "S1-1", "S1-2", "S1-3"])  # file order must be kept
    tsv(data / "test_source2.tsv", ["S2-10", "S2-11"])
    tsv(data / "test_source3.tsv", ["S3-20", "S3-21"])
    cands = [("S1-1", "S3-20"), ("S1-1", "S2-10"), ("S1-3", "S2-11"), ("S1-4", "S3-21")]
    put(work, "s4", cands)
    put(work, "s6", cands[::-1])
    put(work, "s8", [("S1-1", "S3-20"), ("S1-1", "S2-10")])

    kw = dict(test_dir=data, out=out, work=work, reports=rep)
    assert not write("t", **kw)  # first run: only V9.4 fails ("rerun to confirm")
    checks = {c["id"]: c for c in json.loads((rep / "verify_stage9.json").read_text())}
    assert [i for i, c in checks.items() if not c["pass"]] == ["V9.4", "V9.5"], checks
    assert checks["V9.4"]["value"]["status"] == "first run, rerun to confirm"
    assert write("t", **kw)  # rerun: identical hashes -> all HARD pass
    assert json.loads((rep / "verify_stage9.json").read_text())[3]["value"]["status"] == "identical"

    m = (out / "matching_results.tsv").read_text()
    c = (out / "candidate_pairs.tsv").read_text()
    assert m == "source1_entity_id\tmatched_entity_ids\nS1-4\t\nS1-1\tS2-10,S3-20\nS1-2\t\nS1-3\t\n", repr(m)
    assert c == "source1_entity_id\tcandidate_entity_ids\nS1-4\tS3-21\nS1-1\tS2-10,S3-20\nS1-2\t\nS1-3\tS2-11\n", repr(c)

    put(work, "s8", [("S1-2", "S2-10")])  # a match that was never a candidate
    try:
        write("t", **kw)
        raise RuntimeError("subset guard did not fire")
    except AssertionError as e:
        assert "subset" in str(e)
    put(work, "s8", [])
    put(work, "s6", cands[:3])  # scored set != candidate set (V4.6)
    try:
        write("t", **kw)
        raise RuntimeError("V4.6 guard did not fire")
    except AssertionError as e:
        assert "V4.6" in str(e)

    (tmp / "src").mkdir()
    (tmp / "src" / "a.py").write_text("import os, socket  # net\nx = 1  # see http" "s://example.org\n")
    (tmp / "src" / "b.py").write_text("M = 'intfloat/multilingual-e5-small'\nfrom pathlib import Path\n")
    assert len(scan(tmp / "src")) == 1  # the import; the URL in a comment is ignored
    print("ok")


if __name__ == "__main__":
    test()
