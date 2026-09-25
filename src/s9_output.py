"""Stage 9 — OUTPUT: submission TSVs + official validator (V9.*), release zip, run summary.

Usage:
  python src/s9_output.py write NAME [--check-ids] [--test-dir D] [--out D] [--work D] [--reports D]
        # work/s4/NAME/pairs.parquet + work/s6/NAME/scores.parquet + work/s8/NAME/final.parquet
        # -> OUT/{matching_results,candidate_pairs}.tsv + reports/verify_stage9.json (V9.1-V9.5)
        # --test-dir is relative to student_resource/ (default dataset/test); V9.4 passes only on a rerun
  python src/s9_output.py package TEAM   # D4 compliance scan, then ROOT/TEAM_submission.zip
  python src/s9_output.py summary        # reports/verify_summary.md (archi.md D3)
"""
import argparse
import io
import json
import re
import subprocess
import sys
import tokenize
import zipfile
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import DATA, REPORTS, ROOT, read_tsv, sha256  # noqa: E402

SR = DATA.parent  # student_resource/: the validator's cwd
OUT = ROOT / "output"
HEADS = {"matching_results.tsv": "matched_entity_ids", "candidate_pairs.tsv": "candidate_entity_ids"}
MODS = r"(?:requests|urllib\d?|http|httpx|aiohttp|socket)"
IMPORT = re.compile(rf"^\s*(?:from\s+{MODS}\b|import\s+(?:.*,\s*)?{MODS}\b)")
URL = re.compile(r"https?://(?!huggingface\.co/)")  # a model id by name / its HF page is not a data lookup


def guard(pairs: pl.DataFrame, scores: pl.DataFrame, final: pl.DataFrame) -> None:
    for name, df in (("pairs", pairs), ("scores", scores), ("final", final)):
        assert not df.is_duplicated().any(), f"duplicate (s1, rec) in {name}"
    assert pairs.join(scores, on=["s1", "rec"], how="anti").is_empty() and \
        scores.join(pairs, on=["s1", "rec"], how="anti").is_empty(), "V4.6: scored pairs != candidate pairs"
    assert final.join(pairs, on=["s1", "rec"], how="anti").is_empty(), "final matches not a subset of candidates"
    assert pairs["rec"].str.contains(r"^S[23]-").all(), "a candidate id is not S2-/S3- (self-match?)"
    for col in ("s1", "rec"):
        assert not pairs[col].str.contains(r"[\t,\n\"]").any(), f"{col} id contains tab/comma/newline/quote"


def write_tsv(s1: pl.DataFrame, pairs: pl.DataFrame, path: Path) -> None:
    lists = pairs.group_by("s1").agg(pl.col("rec").sort().str.join(","))
    (s1.join(lists, on="s1", how="left", maintain_order="left")
       .with_columns(pl.col("rec").fill_null(""))
       .rename({"s1": "source1_entity_id", "rec": HEADS[path.name]})
       .write_csv(path, separator="\t", quote_style="never", line_terminator="\n"))


def shape(path: Path) -> tuple[bool, int, bool]:
    """(exact header, data rows, no quote chars)."""
    with open(path, encoding="utf-8", newline="") as f:
        head = f.readline() == f"source1_entity_id\t{HEADS[path.name]}\n"
        rows, quotes = 0, False
        for line in f:
            rows += 1
            quotes |= '"' in line
    return head, rows, not quotes


def validate(files: dict, test_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "utils/validate_submission.py", "--matching", str(files["matching_results.tsv"]),
           "--candidate", str(files["candidate_pairs.tsv"]), "--test-dir", str(test_dir), *extra]
    return subprocess.run(cmd, cwd=SR, capture_output=True, text=True)


def write(name: str, test_dir: Path = Path("dataset/test"), out: Path = OUT, work: Path = ROOT / "work",
          reports: Path = REPORTS, check_ids: bool = False) -> bool:
    s1 = read_tsv(SR / test_dir / "test_source1.tsv", ["entity_id"]).rename({"entity_id": "s1"})
    pairs = pl.read_parquet(work / f"s4/{name}/pairs.parquet", columns=["s1", "rec"])
    scores = pl.read_parquet(work / f"s6/{name}/scores.parquet", columns=["s1", "rec"])
    final = pl.read_parquet(work / f"s8/{name}/final.parquet", columns=["s1", "rec"])
    guard(pairs, scores, final)
    del scores
    out.mkdir(parents=True, exist_ok=True)
    files = {f: (out / f).resolve() for f in HEADS}
    write_tsv(s1, final, files["matching_results.tsv"])
    write_tsv(s1, pairs, files["candidate_pairs.tsv"])

    checks = []

    def check(cid: str, value, expected: str, ok, level: str = "HARD") -> None:
        checks.append({"id": cid, "value": value, "expected": expected, "pass": bool(ok), "level": level})
        print(f"{'PASS' if ok else 'FAIL'}  {cid:<5} {value}  (expected {expected}) [{level}]")

    run = validate(files, test_dir)
    print(run.stdout, run.stderr, sep="")
    check("V9.1", {"exit": run.returncode, "PASS": "PASS" in run.stdout}, "validator PASS, exit 0",
          run.returncode == 0 and "PASS" in run.stdout)
    warn = "not present in candidate_pairs.tsv" in run.stdout
    check("V9.2", {"matched_not_in_candidates_warning": warn}, "no such warning", not warn)
    shapes = {f: dict(zip(("header", "rows", "no_quotes"), shape(p))) for f, p in files.items()}
    check("V9.3", shapes, f"exact header, {s1.height:,} data rows (one per test S1), no quotes",
          all(s["header"] and s["rows"] == s1.height and s["no_quotes"] for s in shapes.values()))

    prev_path = reports / "verify_stage9.json"
    prev = {c["id"]: c["value"] for c in json.loads(prev_path.read_text())} if prev_path.exists() else {}
    hashes = {f: sha256(p) for f, p in files.items()}
    old = {f: prev.get("V9.4", {}).get(f) for f in HEADS}
    status = "first run, rerun to confirm" if not any(old.values()) else "identical" if old == hashes else "changed"
    check("V9.4", {**hashes, "status": status}, "identical to the previous run's hashes", status == "identical")

    if check_ids:
        ids = validate(files, test_dir, "--check-ids")
        print(ids.stdout, ids.stderr, sep="")
        check("V9.5", {"exit": ids.returncode, "PASS": "PASS" in ids.stdout}, "--check-ids PASS",
              ids.returncode == 0 and "PASS" in ids.stdout, "SOFT")
    else:
        check("V9.5", "skipped", "--check-ids PASS (run write with --check-ids)", False, "SOFT")

    reports.mkdir(parents=True, exist_ok=True)
    prev_path.write_text(json.dumps(checks, indent=1, default=str) + "\n")
    hard = [c for c in checks if c["level"] == "HARD"]
    print(f"Stage 9  {sum(c['pass'] for c in hard)}/{len(hard)} HARD pass")
    return all(c["pass"] for c in hard)


def scan(src: Path) -> list[str]:
    """D4: network imports or URLs in code (comments ignored; docstrings count)."""
    hits = []
    for path in sorted(src.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                r, c = tok.start
                lines[r - 1] = lines[r - 1][:c]
        hits += [f"{path.name}:{i}: {ln.strip()}" for i, ln in enumerate(lines, 1) if IMPORT.search(ln) or URL.search(ln)]
    return hits


def package(team: str) -> Path:
    v9 = REPORTS / "verify_stage9.json"
    assert v9.exists(), "no reports/verify_stage9.json: run `s9_output.py write test` first"
    bad = [c["id"] for c in json.loads(v9.read_text()) if c["level"] == "HARD" and not c["pass"]]
    assert not bad, f"Stage 9 HARD check(s) failing: {bad} (V9.4 needs a rerun with identical files)"
    hits = scan(ROOT / "src")
    print("\n".join(["D4 compliance scan hits:", *hits]) if hits else "D4 compliance scan: 0 hits")
    assert not hits, "network import / URL in src/*.py"
    code = "code/business_entity_resolution"
    members = {f"output/{f}": OUT / f for f in HEADS}
    members |= {f"{code}/src/{p.name}": p for p in sorted((ROOT / "src").glob("*.py"))}
    members |= {f"{code}/README.md": ROOT / "RUN.md", f"{code}/requirements.txt": ROOT / "requirements.txt",
                f"{code}/run_all.sh": ROOT / "run_all.sh", "Documentation_template.md": ROOT / "Documentation_template.md"}
    missing = [str(p) for p in members.values() if not p.exists()]
    assert not missing, f"missing: {missing}"
    zpath = ROOT / f"{team}_submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, p in members.items():
            z.write(p, arc)
    print(f"wrote {zpath} ({len(members)} files, {zpath.stat().st_size / 1e6:.1f} MB)")
    return zpath


def short(v, n: int = 90) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"


def summary(reports: Path = REPORTS) -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "no-git"
    stages = {int(p.stem.removeprefix("verify_stage")): json.loads(p.read_text())
              for p in reports.glob("verify_stage[0-9]*.json")}
    worlds = "TBD (reports/verify_stage8.json)"
    v82 = next((c["value"] for c in stages.get(8, []) if c["id"] == "V8.2"), None)
    if isinstance(v82, dict) and all(isinstance(v82.get(w), dict) and "macro_f05" in v82[w] for w in ("A", "B", "Bp")):
        worlds = " / ".join(f"{v82[w]['macro_f05']:.4f}" for w in ("A", "B", "Bp"))
    lines = [f"Run: {sha} {date.today()}  Worlds: A / B / B′ macro F0.5 = {worlds}"]
    for n in range(10):
        if n not in stages:
            lines.append(f"Stage {n}  no report (reports/verify_stage{n}.json missing)")
            continue
        hard = [c for c in stages[n] if c["level"] == "HARD"]
        fails = [f"{c['level']} fail: {c['id']} ({short(c['value'])}; expected {c['expected']})"
                 for c in stages[n] if not c["pass"]]
        lines.append(f"Stage {n}  {sum(c['pass'] for c in hard)}/{len(hard)} HARD pass"
                     + "".join(f"\n         {f}" for f in fails))
    gates = reports / "gates.md"
    table = [ln for ln in gates.read_text().splitlines() if ln.startswith("|")] if gates.exists() else []
    md = "# Verification summary\n\n```\n" + "\n".join(lines) + "\n```\n\n## Gates\n\n"
    md += "\n".join(table) + "\n\nSource: [reports/gates.md](gates.md)\n" if table else "reports/gates.md: not written yet\n"
    (reports / "verify_summary.md").write_text(md)
    print(md)
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("name")
    w.add_argument("--check-ids", action="store_true")
    w.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    w.add_argument("--out", type=Path, default=OUT)
    w.add_argument("--work", type=Path, default=ROOT / "work")
    w.add_argument("--reports", type=Path, default=REPORTS)
    sub.add_parser("package").add_argument("team")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "write":
        ok = write(a.name, a.test_dir, a.out.resolve(), a.work.resolve(), a.reports.resolve(), a.check_ids)
        sys.exit(0 if ok else "Stage 9 HARD check failed (see reports/verify_stage9.json)")
    elif a.cmd == "package":
        package(a.team)
    else:
        summary()
