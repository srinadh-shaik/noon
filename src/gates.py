"""reports/gates.md: one row per gate (archi.md C3/D2). Each stage fills in its own gates via set_gate()."""
from s0_harness import REPORTS

GATES = {
    "G1": "One owner per record?", "G2": "How many S1 have no match?", "G3": "How much of test is France?",
    "G4": "Cross-encoder on uncertain pairs beats LightGBM?", "G5": "Owner-or-none model beats best-pick?",
    "G6": "Expected-F0.5 rule beats two thresholds?", "G7": "Why does test have more records?",
    "G8": "Learned transliteration dictionary?", "G9": "Embedding view finds matches V1-V3 miss?",
    "G10": "France house-number mismatches: stricter France tau2?", "G11": "Ownership margin delta: abstain vs argmax?",
    "G12": "Transductive French aliases lift France?", "G13": "Pointwise vs LambdaRank for top-1?",
}
HEAD = "# Gate verdicts (archi.md D2)\n\n| Gate | Question | Verdict | Evidence |\n|---|---|---|---|\n"


def set_gate(gid: str, verdict: str, evidence: str) -> None:
    path = REPORTS / "gates.md"
    rows = {g: f"| {g} | {q} | ⏳ open | |" for g, q in GATES.items()}
    if path.exists():
        rows |= {ln.split("|")[1].strip(): ln for ln in path.read_text().splitlines() if ln.startswith("| G")}
    rows[gid] = f"| {gid} | {GATES[gid]} | {verdict} | {evidence} |"
    REPORTS.mkdir(exist_ok=True)
    path.write_text(HEAD + "\n".join(rows[g] for g in sorted(rows, key=lambda g: int(g[1:]))) + "\n")
