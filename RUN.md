# Business Entity Resolution: how to reproduce

One command turns the raw challenge data into `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. No external data, APIs or downloads: the pipeline reads
only the provided TSVs.

## 1. Layout

Put the challenge's `student_resource/` folder (with `dataset/` and `utils/`) here:

```
business_entity_resolution/          # = repo root (ROOT)
├── src/                             # s0_harness.py … s9_output.py, test_*.py
├── run_all.sh
├── requirements.txt
└── data/6ab10eb3b23ba_student_resource/student_resource/
    ├── dataset/{train,test}/*.tsv
    └── utils/validate_submission.py
```

## 2. Environment

Python **3.12.3**, dependencies pinned in `requirements.txt`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Unit tests (seconds, no data needed except Stage 9's validator script):

```bash
for t in src/test_*.py; do .venv/bin/python "$t"; done
```

## 3. Run

```bash
bash run_all.sh                 # all 22 steps, prints wall-clock per step
FROM=9 bash run_all.sh          # resume from step 9 (artefacts of steps 1–8 are reused)
PY=/path/to/python bash run_all.sh
```

| Step | Command (`python src/…`) | Writes |
|---:|---|---|
| 1 | `s0_harness.py build` | `work/s0/` folds + Worlds A/B/B′ |
| 2 | `s1_normalise.py` | `work/s1/` normalised records |
| 3 | `s2_knowledge.py` | `work/s2/` counts + lexicon |
| 4 | `s3_retrieve.py curve` | `reports/recall_curve.csv` |
| 5–6 | `s4_candidates.py full` / `test` | `work/s4/{full,test}/{s1,pairs,reverse}.parquet` |
| 7–8 | `verify_s3_6.py stage3 full test` / `stage4 full test` | `reports/verify_stage3.json`, `verify_stage4.json` (V3.*, V4.1–4.5) |
| 9–10 | `s5_features.py full` / `test` | `work/s5/{full,test}/features.parquet` |
| 11 | `verify_s3_6.py stage5 full test` | `reports/verify_stage5.json` (V5.*), V4.6 features == pairs |
| 12–13 | `s6_score.py full` / `test` | `work/s6/{full,test}/scores.parquet` (OOF on train; test = mean of 5 fold models) |
| 14 | `verify_s3_6.py stage6 full test` | `reports/verify_stage6.json` (V6.*), V4.6 scores == pairs |
| 15–16 | `s7_owner_model.py fit full` / `predict test full` | G5 owner-or-none q: `work/s7/{full,test}/owner_q.parquet` |
| 17 | `s8_decide.py tune full --owner-model` | G5 candidate thresholds: `work/s8/full_owner/` |
| 18 | `s8_decide.py tune full` | thresholds tuned on OOF in Worlds A/B/B′ (δ, τ₁, τ₂, G6, G11) + V7/V8 reports |
| 19 | `s7_owner_model.py gate full` | G5 verdict in `reports/gates.md` |
| 20 | `s8_decide.py apply test full` (or `full_owner --owner-model` if G5 kept) | `work/s8/test/final.parquet` |
| 21 | `s9_output.py write test` | `output/*.tsv` + official validator |
| 22 | `s9_output.py summary` | `reports/verify_summary.md` |

Every stage writes `reports/verify_stage<N>.json` and exits non-zero when a HARD check
fails, which stops `run_all.sh`. Read the failing check in the report; to continue anyway
(a deliberate override), resume at the next step: `FROM=<failed step + 1> bash run_all.sh`.

**V9.4 (reproducibility):** the first run records the output hashes as `first run, rerun to
confirm` (pending, not failing). `package` refuses until a rerun gives `identical`:
`FROM=20 bash run_all.sh` (minutes) re-derives the final lists and files; a full clean
rerun (`bash run_all.sh` from a fresh checkout) is the strict release check (D4).

**Leaderboard probes (G7, G10),** two-threshold rule, same model:
`s8_decide.py apply test full --tau2 <X>` with X = `probe.tau2_A` (P1) or `probe.tau2_Bp` (P2)
from `work/s8/full/thresholds.json`; P3 adds `--france-tau2 <stricter>`. Then `s9_output.py write test`.

Optional ID-existence check (a few GB of RAM): `python src/s9_output.py write test --check-ids` (V9.5).
Package: `python src/s9_output.py package TEAM` → `TEAM_submission.zip` (refuses if a Stage 9
HARD check fails or the no-network scan finds a hit).

**Rule:** if anything in Stages 1–6 changes, re-run steps 15–19 (G5 model + `s8_decide.py tune full`)
before applying. Old thresholds never carry over.

## 4. AWS

- **Instance:** `g5.8xlarge` or `g6.8xlarge` (32 vCPU, 128 GB RAM, one 24 GB GPU). A CUDA GPU is
  **required**: Stage 4 runs the TF-IDF search on GPU (`search_gpu`). Memory peaks in `s4_candidates.py full`
  (~90M forward + ~300M reverse rows held at once, ≥ 64 GB); `s8_decide.py tune full` needs ~22 GB.
  A 64 GB box (`g5.4xlarge`) may work but has no headroom. Use ≥ 200 GB gp3 disk.
- `s3_retrieve.py curve` (step 4, ~20 min CPU) is the recall report (V3.1); Stage 4 does its own search.
- **Setup:**
  ```bash
  git clone <repo> noon && cd noon          # or unzip code/business_entity_resolution/
  mkdir -p data/6ab10eb3b23ba_student_resource
  aws s3 cp --recursive s3://<bucket>/student_resource data/6ab10eb3b23ba_student_resource/student_resource
  # or: rsync -a student_resource/ ec2-user@<host>:noon/data/6ab10eb3b23ba_student_resource/student_resource/
  curl -LsSf https://astral.sh/uv/install.sh | sh    # then section 2
  nohup bash run_all.sh > run.log 2>&1 &             # tail -f run.log
  ```
- **Artefacts:** `output/matching_results.tsv`, `output/candidate_pairs.tsv` (one row per
  test S1, 1,732,544 rows each); intermediate stages in `work/`; checks in `reports/`
  (`verify_stage*.json`, `verify_summary.md`, `recall_curve.csv`, `gates.md`). Copy back:
  `aws s3 cp --recursive output s3://<bucket>/output` and the same for `reports`.
