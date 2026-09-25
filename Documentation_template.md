# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TBD
**Team Members:** TBD
**Submission Date:** TBD

> Numbers marked *measured* come from the full data or a deterministic sample (see
> `DATA_NOTES.md`, `reports/`). Anything not yet measured is written as
> `TBD (<report file>)`; nothing here is estimated.

---

## 1. Executive Summary

A ten-stage pipeline: normalise and romanise the text (stdlib `unicodedata` only), learn
noise tables from train true pairs, retrieve candidates per country with ranked
char-3-gram TF-IDF search in **both directions** (S1 → record and record → S1) plus a
street+house-number key, score each pair with LightGBM on ~60 country-neutral
features, give each S2/S3 record at most one owner, and pick each S1's list with two
thresholds derived from the F0.5 maths ("top-1 cheap, extras strict"). Every stage has
written HARD/SOFT checks (`reports/verify_stage*.json`), and thresholds are tuned in three
validation worlds built from full train, two of them at test-like density.

---

## 2. Methodology

### 2.1 Problem Analysis

*Measured* (DATA_NOTES §§1–9c):

- **Scale:** train 2.21M S1 / 10.3M S2+S3; test 1.73M S1 / 9.97M S2+S3. Test has ~24% more
  S2/S3 records per S1 than train in every country (4.7 → 5.8).
- **France is 15.0% of test S1 and absent from train.** The country label is clean and
  agrees in 100% of true pairs, so it is used only to split the search, never as a feature.
- **Cardinality:** mean 3.46 true copies per S1; 5.58% singletons (all-empty scores 0.0558).
- **One owner per record:** all 7,638,365 matched ids belong to exactly one S1.
- **Names repeat:** 35.8% (US) / 44.4% (India) of S1 share their exact normalised name with
  another S1 (`primary care group` ×253). A name alone never decides a match.
- **Addresses are shared:** among close-address pairs, 6,622 non-matches vs 8,311 matches
  (*sample*). An address alone never decides a match either.
- **Near-twins** (similar name, neighbouring house number, owned by nobody) are the
  precision killers: "shares a number, another ≤ 20 apart" is 253 decoys vs 27 copies,
  identical numbers 5,397 copies vs 28 decoys (§9c).
- **Indian scripts:** 27% of Indian S2 names are in an Indic script (Devanagari, Telugu,
  Tamil, Gujarati, …), phonetic transliterations of English words (`प्राइवेट लिमिटेड` =
  "private limited").
- **No postcodes** in practice; state codes vs full names, city aliases, dropped components,
  perturbed house numbers in true copies (`2620`→`262`, `44`→`44D`).
- **Blocking cost:** boolean "shares a rare token" blocking needs ~11k candidates per S1 for
  98% recall, so retrieval must be ranked.

### 2.2 Solution Strategy

**Approach Type:** Blocking (ranked retrieval) + pairwise classifier + one-owner-per-record
constraint + per-S1 decision rule.
**Core Innovation:** Validation worlds that reproduce test density (World B: hidden S1 whose
copies become decoys; World B′: one-off decoys like the real ones), an explicit
house-number-difference taxonomy against near-twins, library-free romanisation of Indic
scripts from Unicode character names plus a dictionary learned from train pairs, and
reverse (record → S1) retrieval that also yields a "how contested is this record" signal.

| Stage | What it does |
|---|---|
| 0 Harness | Exact macro-F0.5 scorer (0.714 on the README example), 5 entity-grouped folds, Worlds A/B/B′ |
| 1 Normalise | Clean text, romanise non-Latin letters, split `dba`/`formerly`/domain alternates, parse house numbers (value, fraction, letter, `bis`) |
| 2 Knowledge | Per-split, per-country counts (IDF, name frequency, co-location), aliases, leftover-word log-odds, number-change stats |
| 3 Retrieve | Per country: V1 name and V2 address char-3-gram TF-IDF (forward and reverse), V3 street key |
| 4 Candidates | Union + dedupe = `candidate_pairs.tsv` |
| 5 Evidence | ~60 features in 8 groups |
| 6 Score | LightGBM, 5-fold OOF on train |
| 7 Ownership | Each record keeps its best S1 if it beats the runner-up by δ, else nobody |
| 8 Decide | Top-1 if q ≥ τ₁; each extra if q ≥ τ₂ |
| 9 Output | TSVs, official validator, zip |

**Validation worlds** (all relabelings of full train, Stage 0):

| World | Queries | Index | Records per queried S1 |
|---|---|---|---:|
| A | all 2.21M train S1 | all train S2/S3 | 4.7 |
| B | ~80% of S1 (20% hidden; their copies stay as decoy clusters) | all | 5.8 (V0.9) |
| B′ | ~66% of S1 (34% hidden; one random copy each stays as a one-off decoy) | reduced | 5.8 (V0.10) |

Every threshold must do well in all three; "better" means ≥ +0.002 macro F0.5 in A and B′
with no loss worse than −0.001 in B.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - Search is split by country (India / US / France); an assert guards unexpected labels.
  - **V1 name:** character-3-gram TF-IDF cosine over the cleaned, romanised and alternate names.
  - **V2 address:** character-3-gram TF-IDF cosine over the cleaned address.
  - **V3 street key:** exact (rare address word with S1-df ≤ 200, house number) match; a
    precision supplement.
  - **Two directions:** forward S1 → top-20 records per view, reverse record → top-20 S1
    per view. Top-k sparse products via `sparse-dot-topn`; no n-gram pruning.
- **Recall** (*measured*, recall@K on a 5k train-S1 sample against the full train index,
  `reports/recall_curve.csv`): union of all views and directions at K = 20:
  India **0.967**, US **0.993** (forward only: 0.951 / 0.988). Single views at K = 20
  forward: V1 name 0.59 / 0.71, V2 address 0.81 / 0.87 (India / US). Weakest slices at
  K = 20: India Indic-script names 0.888, empty address 0.891 (India) / 0.888 (US).
- **Candidate pairs generated:** TBD (reports/verify_stage4.json, V4.4)
- **Recall ceiling on full train (World A):** TBD (reports/verify_stage4.json, V4.3)
- **How you ensured true matches were not lost:**
  - K was chosen from the measured recall curve, not by feel (V3.1); reverse retrieval was
    kept because it adds unique recall (India @20 0.951 → 0.967).
  - Per-noise-slice recall is a HARD check (V3.5): Indic-script names, empty addresses,
    renamed/domain names and `dba` names must all be found.
  - `candidate_pairs.tsv` is exactly the set the model scores: Stage 9 asserts
    scored pairs == candidate pairs (V4.6) and final ⊆ candidates before writing.

---

## 4. Matching Model

**Features used** (all country-neutral; the country name is never a feature):
- **Name (F1):** rarity-weighted word overlap, 3-gram cosine, fuzzy ratio, each on clean,
  romanised and alternate names (best of).
- **Name leftovers (F2):** words left after matching, scored by learned log-odds of
  "leftover in true pairs vs leftover in hard negatives" (`center` harmless;
  `group`/`holdings`/`overseas` suspicious). Unseen words (e.g. French `sarl`) get a
  frequency-based near-neutral weight, never "suspicious".
- **Address (F3):** rarity-weighted word overlap, order-free per-component alignment,
  street and city match, alias-aware (`rd↔road`, `mn↔minnesota`).
- **House numbers (F4):** relation class (identical / one side missing / dropped or
  injected / truncation / shared + another ≤ 20 apart / far apart / letter added /
  fraction changed) plus the numeric gap.
- **Rarity and crowding (F5):** chain-name count and co-location count, size-normalised
  (per 100k S1, percentile) and local (same city), because test US is half the size of train US.
- **Missing info (F6):** empty address, no numbers, non-Latin name (missing ≠ different).
- **Competition (F7):** views/directions/ranks that found the pair; the record's best S1
  score vs this one; number of S1 claiming the record.
- **Source (F8):** S2 vs S3.
- Final feature list and importances: TBD (reports/verify_stage6.json, V6.5)

**Model type:** LightGBM binary classifier (MIT licence; far below the 8B-parameter cap),
trained on entity-grouped folds with retrieved (hard) negatives; out-of-fold scores for
all of train; isotonic calibration on OOF if ECE > 0.02 (V6.4). No pretrained text model
is used unless gates G4/G9 are won (multilingual MIT/Apache model only).
**Threshold selection method:** tuned on OOF scores to maximise macro F0.5 in Worlds A, B
and B′ together (best worst-case), using the exact scorer:
- **Ownership margin δ:** TBD (reports/verify_stage8.json, V8.1)
- **τ₁ (accept top-1):** TBD (reports/verify_stage8.json, V8.1)
- **τ₂ (accept each extra):** TBD (reports/verify_stage8.json, V8.1)

Why two thresholds: per S1, F0.5 = 1.25c / (c + 0.25k + w). A wrong ID costs 1, a missed
ID 0.25, so the first pick only has to beat the singleton hypothesis while each extra
needs p ≳ 0.75. τ₁ is still a real threshold because 45% of singletons have a plausible
look-alike. No country-specific thresholds (V8.4).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), OOF:** World A TBD / World B TBD / World B′ TBD
  (reports/verify_stage8.json, V8.2). Baselines: all-empty 0.0558 (V0.3); rule baseline B0
  TBD (V8.3).
- **Leaderboard:** TBD
- **Common false positives (wrong merges):** by design the risk concentrates in
  near-twins (same name, neighbouring house number, owned by no S1), chain names with an
  empty or weak address, and singletons with a look-alike. Measured FP breakdown: TBD
  (reports/verify_stage8.json, V8.6).
- **Common false negatives (missed matches):** records outside the candidate set (recall
  ceiling, §3), Indic-script names with little address overlap, and renamed records
  (`KELOONYX`-style) whose only link is the address, where decoys outnumber copies 5:1.
  Measured FN breakdown: TBD (reports/verify_stage8.json).

---

## 6. Conclusion

The design is driven by measured data facts: names and addresses are each ambiguous
alone, near-twin decoys are separated by house-number evidence, each record has at most
one owner, and the F0.5 maths makes the first pick cheap and extras expensive. Every
stage is gated by written checks and every threshold is tuned in test-density worlds
rather than on the leaderboard. Final scores: TBD (reports/verify_summary.md).

**Compliance:** no external data, APIs, geocoding or web lookups; the code has no network
imports or URLs (scan in `s9_output.py package`). **No third-party transliteration
library** (`unidecode`, `anyascii`, …): romanisation uses the Unicode character names in
Python's `unicodedata` plus a dictionary learned from train pairs (archi.md C6). Every
statistic on test uses test records only (no labels exist). Dependencies:
polars, numpy, pyarrow, scikit-learn, scipy, sparse-dot-topn, LightGBM (all permissive licences).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/s0_harness.py` … `src/s9_output.py`: one file per stage; each writes its
  artefacts to `work/s<N>/` and its checks to `reports/verify_stage<N>.json`.
- `src/test_*.py`: assert-based unit tests (seconds).
- `run_all.sh`: the single entry point, raw data → `output/matching_results.tsv` and
  `output/candidate_pairs.tsv`; stops on any failing HARD check.
- `README.md`: environment setup (Python 3.12.3, `uv pip install -r requirements.txt`),
  run commands, AWS instructions.
- `requirements.txt`: pinned versions.

### B. Additional Results

- Verification summary (all stages, gate verdicts): `reports/verify_summary.md`,
  `reports/gates.md`: TBD at final run.
- Per-check values and pass/fail: `reports/verify_stage0.json` … `verify_stage9.json`.
- Recall@K curve per view × direction × country × noise slice: `reports/recall_curve.csv`.
