# Business Entity Resolution — System Design Report
Amazon ML Challenge 2026 · Design specification (the "what", not the "how")

---

## 1. Objective, translated into system requirements

### 1.1 What is actually scored
The unit of scoring is the **Source 1 entity**. F0.5 is computed per S1 entity and then averaged, so every S1 entity counts equally. An entity with 20 candidates counts no more than a singleton.

This creates two separate scoring regimes, one for each kind of entity:

| S1 entity type | Prediction | Entity score |
|---|---|---|
| Singleton (no true matches) | Empty list | **1.0** |
| Singleton | Any non-empty list | **0.0** |
| Has true matches | Empty list | **0.0** |
| Has 1 true match {a} | {a} | 1.0 |
| Has 1 true match {a} | {a, wrong} | 0.556 |
| Has 1 true match {a} | {wrong} | 0.0 |
| Has 2 true matches {a, b} | {a} (under-predict by 1) | 0.833 |
| Has 2 true matches {a, b} | {a, b, wrong} (over-predict by 1) | 0.714 |

Three consequences follow for the system:
1. **The single most consequential decision per entity is "does it have any match?"** Getting it wrong scores 0 in both directions.
2. **Once an entity is known to have matches, adding a wrong ID costs more than missing a right one** (0.286 lost vs 0.167 lost in the examples above).
3. **Entities are scored independently, but mistakes are not independent.** One S2 record wrongly given to two S1 entities creates a false positive in both.

### 1.2 System requirements derived from the metric

| ID | Requirement | Why |
|---|---|---|
| R1 | **Candidate recall ceiling of about 99% or more** | A true match that never reaches the model cannot be recovered. |
| R2 | **Precision-first final selection** | F0.5 weights precision 2× over recall. |
| R3 | **An explicit answer to "does this entity have any match?"** | This decision is worth a full 1.0 or 0.0 per entity. |
| R4 | **Globally consistent assignments** | One false merge can damage two entities. |
| R5 | **Open-set country generalisation** | France appears in test but not in train. |
| R6 | **Leakage-free local evaluation that reproduces the official metric** | Public LB is a subset; final rank is private. |
| R7 | **Compliance** | Final model ≤ 8B params, MIT/Apache-2.0; no external lookup; reproducible; `candidate_pairs.tsv` = the exact set the model scores. |

Efficiency is a **constraint**, not an objective. The system only needs to run within the available compute. Extra speed earns no points.

"High accuracy, high precision, high recall everywhere" is **not** the target. The target is:
- **maximal recall** at the candidate stage,
- **maximal precision** at the decision stage,
- **correct empty predictions** for singletons.

---

## 2. Design evolution — what was proposed and what changed

### 2.1 v0 — Original design (yours)
- Concatenate `[name]_[address]` into one string and embed it.
- Store the embeddings in a vector index. For each S1 record, retrieve the nearest neighbours into a "bucket".
- Keep the bucket large for recall and let stage 2 remove the false positives.
- Stage 2 is a "System One" decision model (Laya / Jev) that outputs match probabilities cheaply at inference.

**Assessment:** a sound skeleton, and the industry-standard retrieve-then-classify pipeline. Most competitors will build it, so it is not a differentiator.

### 2.2 v1 — First review (weak spots and differentiators)

| Issue in v0 | Change proposed |
|---|---|
| A single fused embedding lets address noise dominate | Retrieve name and address separately and take the union |
| Dense retrieval misses typos, abbreviations and transliterations | Add lexical (character n-gram) retrieval → hybrid blocking |
| "Bigger bucket is always better" | Choose the bucket size from the recall@K curve. Bigger buckets add hard negatives and false-positive risk. |
| English-centric assumptions | Multilingual encoders; country treated as an open string |
| Independent pair decisions | One-to-one constraint (S1 is deduplicated), listwise competition features |
| Generic thresholding | Explicit singleton handling; tune on macro F0.5, not on pair metrics |
| Random negatives | Hard negatives mined from the system's own blocker |

### 2.3 v1.5 — After verifying Laya and Jev
- **Jev: excluded.** It is closed, hosted and waitlist-only, which fails both the licence rule and the no-external-service rule.
- **Laya: eligible.** It has 421M parameters, an Apache-2.0 licence and a ModernBERT-large base. Its model card reports two limits:
  - **Zero-shot accuracy is near random (0.362).** It must be fine-tuned.
  - **Calibration is poor as shipped (ECE 0.466, falling to 0.081 after temperature refitting).** It must be recalibrated.
- **Architecturally, Laya is a cross-encoder with a decision head.** Using it does not make the pipeline unique.
- **The idea worth keeping:** use its option-scoring design **listwise**, with an explicit "none" option. Also run an ablation against a plain encoder.

### 2.4 v2 — Research-frontier design
This version added:
- a learned address/name parser;
- multi-view blocking with a learned pruner;
- a stacked cross-encoder and feature model;
- a listwise selector;
- collective resolution (assignment, S2↔S3 transitivity, correlation clustering);
- **expected-F0.5-optimal decisions**;
- an LLM teacher with distillation;
- synthetic noise augmentation;
- a System-1/System-2 inference cascade.

### 2.5 v3 — Final design (after the double-check)

| v2 element | v3 status | Reason |
|---|---|---|
| Learned field parser | **Replaced** by rule-based structuring | There are no labelled address components, and rules capture most of the value. |
| One-to-one constraint | **Kept, but gated** on a data check | It must be verified in the ground truth, not assumed. |
| Listwise selector model | **Optional** escalation stage | Listwise *features* in the scorer capture most of the gain. |
| LLM teacher / distillation | **Dropped** | Risk under the "external data" ban, and high cost. |
| Expected-F0.5 decisions | **Kept, but must beat** a tuned threshold on validation | It depends on calibration quality. |
| Synthetic augmentation | **Optional** — only transformations of the provided data | Keeps it compliant. |
| Validation | **Upgraded** to entity-grouped splits with distractors | Prevents inflated local scores. |
| S2↔S3 transitivity / correlation clustering | **Optional** | Useful, but secondary to the one-to-one constraint. |

---

## 3. The system at a glance

```
                  ┌──────────────────────────────────────────────┐
 Raw TSVs ───────►│ S0  Data Audit & Hypothesis Gate             │──► A1 Data Facts + Gates G1–G3
                  └──────────────────────────────────────────────┘
                  ┌──────────────────────────────────────────────┐
 Raw + GT ───────►│ S1  Evaluation Harness                       │──► Folds, Scorer, Diagnostics
                  └──────────────────────────────────────────────┘      (judges every stage below)
                  ┌──────────────────────────────────────────────┐
 Raw records ────►│ S2  Normalisation & Structuring              │──► A2 Structured Records + Token Stats
                  └──────────────────────────────────────────────┘
                  ┌──────────────────────────────────────────────┐
 A2 ─────────────►│ S3  Multi-View Candidate Generation          │──► A3 Ranked lists per view
                  └──────────────────────────────────────────────┘
                  ┌──────────────────────────────────────────────┐
 A3 ─────────────►│ S4  Candidate Consolidation & Pruning        │──► A4 Candidate Set  ═► candidate_pairs.tsv
                  └──────────────────────────────────────────────┘      (recall ceiling fixed here)
                  ┌──────────────────────────────────────────────┐
 A4 + A2 ────────►│ S5  Evidence Construction                    │──► A5 Pair Evidence Table
                  └──────────────────────────────────────────────┘
                  ┌──────────────────────────────────────────────┐
 A5 (+ text) ────►│ S6  Match Scoring  (+ optional S6b Listwise) │──► A6 Raw scores (out-of-fold + test)
                  └──────────────────────────────────────────────┘
                  ┌──────────────────────────────────────────────┐
 A6 + labels ────►│ S7  Calibration                              │──► A7 Calibrated P(match)
                  └──────────────────────────────────────────────┘
                  ┌──────────────────────────────────────────────┐
 A7 + G1 ────────►│ S8  Global Consistency Resolution            │──► A8 Consistent probabilities
                  └──────────────────────────────────────────────┘
                  ┌──────────────────────────────────────────────┐
 A8 ─────────────►│ S9  Decision Layer                           │──► A9 Final match set per S1
                  └──────────────────────────────────────────────┘
                  ┌──────────────────────────────────────────────┐
 A9 + A4 ────────►│ S10 Output Assembly & Validation             │──► matching_results.tsv, candidate_pairs.tsv
                  └──────────────────────────────────────────────┘
                  ┌──────────────────────────────────────────────┐
 All ────────────►│ S11 Reproducibility & Evidence Package       │──► submission zip + methodology doc
                  └──────────────────────────────────────────────┘
```

The design divides into three zones:
- **Recall zone:** S2–S4.
- **Precision zone:** S5–S9.
- **Assurance zone:** S0, S1, S10 and S11.

Each zone owns different metrics (Section 6). A change in an upstream zone invalidates the tuning of every zone below it.

---

## 4. Stage contracts

Each stage is specified by six fields:
- **Role:** what the stage is for.
- **Input:** what it consumes.
- **Output:** what it produces.
- **Invariants:** what must always hold.
- **Owned metric:** how the stage is judged.
- **Failure modes:** what typically goes wrong.

The final field, **Open choices**, lists the decisions left to you.

---

### S0 — Data Audit & Hypothesis Gate

**Role.** Turns assumptions into measured facts. Its outputs decide which later stages are enabled and how aggressive they are.

**Input.**
- train_source1/2/3, train_ground_truth, test_source1/2/3 (raw TSVs).

**Output — A1 Data Facts Sheet.**
- **Record counts** per source and per split.
- **Country distribution** per source and split, including France's share of the test set.
- **Singleton rate** in train. This equals the local score of an all-empty prediction and is the floor every system must beat.
- **Match-cardinality distribution** (0, 1, 2, … matches per S1) and the S2 vs S3 share of matches.
- **One-to-one violation count:** how many S2/S3 IDs appear under more than one S1 in the ground truth.
- **Distractor ratio:** the share of S2/S3 records that match no S1 entity.
- **Field missingness** per source (empty addresses, missing postcodes, and so on).
- **Noise pattern catalogue:** observed name and address variation types, with rough frequencies.

**Output — Gates.**
- **G1 One-to-one mode:** *hard* (enforced in S8), *soft* (used as a feature) or *off*.
- **G2 Singleton pressure:** how conservative S9 should be.
- **G3 Unseen-country investment:** how much robustness work France justifies, given its share of the test set.

**Invariants.** Read-only. Facts come from the data, never from assumptions.

**Owned metric.** None directly. This stage protects every downstream metric from wrong assumptions.

**Failure modes.**
- Skipping it and building S8 on an unverified one-to-one assumption.
- Misreading IDs as numbers.
- Reading the TSVs without a tab separator.

---

### S1 — Evaluation Harness

**Role.** The single source of truth for "is this change better?". Every stage is judged here, not on the public leaderboard.

**Input.**
- Training records and ground truth.
- Any prediction in submission format.
- Any candidate set in candidate format.

**Output.**
- **Fold assignment.**
  - Folds are grouped by S1 entity.
  - Each matched S2/S3 record goes to the fold of its S1 entity.
  - Unmatched S2/S3 records (distractors) are spread across folds.
  - Result: each validation fold looks like a miniature test set.
- **Official-metric scorer.** Reproduces the per-entity F0.5 macro-average exactly, including the singleton rules.
- **Blocking diagnostics.** Recall@K, reduction ratio and mean candidates per entity, broken down by view and by country.
- **Error slicer.** Groups false positives and false negatives by country, source, match cardinality and noise pattern.

**Invariants.**
- No S2/S3 record appears in more than one fold.
- The scorer agrees with the official formula on the worked example in the problem statement (0.714).
- The fold split is fixed and seeded.

**Owned metric.** The trustworthiness of every other metric.

**Failure modes.**
- **Random pair-level splits:** the same entity leaks across folds, so validation scores come out too optimistic.
- **Leaving singletons out of the validation score.**
- **Tuning on the public leaderboard.**

---

### S2 — Normalisation & Structuring

**Role.** Converts free text into comparable, field-level representations. Most entity-resolution errors come from comparing unstructured strings.

**Input.**
- Raw records: `entity_id`, `business_name`, `business_address`, `country`, from all sources and both splits.

**Output — A2 Structured Record Table** (one row per record).
- **Preserved raw fields:** raw name, raw address, raw country.
- **Canonical fields:** a canonical name and address (case, Unicode, punctuation and symbol normalisation, "&"→"and", known abbreviations expanded) plus an accent-folded variant.
- **Name structure:**
  - `name_core`: the name with legal form removed.
  - `legal_form`: Pvt Ltd, LLC, Inc, SARL, SAS, and so on.
  - `trade_name`: the DBA or t/a part, if present.
- **Address structure:**
  - `postcode`, `house_number`, `landmark_text` and `landmark_flag`, `locality_hint`, `city_hint`.
  - `address_core`: the remainder.
- **Country:** the raw country string, treated as an open-set label.

**Output — shared resources.**
- **Canonicalisation dictionary.** Variant ↔ canonical token pairs mined from aligned matched pairs, plus a small seeded list covering unseen-country legal forms and street types.
- **Token statistics.** Per-field, per-country IDF with a global fallback, plus name-frequency counts (the chain signal).

**Invariants.**
- **Deterministic** and **lossless:** raw fields are always kept.
- **Applied identically** to all sources, train and test.
- **Unknown-country fallback:** an unseen country goes through the global path and never errors or is dropped.
- **Built from the provided data only:** the dictionary and statistics use no external data.

**Owned metric.** Indirect: measured as the change in S3 recall and S9 F0.5 when S2 changes.

**Failure modes.**
- **Over-normalisation:** deleting distinguishing words. For example, dropping "Hospital" vs "Pharmacy" merges two different businesses.
- **Stripping legal forms too early:** removing them before they have been captured as their own feature.
- **Script- and accent-sensitive comparisons:** comparisons that break on French text.

**Open choices.**
- Which parsing rules to use and how far to expand abbreviations.
- How to mine the dictionary and where to set its minimum support.
- Whether to keep several canonical variants in parallel.

---

### S3 — Multi-View Candidate Generation

**Role.** Produces candidate matches for each S1 record from several **independent views** that fail in different ways, so their union reaches the recall target.

**Input.**
- **Queries:** S1 structured records (A2).
- **Index:** S2 ∪ S3 structured records (A2) for the same split.

**Output — A3 View Lists.** For each view, rows of (S1 id, candidate id, view name, rank within view, view score). The minimal view set is:

| View | Signal it catches | Signal it misses |
|---|---|---|
| Lexical-name | Typos, abbreviations, character-level transliteration | Synonyms, reordering at scale |
| Lexical-address | Shared street or locality tokens, partial addresses | Reformatted or landmark-only addresses |
| Semantic-name | Paraphrase, trade names, cross-lingual variants | Exact codes, rare tokens |
| Semantic-address | Reordered or partially described addresses | Numeric precision |
| Key-based | Exact postcode, rare shared tokens, phonetic name keys | Anything lacking the key |

**Invariants.**
- **Separate measurement:** every view is measured on its own, reporting its recall@K and its **unique recall contribution** (true matches found only by that view).
- **Independent failures:** views are chosen so that they fail independently. A view whose unique contribution is near zero is redundant.
- **Soft country rule:** country is a soft rule, never a hard filter that could drop France.
- **No self-matches:** S1 records are never candidates.

**Owned metric.** Union recall@K per view and combined, against the R1 target of about 99% or more.

**Failure modes.**
- **Redundant views:** all views are variations of the same signal, so the union adds no recall.
- **Wrong K:** K is set by gut feeling instead of by the recall curve.
- **English-only encoder:** the semantic view uses an English-only encoder and fails on French.

**Open choices.**
- Which views to include, their representations and K per view.
- Whether the semantic encoder is used off-the-shelf or fine-tuned contrastively on training pairs.

---

### S4 — Candidate Consolidation & Pruning

**Role.** Merges all views into a single candidate set that meets the recall target at the smallest workable size. **This stage sets the system's recall ceiling.**

**Input.**
- A3 view lists.
- Optionally, cheap pair signals, if a learned pruner is used.

**Output — A4 Candidate Set.** Rows of (S1 id, candidate id) with **provenance**:
- which views found the candidate;
- the best rank and score in each view;
- the number of views that agree.

A4 **is** `candidate_pairs.tsv`: the exact set the scoring model runs inference on.

**Invariants.**
- Duplicates are removed.
- Only S2/S3 IDs appear.
- Every test S1 entity has a row, which may be empty.
- On validation, candidate recall ≥ the chosen target.
- Nothing downstream may add a pair that is not in A4.

**Owned metric.**
- Recall ceiling.
- Reduction ratio.
- Mean and 95th-percentile candidates per entity.

**Failure modes.**
- **Over-pruning:** capping candidates per entity so tightly that entities with several true matches lose some.
- **Undocumented filters:** filtering after A4 without updating the file, which violates the rule that `candidate_pairs.tsv` is the last stage before the model.

**Open choices.**
- Union vs weighted fusion of the views.
- Whether to use a learned pruner.
- The budget per entity.

---

### S5 — Evidence Construction

**Role.** Describes each candidate pair as **explicit evidence**: agreement, contradiction, specificity and competition. The scorer then learns *why* two records match, not just how similar they look.

**Input.**
- A4 candidate pairs with provenance.
- A2 structured records and token statistics.
- For competition features, first-pass scores (see the invariants below).

**Output — A5 Pair Evidence Table** (one row per candidate pair), in six evidence families:

| Family | What it captures |
|---|---|
| **Name agreement** | Similarity of `name_core`, full name, trade name and acronym; legal-form agreement; first-token agreement |
| **Address agreement** | Per-field similarity (postcode, house number, street, locality, city); full-address similarity |
| **Contradiction** | Both sides present **and** different: postcode, house number, legal form, city. Encoded as three states: agree / disagree / missing |
| **Specificity** | Name frequency (chain signal); rarity of shared tokens; how generic the name is |
| **Provenance** | Which views retrieved the pair, the ranks and scores, and how many views agree |
| **Competition** | Within the S1 entity: rank, gap to the best candidate, number of strong candidates. Reverse direction: how strongly this candidate is claimed by other S1 entities |

**Invariants.**
- **Missing is never the same as mismatch.** An absent postcode is not a contradiction.
- **Train/test parity:** the table is computed identically for train and test.
- **No leakage in competition features:** features based on model scores must use **out-of-fold** first-pass scores on train. This implies a two-pass scoring scheme.
- **Context-free families:** every family except competition depends only on the pair and corpus statistics.

**Owned metric.**
- Pair-level separability as a diagnostic.
- Feature-family ablations measured by change in macro F0.5.

**Failure modes.**
- **Chain businesses:** the same name at a different address is treated as a match because name similarity dominates.
- **Missing values treated as mismatches:** records with sparse addresses get penalised for missing data.
- **In-fold competition features:** computed with in-fold scores, which inflates validation.

**Open choices.**
- The exact similarity functions.
- Whether to use two-pass competition features or leave them out.

---

### S6 — Match Scoring  (+ optional S6b Listwise Selector)

**Role.** Turns the evidence into a match score for each pair.

**Input.**
- The A5 evidence table.
- For a text-pair scorer, the raw and canonical text of both records.
- Labels for training pairs: A4 pairs from the training folds, so negatives are the system's own hard negatives.

**Output — A6 Raw Scores.** One score per pair from each scorer, plus a stacked score. Scores cover:
- **all training pairs, out-of-fold**, which S7–S9 need for tuning;
- **all test pairs.**

**Components (by class, not by tool).**
- **Structured scorer.** Learns from A5. Fast, interpretable, and strong on contradiction and specificity signals.
- **Text-pair scorer (optional but likely valuable).** A fine-tuned bidirectional encoder over the record pair, such as the Laya class. Strong on transliteration and paraphrase.
- **Stacker.** Combines the scorers. It must be trained on out-of-fold outputs.

**S6b — Listwise Selector (optional escalation).**
- **Input:** one S1 record, its top-N candidates and an explicit "none" option. It runs only for entities in the uncertainty band.
- **Output:** a distribution over {candidates, none}, used as extra evidence in S9.
- **Gate G5:** keep it only if it improves macro F0.5 on the uncertainty band in validation.

**Invariants.**
- Final model is ≤ 8B parameters with an MIT/Apache-2.0 licence.
- Out-of-fold predictions exist for every training pair.
- There is no test-time training on test labels (there are none to use).

**Owned metric.**
- Pair-level PR-AUC as a diagnostic.
- Macro F0.5 after S7–S9 as the real measure.

**Failure modes.**
- Training on random negatives.
- Stacking on in-fold predictions.
- Adding a neural scorer that does not beat the structured scorer. An ablation is required.

**Open choices.**
- The model families.
- Whether to use the text-pair scorer.
- Whether to build S6b.

---

### S7 — Calibration

**Role.** Makes scores behave as **probabilities**, because S8 and S9 reason with them as probabilities.

**Input.**
- A6 out-of-fold raw scores with labels.
- A6 test scores.

**Output.**
- **A7 Calibrated P(match)** for each pair.
- **Calibration report:** reliability by country, source and candidate rank, plus ECE.

**Invariants.**
- The calibrator is fitted only on out-of-fold data.
- Calibration is monotonic, so it preserves the ranking within each scorer.
- For unseen countries, fall back to the global calibrator.

**Owned metric.** ECE and reliability curves.

**Failure modes.**
- Skipping this stage while using S9's expected-F rule, which then optimises on wrong probabilities.
- **Laya specifically:** its card reports that it needs a temperature refit, so its shipped probabilities cannot be used directly.

---

### S8 — Global Consistency Resolution

**Role.** Removes false merges that only become visible when you look **across S1 entities**.

**Input.**
- A7 calibrated probabilities for all pairs across all S1 entities.
- Gate G1, from S0.

**Output — A8 Consistent Probabilities.** Probabilities adjusted or pruned so that they respect the structural constraints:
- **C1 One-to-one (if G1 = hard).** Each S2/S3 record is assigned to at most one S1 entity.
- **C2 Cross-source coherence (optional).** If an S2 record and an S3 record are strongly the same business, they should go to the same S1 entity.

**Invariants.**
- It only **removes or down-weights** pairs and never adds any.
- The output stays a subset of A4.
- The result is deterministic.

**Owned metric.** Change in macro F0.5 compared with S7 → S9 without S8.

**Failure modes.**
- Enforcing one-to-one when the ground truth violates it.
- Resolving conflicts greedily in an order that systematically favours one source.

**Open choices.**
- **The resolution mechanism:** greedy best claim, optimal assignment, or a soft penalty fed back as a feature.
- **Whether to include C2.**

---

### S9 — Decision Layer

**Role.** Chooses the **final match set for each S1 entity**, possibly empty, to maximise the expected per-entity F0.5. This is where R2 (precision first) and R3 (the "any match?" question) are delivered.

**Input.**
- For each S1 entity: its surviving candidates with probabilities from A8.
- Optionally, the S6b listwise output.

**Output — A9 Final Match Sets.** One set per S1 entity, possibly empty.

**Candidate policies (chosen by validation, not by preference).**

| Policy | Behaviour |
|---|---|
| **P1 Global threshold** | Accept every candidate with p ≥ t. Tune t on validation macro F0.5. |
| **P2 Threshold + structure** | P1 plus entity-level rules, such as a stricter bar when the top candidate is weak or when the gap between the top two candidates is small. |
| **P3 Expected-F0.5 maximisation** | Sort the candidates by p. Compute the expected F0.5 of predicting the top k, for k = 0…n, where k = 0 is the empty set. Output the best k. This handles singletons without a separate rule. |

**Invariants.**
- The output is a subset of the entity's A4 candidates.
- Decisions are made per entity. There are no cross-entity effects here, because those belong to S8.
- Policy parameters are tuned only on out-of-fold data.
- Unseen countries use the global parameters.

**Owned metric.** Macro F0.5, the official metric.

**Failure modes.**
- Choosing t from pair-level F1.
- P3 running on uncalibrated probabilities.
- Per-country thresholds overfitting to small slices.

**Open choices.**
- Which policy to use.
- Whether to stratify by country or source.

---

### S10 — Output Assembly & Validation

**Role.** Converts decisions into files that are guaranteed to be accepted.

**Input.**
- The A9 final match sets.
- The A4 candidate set.
- The list of test S1 IDs.
- The test S2/S3 ID lists.

**Output.**
- `matching_results.tsv`
- `candidate_pairs.tsv`
- A validation report from the official validator script.

**Invariants.**
- Every test S1 entity appears exactly once, and an empty list is allowed.
- Only existing test S2/S3 IDs are used, with no duplicates within a list.
- Matches ⊆ candidates.
- Files are tab-separated, with comma-joined ID lists and no quoting.

**Owned metric.** Pass/fail from the validator. Pass is required.

---

### S11 — Reproducibility & Evidence Package

**Role.** Makes the result reproducible and **provable**. For the top-100 review, evidence about the pipeline is part of the product.

**Input.**
- Code and configuration.
- Seeds.
- The artefacts A1–A10.
- The S1 diagnostics.
- The submission ledger.

**Output.**
- **Submission zip:** `output/`, `code/business_entity_resolution/` (src, README, pinned requirements) and the methodology document.
- **Methodology evidence:**
  - recall@K curves per view and for the union;
  - reduction ratio;
  - ablations per stage (S3 views, S5 families, S6 text scorer, S8 on/off, S9 policy);
  - a calibration plot;
  - an error taxonomy;
  - how France was handled.
- **Version ledger:** submission → config → validation score → leaderboard score.

**Invariants.**
- A single entry point regenerates both output files from the raw data.
- No external data or services are used anywhere.

---

## 5. Artefact contracts (what flows between stages)

| Artefact | Produced by | Consumed by | Grain | Key contents |
|---|---|---|---|---|
| A0 Raw data | — | S0, S1, S2 | record | Raw TSV fields |
| A1 Data Facts + Gates | S0 | All stages (G1–G3) | dataset | Rates, distributions, gate decisions |
| Folds / Scorer | S1 | S3–S9 tuning | entity | Fold id per record; scoring functions |
| A2 Structured Records | S2 | S3, S5, S6 | record | Raw + canonical + parsed fields; token stats; dictionary |
| A3 View Lists | S3 | S4 | (S1, cand, view) | Rank, score per view |
| A4 Candidate Set | S4 | S5, S6, S8, S9, S10 | (S1, cand) | Provenance; = `candidate_pairs.tsv` |
| A5 Evidence Table | S5 | S6 | (S1, cand) | Six evidence families |
| A6 Raw Scores | S6 | S7 | (S1, cand) | Out-of-fold (train) + test scores per scorer |
| A7 Calibrated P | S7 | S8 | (S1, cand) | P(match) |
| A8 Consistent P | S8 | S9 | (S1, cand) | Constraint-respecting P |
| A9 Match Sets | S9 | S10 | S1 entity | Final set (possibly empty) |
| A10 Outputs + Package | S10, S11 | Leaderboard, reviewers | file | TSVs, zip, methodology document |

**Design rule:** each artefact is persisted and versioned, so any stage can be re-run and evaluated in isolation without recomputing the stages before it.

---

## 6. Metric ownership and feedback loops

### 6.1 Who owns what

| Outcome | Owning stage(s) | Measured by |
|---|---|---|
| Recall ceiling | S2, S3, S4 | Candidate recall@K, unique recall per view |
| Pair-level discrimination | S5, S6 | PR-AUC (diagnostic), family ablations |
| Probability quality | S7 | ECE, reliability curves |
| Cross-entity false merges | S8 | Change in macro F0.5 from S8 |
| Per-entity set optimality and singleton accuracy | S9 | Macro F0.5; singleton accuracy; accuracy on "has matches" entities |
| Acceptance and compliance | S10, S11 | Validator PASS; reproducibility check |

### 6.2 Feedback loops

- **L1 Recall loop.** True matches missing from A4 → classify why each was missed → fix S2 (normalisation) or S3 (add or retune a view).
- **L2 Error-taxonomy loop.** False positives and false negatives at S9 → group them by noise pattern, country and cardinality → find the missing evidence family in S5 or the scorer weakness in S6.
- **L3 Re-tuning loop.** **Any** change in S2–S6 shifts the score distributions, so S7, S8 and S9 must always be re-fitted afterwards. Never carry thresholds over from an older configuration.
- **L4 Leaderboard loop.** The public leaderboard is used only as a sanity check that local validation tracks it. Private-leaderboard rank depends on generalisation, not on public-leaderboard fitting.

---

## 7. Cross-cutting concerns

- **Open-set country.**
  - Country is always a string feature with a soft rule.
  - Every country-specific component (dictionary, IDF, calibrator, threshold) has a global fallback.
  - France must flow end-to-end without special-case code paths that might break.
- **Leakage.**
  - Folds are grouped by entity.
  - Stacking, calibration, competition features and policy tuning all use out-of-fold data only.
- **Train/test parity.** S2–S6 use identical code for train and test, and every statistic comes from the provided data.
- **Determinism.** Seeds are fixed, sorting is stable, and conflict resolution in S8 is order-independent.
- **Compliance.**
  - Final model ≤ 8B, MIT/Apache-2.0.
  - No external lookup, geocoding or augmentation from outside sources.
  - `candidate_pairs.tsv` equals the set the model actually scores.
- **Compute.**
  - Cost scales with (number of S1 entities × candidates per entity) × scorer cost.
  - The S4 budget and the optional S6b escalation band are the two controls.
  - Heavy scoring should only be spent where decisions are uncertain.

---

## 8. Decision gates (where the data decides the design)

| Gate | Decided in | Question | Effect |
|---|---|---|---|
| G1 | S0 | Does the ground truth respect one-to-one? | S8 C1 hard / soft / off |
| G2 | S0 | How high is the singleton rate? | How conservative S9 is |
| G3 | S0 | What share of the test set is France? | How much to invest in unseen-country robustness |
| G4 | S6 | Does the text-pair scorer beat the structured scorer alone? | Include or drop it |
| G5 | S6b | Does listwise selection improve the uncertainty band? | Include or drop S6b |
| G6 | S9 | P1 vs P2 vs P3 on validation | Final decision policy |

---

## 9. Deliberately excluded (and why)

| Excluded | Reason |
|---|---|
| Jev | Closed, hosted and waitlist-only; fails the licence and no-external-service rules |
| LLM teacher / distillation | Its world knowledge is close to the external-data ban; cost is high relative to the challenge window |
| Learned field parser | No component labels exist; rule-based structuring captures most of the value |
| Geocoding / external address normalisation | Explicitly prohibited |
| Hard country filters | Would silently drop or mishandle France |

---

## 10. Final double-check — why this design wins

1. **The recall ceiling is engineered, not hoped for.** Independent views, with unique contributions measured, reach R1. v0's single fused embedding could not guarantee it.
2. **Precision comes from structure, not from model size.**
   - Contradiction evidence handles records that look similar but aren't the same.
   - Specificity evidence handles chains.
   - Global consistency handles records claimed by more than one S1 entity.
   - These cover the main false-merge patterns that pair similarity alone misses.
3. **The decision layer optimises the real metric.** Scoring per entity makes "any match?" worth a full 1.0 or 0.0 per entity. S9 treats it as a decision-theory problem on calibrated probabilities instead of a fixed threshold.
4. **Every expensive or risky component is gated by evidence.** One-to-one, the text scorer, the listwise selector and the decision policy are each included only if validation shows a gain.
5. **The assurance zone protects the private-leaderboard result.**
   - Leakage-free, metric-exact validation reduces the risk of overfitting the public leaderboard.
   - The evidence package covers what the top-100 review asks for.

**Residual risks.**
- **France:** it has zero training signal, so it depends on S2 fallbacks, multilingual semantic views and global calibration.
- **Transliteration-heavy Indian names:** these may force the text-pair scorer (G4) from optional to essential.
- **The G1 check:** if the ground truth breaks one-to-one, the biggest structural precision gain shrinks to a soft feature.
