# Deep Critique of the v3 architecture (`archi_v3.md`), against the measured data

Every factual claim below points to a measured number in `DATA_NOTES.md` (§ refs).
The design is judged on one question: **does it spend effort where this dataset
actually loses F0.5?**

---

## 0. Verdict

**The skeleton is right; the emphasis is wrong.** `archi.md` is a strong generic entity-resolution design: retrieve then classify, leakage-free evaluation, calibrated decisions, and assumptions checked by gates. But it was written before seeing the data, and the data turns out to be a *specific* synthetic generator with specific traps. Several of the design's biggest investments target problems this data doesn't have. Several real problems have no stage at all.

| The design invests heavily in… | …but the data says | Status |
|---|---|---|
| Singletons and the "any match?" decision (R3, G2, P3) | 5.6% singletons; mean 3.5 matches per S1 (§3) | **Over-weighted** |
| Postcode keys, postcode contradiction | Postcodes are essentially absent: missing on one side in 91–99.8% of pairs (§4) | **Dead feature** |
| Legal-form contradiction | Legal forms are *noise*: added, dropped, swapped, moved in true pairs (§4) | **Harmful as stated** |
| Soft country rule, never partition | Country agrees in 100% of pairs and the label is clean in all sources, France included (§2) | **Too cautious; costs 2–3× compute** |
| A ModernBERT-class text scorer (Laya) | 27% of India S2 names are in Indic scripts (§6); French in test | **Wrong base model** |
| Listwise "pick one or none" (S6b) | Entities have many matches: a softmax over S1's candidates is the wrong objective | **Mis-framed** (right idea, wrong direction; see §3.7) |
| Hungarian or optimal assignment (S8) | One-to-one binds only on the record side, so assignment is a per-record argmax | **Overbuilt** |
| S2↔S3 transitivity, correlation clustering | S1 is the clean hub; copies are 1 noise-hop from S1 and 2 hops from each other | **Low value** (to verify) |

| The data's real problems… | Stage in `archi.md` |
|---|---|
| **Near-twin negatives:** similar name, adjacent house number (§8) | None; "contradiction" is too coarse |
| **Co-location:** 44% of close-address pairs are different businesses (§8) | Partly (address and name families), not named as a failure mode |
| **Script and transliteration:** Indic names look exactly like co-located negatives (§6, §8) | None; listed only as a "residual risk" |
| **Name non-uniqueness:** 36–44% of S1 names repeat (§8) | Specificity family: right, but under-emphasised |
| **Test density shift:** +24% S2/S3 per S1 (§2) | None; validation would not reproduce it |
| **Blocking cost:** single-token blocks need about 11k candidates for 97.6% recall (§9) | S3 views are named, but there's no cost model |
| **Compute feasibility:** ~20M records, ~50–100M candidate pairs | Treated as "a constraint, not an objective" and never sized |
| **Unlabelled France:** region↔department naming, generic names (§7) | "Global fallback" everywhere, which is the wrong default (§3.3) |

**The ten changes that matter most**, ranked by the F0.5 at stake:
1. **Build a near-twin discriminator.** Model house-number noise versus a genuinely different number, and filler words versus a genuinely different name (S5).
2. **Make transliteration a first-class normalisation step,** for India precision as well as recall (S2).
3. **Change the decision rule to a free top-1 plus strict extras,** replacing the plain global threshold. It comes out of the metric algebra in §1 (S9).
4. **Add record-centric reverse retrieval and a per-record choice between one S1 or none.** This turns the one-to-one fact into both a candidate generator and the natural listwise model (S3, S6b, S8).
5. **Run validation as a full-train, density-matched "world"**, not per-fold miniatures (S1).
6. **Partition hard by country label,** with a fallback only for unseen or empty labels (S3).
7. **Delete postcode keys and legal-form contradictions;** replace them with learned noise-token weights (S2, S5).
8. **Size the compute and restrict any neural cross-encoder to an uncertainty band, on a multilingual base** (S6).
9. **Compute unsupervised statistics transductively per split,** including test France (IDF, name frequency) (S2).
10. **Spend 1–2 leaderboard submissions as test-prior probes** before the final one (S1, S7).

---

## 1. Critique of §1 (objective → requirements)

### 1.1 The metric algebra is correct but read in the wrong regime

For an entity with **k** true matches, predicting **c** correct and **w** wrong gives:

```
F0.5 = 1.25·c / (c + 0.25·k + w)
```

(Check: k=2, c=2, w=1 gives 2.5/3.5 = 0.714, matching the official example.)

This form shows that **a wrong ID weighs 4× a missed ID in the denominator**. Now apply it to the entities that dominate this data (k ≈ 3–4, which covers 70% of entities, §3):

| Entity with k=4 | F0.5 | Change |
|---|---:|---|
| all 4 correct | 1.000 | |
| 3 correct, 0 wrong | 0.938 | missing one costs 0.06 |
| 4 correct, 1 wrong | 0.833 | a wrong one costs 0.17 |
| **only 1 correct** | **0.625** | |
| empty | 0.000 | |

Three consequences the design misses:

1. **The first slot is nearly free.** For any non-singleton, an empty prediction scores 0, and so does a wrong single prediction. Including the top-1 candidate can only help, *unless the entity is a singleton*. With a 5.6% singleton prior, the bar for top-1 is "is P(this entity has a match and top-1 is it) larger than P(singleton)?", not "p ≥ t".
2. **Later slots need high confidence.** Break-even for adding the 4th item when 3 are already correct: p·0.0625 = (1−p)·0.1875, so **p > 0.75**. The general result (Lipton et al.) is that the F-optimal marginal threshold ≈ F\*/(1+β²) = **0.8·F\***. If the achievable per-entity F is about 0.9, extras need p ≳ 0.72.
3. **"The single most consequential decision is: does it have any match?"** (archi §1.1 point 1) is wrong for 94% of entities. The most consequential thing is **getting the first match right and the whole set complete without twins**. A non-singleton that predicts only its top-1 gets 0.625 at k=4, so recall of *all* copies is worth up to 0.375 per entity. That's far more total F0.5 than the singleton slice can ever hold (5.6% × 1.0).

**So:** precision-first is right for *extra* slots, but the architecture must be **recall-hungry at the candidate stage and "complete-the-set" at the decision stage**. Most of the score is in finding all 3–4 copies while rejecting the one near-twin.

### 1.2 Requirements table

| Req | Assessment |
|---|---|
| R1: ~99% candidate recall | **Keep the spirit, change the target.** Some positives can't be recovered no matter how many candidates you keep: a renamed or Indic name **and** an empty address, where the name is ambiguous. At the same address with zero name overlap, non-matches outnumber matches 5:1 (§8). Set K by the **marginal macro-F0.5** of added recall, not by a fixed 99%. |
| R2: precision-first | Correct for slots 2+. **Wrong for slot 1** (§1.1). |
| R3: explicit "any match?" | Demote. It's a 5.6% slice. Handle it inside the decision rule (a singleton probability versus the top-1 score), not as a stage requirement. |
| R4: global consistency | **Confirmed and upgraded** (G1 = hard, exactly one owner for 7.64M ids). But the structure is simpler than §S8 assumes (§3.9). |
| R5: open-set country | Keep the requirement, but the implementation should be **partition by the observed label**, not "soft rule" (§3.4). |
| R6: leakage-free evaluation | Keep; the design itself is insufficient (§3.2). |
| R7: compliance | Keep. Add: **verify every model card offline before investing** (Laya's claims in §2.3 are unverified hearsay in the doc). |
| **Missing R8** | **Near-twin precision:** reject similar-name, adjacent-number records. |
| **Missing R9** | **Script invariance:** the same business in Latin and Indic scripts must score as the same name. |
| **Missing R10** | **Density and shift robustness:** thresholds must hold when test has about 24% more S2/S3 per S1. |
| **Missing R11** | **A compute budget,** sized in pairs × model cost, with a stated hardware assumption. |

"Efficiency is a constraint, not an objective" is true, but with ~20M records per split and a possible cross-encoder, **compute is the binding constraint on the design**. It has to be designed in, not assumed away.

---

## 2. Critique of §2 (design evolution)

- **v0 → v1** (split name and address retrieval, hybrid lexical plus dense): **supported by the data.** The address alone reaches 95.5% single-token recall versus 86.7% for the name (§9), so they fail differently. Fusing them into one embedding would let the long Indian addresses (78 characters vs 26 for the name) swamp the name signal.
- **"Multilingual encoders" (v1):** right instinct, and more important than the doc realises (27% of India S2 names are Indic, §6). **"Country as open string":** right, but see §3.4.
- **v1.5 Laya:** the doc says "ModernBERT-large base". **ModernBERT is an English (plus code) model.** On Devanagari, Telugu or Tamil it sees mostly byte fragments with no phonetic grounding, and French is out of distribution. That rules it out as the text-pair backbone for 47% of test (India) and it's weak on 15% (France). If a cross-encoder is used, the base should be multilingual (for example XLM-R or mDeBERTa-v3, both MIT; verify the licence and size offline). Also, **"421M params, Apache-2.0, ECE 0.466 → 0.081, zero-shot 0.362" have not been verified** in this repo. Treat them as unverified until someone reads the card.
- **v1.5 "listwise with an explicit none":** see §3.7. It's a good idea pointed in the wrong direction.
- **v2 → v3 drops:** agree with dropping the learned parser, the LLM teacher (on cost grounds; the "world knowledge ≈ external data" argument is weak, since every pretrained encoder has world knowledge) and heavy augmentation *for US/India*. There are 7.6M positive pairs, so data is not the bottleneck.

---

## 3. Stage-by-stage

### 3.1 S0 Audit — ✅ done; extend it

The gates are resolved: **G1 hard, G2 low (5.6%), G3 = 15% France.** Add two new gates:
- **G7 density shift:** test S2+S3 per S1 = 5.8 vs 4.67 in train. It's unknown whether that's more matches or more distractors, and it decides how the decision thresholds must move.
- **G8 script share:** India S2 names are 27% Indic and S3 18%. This decides whether transliteration is on the critical path. **It is.**

### 3.2 S1 Evaluation harness — ⚠️ the fold design is subtly broken for this data

The problem: **validation difficulty depends on index density.** A 1/5 fold validated against a 1/5 index sees about 5× fewer competing records per query than test does. Precision, the one-to-one resolution and the thresholds would all be tuned in an easier world than test.

It also breaks **S8** specifically. If only fold-k S1 entities are queried, a record owned by an out-of-fold S1 has **no competing claimant** in validation, but in test its owner is present and competing. The one-to-one gain comes out wrong in both directions.

**Replace with a "full-world OOF" harness:**
1. Retrieval and scoring run over the **entire train world**: all 2.2M S1 queries against all 10.3M S2/S3 records.
2. The scorer is trained k-fold **grouped by S1 entity**, and each fold's pairs are scored by the model that did not see them. So every train pair has an out-of-fold score *computed in a test-density world*.
3. S7, S8 and S9 are fitted and evaluated on this full world. S8 then sees real competition.
4. **Density stress test:** drop about 19.5% of train S1 entities from the *queries*, but keep their matched records in the index as extra distractors. That reproduces test's 5.8 S2/S3 per S1:

   (4.67 / (1 − 0.195) ≈ 5.8)

   Tune the thresholds to be robust in **both** worlds (the "more matches" and the "more distractors" readings of the +24%).
5. The blocking diagnostics must also be sliced by **noise slice**, not only country: Indic-script name, empty address, renamed or domain name, dba form, near-twin present.

Also: *"Do not use the public leaderboard"* is good discipline, but **1–2 submissions spent as probes of the test prior are rational**. For example, the same model at two extra-slot thresholds; the difference shows which way test's +24% leans. The goal is one best final submission, and probes serve that goal.

### 3.3 S2 Normalisation — ⚠️ right principles, wrong priorities

**Keep:** lossless, deterministic, identical for train and test, an accent-folded variant, and a dictionary mined from aligned pairs (§2 of archi). The pair-mined dictionary is the **best idea in the stage**. With 7.6M positive pairs it can learn `MN↔Minnesota`, `UP↔Uttar Pradesh`, `Calcutta↔Kolkata`, `Rd↔Road`, `CDP/City/Township`, and so on, entirely from provided data.

**Change or add:**

| Item | Why (data) | What |
|---|---|---|
| **Transliteration layer** | 27% of India S2 and 18% of S3 names are Indic; 19–28% of India true pairs share *no* name token (§5, §6). S1 India is 99.94% ASCII. | Romanise every Indic-script span into a Latin phonetic key, and keep both forms. Two in-data options. **(a)** A deterministic Unicode-to-romanisation table: it's an algorithm, not data lookup, but confirm compliance via the organisers' query form. **(b)** Learn char or char-cluster alignments from train pairs (Latin S1 ↔ Indic S2/S3): millions of parallel examples, fully compliant. Then match on phonetic keys (vowel-collapsed consonant skeletons), because transliteration is lossy (प्राइवेट → "praivet" ≈ "private"). |
| **Unicode-aware tokenisation** | Python's `\w+` splits Indic words at vowel signs and viramas (§11). | Tokenise on whitespace and punctuation, keeping Mn/Mc marks inside tokens. |
| **Legal forms: don't treat as identity** | In true pairs, suffixes are added, dropped, swapped, abbreviated or moved to the front (§4). | Canonicalise to a *family* per country and use it only as a weak feature. **Never** as a "contradiction". |
| **A noise-token lexicon (new)** | The generator injects filler (`Center`, `Services`, `Group`, `Holding`, `Co`, `International`), honorifics, junk prefixes and country tags (§4, §11). | Mine per-token **insertion and deletion rates** from positive pairs and compare them with rates in hard negatives. The token weight in name similarity becomes "how often is this token noise?" rather than IDF alone. This is the Fellegi–Sunter m/u idea at token level, and it's the single most data-grounded name feature available. |
| **Domain names → words** | 3–4% of S2/S3 names are domains (`millerpurpose.com`) (§11). | Segment the domain body into words using the S1 name vocabulary of the same country. |
| **Alternate names** | `X dba Y`, `formerly`, 1.3–2.5% in S3 (§11). | Split them into name alternatives; similarity = max over alternatives. |
| **Junk in names** | Phone numbers, `(ID: 30420)`, `***`, `--`, `[INCORPORATED]`. | Strip them, keeping the raw form. Be careful with legitimate digits (`Studio 90`). |
| **Numbers** | Numbers are shared in 82% of true pairs; formats vary (`#`, `No`, `N°`, `(12)`, `0070`, `bis`/`ter`, `1/2`, `2151/8`) (§4, §8). | Parse house numbers into a structure: number, fraction, letter suffix, bis/ter, zero-stripped. This feeds the near-twin discriminator (§3.6). |
| **Postcode field** | Absent (§4). | **Drop it** as a structured field. |
| **Junk literals** | `null`, `N/A` inside addresses (§1). | Treat them as missing. |
| **France region↔department** | S1 uses the region, S2/S3 the department; train can't teach it (§7). | Don't hand-code a geography table (grey area for compliance). Instead: **(a)** give admin-area tokens low weight, since the street, number and city carry France; **(b)** optionally mine region↔department co-occurrence **transductively** from high-confidence first-pass France matches in test (unsupervised, provided data only). |
| **Token statistics** | The doc says "per-country IDF with global fallback". | For France the fallback is the wrong default. IDF and name frequency are *unsupervised*, so **compute them per split per country, including test France**. Only *learned* components (scorer, calibrator) need a global fallback. |

**Over-normalisation warning (keep):** the doc's "Hospital vs Pharmacy" point is exactly the co-location case (§8). The type word is often the only thing separating co-located businesses. That's another reason to use pair-mined noise weights instead of hand stopword lists.

### 3.4 S3 Candidate generation — ⚠️ views need re-selection, direction, and a cost model

**What the data says about views** (§9):
- **Postcode view: remove** (no postcodes).
- **Phonetic name keys** (Soundex, Metaphone) are English-only, blind to Indic and weak for French, and char-n-gram retrieval already covers typos. Remove, or replace with *transliteration-phonetic* keys.
- **Single-token boolean blocking is far too coarse.** 97.6% recall costs about 11k postings per S1 (p95 25k). All lexical views must be **ranked top-K retrieval** (IDF-weighted char n-gram or BM25), not blocks.
- **Name recall saturates at about 87%** (no shared name token in 13% of true pairs). **Address is the recall carrier.** An address view **plus a number-anchored key view** (street token + house number) is probably where the unique recall is. 82% of positives share a number token.
- **Semantic views:** a dense encoder mainly buys cross-script matching, which transliteration does more cheaply and more controllably. **Gate it on measured unique recall after transliteration.** Don't pre-commit.

**Country:** partition hard by the observed label (India, US, France). It holds in 100% of true pairs, the label is clean in every source, and France is present in test S2/S3 (§2). This isn't hard-coding `{US, India}`: the partitions are whatever labels exist. Fallback: a record with an empty or unseen label searches all partitions. It cuts index size and hard-negative exposure 2–3×.

**Add reverse (record-centric) retrieval.** This is the single biggest structural idea the doc misses:
- Every S2/S3 record belongs to **at most one** S1 (G1). So for each record, retrieve its **top-k S1 entities** from the S1 index: 0.26M–0.81M per country, far smaller than the S2/S3 side.
- With k ≈ 3–5, that's about 30–50M pairs for 10M records, and it directly gives the **"which S1 owns this record, or none?"** structure that S8 and S6b need.
- Forward retrieval (S1 → top-K records) covers entities with many copies. Reverse covers records whose best S1 is unambiguous. The **union of both directions** is the candidate set.
- It also yields leak-free competition features for free: rank and margin of this S1 among the record's retrieved S1s, from *retrieval scores* rather than learned scores. No two-pass OOF is needed.

**Cost model (must be written down):** test has 1.73M S1 and 9.97M S2/S3. At forward K=30 plus reverse k=5, expect about 60–100M candidate pairs, and a similar number for the full train world. Sparse char-n-gram top-K per country partition is feasible on a 15 GB machine *per partition, in chunks*. Dense ANN over 10M × 384-dim needs about 7.7 GB at fp16 per split, so it needs a bigger box or product quantisation.

### 3.5 S4 Consolidation — ✅ mostly right

- **Keep:** provenance, `candidate_pairs.tsv` = the scored set, nothing added downstream.
- **Change:** the per-entity budget must be **adaptive**. 0.6% of entities have ≥8 matches, and a fixed small K silently caps them. Budget by score gap or elbow, not by a flat K.
- **Add:** the reverse-direction pairs (§3.4) and a per-record "claimed-by" count.
- **Note:** at about 30 candidates × 1.73M rows, `candidate_pairs.tsv` is roughly 600 MB. Fine, but the validator's `--check-ids` then needs several GB of RAM.

### 3.6 S5 Evidence — ⚠️ the "contradiction" family is the wrong abstraction

This is where the dataset's precision is decided, and the doc's framing ("both present **and** different → contradiction") fits the data poorly:

| Evidence (archi) | Data verdict |
|---|---|
| Postcode contradiction | Dead: no postcodes. |
| Legal-form contradiction | **Harmful.** Suffixes are swapped in true pairs. |
| House-number contradiction | **Too coarse.** True pairs have disjoint numbers in 5% (`2620→262`, `L1→L2`, `44→44D`), while near-twin negatives differ by *adjacent* numbers (`204 1/2 vs 1/9`, `10084 vs 10105`, `109 vs 111`). "Agree / disagree / missing" can't separate them. |
| City contradiction | Weak. City aliases, suffixes and dropped components are common noise. |

**Replace "contradiction" with a noise-channel model.** The data is generated by operators applied to S1, so learn what *each operator looks like* and score a pair by "can the noise process explain this difference?":
- **Number-transform features:** exact / zero-pad / truncation (prefix or suffix match) / letter-suffix added / extra injected number / **numeric distance |a−b|** / fraction change / fully different.
  - *Hypothesis to verify:* noise ops are **string edits** (drop a digit, pad, append a letter), while twins are **arithmetic neighbours** (109→111, 10084→10105). If that holds, it is the near-twin discriminator.
- **Name-diff features:**
  - The set of tokens in one name but not the other, each weighted by its **pair-mined noise probability** (§3.3). A leftover `Holding` or `Summit` that the noise process rarely inserts is strong negative evidence; a leftover `Center` or `Co` is weak.
  - Order-invariant token alignment.
  - Transliteration-key similarity.
  - Domain-segmented similarity.
- **Specificity (chain) features: essential, not optional.**
  - Name frequency among S1 in the same country: `primary care group` ×253.
  - Name frequency among S2/S3.
  - Number of S1 entities with the same name *on the same street*.
  - With 36–44% of names repeated, a name match is worth little without this.
- **Co-location features:**
  - How many S1 entities (and records) share this address or street + number.
  - When the address is shared, the name must carry the decision, and vice versa. Let the model learn the interaction explicitly (for example, name-sim × address-specificity).
- **Script flags:** either side Indic; name similarity computed in romanised space. Without this, an Indic-name true match at a shared address is **indistinguishable from a co-located different business** (5:1 negatives, §8). So transliteration is a **precision** requirement for India, not just a recall one.
- **Competition (keep, but change the source):** use **retrieval-score** competition in both directions (§3.4). That removes the doc's two-pass OOF requirement and its leakage risk.

### 3.7 S6 Scoring (and S6b) — ⚠️ right model class, wrong neural bet

- **Structured GBDT scorer: keep as the primary model.** The data is tabular-noise-shaped (§4). A GBDT on the evidence above is likely most of the achievable F0.5.
- **Training volume:** about 100M train pairs won't fit in memory as a dense float matrix on a 15 GB box (roughly 24 GB at 60 float32 features). **Subsample entities** for training (a few hundred thousand S1 with all their candidates), and score the full world out-of-fold. Don't train on all pairs.
- **Text-pair cross-encoder:**
  - **Base must be multilingual** (§2).
  - **Compute:** at an optimistic ~3k pairs/s on a data-centre GPU, 60–100M pairs ≈ 6–9 hours per world (train OOF plus test ≈ 2×), and several times slower on consumer GPUs. **It only fits as an escalation on an uncertainty band** (say 5–15% of pairs), trained on a subsample.
  - **Its likely gain:** transliteration and paraphrase. If the transliteration layer and name-diff features are done well, G4 may fail. Keep the ablation gate.
- **S6b listwise selector — mis-framed as written.**
  - "One S1, its top-N candidates, pick a distribution over {candidates, none}" assumes **one** correct answer per S1. Here S1 has about 3.5, so a softmax over an S1's candidates punishes correct co-matches.
  - **Flip the direction:** for **one S2/S3 record**, choose among its **candidate S1 entities plus "none"**. By G1, *exactly one* answer is correct: an owner or none (distractors, 26%).
  - That's a proper single-choice listwise problem. It directly produces the probabilities S8 needs, and it is where "none" genuinely carries mass (26% of records). This is the right home for the Laya "option + none" idea.
- **Stacker:** probably unnecessary. Feed a cross-encoder score (OOF) into the GBDT as a feature instead of adding a separate stacking stage.

### 3.8 S7 Calibration — ⚠️ add prior shift

- GBDT log-loss outputs are usually close to calibrated. Isotonic on full-world OOF is enough. **Per-country calibration is impossible for France**, so the global one is the only option (fine).
- **Missing:** test's +24% density may mean more distractors, a lower match prior per candidate. Calibrated train probabilities then **overstate** P(match) on test. Two cheap mitigations:
  - **(a)** estimate the test prior from unlabelled test score distributions (EM prior re-estimation in the style of Saerens et al.) and re-weight;
  - **(b)** choose thresholds that are robust across the density-stress validation world (§3.2).
- Leaderboard probes (§3.2) are the empirical check.

### 3.9 S8 Global consistency — ✅ the biggest win, but simpler than drawn

- G1 is **hard and exact** (7,638,365 ids, each owned once). This stage is justified.
- **The structure is many-to-one, not one-to-one.** An S1 can own many records; a record has at most one S1. There's **no capacity constraint on the S1 side**, so the "optimal assignment" is simply: **for each record, keep only its best S1 claim (or none)**. That's a per-record argmax: deterministic, order-independent and linear time. The Hungarian algorithm is unnecessary, and the doc's worry about greedy ordering favouring a source is moot.
- **Better than a hard argmax:** normalise claims per record (the reverse listwise model of §3.7), so that p(record → S1) + p(record → none) = 1. Feed these *competition-aware* probabilities to S9.
- **How much can it fix?** It fixes negatives that are **owned by another S1**. It can't fix negatives that are distractors (owned by nobody). Measured (§6): it can fix **87–92%** of co-location and same-name negatives, and **~0%** of near-twins.
- **C2 cross-source coherence: likely low value.** The generator makes each S2/S3 copy from S1, so two copies are two noise-hops apart and less similar to each other than to S1. Verify with one measurement (sim(copy, S1) vs sim(copy, copy)) before building it.

### 3.10 S9 Decision — ⚠️ P1 alone is provably suboptimal here

- From §1.1: **P1 (a single global threshold)** treats slot 1 and slot 4 the same. The algebra says slot 1 is nearly free (only the singleton prior opposes it) and slot 2+ needs p ≳ 0.72–0.8.
- **Minimum viable policy, P2′:**
  - Include the top-1 if p(top-1) > τ₁, with τ₁ low and tuned. It has to beat only the singleton hypothesis.
  - Include further candidates if p > τ₂, with τ₂ high.
  - Both are tuned on the full-world OOF (and the density-stress world).
- **P3 (expected-F0.5):** keep as the upgrade, but note the doc's description is incomplete.
  - The expected F of "predict top-j" needs the distribution of the **number of true matches among the unselected** candidates too, because recall's denominator is k.
  - With per-pair probabilities, that's a Poisson-binomial DP: O(n²) per entity, cheap for n ≈ 30.
  - It's only as good as the calibration and the (post-S8) independence assumption.

### 3.11 S10 and S11 — ✅ fine

- Add: run the validator in *default* mode for the gate. Run `--check-ids` only on a machine with enough RAM, since it holds 10M ids plus the candidate lists.
- The guidelines ask for a 1–2 page write-up while the problem statement says there's no page limit. Write a 2-page core plus appendices.

---

## 4. Blind spots (nothing in `archi.md` addresses these)

1. **Near-twin negatives:** similar name, adjacent number, extra filler word (§8). This is the main precision loss once the name and address basics are in place.
2. **Co-location:** 44% of close-address pairs are different businesses (§8). The design never says that the address alone is insufficient.
3. **Transliteration as precision:** an Indic-name true match looks exactly like a co-located different business unless names are compared in a common script.
4. **Test density shift:** +24% (§2). Validation, calibration and thresholds all assume train-like density.
5. **A compute model:** no pair counts, no hardware assumption, no per-stage time budget.
6. **Intra-source duplicates:** S2 and S3 each hold 2–3 copies per entity (§3). Recall must find *all* copies, and an adaptive K matters.
7. **Transductive unsupervised statistics:** IDF, name frequency and co-location counts should be computed on the split being predicted (test France especially). It's allowed, since it's provided data without labels.
8. **Learning the generator:** the data is synthetic, and 7.6M positive pairs expose the noise operators. The design treats noise generically; it should **estimate operator statistics** (token insertion rates, number transforms, state and city alias maps) and build features on them.

---

## 5. Proposed revised flow (v4)

```
                      ┌─ per split, per country partition (India | US | France | fallback) ─┐
 Raw TSVs ──► N  Normalise: lossless raw + folded + romanised(Indic) + parsed numbers,
                 alt-names (dba/formerly/domain-split), junk stripped, admin tokens tagged
          ──► ST Unsupervised stats ON THIS SPLIT: IDF, name freq (chain), co-location counts
          ──► LX Lexicon mined from TRAIN positive pairs: noise-token insert/delete rates,
                 state/city alias maps, number-transform distribution   (applied to test)
 ─────────────────────────────── recall zone ───────────────────────────────────────────
          ──► R→ Forward retrieval: S1 → top-K records   (name char-ngram, addr char-ngram,
                                                         street+number key; dense only if G-gated)
          ──► R← Reverse retrieval: record → top-k S1    (same views, S1 index is small)
          ──► C  Union + provenance + adaptive budget  ═► candidate_pairs.tsv
 ─────────────────────────────── precision zone ────────────────────────────────────────
          ──► E  Evidence: name-diff w/ noise weights, romanised sim, number-transform class,
                 chain + co-location specificity, retrieval competition (both directions)
          ──► M  GBDT scorer (entity-grouped OOF over the full train world)
                 [+ optional multilingual cross-encoder on uncertainty band, gated]
          ──► O  Ownership: per-record normalisation over {candidate S1s, none}  (G1 hard)
          ──► D  Decision: top-1 vs singleton prior (τ₁) + extras at τ₂  →  P3 if it wins
 ─────────────────────────────── assurance ─────────────────────────────────────────────
   H  Full-world OOF harness + density-stress world + noise-slice diagnostics + LB prior probes
```

**Stage count:** 12 → about 9. The chain is shorter, and each stage has a clean artefact boundary for teammates to own.

---

## 6. Measured reach of the one-to-one rule (S8)

*Sample: 5,570 train S1.* For each class of hard candidate: is the non-matching record **owned by another S1** (so the one-to-one rule can remove it, provided the rightful owner scores higher) or a **pure distractor** (so only the pair scorer can reject it)?

| Hard-candidate class | True matches | Neg: owned by another S1 | Neg: distractor | S8 can fix |
|---|---:|---:|---:|---:|
| Same name, far address (name J = 1, addr J < 0.3) | 737 | 52,861 | 4,572 | **92%** of negatives |
| Co-located, zero name overlap (addr J ≥ 0.5, name J = 0) | 1,041 (35% Indic, 65% renamed or domain) | 4,619 | 687 | **87%** |
| **Near-twin** (addr J ≥ 0.5, name J ≥ 0.5) | 7,693 | **2** | **303** | **~0%** |

What follows:
1. **S8 is the main fix for co-location and same-name confusion.** Those negatives are other real S1 entities, so correct ownership removes them. That's why the reverse "record → which S1 or none" model (§3.7) is worth building. The competition happens between real claimants.
2. **Near-twins are generated as distractors, owned by nobody.** No consistency step can remove them. **Only the pair scorer's evidence** (number transform, residual name tokens) can reject them. This confirms that the near-twin discriminator (§3.6) has to be built; there is no structural shortcut.
3. **Near-twins affect about 5.4% of S1 entities** (303 / 5,570) in train. If test's +24% extra records are mostly distractors, **the near-twin rate on test is likely higher**. So the density shift and near-twin precision are the *same* risk.
4. **The dangerous positives:** 737 true pairs have an identical name but a far or empty address. They sit among 57k same-name negatives. Name-only acceptance loses badly, and S8 ownership plus the chain-frequency features are what can rescue them.
5. **Co-located Indic-name matches** (365) vs co-located Indic negatives (563, of which 420 are owned elsewhere): transliteration plus ownership is what separates them.

---

## 7. Revised gates

| Gate | Question | Status or decision rule |
|---|---|---|
| G1 | One-to-one holds? | **Hard** (measured). S8 = per-record ownership. |
| G2 | Singleton pressure | **Low** (5.6%). Handled by τ₁ in S9, not a stage. |
| G3 | France share | **15%**. Transductive stats, low-weight admin tokens, no special code path. |
| G4 | Cross-encoder beats GBDT-plus-transliteration on the uncertainty band? | Ablation. Multilingual base only. |
| G5 | Reverse listwise ownership beats a plain argmax? | Ablation on full-world OOF. |
| G6 | P2′ vs P3 | Validation, in *both* the normal and density-stress worlds. |
| **G7** | Test density shift: more matches or more distractors? | **Proxy answer (DATA_NOTES §9b): both.** Matches per S1 ≈ +9–16%; distractors per S1 ≈ +50–65% (share ~26% → ~31–35%). Consequence: tune extra-slot thresholds in the density-stress world; confirm with 1–2 LB probes. |
| **G10** | France house numbers: are near-twins more common? | 11.4% of strong France pairs have disjoint numbers, vs 0.6–1.7% for train countries. Treat France number-mismatch as *strong* negative evidence unless the number-transform features explain it. Watch France precision on the LB. |
| **G8** | Transliteration: table vs learned alignment | Measured by India name-recall and precision lift. |
| **G9** | Dense view adds unique recall after transliteration? | Unique-recall measurement. Drop it if about 0. |

---

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Thresholds tuned on train density fail on test (+24%) | High | High | Density-stress world, prior re-estimation, LB probes |
| Near-twins accepted as matches | High | Medium–High | Number-transform and name-diff features. **S8 can't help**: 303 of 305 near-twins are unowned distractors (§6) |
| India Indic-script names mis-scored | High | High (India = 47% of test S1) | Transliteration layer (G8) |
| France generic names + region/department | Medium | Medium (15% of test) | Transductive IDF and chain stats, admin tokens down-weighted |
| **France near-twins**: 11.4% of strong France pairs have disjoint house numbers (vs ≤1.7% in train countries) | Medium–High | Medium–High | Number-transform features; a stricter extra-slot τ₂ if France precision lags; country-specific monitoring |
| Test distractor share rises from ~26% to ~31–35% (proxy) | High | High | Density-stress validation world; thresholds robust in both worlds |
| Cross-encoder compute blows the window | High if attempted broadly | High | Band-only, gated, multilingual, subsampled training |
| Transliteration-table compliance questioned | Low–Medium | Disqualification-level | Ask via the organisers' query form, or use the learned alignment from train pairs (fully in-data) |
| 12-stage chain too long for the team in 3 days | Medium | High | The v4 flow above: ~9 stages, clear artefacts, GBDT-first |

---

## 9. Measurements that would settle the remaining design choices

1. **Number-transform taxonomy:** positives vs near-twin negatives. Is noise string-edit and twin arithmetic? That decides the near-twin features.
2. **Copy-to-copy vs copy-to-S1 similarity:** decides whether C2/transitivity is worth anything.
3. **Ranked-retrieval recall@K:** char n-gram name, address and street+number key, forward vs reverse, per country and per noise slice. This sets K and view selection.
4. **Noise-token insertion rates** per token from positive pairs: the lexicon for the name-diff features.
5. **Size of the unreachable slice:** true pairs with a renamed or Indic name *and* an empty or zero-overlap address.
