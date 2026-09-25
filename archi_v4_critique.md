# Critique of Architecture v4: every choice, questioned against the data

Each v4 decision gets the same four lines:

- **Q:** the question a sceptic would ask.
- **Data:** what we measured (the `DATA_NOTES.md` section).
- **Verdict:** ✅ supported · ⚠️ partly right, needs a change · ❌ contradicted · ❓ not measured yet (a guess).
- **Change:** what to do about it.

---

## 0. Scoreboard

| # | v4 decision | Verdict |
|---|---|---|
| 0.1 | Validate on the full train "world", not small folds | ✅ |
| 0.2 | World B = hide 20% of S1 to mimic test | ⚠️ its decoys are the wrong *kind* |
| 1.1 | Transliterate Indian scripts | ✅ strongly |
| 1.2 | Rule table first, learned dictionary second | ❓ |
| 1.3 | Keep legal forms only as a weak signal | ✅ |
| 1.4 | Parse house numbers into a structure | ✅, but the classes must be finer |
| 2.1 | Count statistics separately per split | ✅, but counts must be size-normalised |
| 2.2 | Noise-word rates learned from true pairs | ✅, but contrast them with negatives |
| 2.3 | Alias tables from train pairs | ✅, and more important for US than v4 says |
| 2.4 | France: nothing learnable, down-weight admin words | ⚠️ French abbreviations are missed entirely |
| 3.1 | Split by country | ✅ (the empty-label fallback is unneeded) |
| 3.2 | V3 street+number key "82% share a number" | ❌ the key finds only 40–58% |
| 3.3 | V1/V2 char-3-gram ranked retrieval | ❓ recall@K never measured, the biggest unknown |
| 3.4 | Reverse retrieval (record → S1) | ❓ logical, unmeasured |
| 3.5 | K = 50 / k = 5, budget 30–40 | ❓ guesses |
| 5.1 | F4 house-number evidence | ✅ the strongest near-twin signal |
| 5.2 | F2 leftover-word evidence | ✅ |
| 5.3 | F5 chain / co-location counts | ✅, normalise scale |
| 5.4 | F6 "missing ≠ different" | ✅ |
| 5.5 | No country feature | ✅, with a caveat |
| 6.1 | LightGBM as the scorer | ❓ a sound default, no result yet |
| 6.2 | Train on a 300–500k S1 subsample | ✅ |
| 7.1 | Ownership: one owner per record | ✅ |
| 7.2 | Ownership = plain argmax | ⚠️ needs an "abstain if unclear" rule |
| 8.1 | Top-1 "nearly free", τ₁ low | ⚠️ 45% of singletons have a look-alike |
| 8.2 | Extras need p ≳ 0.75 | ✅ |
| C6 | Don't link S2↔S3 copies | ✅ confirmed |

**In one line:** the precision half of v4 (Stages 5–8) is mostly *confirmed* by the data. The recall half (Stage 3) is mostly *unmeasured*. Five concrete fixes follow.

---

## Stage 0 — Harness

### 0.1 Full-train world instead of small folds — ✅
- **Q:** Why not ordinary 5-fold validation on a fifth of the data?
- **Data:** Competition depends on density. Train has 4.67 S2/S3 per S1 and test 5.8 (§2). A fifth-size index has a fifth of the competitors per query. Ownership also needs every rival S1 present: 87–92% of confusing negatives are owned by *another* S1 (§8).
- **Verdict:** ✅
- **Change:** fix the wording. World A is *not* "the same size as test": it has 2.2M S1 vs 1.73M, at a lower density.

### 0.2 World B: hide 20% of S1 so their records become decoys — ⚠️
- **Q:** Do the decoys we create look like the extra decoys in test?
- **Data:**
  - **No.** Hidden S1 copies arrive as *clusters*: 18.5% of owned records have a same-name, similar-address sibling. Real decoys are **one-off records** (0.24% have a sibling) (§9c).
  - World B keeps 3.45 matches per S1, while test has about 3.8–4.0 (§9b).
  - World B mostly turns "owned by another S1" negatives into ownerless ones. That tests the scorer *without ownership help*, which is a useful stress test but a different failure from test's.
- **Verdict:** ⚠️ It's a stress test, not a replica.
- **Change:**
  - Add **World B′:** hide about 34% of S1, keep only **one** random copy of each as a decoy, and drop the rest. The decoys are then one-offs, like real ones, and the density still reaches ≈5.8:
    `(0.66·3.45 + 0.34·1 + 1.21)/0.66 ≈ 5.8`.
  - Pick thresholds that are good in A, B and B′.
  - Treat the leaderboard probes as the real check on test cardinality.

---

## Stage 1 — Normalise

### 1.1 Transliterate Indian scripts — ✅ strongly
- **Q:** Could the address alone handle Indian-script names, so we skip transliteration?
- **Data:**
  - Indian-script names make up 16.7% of India true pairs. Of those, 71% have a good address and 28% don't (§9c).
  - Without transliteration, the first group becomes "same address, different name" pairs, where negatives outnumber matches 5:1 (§8).
  - The second group, 4.7% of India pairs, is almost unreachable.
  - India is 47% of test S1.
- **Verdict:** ✅ It's both a precision and a recall fix, and it belongs in Tier 1.
- **Change:** none. Almost all Indian-script names (1,254 of 1,263) are *whole-name* transliterations, which makes the learned dictionary easy: align words in pairs with the same word count.

### 1.2 Rule table first, dictionary second — ❓
- **Q:** Is a rule-based romanisation good enough on its own?
- **Data:** Not measured. We only know the pairs are whole-name transliterations of English words (`प्राइवेट लिमिटेड`).
- **Verdict:** ❓ (gate G8)
- **Change:** Stream A's first test: rule-table romanisation, then name similarity on the 1,263 sampled Indian-script pairs vs same-address negatives.

### 1.3 Legal forms as a weak signal only — ✅
- **Data:** After removing legal words, the names are identical in 72% of true copies (§9c). Legal words differ constantly in true pairs (§4).
- **Verdict:** ✅

### 1.4 Parse house numbers — ✅, but finer classes are needed
- **Q:** Is "agree / disagree / missing" enough?
- **Data** (§9c), inside the "similar name and address" group:
  - identical numbers: 5,397 matches vs 28 negatives;
  - **a shared number plus another number ≤20 apart: 27 vs 253 (90% negative)**;
  - truncation or a dropped number: mostly matches;
  - **"same digits, letter or fraction differs": 100 vs 112, a coin flip.** That bucket merges `44→44D` (noise) with `1/2→1/9` (twin).
- **Verdict:** ✅ the principle; ⚠️ the resolution.
- **Change:** keep **fraction** and **letter suffix** as separate parsed parts, and make "fraction changed" and "letter added" separate classes.

---

## Stage 2 — Knowledge

### 2.1 Count statistics per split — ✅, normalise scale
- **Q:** Won't counts from test differ from train?
- **Data:**
  - France exists only in test (§2), so train-only counts give France nothing. That confirms counts must be computed per split.
  - But **test US S1 is half the size of train US S1** (0.66M vs 1.32M), so raw counts roughly halve. The chain rate is 29% in test vs 36% in train (§9b).
  - A model split like "name count > 50" learned on train means something different on test.
- **Verdict:** ✅ per split; ⚠️ raw counts.
- **Change:** feed **size-normalised** counts (per 100k S1, or a within-split percentile) and *local* counts (same name **in the same city**). Local counts depend much less on split size.

### 2.2 Noise-word rates from true pairs — ✅, contrast with negatives
- **Data:**
  - Near-twin leftover words are *business words*: `group`, `holdings`, `industries`, `overseas`, `exports`, `ventures`, `enterprises`, at 5–7% of negatives each. Each is ≤0.2% in positives.
  - Positive leftovers are *filler*: `center`, `services`, `partners`, `dba`, `shri`, `the` (§9c).
- **Verdict:** ✅ The signal is strong.
- **Change:** compute each word's **log-odds of being a leftover in positives vs hard negatives**, not just its rate in positives. Both are available in train.

### 2.3 Alias tables — ✅, and more important for US than v4 says
- **Data:** **29% of US true pairs have a similar name but a weak address match (J < 0.5)** (§9c). S3 spells US states out (only 4% use a 2-letter code, vs 86% in S1) (§11). S3 India uses codes (`UP`, `KL`) 57% of the time.
- **Verdict:** ✅ Aliases are a Tier-1 dependency for the address features in both countries.

### 2.4 France: down-weight admin words, learn nothing — ⚠️
- **Q:** What about French street words?
- **Data:** France uses `R.`/`R` for Rue, `BD`/`Bd.` for Boulevard, `Pl` for Place, `AV.` for Avenue, `ALL.` for Allée. None of these appear in train, so no alias can be learned (§7). France is 15% of test.
- **Verdict:** ⚠️ v4 leaves French abbreviations unhandled.
- **Change:**
  - **Transductive alias mining on test France.** Take pairs that are near-certain from the name and the house number alone, align their address words, and keep frequent pairs (`r↔rue`, `bd↔boulevard`). It uses no labels and only provided data.
  - Tier 2, but it's the main France lever next to house numbers.
  - Character-3-gram similarity partly covers it meanwhile.

---

## Stage 3 — Retrieve

### 3.1 Split by country — ✅
- **Data:** Country agrees in 100% of true pairs, and the label is never empty in any file (§1, §2).
- **Change:** the "empty or unknown label searches everything" fallback never triggers. Keep it only as a tiny assert-style guard.

### 3.2 V3 street+number key — ❌ the claim is wrong
- **Q:** v4 says "82% of true pairs share a number", which implies V3 has high recall. True?
- **Data:** The key (a rare address word plus a number) finds only **40% of US and 58% of India** true pairs. Exact name or key together find 59–66% (§9c). Sharing *some* number isn't the same as sharing *the key*.
- **Verdict:** ❌ V3 is a precise *supplement*, not a recall carrier. V1/V2 must carry recall.
- **Change:** fix the text; keep V3 as a cheap high-precision view.

### 3.3 V1/V2 ranked character-3-gram retrieval — ❓ the biggest unknown
- **Q:** What recall does top-K reach, and at what K?
- **Data:**
  - We only know word-level boolean blocking: 97.6% at about 11k candidates (§9).
  - Cheap exact keys miss **8.9% of non-singleton S1 entirely** (§9c).
  - Ranked 3-gram retrieval has **never been measured**.
- **Verdict:** ❓ Every recall number in v4 is a hope until this curve exists.
- **Change:** **the first deliverable of Stream B** is a recall@K curve per view, per direction, per country and per noise slice, on a 5k-S1 sample against the full index.

### 3.4 Reverse retrieval — ❓
- **Data:** The logic holds (each record has at most one owner, §3), but its *unique* recall and its cost (10M queries) are unmeasured.
- **Verdict:** ❓ Keep it, but gate it on unique recall like any other view.

### 3.5 K = 50, k = 5, budget 30–40 — ❓
- **Data:** None. These are placeholders.
- **Note:** test has more copies per S1 (≈3.8–4.0 vs 3.45, §9b), so a budget sized on train is *tighter* on test. Size it with headroom.

---

## Stage 5 — Evidence

### 5.1 F4 house numbers — ✅ the single best near-twin signal
- **Data:** Identical numbers are 99.5% matches. "Shared number + another ≤20 apart" is 90% negatives (§9c).
- **Change:** use exactly the classes in §9c (with the letter/fraction split from 1.4), plus the numeric gap.

### 5.2 F2 leftover words — ✅
- **Data:** Only 10.8% of near-twin negatives have an empty leftover (identical name apart from legal words), vs 72% of positives. So about 89% of near-twins carry a telltale leftover word (§9c).
- **Note:** the other 11% (for example `First Apex Premium` vs `CORP. FIRST APEX PREMIUM`) can **only** be caught by F4.

### 5.3 F5 chain and co-location counts — ✅
- **Change:** normalise the scale (see 2.1).

### 5.4 F6 missing ≠ different — ✅
- **Data:** When one side has no number: 472 matches vs 1 negative (§9c). Treating "missing" as "disagree" would throw away true copies.

### 5.5 No country feature — ✅, with a caveat
- **Q:** Is the model then really country-neutral?
- **Data:** Not fully. Address length and component counts differ by country: India 3–12 components, US 3–4, France 2–3 (§11). A model can learn "short address → US-like behaviour" and apply it to France.
- **Change:** accept the risk, but read the France slice on a leaderboard probe. Prefer *relative* features (overlap shares) over absolute lengths.

---

## Stage 6 — Score

### 6.1 LightGBM — ❓, a sound default
- **Q:** Why LightGBM and not a neural matcher?
- **Data:** The decisive signals are structured: number classes, leftover words, counts (§9c). There's no model result yet.
- **Verdict:** ❓ A reasoned default (CPU, scale, tabular signals), not a measured win. G4 can overturn it.

### 6.2 300–500k S1 training subsample — ✅
- **Q:** Enough hard negatives?
- **Data:** About 520 strong-class negatives per 5,570 S1 (§9c), so about 37k near-twin-type negatives per 400k S1. Plenty. And all ~100M pairs don't fit in memory.

---

## Stage 7 — Ownership

### 7.1 One owner per record — ✅
- **Data:** It's exact for all 7.64M ids. It removes 87–92% of same-name and co-located negatives (§8).

### 7.2 Plain argmax — ⚠️ add abstention
- **Q:** What if the claims are nearly tied?
- **Data:**
  - A true copy with a chain name and an empty or weak address (3.5–4.6% of true pairs, §9c) looks equally good for many S1: `primary care group` ×253 (§8).
  - Argmax then hands it confidently to *some* S1, often the wrong one. That's a wrong ID (weight 1) plus a missed ID (weight 0.25).
- **Verdict:** ⚠️
- **Change:** **argmax with a margin.** If the best claim doesn't beat the second by δ (tuned), give the record to **nobody**. A miss costs a quarter of a wrong pick.

---

## Stage 8 — Decide

### 8.1 "The top-1 is nearly free", τ₁ low — ⚠️
- **Q:** Free against what?
- **Data** (§9c):
  - It's free against an *empty answer for a non-singleton*.
  - But **45% of singletons have a plausible look-alike** (name J + addr J ≥ 1.0).
  - In that middle band, a third of top-1 picks for real entities are wrong.
  - The band holds 1,105 correct, 498 wrong and 131 singletons. Accepting still wins: about +660 F0.5 points vs −131. So the rule is right, but **τ₁ is a real tuned threshold, not "near zero"**.
- **Change:**
  - Say so in the doc, and tune τ₁ in World A/B/B′.
  - Top-1 **ranking** quality (which candidate comes first) matters as much as the threshold. It's an argument for a ranking-aware objective or rank features in Stage 6.

### 8.2 Extras need p ≳ 0.75 — ✅
- **Data:** It follows from the scoring formula `F = 1.25c/(c + 0.25k + w)` (archi C2). It's arithmetic, not an assumption.

---

## Part C checks

- **C6 "don't link S2↔S3 copies":** ✅ Copies are more similar to S1 (0.62) than to each other (0.51) (§9c).
- **C4 compute budget:** ❓ Estimates only. Record the actual pair counts and runtimes once Stage 3 runs.
- **Tiering:** ✅ Transliteration, number features, leftover words and aliases are all in Tier 1, and the data supports each. **Move French alias mining (2.4) to Tier 2** explicitly.

---

## New issues v4 doesn't mention

1. **World B's decoys have the wrong shape.** Add World B′ (one-off decoys) (0.2).
2. **Count features shift with split size.** Normalise them (2.1).
3. **French street abbreviations aren't handled.** Transductive alias mining (2.4).
4. **Ownership needs abstention** for tied claims (7.2).
5. **The letter vs fraction house-number split** is the difference between noise and a twin (1.4).
6. **US addresses are rewritten in 29% of true pairs,** so alias quality drives US recall and precision (2.3).
7. **Cheap keys miss 8.9% of real entities entirely,** so the V1/V2 recall curve is the top priority (3.3).

---

## What the data still can't answer (and who measures it)

| # | Unknown | Why it matters | Who | How |
|---|---|---|---|---|
| U1 | Recall@K of V1/V2, forward and reverse | Sets the recall ceiling and the budget | Stream B | 5k-S1 sample vs the full index, per view and slice |
| U2 | Rule-table transliteration quality | India precision and recall | Stream A | Similarity on the 1,263 Indian-script pairs vs same-address negatives |
| U3 | Baseline F0.5 of Tier 1 in World A/B/B′ | Is the whole design working? | Stream C | First end-to-end run |
| U4 | Yield of French alias mining | France precision | Stream A | Count aligned pairs above support |
| U5 | Test cardinality and France behaviour | Thresholds under shift | Stream C | 1–2 leaderboard probes (same model, two τ₂) |
| U6 | Real compute per stage | Whether Tier 2/3 fit | All | Log wall-clock and memory per stage |
