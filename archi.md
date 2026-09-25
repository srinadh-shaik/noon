# Business Entity Resolution — Architecture v4.1

Amazon ML Challenge 2026. This document says **what** the system is and **why**
each part exists. The evidence behind every "why" is in `DATA_NOTES.md` (cited as
§n). Reviews: `archi_critique.md` (of v3) and `archi_v4_critique.md` (of v4, which
produced this v4.1). The old design is kept in `archi_v3.md`.

**How to read it:** Part A (pages 1–2) is the whole idea. Part B is one card per
stage, all in the same format. Part C covers validation, the decision maths,
compute, who builds what, and what is still unknown. **Part D is the verification
playbook:** the checks the building agent must run after each stage and for each
gate, with expected values and pass/fail rules.

**What changed from v4 → v4.1** (each change is backed by DATA_NOTES §9c):

| # | Change | Where | Why (measured) |
|---|---|---|---|
| 1 | House-number classes are finer: fraction and letter suffix are split apart | Stages 1, 5 | "Letter or fraction differs" was a 100-vs-112 coin flip; `44→44D` is noise, but `1/2→1/9` is a twin |
| 2 | Leftover-word weights = log-odds of positives vs **hard negatives** | Stages 2, 5 | `group`, `holdings`, `industries` are leftovers in 5–7% of near-twins vs ≤0.2% of copies |
| 3 | Count features are size-normalised and local (same city) | Stages 2, 5 | Test US S1 is half the size of train US S1; raw counts halve |
| 4 | French street abbreviations are learned transductively from test France | Stage 2 (Tier 2) | `R.`, `BD`, `Pl`, `ALL.` never occur in train |
| 5 | V3 (street+number key) is a precision supplement, not a recall carrier | Stage 3 | It finds only 40–58% of true pairs |
| 6 | The recall@K curve for V1/V2 is the **first** Stream B deliverable | Stage 3, C5 | It's never been measured, and every K in the doc is a placeholder |
| 7 | Ownership abstains when claims are nearly tied | Stage 7 | Chain names with empty addresses (3.5–4.6% of copies) tie across many S1 |
| 8 | τ₁ is a real tuned threshold, not "near zero" | Stage 8, C2 | 45% of singletons have a plausible look-alike |
| 9 | World B′ (one-off decoys) added next to World B | Stage 0 | Real decoys are one-offs (0.24% have a sibling), unlike hidden-S1 copies (18.5%) |
| 10 | Prefer relative features to absolute lengths | Stage 5 | Address shape differs by country (India 3–12 parts, US 3–4, France 2–3) |

---

# PART A — The whole idea

## A1. The task in one paragraph

We get business records from three sources. **S1** is clean and deduplicated: one
row per real business. **S2** and **S3** are noisy copies of those businesses, mixed
with unrelated businesses (**distractors**). For every S1 business we must list its
copies in S2/S3, or an empty list if it has none. The score is **F0.5 per S1
business, averaged**, and precision counts double.

## A2. Eight facts about the data that shape everything

| # | Fact (measured) | What it forces |
|---|---|---|
| 1 | The average S1 business has **3.5 copies**; only **5.6%** have none (§3) | Finding *all* copies matters most. Empty answers are a small edge case. |
| 2 | **Names repeat:** about 40% of S1 names are shared with another S1 business (`primary care group` ×253) (§8) | A name alone can never decide a match. |
| 3 | **Addresses are shared too:** 44% of close-address pairs are *different* businesses (§8) | An address alone can never decide a match either. |
| 4 | Name **and** address together almost always decide it. The traps are **near-twins**: almost the same name, a neighbouring house number (`204 1/2` vs `204 1/9 Crawford St`). They're decoys owned by nobody (§8). Their signature: a shared number plus another number ≤20 apart is 90% decoy, while identical numbers are 99.5% true copies (§9c) | We need precise house-number evidence. |
| 5 | **Each S2/S3 record belongs to at most one S1** (true for all 7.6M) (§3) | A free precision tool: every record picks at most one owner. |
| 6 | **27% of Indian S2 names are in Indian scripts** (`ईस्ट सॉल्यूशंस प्राइवेट लिमिटेड` = East Solutions Private Limited) (§6) | Names must be compared in one script (transliteration). |
| 7 | **No postcodes.** Country labels are clean and **always agree** in true pairs (§2, §4) | No postcode logic. Work inside one country at a time. |
| 8 | **Test is harder:** about 24% more S2/S3 records per S1 (more copies *and* more decoys). **France** (15% of test) has no training data and many house-number mismatches (§2, §7, §9b) | Validate in a "harder world" and keep France on country-neutral features. |

## A3. The pipeline at a glance

```
            ┌──────────── build first: the scoreboard ────────────┐
   Stage 0  │ HARNESS    full-train "world", folds, exact scorer,  │  judges every stage
            └──────────────────────────────────────────────────────┘

   Stage 1   NORMALISE     clean text, transliterate Indian scripts, parse house numbers
   Stage 2   KNOWLEDGE     (a) counts from each split  (b) noise patterns learned from train pairs
 ─── find candidates (recall) ──────────────────────────────────────────────────────────
   Stage 3   RETRIEVE      per country: S1 → records  AND  record → S1   (3 lexical views)
   Stage 4   CANDIDATES    union + budget  ═►  candidate_pairs.tsv
 ─── decide (precision) ─────────────────────────────────────────────────────────────────
   Stage 5   EVIDENCE      ~60 features per pair: name, address, numbers, rarity, competition
   Stage 6   SCORE         LightGBM → P(match) per pair
   Stage 7   OWNERSHIP     each record keeps at most one S1 owner
   Stage 8   DECIDE        per S1: take the best one if plausible, extras only if confident
   Stage 9   OUTPUT        matching_results.tsv + validator + submission zip
```

## A4. One real record through the whole pipeline

S1 entity `S1-430295778` (train, India). Its true copies are marked ✓. The
decisions below show what each stage is *meant* to do.

| Record | Name | Address |
|---|---|---|
| **S1** | Mayer & Sons Private Limited | Plot No 9 Khasra No 123, Mauja Lakhanpur, Sadar, Agra, Uttar Pradesh |
| ✓ S2-749504851 | Mayer + Sons Private Limited | PLOT NO 9 KHASRA NO 123, MAUJA LAKHANPUR, SADAR, Uttar Pradesh |
| ✓ S2-204531508 | MAYER + SONS PHFITE LIMITED | H.NO 9 KHASRA NO 123, MAUJA LAKHANPUR, SADAR, उत्तर प्रदेश |
| ✓ S3-671671038 | Mayer and Sons Private  Limited | उत्तर प्रदेश, Sadar, Bhainpur, Plot No 9 Khasra No 123 |
| ✓ S3-499631810 | mayersons.com | Plot No 9 Khasra No 123, Mauja Lakhanpur, Agra, Bhainpur, UP |
| ✓ S2-770104653 | KELOONYX | #9 KHASRA NO 123, MAUJA LAKHANPUR, AGRA, Uttar Pradesh |

1. **Normalise:**
   - `+` and `&` become `and`.
   - `उत्तर प्रदेश` is transliterated and mapped to `uttar pradesh`; `UP` also maps to `uttar pradesh`.
   - `mayersons.com` is split into `mayer sons`.
   - House numbers are parsed to {9, 123} in every row.
2. **Knowledge:** it knows `PHFITE` is a typo-like variant (close to `PRIVATE` in characters), that `private` and `limited` are low-information words, and how many S1 businesses are called "mayer and sons".
3. **Retrieve:** the name view finds the first four rows. The street+number key (`lakhanpur` + `123`) finds all five, including `KELOONYX`.
4. **Evidence:** four rows score high on name and address. `KELOONYX` has zero name similarity but an exact address and exact numbers.
5. **Score:** four rows get p ≈ 0.95+. `KELOONYX` gets a middling p, because "same address, different name" is usually a *different* business (5:1 in train).
6. **Ownership:** no other S1 claims these records, so they stay.
7. **Decide:** take the best one (free), then the three other confident ones. `KELOONYX` stays out unless its p clears the extras bar. Missing it costs little (F0.5 0.95 instead of 1.0); a wrong add would cost more.

---

# PART B — Stage cards

Every card uses the same headings: **Purpose · Why (data) · In → Out · How ·
Defaults · Pitfalls · Judged by.**

---

## Stage 0 — HARNESS (the scoreboard) — *build first*

**Purpose.** A local copy of the leaderboard that we trust more than the real one.

**Why.** Test is harder than train: more records per S1 and more decoys (§9b). A naive validation split is *easier* than test, which makes everything look better than it is.

**In → Out.** Train files + ground truth → fold ids, an exact F0.5 scorer, two validation "worlds", and slice reports.

**How.**
1. **Exact scorer.** It reproduces the official per-S1 F0.5 average, singletons included. It must give 0.714 on the example in the problem statement.
2. **World A, "full train":** all 2.2M train S1 against all 10.3M train S2/S3 records. It's test-*scale* (test has 1.73M S1 and 9.97M records), so every rival S1 is present and ownership sees real competition. It's still *less dense* than test: 4.7 vs 5.8 records per S1.
3. **World B, "no-owner stress":** hide about 20% of train S1 entities from the queries, but keep all their records in the index, so they become decoys for everyone else. Density rises to 5.8. These decoys arrive as clusters of copies, so World B mainly tests the scorer *without ownership help*.
4. **World B′, "test-like decoys":** hide about 34% of S1, keep **one** random copy of each as a one-off decoy, and drop their other copies. Density ≈ 5.8:

   `(0.66·3.45 + 0.34·1 + 1.21) / 0.66 ≈ 5.8`

   Real decoys are one-offs: only 0.24% have a sibling, vs 18.5% of owned records (§9c). So B′ is the closer replica.
5. **Folds:** 5 folds grouped by S1 entity. Each record goes with its owner's fold, and decoys are spread evenly. The model for fold k never sees fold k's labels, so every train pair gets an honest **out-of-fold (OOF)** score.
6. **Slices:** report the score by country, by source, and by noise type: Indian-script name, empty address, renamed or domain name, near-twin present, number of true copies.

**Defaults.** Seeded folds. Every threshold is chosen to do well in **all three** worlds (A, B, B′). None of them reproduces test's extra copies per S1 (≈3.8–4.0 vs 3.45, §9b), so the leaderboard probes remain the final check.

**Pitfalls.**
- Validating a small fold against a small index (too easy).
- Tuning on the public leaderboard. Use at most 1–2 submissions as deliberate probes of the test difficulty.

**Judged by.** Nothing; it's the judge.

**Verify after build.** Part D, checks `V0.*` (all HARD). Nothing downstream is trusted until they pass.

---

## Stage 1 — NORMALISE

**Purpose.** Turn messy text into comparable pieces, **without losing the original**.

**Why.**
- Typos, accents, reordering, abbreviations and Indian scripts hide true matches (§4, §6).
- House numbers are the key evidence against near-twins, so they must be parsed carefully (§8).

**In → Out.** Raw record → normalised record. Every raw field is kept, plus:

| Field | Example | Notes |
|---|---|---|
| `name_clean` | `mayer and sons private limited` | Lowercase; accents stripped (`Flóating`→`floating`); `&`/`+`→`and`; junk removed (`***`, `--`, `[INCORPORATED]`, phone numbers, `(ID: 30420)`) |
| `name_roman` | `ram teknolojis praivet limited` | Non-Latin parts romanised into Latin letters (see below) |
| `name_alts` | [`nexaria labs`, `allen horizon floating inc`] | Split on `dba` / `formerly`; real domains (dot-TLD) and `@handles` split into words (`millerpurpose.com`→`miller purpose`, `@sarsa_consultants`→`sarsa consultants`). A bare last word is **not** an alternate. |
| `name_tokens` | `{mayer, and, sons, private, limited}` | **All words kept**, legal and stop words included. **No `legal_family` column** (decision 2026-09-25): legal forms are shuffled in true pairs, and a hand list would need French forms that train can't teach. Stage 2 weights every word from the data instead (rarity + leftover log-odds, with the unseen-word fallback). |
| `addr_clean`, `addr_tokens` | `plot no 9 khasra no 123 mauja lakhanpur …` | Same cleaning; `null` and `N/A` become empty |
| `numbers` | `[{v:9}, {v:123}]`, `[{v:204, frac:"1/2"}]`, `[{v:44, letter:"d"}]`, `[{v:36, bis:true}]` | Parsed house numbers: value with leading zeros stripped, **fraction and letter suffix kept as separate parts** (a fraction change signals a twin; an added letter is usually noise, §9c), `bis`/`ter` |
| `admin` | state/region token tagged | `MN`/`Minnesota` and `UP`/`उत्तर प्रदेश`/`Uttar Pradesh` are mapped to one form using the Stage 2 alias list |
| `name_nonlatin`, `addr_nonlatin` | name/address contains letters outside the Latin script: yes/no | A generic flag (no script list); used as evidence later |

**How: transliteration**, in two steps:
1. Convert every non-Latin word to Latin letters **from its Unicode character names**, using Python's built-in `unicodedata` (for example `ट` = "DEVANAGARI LETTER TTA" → `ta`). This uses no per-language table and **no third-party transliteration library** (decision recorded in C6).
2. Fix the common words with a **dictionary learned from train pairs**. S1 has the English name, and its copy may have the same name in Indian script, so we can align words: `प्राइवेट`→`private`, `लिमिटेड`→`limited`.
   - Indian-script names are almost always *whole-name* transliterations (1,254 of 1,263 sampled pairs, §9c), so pairs with the **same word count** can be aligned word by word.
   - Keep word pairs seen often enough.
   - This step is 100% in-data.
   - Rare proper names fall back to a phonetic key (a consonant skeleton), because transliteration is lossy.

**Why it's Tier 1** (§9c):
- 16.7% of India true pairs have an Indian-script name.
- Without transliteration, 4.7% of India pairs are almost unreachable (weak name *and* weak address).
- Another 12% must rely on the address alone, where decoys outnumber copies 5:1.
- India is 47% of test S1.

**Tokenisation.** Split on spaces and punctuation only, and keep Unicode combining marks inside words. Python's `\w+` breaks Indian words apart (§11).

**Defaults.** Deterministic. The same code runs on train and test. It never drops information; it only adds fields.

**Pitfalls.**
- Deleting words that distinguish businesses. `Hospital` vs `Pharmacy` at the same address is often the *only* difference.
- Treating legal-form differences as proof of different businesses.

**Judged by.** The recall change in Stage 3 and the F0.5 change in Stage 8 when this stage changes.

**Verify after build.** Part D, checks `V1.*`.

---

## Stage 2 — KNOWLEDGE

**Purpose.** Collect the facts that later stages use to weigh evidence.

**Why.**
- A match on a rare word means more than a match on `private`.
- A leftover word like `Center` is usually noise, while a leftover word like `Summit` usually means a different business.
- The data comes from a generator, and its noise habits can be *measured* from 7.6M true pairs (§4).

**In → Out.** Normalised records (+ train pairs) → lookup tables.

**(a) Counts, computed separately for each split and each country** (train counts for train, test counts for test). No labels are needed, so this is allowed on test and essential for France:
- Word rarity (IDF) for name and address.
- **Name frequency:** how many S1 businesses, and how many S2/S3 records, have this exact name. This is the "chain" signal.
- **Co-location count:** how many S1 businesses sit at this street + number.

**Scale rule:** raw counts change with split size. Test US S1 is half of train US S1, and the chain rate drops from 36% to 29% (§9b). A model rule like "count > 50" would shift meaning. So every count is fed **two ways**:
- **normalised** (per 100k S1 in that split and country, or as a within-split percentile);
- **local** (the same name *in the same city*), which depends far less on split size.

**(b) Noise patterns, learned once from train true pairs** and applied everywhere:

| Table | What it holds | Example |
|---|---|---|
| Aliases | Word ↔ word variants seen in true pairs | `rd↔road`, `mn↔minnesota`, `calcutta↔kolkata`, `cdp`/`city`/`township` = noise. **Critical for US:** 29% of US true pairs have a rewritten address (S3 spells states out) (§9c, §11) |
| **Leftover-word log-odds** | For each word: log(rate as a leftover in *true* pairs ÷ rate as a leftover in *hard negatives*). Both come from train. | Measured: `center` 2.9% of copies vs 1.3% of twins (harmless); `group` 0.2% vs 7.3%, `holdings` 0.1% vs 6.7%, `overseas` 0% vs 6.2% (suspicious) (§9c) |
| **Unseen-word fallback** | Weight for a leftover word with **no learned log-odds** (too rare in train, or never seen, e.g. French `sarl`/`sas`/`eurl`): derived from its **frequency in that split and country**. Very common means near-neutral; rare means mildly informative. **Never treat unseen as suspicious.** | Without it, a French true copy swapping `SAS`↔`SARL` would look like a near-twin. This is the condition attached to dropping `legal_family`. |
| Script dictionary | Romanised word → English word, learned from train pairs | `praivet`→`private`, `phuds`→`foods` |
| Number changes | How house numbers change inside true pairs | zero-padding, truncation `2620→262`, letter added `44→44D`, extra number injected |

**(c) French aliases, mined transductively from test France (Tier 2).**
- France abbreviates street words (`R.`/`R`=Rue, `BD`=Boulevard, `Pl`=Place, `AV.`=Avenue, `ALL.`=Allée), and **none of these occur in train** (§7).
- Take test France pairs that are near-certain from the name and the house number alone.
- Align their address words and keep word pairs with enough support (`r↔rue`).
- It uses no labels and only provided data. France is 15% of test.

**Defaults.** Minimum support (e.g. at least 20 occurrences) before an alias or noise rate is trusted.

**Pitfalls.**
- Computing counts on train and reusing them on test (test France would get no statistics).
- Feeding raw counts (see the scale rule).
- Hand-typing geography tables (compliance grey zone). For France, give region/department words **low weight**, and rely on the mined street aliases.

**Judged by.** The feature importance and ablation gain in Stages 5–6.

**Verify after build.** Part D, checks `V2.*`.

---

## Stage 3 — RETRIEVE (find candidates)

**Purpose.** For each S1, find a short list that very likely contains **all** its true copies.

**Why.**
- Comparing everything with everything (2M × 10M) is impossible.
- Simple "shares a word" blocking needs about 11k candidates per S1 to reach 98% recall (§9).
- Names alone reach at most 87% recall (Indian scripts, renames); addresses carry more (§9).

**In → Out.** Normalised records → list of (S1, record, view, direction, rank, score).

**How.**
1. **Split by country** (India / US / France). Country agrees in 100% of true pairs, and no file has an empty label (§1, §2). Keep one assert that fails loudly if a record ever has an empty or unexpected label; no fallback search path is needed.
2. **Three views**, each a *ranked* search (best-first), not a yes/no bucket:
   - **V1 name:** character-3-gram TF-IDF on `name_clean` + `name_roman` + `name_alts`. Catches typos, accents and reordering.
   - **V2 address:** character-3-gram / word TF-IDF on `addr_clean`. Catches renamed businesses and Indian-script names.
   - **V3 street + number key:** exact match on (rare address word, house number). Precise and cheap, but it finds only **40% (US) / 58% (India)** of true pairs (§9c). It's a **supplement**; V1 and V2 must carry recall.
3. **Two directions:**
   - **Forward:** S1 → top-K records. Finds businesses with many copies.
   - **Reverse:** record → top-k S1. The S1 index is small, and it matches fact 5: each record has at most one owner. It also tells us, for free, *how contested* a record is.

**Defaults.** Start with forward K = 50 per view and reverse k = 5 per view, then tune both on World A's recall curve. **Optional V4:** a multilingual embedding view, kept only if it finds true matches that V1–V3 miss (gate G9).

**⚠ Unmeasured, so measure it first.**
- Recall@K of the ranked V1/V2 search has **never been measured**. Every K and recall number here is a placeholder.
- Cheap exact keys miss 8.9% of real entities entirely (§9c).
- **Stream B's first deliverable** is the recall@K curve: per view, per direction, per country and per noise slice, on a 5k-S1 sample against the full index.
- Reverse retrieval stays only if it adds **unique** recall.

**Pitfalls.**
- Postcode keys (there are no postcodes).
- English phonetic codes like Soundex (useless for Indian and French names).
- Fixing K by feel instead of by the recall curve.

**Judged by.** Recall (share of true pairs found) per view, per country and per noise slice, plus each view's **unique** contribution.

**Verify after build.** Part D, checks `V3.*`.

---

## Stage 4 — CANDIDATES (= `candidate_pairs.tsv`)

**Purpose.** Merge all views into the single list the model will score. **This fixes the recall ceiling.**

**In → Out.** Stage 3 lists → one row per (S1, record) with provenance: which views and directions found it, best rank and score in each.

**How.**
1. Take the union and remove duplicates.
2. Apply an **adaptive** per-S1 budget: cut where scores drop sharply, not at a flat K. Some businesses have 8–11 copies.
3. Write `candidate_pairs.tsv` from exactly this set.

**Defaults.** An average budget of about 30–40 per S1 (a placeholder until the recall-vs-cost curve exists). **Leave headroom:** test has about 3.8–4.0 copies per S1 vs 3.45 in train (§9b), so a budget that just fits train is tight on test.

**Pitfalls.** Filtering *after* this file is written. The rules require this file to be exactly what the model scores.

**Judged by.** Recall ceiling, candidates per S1 (mean and p95), and the full pair count against the compute budget.

**Verify after build.** Part D, checks `V4.*`.

---

## Stage 5 — EVIDENCE (features)

**Purpose.** Describe each candidate pair with numbers that explain *why* it would or wouldn't be the same business.

**Why.** The traps are specific (facts 2–4), so the features must target them directly.

**In → Out.** Candidate pairs + normalised records + Stage 2 tables → one row of about 60 features per pair.

**How: eight feature groups.**

| Group | What it measures | Aimed at |
|---|---|---|
| F1 Name similarity | Best of: word overlap (rarity-weighted), 3-gram cosine, fuzzy ratio, computed on clean, romanised and alternate names | Typos, accents, reordering, scripts |
| F2 **Name leftovers** | The words left over after matching (legal words included; their learned weight is near-neutral), scored by their **leftover log-odds** (Stage 2; unseen words use the frequency fallback): leftover `center` ≈ harmless, leftover `group`/`holdings` ≈ suspicious. About 89% of near-twins carry such a word; the other 11% have identical names and only F4 can catch them (§9c) | Near-twins, co-located businesses |
| F3 Address similarity | Rarity-weighted word overlap, per-component best alignment (order-free), street match, city match (alias-aware) | Reordering, dropped parts |
| F4 **House numbers** | The *kind* of difference, plus the numeric gap `|a−b|` (classes and measured match rates below) | **Near-twins** (`109` vs `111`) vs noise (`2620`→`262`) |
| F5 Rarity and crowding | Name frequency (chain count), co-location count, rarity of shared words, all **size-normalised and local** (Stage 2 scale rule) | Repeated names, shared buildings |
| F6 Missing info | Address empty on either side, no numbers, Indian-script name. Measured: one side missing numbers is 472 copies vs 1 decoy, so missing ≠ different (§9c) | So "missing" is never read as "different" |
| F7 Competition | Views and directions that found the pair and their ranks; for the record: best S1 score vs this S1's; how many S1s claim it | Records wanted by several S1s |
| F8 Source | S2 or S3 (they have different styles) | Style differences |

**F4 classes** (measured among pairs with similar name *and* address; true copies vs decoys, §9c):

| House-number relation | copies | decoys | Reads as |
|---|---:|---:|---|
| identical numbers | 5,397 | 28 | match |
| one side has no number | 472 | 1 | match (missing ≠ different) |
| a number dropped or injected | 954 | 49 | mostly match |
| truncation (`2620`↔`262`) | 141 | 25 | mostly match |
| **shares a number, another ≤20 apart** | 27 | **253** | **near-twin** |
| shares a number, another far apart | 34 | 48 | mostly decoy |
| letter suffix added (`44`↔`44D`) | ⎫ 100 | ⎫ 112 | expected noise |
| fraction changed (`1/2`↔`1/9`) | ⎭ | ⎭ | expected twin; the v4.1 split separates these |

**Important rule:** **don't give the model the country name.** France is unseen, and a country feature would put it out of vocabulary. Every feature must be country-neutral (similarities, rarities, counts).

**Caveat:** country-neutral isn't distribution-neutral. Address shape differs (India 3–12 components, US 3–4, France 2–3, §11), and the model can learn "short address → US-like". Prefer **relative** features (overlap shares, percentiles) to absolute lengths, and read the France slice on a leaderboard probe.

**Pitfalls.**
- Treating a missing address as a mismatch.
- Features computed differently on train and test.
- Any feature that uses labels.

**Judged by.** The feature-group ablation: F0.5 with and without each group.

**Verify after build.** Part D, checks `V5.*`.

---

## Stage 6 — SCORE

**Purpose.** Turn the features into P(match) for each pair.

**In → Out.** Feature rows → probability per pair: out-of-fold for train, and for test.

**How.**
1. **LightGBM**, binary.
2. Train on a **subsample of train S1 entities** (e.g. 300–500k, each with *all* its candidates, so the negatives are real hard negatives). All ~100M train pairs won't fit in memory.
3. Give 5-fold grouped OOF scores for the whole of World A (Stage 0).
4. Check calibration: when the model says 0.8, is it right about 80% of the time? Fix with isotonic calibration on OOF if not.

**Optional upgrade (gate G4):** a cross-encoder text model, run **only on uncertain pairs** (say p between 0.2 and 0.8), with a **multilingual** base because of the Indian scripts and French. Its score becomes one more feature. Keep it only if it wins on validation.

**Defaults.** Final model ≤ 8B parameters, MIT or Apache-2.0 licence. LightGBM qualifies trivially.

**Why LightGBM, honestly:**
- It's a reasoned default, **not yet a measured win**.
- The decisive signals are structured (number classes, leftover words, counts; §9c), the data is large, and it runs on CPU.
- A subsample of 400k S1 still holds about 37k near-twin-type negatives (§9c): plenty.
- **Also rank quality:** in the middle score band a third of top-1 picks are wrong (§9c), so add within-S1 rank features, and try a ranking objective (LambdaRank) as an ablation.

**Pitfalls.**
- Random negatives instead of retrieved ones.
- An English-only text model.
- Scoring all pairs with a neural model (it won't fit in the time we have).

**Judged by.** Precision/recall curves as a diagnostic; F0.5 after Stages 7–8 as the real measure.

**Verify after build.** Part D, checks `V6.*`.

---

## Stage 7 — OWNERSHIP (one owner per record)

**Purpose.** Use fact 5: each S2/S3 record belongs to at most one S1.

**Why.** Most confusing negatives, same name elsewhere or same building, are records that **belong to another real S1**: 87–92% of them (§8). Ownership removes them for free. (It can't remove near-twins; those are owned by nobody. Stage 5's F4 handles them.)

**In → Out.** P(match) for all pairs → adjusted probability q per pair.

**How.**
- **Default:** for each record, keep only its **highest-scoring S1**, **but only if it beats the second-best claim by a margin δ**. If the claims are nearly tied, the record goes to **nobody**. Every other S1's claim is dropped. A record can belong to only one S1, so this is simply "pick the best, or abstain"; no complex assignment algorithm is needed.
  - *Why abstain:* a true copy with a chain name (`primary care group` ×253) and an empty or weak address (3.5–4.6% of copies, §9c) fits many S1 equally well. A confident wrong pick costs a wrong ID (weight 1) plus a missed ID (0.25). Abstaining costs only the miss (0.25).
  - δ is tuned in Worlds A, B and B′.
- **Upgrade (gate G5):** a small model that, for each record, chooses among {its candidate S1s, **none**} and outputs probabilities that sum to 1. "None" matters: 26–35% of records are decoys.

**Pitfalls.** Letting ownership add pairs. It may only remove or down-weight.

**Judged by.** F0.5 with ownership on vs off (Worlds A, B and B′).

**Verify after build.** Part D, checks `V7.*`.

---

## Stage 8 — DECIDE (final list per S1)

**Purpose.** Pick the final list of matches for each S1, possibly empty.

**Why.** The scoring formula (Part C2) says:
- the **first** match is nearly free: an empty answer already scores 0 unless the S1 is a singleton (5.6%);
- **every extra** match needs high confidence, because a wrong one costs about 3× what a missed one costs.

**In → Out.** Per S1: candidates with q → final list.

**How (default rule, two thresholds):**
1. Sort the candidates by q.
2. Include the **top one** if q₁ ≥ τ₁. τ₁ is lower than τ₂, but it's a **real tuned threshold**:
   - **45% of singletons have a plausible look-alike** (§9c), and each wrongly accepted singleton costs a full 1.0.
   - Measured in the middle band: 1,105 correct top-1, 498 wrong, 131 singletons. Accepting still wins, but not by a mile.
3. Include **each further** candidate if q ≥ τ₂. τ₂ is high, typically 0.7–0.8.
4. Tune τ₁ and τ₂ on OOF scores so they do well in **all three** worlds (A, B, B′).

**Upgrade (gate G6):** pick, per S1, the list size with the highest *expected* F0.5 given the probabilities. It's exact via a small dynamic programme, but only as good as the calibration. Keep it only if it beats the two-threshold rule in all three worlds (A, B, B′).

**Pitfalls.**
- One global threshold for everything.
- Thresholds from pair-level F1.
- Separate thresholds for France (there's no data to tune them).

**Judged by.** Macro F0.5, the official metric.

**Verify after build.** Part D, checks `V8.*`.

---

## Stage 9 — OUTPUT

**Purpose.** Produce files that are guaranteed to be accepted, plus the package.

**How.**
1. Write `matching_results.tsv`: one row per test S1, possibly empty, tab-separated, IDs comma-joined.
2. Check that the matches are a subset of `candidate_pairs.tsv`.
3. Run `utils/validate_submission.py` and require `PASS`.
4. Build the zip: `output/`, `code/business_entity_resolution/` (src, README, pinned requirements), and the methodology document.

**Judged by.** Validator `PASS`. One command must regenerate both files from raw data.

**Verify after build.** Part D, checks `V9.*` plus the release checklist D4.

---

# PART C — Making it work

## C1. Artefacts (what flows between stages)

| Artefact | Made by | Grain | Contents |
|---|---|---|---|
| `records_norm.parquet` | 1 | record | raw + normalised fields |
| `stats_{split}.parquet` | 2a | token / name / address | IDF, name frequency, co-location counts |
| `lexicon/*.parquet` | 2b | word / pattern | aliases, noise-word rates, script dictionary, number changes |
| `retrieval.parquet` | 3 | (S1, record, view, direction) | rank, score |
| `candidates.parquet` → `candidate_pairs.tsv` | 4 | (S1, record) | provenance |
| `features.parquet` | 5 | (S1, record) | about 60 features |
| `scores.parquet` | 6 | (S1, record) | p (OOF for train) |
| `owned.parquet` | 7 | (S1, record) | q |
| `matching_results.tsv` | 8–9 | S1 | final list |

Every artefact is saved, so any stage can be re-run and re-judged alone.
**Rule:** if anything in Stages 1–6 changes, re-tune Stages 7–8. Old thresholds never carry over.

## C2. The decision maths (why "top-1 free, extras strict")

For one S1 with **k** true copies, predicting **c** right and **w** wrong gives:

```
F0.5 = 1.25·c / (c + 0.25·k + w)
```

A wrong ID adds **1** to the bottom; a missed ID adds only **0.25**. Example with k = 4:

| Prediction | F0.5 |
|---|---:|
| all 4 right | 1.00 |
| 3 right, none wrong | 0.94 |
| 4 right + 1 wrong | 0.83 |
| only 1 right | 0.63 |
| empty | 0.00 |

- **Slot 1:** going from empty to 1 right is worth +0.63. A wrong first pick scores 0, the same as empty. So the first pick only has to beat the singleton hypothesis.
  - The singleton rate is 5.6% overall, but singletons are *not* spread evenly. 45% of them have a plausible look-alike, so exactly where the top candidate is mediocre, the singleton risk is much higher than 5.6%.
  - That's why τ₁ must be tuned, not set near zero.
- **Extra slots:** adding a 4th when 3 are right gains 0.06 if right and loses 0.19 if wrong. So it needs **p > 0.75**. As a rule of thumb, extras need p ≳ 0.8 × (the F0.5 we typically achieve).

## C3. Decisions left to the data (gates)

| Gate | Question | Status |
|---|---|---|
| G1 | One owner per record? | ✅ **Yes, exactly.** Stage 7 is on. |
| G2 | How many S1 have no match? | ✅ 5.6%, handled by τ₁ in Stage 8. |
| G3 | How much of test is France? | ✅ 15%. Country-neutral features, per-split counts. |
| G4 | Does a cross-encoder beat LightGBM on uncertain pairs? | ⏳ Ablation. Multilingual base only. |
| G5 | Does the "record chooses owner or none" model beat plain best-pick? | ⏳ Ablation. |
| G6 | Two thresholds vs expected-F0.5 rule? | ⏳ Validation in A, B and B′. |
| G7 | Why does test have more records? | ✅ Proxy: both more copies (+9–16%) and more decoys (~26% → ~31–35%). World B is required. |
| G8 | Rule-table transliteration enough, or add the learned dictionary? | ⏳ Measure India name recall and precision. |
| G9 | Does an embedding view find matches V1–V3 miss? | ⏳ Unique-recall check; drop it if about 0. |
| G10 | France house-number mismatches (11.4% vs ≤1.7%): near-twins or noise? | ⏳ Watch France on 1–2 leaderboard probes. Consider a stricter τ₂ if its precision lags. |
| G11 | Ownership margin δ: abstain vs plain argmax? | ⏳ Tune δ in A/B/B′; keep abstention only if it wins. |
| G12 | Do the transductive French aliases lift France? | ⏳ Count mined pairs; check the France slice on a leaderboard probe. |
| G13 | Pointwise LightGBM vs a ranking objective (LambdaRank) for top-1 quality? | ⏳ Ablation on top-1 precision and macro F0.5. |

## C4. Compute budget

| Item | Size |
|---|---|
| Records per split | about 12M (train), 11.7M (test) |
| Candidate pairs per split | about 60–100M (at ~35 per S1, plus reverse pairs) |
| LightGBM training | 300–500k S1 subsample ≈ 10–20M rows |
| Feature computation | vectorised or C-backed string metrics (e.g. `rapidfuzz`), run per country in chunks |
| Cross-encoder (optional) | Uncertain band only. Scoring everything would take days. |

**Machine:** 32 GB+ RAM recommended. The 15 GB laptop works only by running each country and chunk separately. A GPU is needed only for the optional G4/G9 stages.

## C5. Build order and team split

**Tier 1 — needed for a strong submission (build in this order):**
*After each step, run that stage's Part D checks. Do not start the next step while any HARD check fails.*
1. Stage 0 scorer + World A folds (Worlds B and B′ are cheap relabelings of A; add them before tuning).
2. Stage 1: cleaning, number parsing (with fraction and letter split), rule-table transliteration.
3. Stage 2: size-normalised and local counts, aliases, leftover-word log-odds, number-change stats.
4. Stages 3–4: **the V1/V2 recall@K curve first**, then V1, V2 and V3 forward + reverse, and the budget from the curve.
5. Stages 5–6: F1–F8, then LightGBM OOF.
6. Stage 7: best-pick ownership with margin abstention.
7. Stage 8: τ₁ and τ₂ tuned in Worlds A, B and B′.
8. Stage 9: output + validator.

**Tier 2 — upgrades once Tier 1 scores:** the learned Indian-script dictionary (G8), **transductive French street aliases (G12)**, the owner-or-none model (G5), the expected-F0.5 rule (G6), a ranking objective (G13), better alias mining.

**Tier 3 — only if time and GPU allow:** the cross-encoder on uncertain pairs (G4), the embedding view (G9).

**Three workstreams with clean hand-offs:**

| Stream | Owns | Hands off |
|---|---|---|
| A: Data | Stages 1–2 | `records_norm`, `stats`, `lexicon` |
| B: Recall | Stages 3–4 | `candidates` + recall report |
| C: Precision | Stages 0, 5–9 | scores, thresholds, submission |

**Leaderboard plan:** 1–2 probe submissions (same model, two τ₂ values) to read the test difficulty, then **one** final best submission.

## C6. Deliberately left out

| Left out | Why |
|---|---|
| Postcode logic | There are no postcodes (§4). |
| "Legal form differs → not a match" | Legal forms are shuffled in true matches (§4). |
| Country as a model feature | France would be unseen. We use country only to split the search. |
| Hand-typed geography tables | Compliance grey zone. France admin words get low weight instead. |
| Complex assignment algorithms | One owner per record is just "pick the best" (Stage 7). |
| Linking S2↔S3 copies to each other | Measured: copies are closer to S1 (mean similarity 0.62) than to each other (0.51) (§9c). |
| A fallback search for empty country labels | No file has an empty label (§1); an assert is enough. |
| English-only text models (e.g. ModernBERT-based) | 27% of Indian S2 names are in Indian scripts, and France is in test. |
| **Third-party transliteration libraries** (`unidecode`, `anyascii`, `indic-transliteration`, …) | **A deliberate decision.** (1) Compliance: a library is effectively an external lookup table, closer to the "no external data" line than logic built on the Unicode standard; `unidecode` is also GPL. (2) Not needed: romanising from Unicode character names (stdlib `unicodedata`) keeps 96% of a hand-built table's gain (39.9% vs 41.4% of Indian-script true pairs share a real name word). (3) No extra dependency to install or pin. The remaining gap (`phuds`↔`foods`, `sh`↔`s`) is closed by the learned dictionary from train pairs (G8), which is also library-free. **Ceiling:** it fits scripts where each consonant carries a vowel, which covers every non-Latin script in this data (§6). Greek or Cyrillic would need revisiting. |
| LLM teacher / distillation | Cost, relative to the time window. |
| Any external lookup (APIs, geocoding, web data) | Prohibited; disqualification. |

## C7. Still unknown (measure these; don't assume them)

| # | Unknown | Why it matters | Who | How |
|---|---|---|---|---|
| U1 | Recall@K of V1/V2, forward and reverse | Sets the recall ceiling, K and the budget | Stream B | 5k-S1 sample vs the full index, per view, country and slice |
| U2 | Rule-table transliteration quality | India precision and recall | Stream A | Name similarity on the 1,263 sampled Indian-script pairs vs same-address decoys |
| U3 | Baseline F0.5 of Tier 1 | Is the whole design working? | Stream C | First end-to-end run in A, B and B′ |
| U4 | Yield of French alias mining | France precision | Stream A | Mined word pairs above support |
| U5 | Test copies per S1 and France behaviour | Thresholds under shift | Stream C | 1–2 leaderboard probes (same model, two τ₂) |
| U6 | Real compute per stage | Whether Tier 2/3 fit | All | Log wall-clock and memory per stage |

## C8. Glossary

- **Entity / S1 business:** one real business; S1 has exactly one row per business.
- **Record / copy:** a row in S2 or S3. A copy is a noisy version of an S1 business; a **decoy** (distractor) is an S2/S3 row that belongs to no S1.
- **Candidate:** a (S1, record) pair we bother to score.
- **Recall ceiling:** the share of true pairs that make it into the candidates. Nothing later can recover the rest.
- **Near-twin:** a decoy with nearly the same name and a neighbouring house number.
- **Co-location:** different businesses at the same address.
- **Chain name:** a name shared by many different businesses (`primary care group`).
- **Transliteration:** writing a word from one script in another (`लिमिटेड` → `limited`).
- **IDF / rarity:** how rare a word is. Rare shared words are strong evidence.
- **OOF (out-of-fold):** a score from a model that never saw that pair's label. It's the only honest input for tuning.
- **World A / World B:** full-train validation at train density / at test-like density.
- **Macro F0.5:** F0.5 computed for each S1, then averaged. Precision counts twice as much as recall.

---

# PART D — Verification playbook (run after building each stage)

## D0. How the agent uses this

1. **After building a stage, run all of that stage's `V` checks** and write the results to `reports/verify_stage<N>.json`, one entry per check:
   `{"id": "V3.2", "value": 0.412, "expected": "0.35–0.65", "pass": true, "level": "HARD"}`
2. **HARD** check fails → **stop.** Don't build the next stage, and report the failing check with its value.
   **SOFT** check fails → continue, but flag it in the report and in the stage summary.
3. **Anchors:** many expected values are numbers we already measured (DATA_NOTES §). If a rebuilt pipeline doesn't reproduce them, the bug is in the pipeline, not the data.
4. **Unit checks** (fixed input → exact output) live as small `assert`-based tests next to the code. They run in seconds and must pass before any full-data run.
5. **Re-verify downstream after any change.** Changing stage N means re-running the checks for N and every stage after it (see the re-tuning rule in C1).
6. **"Better" means ε:** a gain of **≥ +0.002 macro F0.5** in World A **and** World B′, and no loss worse than −0.001 in World B. Anything smaller is noise, and the simpler option wins.

---

## D1. Stage checks

### Stage 0 — HARNESS (all HARD)

| ID | Check | Expected |
|---|---|---|
| V0.1 | Scorer on the problem statement's example (predict {S2-00047, S2-00193, S3-00812}, truth {S2-00047, S3-00812}) | 0.714 (±0.001) |
| V0.2 | Scorer: perfect prediction on train GT | 1.000 |
| V0.3 | Scorer: all-empty prediction on train GT | **0.0558** (= singleton rate, §3) |
| V0.4 | Scorer: a singleton with any prediction scores 0; a singleton with empty scores 1; a non-singleton with empty scores 0 | exact |
| V0.5 | Row counts read from raw files | train S1 2,206,821 · S2 5,034,616 · S3 5,285,603 · GT 2,206,821 · test S1 1,732,544 · S2 4,887,273 · S3 5,082,316 |
| V0.6 | IDs read as strings; no id parsed as a number; 0 duplicate ids per file | exact |
| V0.7 | GT: every matched id has exactly one owner | 7,638,365 ids, max owners = 1 (G1) |
| V0.8 | Folds: every S1 in exactly one fold; every owned record in its owner's fold; decoys spread with fold sizes within ±1% | exact |
| V0.9 | World B density (records per queried S1) | 5.8 ± 0.1 |
| V0.10 | World B′ density; the share of B′ decoys having a same-name, similar-address sibling | 5.8 ± 0.1; < 1% (real decoys: 0.24%, §9c) |
| V0.11 | Seeded: rebuilding folds and worlds twice gives identical files (hash) | identical |

### Stage 1 — NORMALISE

| ID | Check | Expected | Level |
|---|---|---|---|
| V1.1 | Unit: `Flóating Désert Ámc` → `name_clean` | `floating desert amc` | HARD |
| V1.2 | Unit: `Foot + Ankle Care` and `Foot & Ankle Care` → the same `name_clean` | equal | HARD |
| V1.3 | Unit: junk stripped: `*** Allen Horizon`, `-- Regional California`, `[INCORPORATED] PEAK`, `Downtown Seafood - 6215889221`, `Peak Nexpoint (ID: 30420)` | junk removed; raw kept | HARD |
| V1.4 | Unit: `Studio 90 Pub` keeps `90` | kept | HARD |
| V1.5 | Unit: `millerpurpose.com` → alt name | `miller purpose` | SOFT |
| V1.6 | Unit: `Gildriza dba EYF Pharmaceutical…` and `Nexaria Labs formerly Allen Horizon…` → 2 alternates each | 2 | HARD |
| V1.7 | Unit numbers: `0070`→{v:70}; `44D`→{v:44, letter:d}; `204 1/2`→{v:204, frac:1/2}; `36 bis`→{v:36, bis}; `#9 KHASRA NO 123`→{9},{123}; `2151/8`→{v:2151, sub:8} (a slash house number, *not* a fraction) | exact | HARD |
| V1.8 | Unit: `null`, `N/A` in an address → treated as empty | empty | HARD |
| V1.9 | Unit tokeniser: `लिमिटेड` stays **one** token | 1 token | HARD |
| V1.10 | Unit transliteration: `प्राइवेट लिमिटेड` → `name_roman` 3-gram similarity to `private limited` | ≥ 0.5 (after the dictionary: exact) | SOFT |
| V1.11 | Coverage: every non-Latin name gets a non-empty `name_roman`, with no non-Latin letters left | 100%; 0 residual | HARD |
| V1.12 | Output rows = input rows per file; raw fields byte-identical to input | exact | HARD |
| V1.13 | Deterministic: two runs → identical output hash | identical | HARD |
| V1.14 | Same code path for train and test: the same rows normalised with train's vs test's vocabulary agree on every column except `name_alts` (behavioural test, not a claim) | identical | HARD |
| V1.15 | Unit: ordinals aren't house numbers: `78 BD Albert 1er`→{78}; `2Nd Floor 12 Park St, 213th Drive`→{12}; `8BIS`→{8, bis}; `74SECTOR-33`→{74},{33} (missing space keeps the number) | exact | HARD |
| V1.16 | Unit: a bare last word is not an alternate (`Hernandez Pipeline`→[]); an `@handle` gives one segmented alternate (`@midwestinterstate`→`midwest interstate`, `@sarsa_consultants`→`sarsa consultants`) | exact | HARD |
| V1.17 | Unit: general romanisation across scripts: `लिमिटेड`, `லிமிடெட்`, `ટેક્નોલોજીસ`, `প্রাইভেট`, `ਪ੍ਰਾਈਵੇਟ` → `limited`, `limitet`, `teknolojis`, `praibhet`, `praivet` | exact | SOFT |

### Stage 2 — KNOWLEDGE

| ID | Check | Expected | Level |
|---|---|---|---|
| V2.1 | Statistics exist per split and per country, including **test France** | non-empty for all 5 combos | HARD |
| V2.2 | Raw chain rate reproduces | train US ≈ 35.8%, India ≈ 44.4%; test US ≈ 29.1%, India ≈ 43.8%, France ≈ 34.4% (§9b) | HARD |
| V2.3 | Normalised count features: train vs test distribution shift (PSI) for India and US | PSI < 0.2 (raw counts will fail this; normalised must pass) | SOFT |
| V2.4 | The alias table contains `rd↔road`, `st↔street`, `mn↔minnesota`, `up↔uttar pradesh`, `calcutta↔kolkata`; `cdp`/`township` flagged as noise | all present | HARD |
| V2.5 | Leftover log-odds signs: `group`, `holdings`, `industries`, `overseas` clearly negative; `center`, `services`, `shri` near 0 or positive | signs as stated (§9c) | HARD |
| V2.6 | Number-change stats contain truncation, zero-pad, letter-added, injected-number classes with non-zero counts | present | SOFT |
| V2.7 | Minimum support enforced (no alias or log-odds entry below support) | 0 violations | HARD |
| V2.8 | Built from **train pairs only**; nothing uses test labels (there are none); test counts use test records only | code review | HARD |
| V2.9 | (Tier 2) French alias mining yields `r↔rue`, `bd↔boulevard`, `av↔avenue`, `pl↔place` | all 4 present | SOFT |
| V2.10 | **Unseen-word fallback:** leftover words with no learned log-odds get a frequency-based weight, and very common unseen words are near-neutral. On test France, `sarl`, `sas`, `eurl`, `sasu` all get \|weight\| ≤ the weight of train's most neutral legal word (e.g. `limited`) | holds for all 4 | HARD |

### Stage 3 — RETRIEVE

| ID | Check | Expected | Level |
|---|---|---|---|
| V3.1 | The recall@K curve file exists: per view × direction × country × slice, K ∈ {5, 10, 20, 50, 100, 200} | exists | HARD |
| V3.2 | V3-key recall reproduces the anchor (rare word + number, S1-df ≤ 200) | US ≈ 0.40, India ≈ 0.58 (§9c) | SOFT |
| V3.3 | Unique recall per view reported; a view with < 0.1% unique recall is flagged as redundant | report | SOFT |
| V3.4 | Candidates never cross countries; never contain S1 ids; the country assert never fires | 0 | HARD |
| V3.5 | Recall per noise slice (Indian-script name, empty address, renamed/domain, dba) reported; none silently 0 | report, each > 0 | HARD |
| V3.6 | Runtime and memory logged per country | logged | SOFT |

### Stage 4 — CANDIDATES

| ID | Check | Expected | Level |
|---|---|---|---|
| V4.1 | No duplicate (S1, record) pairs; only S2/S3 ids | 0 | HARD |
| V4.2 | Every test S1 has exactly one row in `candidate_pairs.tsv` (empty allowed) | 1,732,544 rows | HARD |
| V4.3 | Recall ceiling on World A reported, overall and per slice | report. **Target set from the V3.1 curve**, then HARD: ceiling ≥ target | HARD |
| V4.4 | Candidates per S1: mean, p95, max; total pairs within the compute budget (C4) | within budget | HARD |
| V4.5 | S1 entities with ≥ 8 true copies keep ≥ 95% of their copies (adaptive budget works) | ≥ 95% | SOFT |
| V4.6 | The scored pair set in Stage 6 == the pairs in `candidate_pairs.tsv` (set equality) | equal | HARD |

### Stage 5 — EVIDENCE

| ID | Check | Expected | Level |
|---|---|---|---|
| V5.1 | Train and test feature tables have identical columns and dtypes | identical | HARD |
| V5.2 | No column encodes the country, the source file name or any label | 0 | HARD |
| V5.3 | F4 classes on train reproduce §9c: identical numbers ≥ 99% positive; "shared + another ≤ 20 apart" ≈ 90% negative; after the split, "fraction changed" leans negative and "letter added" leans positive | within ±5 points | HARD |
| V5.4 | Missing is not disagreement: pairs with a number missing on one side are ≥ 95% positive on train (anchor: 472 vs 1) | ≥ 95% | HARD |
| V5.5 | NaN only in documented "missing" columns; no infinities | 0 undocumented | HARD |
| V5.6 | Per-feature PSI train vs test (India, US) | < 0.25 each, or listed with a reason | SOFT |
| V5.7 | France feature ranges fall inside the train ranges for ≥ 95% of rows per feature | ≥ 95% | SOFT |

### Stage 6 — SCORE

| ID | Check | Expected | Level |
|---|---|---|---|
| V6.1 | OOF scores exist for 100% of World A pairs | 100% | HARD |
| V6.2 | Fold hygiene: the model for fold k saw no S1 from fold k | 0 overlap | HARD |
| V6.3 | **Label-shuffle control:** train on shuffled labels → OOF AUC | 0.50 ± 0.01 (higher = leakage) | HARD |
| V6.4 | Calibration: expected calibration error on OOF (after isotonic if used) | ECE ≤ 0.02 | HARD |
| V6.5 | Top features include F4 (numbers) and F2 (leftovers) groups | in top 10 by gain | SOFT |
| V6.6 | Test scores exist for every test candidate pair, France included; France score distribution reported | 100%; report | HARD |
| V6.7 | Licence and size of every model used ≤ 8B params, MIT or Apache-2.0 (LightGBM: MIT) | recorded | HARD |

### Stage 7 — OWNERSHIP

| ID | Check | Expected | Level |
|---|---|---|---|
| V7.1 | After ownership, every record belongs to ≤ 1 S1 | max = 1 | HARD |
| V7.2 | Ownership only removes or down-weights: output pairs ⊆ input pairs; no q > p | exact | HARD |
| V7.3 | Macro F0.5 with ownership on vs off, in A, B and B′ | on ≥ off in A and B′ | HARD |
| V7.4 | Abstention rate (records given to nobody by the margin rule) reported, and its precision effect | report | SOFT |

### Stage 8 — DECIDE

| ID | Check | Expected | Level |
|---|---|---|---|
| V8.1 | Thresholds (τ₁, τ₂, δ) tuned on OOF only; the values saved with the model version | saved | HARD |
| V8.2 | Macro F0.5 in A, B and B′ reported, overall and per slice | report | HARD |
| V8.3 | Beats the all-empty baseline (0.0558) and the rule baseline **B0** (name J ≥ 0.5 and addr J ≥ 0.5 among candidates, then ownership) | ≥ B0 + 0.05 in World A | HARD |
| V8.4 | No country-specific thresholds (unless G10 decides otherwise via leaderboard) | none | HARD |
| V8.5 | The final list ⊆ candidates for every S1 | exact | HARD |
| V8.6 | Singleton accuracy and top-1 precision reported (anchor: 45% of singletons have a look-alike) | report | SOFT |

### Stage 9 — OUTPUT

| ID | Check | Expected | Level |
|---|---|---|---|
| V9.1 | `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` | `PASS`, exit 0 | HARD |
| V9.2 | Validator warnings: matched ids not in candidates | 0 | HARD |
| V9.3 | `matching_results.tsv` has 1,732,544 data rows, the exact header, tabs, no quotes | exact | HARD |
| V9.4 | Clean-checkout rerun (one command, raw data → both files) gives identical file hashes | identical | HARD |
| V9.5 | `--check-ids` run on a machine with enough RAM | PASS | SOFT |

---

## D2. Gate verification (how each open decision is settled)

Each gate ends with a written verdict in `reports/gates.md`: the numbers, the rule applied, and keep/drop.

| Gate | Evidence to produce | Keep / decide rule |
|---|---|---|
| G1 one owner | V0.7 | Already settled: hard. Re-assert on every build. |
| G2 singletons | V0.3 | Settled: 0.0558. Re-assert. |
| G3 France share | Share of test S1 labelled France | Settled: 0.1498 ± 0.001. Re-assert. |
| G4 cross-encoder | F0.5 with vs without its score feature, in A/B/B′; runtime on the uncertain band | Keep only if ≥ ε (D0.6) **and** the runtime fits C4 |
| G5 owner-or-none model | F0.5 vs plain best-pick with margin | Keep only if ≥ ε |
| G6 expected-F0.5 rule | F0.5 vs the two-threshold rule | Keep only if ≥ ε; ties go to two thresholds (simpler) |
| G7 test density | Leaderboard probes P1 (τ₂ as tuned in A) and P2 (τ₂ as tuned in B′) | If P2 > P1 on the leaderboard, test looks like B′: ship B′-tuned thresholds. Otherwise ship the robust choice (the best worst-case across A/B/B′). |
| G8 transliteration dictionary | India slice F0.5 and Indian-script-slice recall, with vs without the dictionary | Keep if the India slice gains ≥ +0.003 or Indian-script recall gains ≥ 2 points without a precision loss |
| G9 embedding view | Unique recall at the chosen budget; F0.5 gain | Keep only if unique recall ≥ 0.3% of true pairs **and** F0.5 ≥ ε |
| G10 France numbers | Leaderboard probe P3 = P1 but with a stricter France τ₂ | Keep the stricter France τ₂ only if P3 > P1 on the leaderboard. It's the one allowed country-specific setting, decided by leaderboard evidence alone. |
| G11 ownership margin | F0.5 across δ ∈ {0, 0.05, 0.1, 0.2} in A/B/B′ | Pick the δ with the best worst-case; δ = 0 (plain argmax) if the gain < ε |
| G12 French aliases | Number of mined pairs; V2.9; leaderboard probe with vs without | Keep if V2.9 passes and the probe doesn't lose |
| G13 ranking objective | Top-1 precision and macro F0.5, LambdaRank vs pointwise | Keep only if F0.5 ≥ ε (top-1 precision alone isn't enough) |

**Leaderboard budget for probes:** at most 3 probes (P1, P2, P3) across days 2–3. That leaves room for the final submission plus a fallback.

---

## D3. Verification report format

`reports/verify_summary.md`, regenerated after every full run:

```
Run: <git sha> <date>  Worlds: A / B / B′ macro F0.5 = 0.xxx / 0.xxx / 0.xxx
Stage 0  11/11 HARD pass
Stage 1  13/14 pass   SOFT fail: V1.10 (roman sim 0.41 < 0.5)
...
Gates:   G4 dropped (+0.0007 < ε) · G8 kept (+0.006 India) · ...
Open:    U1 done (recall@50 union = 0.9xx) · U5 pending (probe P2 tomorrow)
```

---

## D4. Release checklist (before the final submission)

- [ ] All HARD checks V0–V9 pass on the final configuration.
- [ ] Every gate has a written verdict in `reports/gates.md`.
- [ ] Thresholds were tuned on OOF after the **last** change to Stages 1–6 (re-tuning rule).
- [ ] The validator prints `PASS` with 0 warnings (V9.1, V9.2).
- [ ] The clean-checkout rerun reproduces identical files (V9.4).
- [ ] The zip has `output/`, `code/business_entity_resolution/` (src, README with the exact commands, pinned `requirements.txt`) and the filled `Documentation_template.md`.
- [ ] No external data, API or lookup anywhere in the code (grep for `http`, `requests`, `urllib`: 0 hits outside comments).
- [ ] Model licences and sizes recorded (V6.7).
- [ ] The submission ledger is updated: version → config → World A/B/B′ scores → leaderboard score.
