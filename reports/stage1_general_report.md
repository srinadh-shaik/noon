# Stage 1: general version, before/after report

**Branch:** `stage1-general` (from `17eb4cf`). **Not merged.** The other agent's worktree and outputs (`stage0-harness`) were not touched.
**Files:** `src/s1_normalise.py`, `reports/verify_stage1.json`. Outputs are in this worktree's `work/s1/` (gitignored).
**Reproduce:** `python src/s1_normalise.py`, same command as before.

## Why
The architect asked for Stage 1 to have **no language-specific hand tables or word lists**, wherever a general mechanism does the same job. The review also found two logic bugs.

## What changed

| # | Before (`17eb4cf`) | After (`stage1-general`) | Evidence |
|---|---|---|---|
| 1 | A hand-built romanisation table for 9 Brahmic scripts (the code-point range also covered Sinhala and produced garbage) | **General romanisation from Unicode character names**: `LETTER X` gives x; a vowel sign replaces the inherent `a`; a virama removes it; the word-final inherent vowel is dropped; doubled letters inside one letter's name collapse (`TTA`→`ta`, `AA`→`a`). No per-script tables. | See §B |
| 2 | `INDIC` code-point range; columns `name_indic`/`addr_indic` | `NONLATIN = [\p{L}&&\P{Latin}]`; columns **`name_nonlatin`/`addr_nonlatin`** | Same count: 1,620,114 names flagged |
| 3 | English stop-words `{and, the, of}` removed from `name_tokens` (French `de`/`la`/`du` were kept) | No stop-word list; **`name_tokens` keeps all words** | Weighting belongs to Stage 2 (per-split rarity + learned leftover log-odds) |
| 4 | A hand-typed `LEGAL` list (including French `sarl`/`sas`, which can't be learned from train); column `legal_family` | **`LEGAL` list and the `legal_family` column removed.** Legal words stay in `name_tokens`; Stage 2 weights them from data | A positional "last word" rule was considered and rejected: it would also flag `group`, which is the strongest near-twin signal (DATA_NOTES §9c) |
| 5 | Ordinal filter for English `st/nd/rd/th` only, so **French `1er` became house number 1** (`78 BD Albert 1er` → {78, 1}) | General rule: digits glued to **2–3 letters** are an ordinal or unit (`1er`, `2nd`, `213th`, `2eme`, `40ft`) and are dropped (`bis`/`ter` excepted). **≥4 glued letters** means a missing space, so the number is kept (`74SECTOR-33` → {74, 33}) | See §C |
| 6 | `m/s` honorific rule (India-specific) | Removed; Stage 2 leftover weights handle it | — |
| 7 | **Bug:** the domain regex treated any final word of 4+ letters as a domain, so **56% of S1 names got a junk alternate** equal to their last word (`limited`, `group`, `trust`). With "max over alternates" similarity, that alternate `limited` would match every "… Limited". | A domain needs a real dot-TLD (`\.[a-z]{2,6}`), with optional `www.` | S1 train rows with alternates: **1,247,235 → 12** |
| 8 | **V1.14 hard-coded `True`** (it checked nothing) | Behavioural: 200k test rows normalised with the train vs test vocabulary must match on every column except `name_alts` | Pass; 1,315 rows differ in `name_alts` only, as expected |

Kept on purpose, because they're data-backed: `ALIAS` markers (`dba`/`formerly`/`aka`; the generator uses English markers in every country, including France S3), `bis`/`ter`/`quater`, and `LEET` (measured on train pairs).

## A. Checks (`reports/verify_stage1.json`)
- **14/14 HARD pass.** The only SOFT fail is V1.10 (`praivet limited` vs `private limited`), the same as before; the G8 learned dictionary is meant to fix it.
- **New proposed checks,** to add to archi.md Part D:
  - **V1.15:** ordinals dropped, `bis` kept, missing-space number kept.
  - **V1.16:** a bare last word is not an alternate.
  - **V1.17 (SOFT):** general romanisation across scripts gives `limited`, `limitet`, `teknolojis`, `praibhet`, `praivet`. These were predicted by hand before running, and all matched.
- **Runtime:** 1:55 at 4.7 GB peak (before: 2.6 min at 4.8 GB).

## B. Romanisation quality (50k Indian-script true pairs, train, same rows)
Share of pairs where the romanised copy shares at least one non-legal name word with its S1:

| | share |
|---|---|
| no romanisation | 0.063 |
| before: hand table | **0.414** |
| after: general | **0.399** |

That's 96% of the hand table's gain. 69.7% of romanised strings are byte-identical to before. The remaining differences are systematic: `ph`↔`f` (`phuds`/`foods`), `s`↔`sh` (`SSA`→`s`), and sometimes better (`alpha` vs `alfa`). These are consistent substitutions, which is exactly what the Stage 2 learned dictionary (G8) is designed to absorb.

## C. House numbers (the rule was chosen on data)
Across 130k train true pairs with numbers on both sides:

| Rule | India identical / disjoint | US identical / disjoint |
|---|---|---|
| before (English ordinals) | 0.7389 / 0.0594 | 0.8011 / 0.0793 |
| drop any glued letters (rejected: loses `74SECTOR`, `905NEW`) | 0.7394 / 0.0598 | 0.8012 / 0.0793 |
| **adopted: 2–3 letters dropped, ≥4 kept** | 0.7392 / 0.0594 | 0.8012 / 0.0793 |

- On train countries all three tie, and the adopted rule keeps the most number evidence.
- Rows whose numbers changed: 0.14–0.21% per file.
- **France (test): 3,918 rows fixed.** Examples: `Albert 1er`, `1ère Armée`, `20eme Siecle`, `6eme etage` no longer add a fake house number.
- **Known ceiling:** a flat number with a 2–3 letter suffix (`5AB`, `255KA/…`) now loses its number. That's rare (about 500 train S2/S3 rows).

## D. Interface changes for downstream stages (read before building Stage 2+)
1. `name_indic`/`addr_indic` are renamed **`name_nonlatin`/`addr_nonlatin`**.
2. **`legal_family` is removed.** Legal-form weighting moves to Stage 2 (learned leftover log-odds plus per-split rarity). archi.md's Stage 1 table still lists `legal_family`, so the architect has to sign off on this.
3. **`name_tokens` now includes legal and stop words.** Any Stage 2+ code that assumed they were stripped must weight them instead.
4. `name_alts` is now almost always empty for S1 (12 rows in train), and non-empty only for real `dba`/`formerly` names or real domains (about 4–6% of S2/S3).
5. Recommendation for Stage 5: compare romanised names with a letter-collapsed key on **both** sides, so geminates like `ottappalam`↔`otapalam` still match.
