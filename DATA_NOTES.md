# Dataset Dossier — Business Entity Resolution (Amazon ML Challenge 2026)

A reference for design discussions. Every number here was **measured** on the full
data (2026-09-25) unless marked *sample*. Samples are deterministic: S1 ids with
`crc32(id) % N == 0`. The analysis scripts were throwaway and not kept.

Normalisation used for all comparisons: casefold → NFKD → strip combining accents →
`\w+` tokens.

---

## 0. The story in plain words

1. **The data is synthetic.** Each business has one clean S1 record. S2 and S3 records are noisy copies of it: typos, accents, a shuffled address, and sometimes the name written in Hindi, Telugu or another Indic script. Other S2/S3 records are unrelated businesses (distractors).
2. **Names are not unique.** About 40% of S1 businesses share their exact name with another S1 business (`primary care group` ×253). A name alone cannot decide a match.
3. **Name and address are each insufficient alone; together they are strong.**
   - A same-name non-match *never* shares much of the address, while a true match usually does (87%).
   - But many *different* businesses sit at the same address: 44% of close-address pairs are non-matches, so the address alone isn't enough either.
   - The residual traps are near-twins (similar name, slightly different house number) and renamed or transliterated names at a shared address.
4. **The hard cases are few and specific.**
   - True matches with an empty or unrelated address, about 9%, where only the name is left and the name is ambiguous.
   - Names in an Indic script: 19–28% of Indian true pairs share no name word.
   - France: no training data, generic names, and region vs department naming.
5. **There are no postcodes**, so a postcode key is useless.
6. **Each S1 entity has about 3.5 true matches** and only 5.6% have none, so **finding all of them (recall) matters more than handling singletons**.
7. **Each S2/S3 record belongs to at most one S1 entity.** This exact one-to-one rule is a free precision tool.
8. **Simple word-based blocking is expensive.** Reaching ~98% recall by sharing a single word costs about 11k candidates per entity, so candidate generation must be ranked retrieval, not word buckets.
9. **Test differs from train:** about 24% more S2/S3 records per S1. Thresholds tuned on train may be off on test.

---

## 1. Files and scale

| File | Rows | Size |
|---|---:|---:|
| train_source1.tsv | 2,206,821 | 210 MB |
| train_source2.tsv | 5,034,616 | 489 MB |
| train_source3.tsv | 5,285,603 | 504 MB |
| train_ground_truth.tsv | 2,206,821 | 127 MB |
| test_source1.tsv | 1,732,544 | 175 MB |
| test_source2.tsv | 4,887,273 | 509 MB |
| test_source3.tsv | 5,082,316 | 506 MB |

- **Schema:** `entity_id, business_name, business_address, country`. The GT file has `source1_entity_id, matched_entity_ids`.
- **ID format:** `S<k>-<integer>`, for example `S1-385631947` or `S2-72648771`. The IDs are random. The correlation between an S1 id number and its match's id number is −0.008 (*sample*, 19k pairs), so **there is no id-order leakage**.
- **Hygiene is perfect:** every file has 0 malformed rows, 0 duplicate ids and 0 wrong prefixes. The GT covers all 2,206,821 train S1 ids, and every matched id exists in S2/S3.
- **Literal junk tokens** occur, for example `null` and `N/A` inside addresses.

## 2. Country mix and train→test shift

| | train S1 | test S1 | train S2+S3 | test S2+S3 | S2+S3 per S1 (train → test) |
|---|---:|---:|---:|---:|---|
| India | 883,188 | 809,986 | 4,133,346 | 4,717,565 | 4.68 → **5.82** |
| US | 1,323,633 | 663,106 | 6,186,873 | 3,817,031 | 4.67 → **5.76** |
| France | — | 259,452 (15.0%) | — | 1,434,993 | — → 5.53 |

- The country label is **clean** (only `India`, `US` and `France`). It appears in all three sources, France included.
- **The country label agrees in 100% of true pairs** (*sample*, 382k pairs).
- **Test has about 24% more S2/S3 records per S1 than train, in every country.** Either the test set has more matches per entity or more distractors, and the labels can't tell us which. Priors, thresholds and calibration fitted on train may therefore be shifted on test. See §11.

## 3. Ground-truth structure (train)

- **Singletons:** 5.58% (123,247 entities), about the same for India (5.59%) and US (5.58%). Predicting no matches for everyone scores **0.056**.
- **Cardinality** (matches per S1):

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 123,247 | 119,157 | 375,212 | **530,841** | 484,115 | 321,957 | 164,868 | 63,968 | 18,680 | 4,205 | 534 | 37 |

  That is a mean of 3.46 matches per S1, or 3.67 per non-singleton.
- **Source mix per entity:** 80.5% of entities have S2 and S3 matches, 7.5% only S3, 6.5% only S2, and 5.6% none.
- **Matches split by source:** 48.4% S2, 51.6% S3.
- **S2 and S3 are NOT deduplicated.** One entity often has 2–3 records in the same source. The top (#S2, #S3) combinations are (1,1) 270k, (1,2) 251k, (2,1) 223k, (2,2) 208k, (1,3) 140k and (3,1) 114k.
- **Strict one-to-one:** all 7,638,365 matched ids belong to exactly one S1. **Gate G1 is hard.**
- **Distractors** (S2/S3 records that match no S1): 26.6% of S2 and 25.4% of S3. They look like ordinary businesses that simply aren't in S1, including Indic-script names.

## 4. How the records look (noise catalogue)

The data is **synthetically generated** from a clean S1 record by applying noise operators.

**Name operators** (all seen in true pairs):
- Case changes. S2 is often ALL-CAPS.
- Accent injection on Latin names, for example `Flóating`, `Cáre`, `Désert`, `Màison`, `Çomite`.
- OCR-style typos and digit swaps, for example `Memoria1`, `Pharmaceutica1`, `TRADIN6`, `Floatign`, `Dehert`, `PHFITE`, `Adatrs`.
- Legal-form changes. Suffixes are added, dropped, swapped, abbreviated or moved to the front: `LLC Floria'S Capital`, `Private Ottappalam Arts Limited`, `L.L.C.`, `Incorporated`.
- Filler words added: `Co`, `Services`, `Center`, `Group`, `Holding`, `International`, `Distribution`.
- Honorific prefixes (`Shri`, `Sri`, `Smt --`) and junk prefixes (`***`, `--`, `[INCORPORATED]`).
- `&` ↔ `and` ↔ `+`.
- Word reordering and duplicated tokens (`BQN BQN Limited Private Infrastructure`).
- Punctuation joins (`Ottappalam-Arts`).
- Trade-name forms: `X dba Y`, `Nexaria Labs formerly Allen Horizon Floating, Inc.`
- The name replaced by a domain (`millerpurpose.com`) or by a phone number (`Downtown Seafood - 6215889221`).
- **The name replaced entirely by an unrelated token** (`KELOONYX` for `Mayer & Sons Private Limited`). Only the address links these.
- **The whole name transliterated into an Indic script** (`રામ ટેક્નોલોજીસ પ્રાઇવેટ લિમિટેડ` = Ram Technologies Private Limited), or partly (`Shree ਫਿਊਚਰ Estate ਪ੍ਰਾ. ਲਿ.`). See §6.
- Inserted country tags: `(India)`, `(France)`, `[FRANCE]`, `(laIna)`.

**Address operators:**
- Components reordered (`AZ, Buckeye, 1989 213th Drive`).
- Components dropped (city, district, state), or the address emptied (about 3% of S2/S3 records, 4.5% of true pairs).
- Abbreviations: Rd/Road, St/Street, Ave, Ct, Dr, BD/Boulevard, R./Rue, Pl/Place, ALL./Allée.
- State shown as a code or in full: `MN`↔`Minnesota`, `UP`↔`Uttar Pradesh`, `KL`, `MH`, `DL`, `WB`.
- State written in an Indic script (`उत्तर प्रदेश`, `കേരളം`, `ఆంధ్రప్రదేశ్`).
- City suffixes (`CDP`, `City`, `Township`), city aliases (`Calcutta`/`Kolkata`, `Newdelhi`), and sometimes a different locality (`Beekman` vs `Hopewell Junction`).
- House-number formatting: `#`, `##`, `No`, `H.NO`, `Door No`, `N°`, `Nº`, `(12)`, zero-padding (`0070`, `023`), `bis`/`ter`.
- **House numbers perturbed in true matches** (`2620`→`262`, `L1`→`L2`, `44`→`44D`, extra `NO 00944` injected). So a disagreeing number is only soft evidence.
- Extra PO Box, PMB, Unit or Apt components.
- Typos (`DLIVE`, `JUNCCTION`, `Plaae`, `Gadn`).
- **No postcodes in practice.** A 5–6 digit token appears in about 11% of US addresses, about 1% of Indian ones and 0.5% of French ones. At least one side lacks a postcode in 91–99.8% of true pairs.

**Source style:**
- S2 addresses are mostly UPPERCASE; S3 is title case.
- S3 carries `dba` forms (India 1.1%, US 1.6%); S2 essentially none.
- S3 rewrites addresses more than S2: the address is identical in 4% of S3 pairs vs 11–13% for S2.

**Country style:**
- India: long addresses (mean 78 characters in S1, many components). Legal forms appear in 82% of S1 names. Landmarks (`near`, `opp`, `behind`) appear in 9–13% of addresses.
- US: short addresses (about 35 characters: number, street, city, state).
- France (test only): see §7.

## 5. True-pair similarity (*sample*, 382k pairs, by source|country)

| Group | name identical | addr identical | name tokens share nothing | name trigrams share nothing | addr tokens share nothing |
|---|---:|---:|---:|---:|---:|
| S2 India | 15% | 11% | 28% | 23% | 4% |
| S3 India | 17% | 4% | 19% | 14% | 4% |
| S2 US | 26% | 13% | 8% | 1.7% | 5% |
| S3 US | 26% | 4% | 8% | 1.8% | 5% |

- ≥99.99% of true pairs share at least one token in name *or* address.
- India's high "no shared name token" rate is mostly **script** (§6). The US rate comes from renamed or domain names plus tokenisation of typos.

## 6. Scripts (train; first non-ASCII letter of each field)

- **India S2 names:** 72% ASCII, **13.3% Devanagari**, and about 11% other Indic scripts (Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Oriya, Gurmukhi). About 4% are Latin with accents.
- **India S3 names:** 82% ASCII, 7.5% Devanagari, about 6% other Indic scripts, 5% accented Latin.
- **India addresses (S2/S3):** about 22% contain an Indic script, usually the state name.
- **India S1 is ASCII** (0.06% non-ASCII).
- **US:** S1 is pure ASCII. About 7% of S2/S3 names are accented Latin, which is noise injection. Addresses are ASCII.
- The Indic scripts carry **phonetic transliterations of English words**: प्राइवेट लिमिटेड = "private limited", ईस्ट सॉल्यूशंस = "East Solutions". The training pairs (Latin S1 ↔ Indic S2/S3) give free parallel data for learning a transliteration.

## 7. France (test only, unlabeled)

- **Name pattern:** a head word, a type word and a legal form. The vocabulary is small and generic: `Club`, `Ecole`, `Maison`, `Transports`, `Petanque`, `Comite`, `Lycée`, `Clinique`, `Centre Médical`.
- **Legal forms:** SAS, SARL, SA, EURL, SCI, SASU, E.I., S.A.S., E.U.R.L., Cie, Ets / Établissements.
- **Name tokens are very weakly discriminative.** For example, 10+ different `Dreamy …`, `Calais …` and `Saad …` businesses share the head word.
- **Near-identical names at nearby addresses exist.** `Baroque Grandeur Centre SA, 9 Rue des Écoles` vs `Baroque Centre Grandeur SAS, 18 Rue des Écoles`. Whether that pair is a sibling or noise is unknown, which is a precision risk.
- **Region vs department:** S1 uses the region (`Nouvelle-Aquitaine`, `Hauts-de-France`, `Pays de la Loire`), while S2/S3 often use the department (`Gironde`, `Nord`, `Pas-de-Calais`, `Loire-Atlantique`). **No training pair teaches this mapping.**
- The same noise operators as train appear: accent injection and removal (`Pétanque`↔`Petanque`), typos (`Gadn`, `PSTES`, `MUNIN`/`MENIN`), and abbreviations (`R.`, `BD`, `Pl`, `AV.`, `ALL.`).
- House numbers appear as `N°`, `Nº`, `(12)`, `0070`, `bis`, `ter`, `B`, `C`. There are no postcodes. `(France)` gets inserted into names.
- About 39–42% of names and addresses contain accented Latin. That's natural French plus injected accents. **No non-Latin scripts.**

## 8. Ambiguity and hard negatives (train, *sample* of 5,570 S1)

- **S1 names are highly non-unique within a country.** 35.8% of US S1 entities and 44.4% of Indian S1 entities share their exact normalised name with another S1 entity. The most repeated: `primary care group` ×253, `ear nose throat group` ×251, `shree trading private limited` ×99.
- 38% of S1 entities have at least one **same-name non-match** in S2/S3.
- 4.4% (India) and 6.1% (US) of distractors carry a name identical to some S1 entity.
- **Names are built from common words.** Almost every S1 name's rarest token is also carried by more than 10 non-matching records.
- **The address separates same-name non-matches:**

| | shared number token | numbers disjoint | numbers missing | addr token Jaccard 0 | Jaccard <0.3 | Jaccard ≥0.3 |
|---|---:|---:|---:|---:|---:|---:|
| All true pairs | 82% | 5% | 13% | 4.6% | 8.6% | **87%** |
| True pairs with an identical name | 82% | 5% | 13% | 5.4% | 8.2% | 86% |
| **Same-name NON-matches** | 1.3% | **88%** | 11% | **86%** | 14% | **0.0%** |

  In train, a same-name negative is always another business at an unrelated address. Name plus any address agreement is strongly decisive. The **dangerous slice is the true pairs with an empty or non-overlapping address** (about 9%). They must be matched on the name alone, which is exactly where names are non-unique.
- **Same or similar address, different business, is common.** Among S2/S3 records whose address token Jaccard with an S1 record is ≥0.5 (sharing a ≥3-digit number), **6,622 are non-matches vs 8,311 matches** (*sample*). Co-location (same building, same street number) is frequent.

| addr Jaccard ≥0.5 and … | name Jaccard 0 | name Jaccard <0.5 | name Jaccard ≥0.5 |
|---|---:|---:|---:|
| non-match | 5,306 | 1,013 | **303** |
| match | 1,041 | 616 | 6,654 |

- Three kinds of hard negative follow:
  - **(A) Same name, far address.** Always separable by the address (table above).
  - **(B) Co-located, different name.** Separable by the name, *unless* the true match's name was replaced (`KELOONYX`) or transliterated. With the same address and zero name overlap, non-matches outnumber matches **5:1**. Address-only evidence is not enough.
  - **(C) Near-twins: similar name and nearby address with a slightly different number.** Examples: `First Apex Premium LLC, 204 1/2 Crawford St` vs `CORP. FIRST APEX PREMIUM, 204 1/9 CRAWFORD ST`; `NGW Better, 10084 Tr 112` vs `NGW Bétter Summit, 10105 TR 112`; `Star Products Pvt Ltd, 109 Angappa Naicken St` vs `Star Tech Pvt Ltd, 111 …`. These are about 4% of the "both similar" region (303 vs 6,654), and they're the precision killers. True matches also get perturbed numbers (`2620`→`262`), so **exact number agreement is informative but not decisive**. The model must learn the difference between number noise and a different business.

- **Who owns the hard negatives?** (*sample*, 5,570 S1)

| Class | matches | neg owned by another S1 | neg distractor |
|---|---:|---:|---:|
| same name, far address | 737 | 52,861 (92%) | 4,572 |
| co-located, name J = 0 | 1,041 (365 Indic, 676 Latin renamed) | 4,619 (87%) | 687 |
| near-twin (addr J ≥ .5, name J ≥ .5) | 7,693 | 2 | **303 (99%)** |

  Near-twins are **generated distractors**, owned by no S1, so only pair evidence can reject them. Co-location and same-name confusion is mostly **between real S1 entities**, so the one-to-one ownership rule resolves it.

## 9. Blocking evidence (train, *sample* of 5,570 S1 / 19,229 true pairs, within country)

The question: if a candidate must share a token whose document frequency (df, over S2∪S3 in that country) is ≤ T, what recall do we get, and at what cost?

| T | recall (name) | recall (address) | recall (name OR address) | postings per S1, mean | p95 |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.19 | 0.33 | 0.45 | 27 | 105 |
| 1,000 | 0.45 | 0.69 | 0.83 | 598 | 1,648 |
| 3,000 | 0.56 | 0.84 | 0.93 | 2,575 | 6,071 |
| 10,000 | 0.70 | 0.92 | 0.976 | 10,867 | 24,553 |
| ∞ | 0.87 | 0.95 | 1.000 | 2.1M | 5.6M |

- **Single-token blocks are too coarse.** Reaching 99% recall with boolean blocks costs tens of thousands of candidates per S1.
- Retrieval must be **ranked**: top-K by a combined, IDF-weighted score over several tokens or n-grams, on name and address.
- **No name token is shared by 13% of true pairs** (script changes and renames), so a name-only view can never exceed 87% recall. The address carries recall.

## 9b. Test-side structure (label-free proxy, identical code on train and test)

**The proxy:**
- For sampled S1 entities, count S2/S3 records in the same country that look like a **strong** match: name token Jaccard ≥0.5 **and** address token Jaccard ≥0.5.
- Candidates are found by exact name, or by (rare address word + house number).
- The proxy is calibrated on train, where labels tell us its precision and recall.

| | train India | test India | train US | test US | test France |
|---|---:|---:|---:|---:|---:|
| sampled S1 | 2,192 | 2,668 | 3,378 | 2,209 | 850 |
| strong candidates per S1 | 1.68 | **1.83** | 1.19 | **1.38** | 2.24 |
| S1 with 0 strong candidates | 27.7% | 25.9% | 36.5% | 30.5% | 17.5% |
| proxy precision (train labels) | 0.88 | — | 0.98 | — | — |
| proxy recall of true matches (train) | 0.43 | — | 0.34 | — | — |
| **estimated true matches per S1** | 3.45 (actual) | **≈3.77** | 3.45 (actual) | **≈4.0** | n/a |
| strong pairs with disjoint numbers | 0.6% | 0.7% | 1.7% | 1.5% | **11.4%** |
| S1 names repeated in S1 (chain) | 44.4% | 43.8% | 35.8% | 29.1% | 34.4% |

What this says:
1. **The +24% test density is both more matches and more distractors.**
   - Estimated matches per S1: India ≈3.77 (+9%) and US ≈4.0 (+16%). The US figure is biased upward, because test US S1 is half the size of train, which makes more address words look "rare" and inflates the proxy's recall.
   - That implies **distractors per S1 rise from ~1.2 to ~1.8–2.0**, so the distractor share goes from ~26% to ~31–35%.
   - **Decision thresholds tuned on train will face a higher false-positive pressure on test.** The density-stress validation world is justified.
2. **For India/US the generator is unchanged.** Script mix, token-length mix, style flags, and the number relation inside strong pairs all match train to within noise.
3. **⚠️ France: 11.4% of strong pairs have *disjoint* house numbers**, versus 0.6–1.7% for train countries. In train, "disjoint numbers" is where near-twin distractors live. Two readings, and labels can't separate them:
   - **(a)** France carries far more near-twin distractors (as in `Baroque Grandeur Centre, 9 Rue des Écoles` vs `18 Rue des Écoles`);
   - **(b)** French noise perturbs numbers more.

   Either way, **house-number handling decides France precision**. And France's short, generic addresses (street + city, 2–3 components) make address Jaccard ≥0.5 easy to reach by chance. That's also why France shows 2.5 "co-located, zero name overlap" records per S1, versus 0.4–1.9 elsewhere.
4. **Chain rates** in test are similar for India (43.8%). They're lower in US (29%), in line with its smaller S1. France is 34%. Name non-uniqueness is a test problem too.

## 9c. Assumption tests for v4 (train, *sample* of 5,570 S1 / 19,229 true pairs)

**Near-twins vs true copies** (the "strong" class: name J ≥0.5 **and** address J ≥0.5; 7,180 positives, 520 negatives):

| House-number relation (S1 vs record) | positives | negatives | reads as |
|---|---:|---:|---|
| identical number sets | 5,397 | 28 | match (99.5%) |
| one side has no number | 472 | 1 | match |
| one set ⊂ the other (number dropped or injected) | 954 | 49 | mostly match (95%) |
| truncation (`2620`↔`262`) | 141 | 25 | mostly match (85%) |
| **shares a number, another differs by ≤20** (`204 1/2`… `10084`↔`10105`) | 27 | **253** | **near-twin (90% negative)** |
| shares a number, another differs a lot | 34 | 48 | mostly negative |
| same digits, letter or fraction differs (`44`↔`44D`, `1/2`↔`1/9`) | 100 | 112 | 50/50: must split letter vs fraction |
| no shared number, small gap | 35 | 3 | positive (!) |

→ **Noise edits numbers as strings** (drop, inject, truncate). **Twins shift a number arithmetically.**

**Leftover name words** (after removing legal forms):
- Names are identical apart from legal-form words in 72% of positives vs 10.8% of near-twin negatives.
- In negatives the leftovers are *business words*: `group` 7.3%, `holdings` 6.7%, `industries` 6.5%, `overseas` 6.2%, `exports` 6.0%, `ventures` 5.8%, `public` 5.6%, `enterprises` 4.6%. Each of these is a leftover in ≤0.2% of positives.
- In positives the leftovers are *filler*: `center` 2.9%, `services` 2.2%, `service`, `partners`, `dba`, `smt`, `the`, `shri`, `sri`, `dr`, `mr`, `india`, `formerly`.

**Street+number key recall** (rare address word with S1-df ≤200 + a house number):
- The key alone finds **40% (US) and 58% (India)** of true pairs.
- Exact name or key together find 59% / 66%.

**Reachability of true pairs** (name J ≥0.5 = "name ok"; address J ≥0.5 = "addr ok"):

| | India (7,558) | US (11,671) |
|---|---:|---:|
| name ok + addr ok | 53.3% | 52.2% |
| name ok + addr weak | 13.6% | **28.8%** |
| name ok + addr empty | 3.5% | 4.6% |
| name weak + addr ok | 22.3% (898 of these are Indic names) | 9.2% |
| **name weak + addr weak/empty** | **7.3%** (356 Indic) | **5.3%** |

→ About 29% of US true pairs have a *rewritten* address (state spelled out, reordered, PO Box added), so address matching must be alias-aware. Transliteration moves about 4.7% of India pairs out of the "both weak" slice and about 12% out of "address only".

**Copy-to-S1 vs copy-to-copy similarity:** mean (name J + addr J)/2 = **0.62 to S1 vs 0.51 between copies**. Copies are closer to S1 than to each other.

**Top-1 behaviour** (cheap retrieval, s = name J + addr J of the best candidate):

| | no candidate | s <1.0 | 1.0 ≤ s <1.5 | s ≥1.5 |
|---|---:|---:|---:|---:|
| singleton (310) | 134 | 35 | **131** | 10 |
| has match, top-1 correct | — | 43 | 1,105 | 3,091 |
| has match, top-1 wrong | — | 17 | **498** | 37 |
| has match, nothing found | 469 | | | |

→ **45% of singletons have a plausible look-alike (s ≥1.0).** In the middle band, a third of top-1 picks are wrong.

**Do decoys come in clusters?**
- 18.5% of *owned* records have another record with the same name and a similar address (a sibling copy).
- Only **0.24% of distractors** do. **Real decoys are one-off records, not copy clusters.**

## 10. Open questions (not measured yet)

1. ~~The test shift~~: **answered by proxy (§9b).** It's both: about +9–16% matches and about +50–65% distractors per S1. Confirm with 1–2 leaderboard probes.
2. How often do "same address, different business" negatives occur? Running (§11).
3. Does test have near-sibling negatives (France: `Baroque Grandeur Centre` vs `Baroque Centre Grandeur`) that train lacks?
4. Is every S2/S3 record in an entity similar to S1, or are some similar only to another record of the same entity? That decides whether S2↔S3 transitivity or clustering adds recall.
5. How often is a name fully in Indic script *and* the address also sparse? That's the unreachable slice.

## 11. Format profile (all 7 files, full pass)

- **Case:**
  - S2 US addresses are 90% UPPERCASE and S2 India 24%. S3 addresses are never all-caps.
  - About 21% of S2 names are uppercase; 3% of S3 names.
- **State format flips by source:**
  - US: S1 ends with a 2-letter state in 86% of records and S2 in 83%, but **S3 spells the state out** (only 4% 2-letter).
  - India: S1 spells the state out, but **S3 uses 2-letter codes in 57% of records** (`UP`, `KL`, `MH`).
  - This mapping is learnable from train pairs for US/India. France's region↔department can't be learned from train.
- **Address component counts:**
  - US is almost always 3–4 components; France 2–3 (S1 is always 3: street, city, region).
  - India ranges from 3 to 12, peaking at 5–6.
  - About 2.9–3.7% of S2/S3 addresses are empty in train (**0 components**); in test it's about 2.3–3.1%.
- **Name markers in S2/S3** (per record):

| Marker | S2/S3 rate |
|---|---|
| domain (`.com`, `www.`) | 3–4% |
| junk prefix | 1–2.5% |
| honorific (`Shri`/`Sri`/`Smt`) | India 5–6% |
| country tag (`(India)`, `(France)`) | 4–8%; it also appears in S1 India (5%) and S1 France (8%), so it's not only noise |
| `dba`/`formerly` | S3 only, 1.3–2.5% |

- **Address markers:**

| Marker | Rate |
|---|---|
| house-number markers (`#`, `No`, `H.No`, `Plot`) | India 44–57%; US S2 5% |
| PO Box / Unit / Apt | US S1 14%, S3 10% |
| zero-padded numbers | S2/S3 5–6% |
| `bis`/`ter` | France 4–5% |
| city suffixes (`CDP`, `City`, `Township`) | 8–11% |

- **Exact duplicate (name, address) records within a source** are rare but present: about 0.3–0.5% of S2/S3 (10k–24k per file).
- **Train/test entity overlap:** 44% of India and 36% of US test S1 normalised names also occur among train S1 names. That's the shared name vocabulary (`shree trading private limited`), not the same entities. France overlaps 0.01%.
- **The legal-form vocabulary per country** (last name token):
  - India: `limited`, `ltd`, `llp`, `co`, `private`, `pvt`.
  - US: `llc`, `inc`, `corp`, `pc`, `pllc`, `lp`, `co`.
  - France: `sarl`, `sas`, `eurl`, `sa`, `sasu`, `sci`, `ei`, `cie`.
  - Filler words also sit in the last position (`center`, `services`, `group`, `holdings`, `partners`).
- **Tokenisation pitfall:** Python's `\w+` splits Indic words at vowel signs and viramas (e.g. `ड` shows up as a "token"). The pipeline tokenizer must keep Unicode combining marks (categories Mn/Mc) inside words.
- **Test distractor-rate proxy: inconclusive.** A "rare S1-name-token anchor" proxy doesn't separate matched from distractor records in India (5.4% vs 5.6%). In US it gives an implausible estimate, because the test S1 is half the size of train S1 and that changes the token df. **The +24% test density question stays open.** The cleanest probe is one or two leaderboard submissions at different thresholds.
