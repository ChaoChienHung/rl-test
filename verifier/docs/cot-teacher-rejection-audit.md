# CoT teacher rejection audit (2026-10-15)

## Scope and method

The `qwen3.8:27b` SoCLaaS teacher returned all 5,895 FEVER train prompts. The corrected `full1/validation-v2` collection accepts 5,177 and rejects 718. This audit samples **10 from each of three rejection reasons**, with a fixed random seed of `20261015` and 3 SUPPORTED, 3 REFUTED, and 4 NOT_ENOUGH_INFO claims per reason. IDs and labels were checked against the [official FEVER train annotations](https://fever.ai/download/fever/train.jsonl): all 30 claim texts and labels match. The judgments below concern whether the **evidence shown in our prompt** justifies that label; they are not a reannotation of the entire FEVER dataset.

The original teacher responses are kept in `full1/responses.jsonl`, and the rejected subset is kept locally in `full1/validation-v2/rejected.jsonl`. Neither is committed. The tables give a reproducible ID list for targeted inspection.

## Citation mismatch: 10 sampled

| Claim ID | Gold | Observation |
|---|---|---|
| 160615 | SUPPORTED | Teacher cites E21 in the list but omits the ID in prose; E21 directly says the film was released in 3D. |
| 149565 | SUPPORTED | Cites E31 only in the list; E31 states the show was renewed. |
| 224094 | SUPPORTED | Lists E43/E44/E63 without naming IDs in prose; E63 directly describes the American singer's tour. The gold evidence bundle also includes E23. |
| 217246 | REFUTED | Cites E14 only in the list; its February start contradicts May. |
| 103174 | REFUTED | Cites E27 only in the list; the sentence places the premiere in New York City. |
| 106736 | REFUTED | Cites E18 only in the list. Cancellation in May does not necessarily establish the show's last airing date, so the gold evidence merits review too. |
| 103781 | NEI | Lists three E IDs without naming any in prose; the explanation says the visible biographies do not establish religion. |
| 52073 | NEI | Lists E12 and E21 but mentions only E12 in prose; the explanation finds no war connecting the two people. |
| 147987 | NEI | Lists E1/E2/E3 but mentions only E1 in prose; the generic name is ambiguous. |
| 228395 | NEI | Lists E2/E6/E7 without naming IDs in prose; the evidence concerns acting and awards, not writing. |

**Reading:** The sampled failures are mainly teacher citation-format failures. Do not infer a wrong FEVER label from this rejection code. Revalidate a repaired answer against all rules before admitting it: one example has a separate evidence concern.

## Teacher says gold is not justified: 10 sampled

| Claim ID | Gold | Observation |
|---|---|---|
| 209568 | SUPPORTED | Shown E3 calls *X-Men* an American film; that does not establish that filming occurred in America. Strong prompt/gold evidence concern. |
| 199096 | SUPPORTED | E7 mentions a variety of social-science and humanities courses; whether that entails “many subject matters” is ambiguous. |
| 34342 | SUPPORTED | Evidence mentions a 2003 film and a 2004 US airing; both are early 2000s. Teacher's `gold_justified=false` appears overly strict. |
| 22287 | REFUTED | E20 identifies *Dark Places* as a film but gives no firing event. A definite refutation is not established by that sentence alone. |
| 151085 | REFUTED | Teacher's own explanation distinguishes Ween's album from Nick Jonas and describes another Jonas album. Its `gold_justified=false` conflicts with its reasoning. |
| 4548 | REFUTED | Prompt E10 calls Amaravati the de facto capital, while gold cites historical Hyderabad transfer (E21). The shown context conflicts with the gold refutation; this also depends on time and wording. |
| 75374 | NEI | Teacher says the shown context cannot determine whether Elizabeth Taylor was a stage actress in 1978. That supports NEI; `gold_justified=false` is inconsistent. |
| 117676 | NEI | Teacher says Emmy evidence does not establish an Academy Award. That supports NEI; false flag is inconsistent. |
| 167767 | NEI | Teacher says the shown occupations do not establish that Jamie Foxx is a dancer. That supports NEI; false flag is inconsistent. |
| 33831 | NEI | Teacher says the text does not link John Hartford to *Summertime*. That supports NEI; false flag is inconsistent. |

**Reading:** `gold_justified=false` is not an independent dataset-quality verdict. In four sampled NEI cases the teacher explicitly reasoned toward NEI but set the Boolean to false. Review non-NEI cases separately, especially where the displayed gold sentence only weakly supports the label.

## Wrong relation: 10 sampled

| Claim ID | Gold → teacher | Observation |
|---|---|---|
| 170342 | SUPPORTED → refutes | Claim says business school; E18 says law school. Clear contradiction between shown evidence and gold. |
| 128386 | SUPPORTED → insufficient | Claim says Jack Black reviewed *High Fidelity*; E3 describes his acting role. Review is not established. |
| 213219 | SUPPORTED → insufficient | Nile drainage-basin countries plus forested hills in Kenya do not explicitly place the river or basin through a forest. Evidence chain is weak. |
| 120636 | REFUTED → insufficient | FX is identified as a cable network but the shown sentence does not say whether it is premium. Refutation is weak from this context alone. |
| 23881 | REFUTED → insufficient | Claim substitutes “pigs” for awards; E7 lists awards. Teacher may be too cautious about the obvious category contradiction. |
| 180114 | REFUTED → insufficient | E24 lists the drummer's bands, but does not explicitly establish that he never worked with a Russian band. Refutation is weak from this context alone. |
| 152612 | NEI → supports | E18/E19 explicitly state the fourth Justin Bieber album spawned three number-one singles. Shown context supports the claim. |
| 35632 | NEI → supports | E2/E5 explicitly identify Tony Blair as Labour Party leader. Shown context supports the claim. |
| 186692 | NEI → supports | E21 explicitly identifies Adam Richard Wiles as professionally known as Calvin Harris. Shown context supports the claim. |
| 133015 | NEI → supports | E3 explicitly lists the named Vikrant Massey TV series. Shown context supports the claim. |

**Reading:** Several disagreements are genuine conflicts between the shown context and original FEVER label. All 30 IDs match the original FEVER train file, so these are not claim/label joins corrupted during local conversion. The NEI conflicts may arise because the constructed prompt retrieves information beyond FEVER's empty NEI evidence annotation. This sample does not estimate the prevalence among all 718 rejected cases.

## Decisions

1. Keep the 718 rejected examples out of CoT SFT. Do not automatically flip original labels or merge teacher disagreements into gold.
2. The 5,177-row `validation-v2` file is a **candidate**, not a semantic-faithfulness certificate. A sample of accepted examples is still needed before full-data training.
3. Prioritize a prompt-versus-label conflict audit for `wrong_relation` and non-NEI `gold_not_justified_by_context`. Separate a teacher error from a valid original label with missing or contradictory displayed evidence.
4. For citation-format failures, repair or regenerate only after checking all other constraints, including whether non-NEI gold evidence IDs are explained. Keep the original response and any revised response as separate records.

## Follow-up: accepted-set audit and recovery ceiling

An independent stratified sample of **30 accepted rows** (10 per gold class, seed `20261016`) shows why passing the structural validator is insufficient:

| Claim ID | Gold | Concern |
|---|---|---|
| 213283 | SUPPORTED | The sentence says the Nile's **drainage basin** covers Eritrea; the claim says the river **runs through** Eritrea. The teacher treats these as equivalent. |
| 89620 | NEI | Shown E8 calls *The Sinatra Family Wish You a Merry Christmas* a **1968 Christmas album by Frank Sinatra**. The teacher nevertheless explains NEI for “In 1968, Frank Sinatra released an album.” |
| 121171 | REFUTED | Award nominations are used to refute the subjective claim that Farrah Fawcett is “untalented”; this inference is debatable. |
| 221583 | REFUTED | E13 names BoA as a judge, but the teacher relies on her being a singer without citing a shown sentence for that fact. |
| 35035 | REFUTED | Being called a television producer is used to infer that a particular visual credit exists; the displayed sentence does not identify one. |

These are **2 clear prompt/label-or-explanation conflicts and 3 judgment calls in 30 inspected accepted rows**, not a population error-rate estimate. The sampled IDs are: SUPPORTED `138276, 66854, 30797, 228304, 51459, 56157, 20503, 27653, 177795, 213283`; REFUTED `137500, 121171, 65774, 35035, 97552, 221583, 149832, 22278, 158670, 111088`; NEI `136618, 85482, 93462, 58186, 89620, 94158, 60715, 64367, 67029, 112778`.

Simulating citation-ID repair on the **259 remaining `citation_mismatch`** rows makes 258 pass the other deterministic rules; one still fails to explain all gold evidence. Changing only the **196 NEI `gold_not_justified_by_context`** flags makes 168 pass the other rules, 27 fail citation checks, and one fail the duplicate/too-many-citations check. Thus `5,177 + 258 + 168 = 5,603` is a **structural upper bound**, not an approved training-set size.

## 5,500-sample repaired dataset build (2026-10-15)

To prepare a balanced 5,500-sample training set without introducing arbitrary unverified samples, `utils/build_repaired_cot_sft_5500.py` applies deterministic structural repairs:
1. **Citation repair (258 rows)**: For teacher answers where `cited_evidence` contained valid shown IDs but the prose omitted literal `E...` tokens, missing IDs were appended as `(E...)`. All 258 pass full validation (50 SUPPORTED, 100 REFUTED, 108 NEI).
2. **NEI justification repair (65 rows of 168 available)**: For NEI claims where the teacher correctly reasoned that the context was insufficient but set `gold_justified=false` due to literal prompt reading, `gold_justified` was set to `true`. 65 rows were selected with seed 0 to reach exactly 5,500.
3. **Validation & dev isolation**:
   - Total rows: **5,500** (`5,177` original accepted + `258` citation repaired + `65` NEI repaired).
   - Label breakdown: SUPPORTED: 1,840 (33.5%), REFUTED: 1,865 (33.9%), NOT_ENOUGH_INFO: 1,795 (32.6%).
   - Overlap with frozen FEVER dev (900 claims): **0**.
   - Output paths:
     - CoT SFT: `verifier/data/train/sft/fever-oracle-cot-soclaas/cot_sft_labeled_prompts.jsonl`
     - Remaining rejected (395 rows): `verifier/data/train/sft/fever-oracle-cot-soclaas/rejected.jsonl`
     - Direct SFT (matched IDs): `verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl`

