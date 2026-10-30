# Phase 3 paired development evaluation

Status: agent-adjudicated complete.
3 paired repetitions of the same 24 frozen questions.
Models: gpt-4o-mini / text-embedding-3-small.

## Recorded intervention

Baseline: dense Top-3; optimized: dense Top-5.
Context character ceiling: 16000 for both. Cosine threshold: 0.35 for both.
The selected variant bundles a larger Top-k evidence allowance and the recorded prompt change. It is not a controlled single-variable retrieval or prompt experiment. Top-5 may supply up to five chunks where Top-3 supplies three.

| Metric | Fresh baseline | Optimized | Difference (optimized − baseline) |
| --- | ---: | ---: | ---: |
| Page Recall@3 | 72.22% | 72.22% | +0.00 pp |
| Page Recall@selected k | 72.22% | 72.22% | +0.00 pp |
| Evidence Recall@3 | 66.67% | 66.67% | +0.00 pp |
| Evidence Recall@selected k | 66.67% | 72.22% | +5.56 pp |
| Evidence Recall after filtering @selected k | 66.67% | 72.22% | +5.56 pp |
| Answerable accuracy | 61.11% | 77.78% | +16.67 pp |
| Overall accuracy | 70.83% | 83.33% | +12.50 pp |
| False refusal rate | 33.33% | 22.22% | -11.11 pp |
| Correct refusal rate | 100.00% | 100.00% | +0.00 pp |
| Returned citation validity (provenance) | 100.00% | 100.00% | +0.00 pp |
| Raw model quote validity | 86.05% | 100.00% | +13.95 pp |
| P50 ask latency (s) | 1.783 | 1.893 | 0.110 |
| P95 ask latency (s) | 2.520 | 2.905 | 0.385 |
| Actual context characters, mean | 2314.667 | 3710.875 | 1396.208 |
| Actual context characters, maximum | 2911.000 | 4568.000 | 1657.000 |

Evidence-label coverage (verified / answerable question-trials):
- baseline: at_3: 54/54; at_k: 54/54; after_filter_at_k: 54/54
- optimized: at_3: 54/54; at_k: 54/54; after_filter_at_k: 54/54

## Per-run variability

| Repeat | Variant | Reviewed accuracy | Answerable accuracy | False refusal | P95 (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | baseline | 70.83% | 61.11% | 33.33% | 2.906 |
| 1 | optimized | 83.33% | 77.78% | 22.22% | 3.259 |
| 2 | baseline | 70.83% | 61.11% | 33.33% | 2.450 |
| 2 | optimized | 83.33% | 77.78% | 22.22% | 2.216 |
| 3 | baseline | 70.83% | 61.11% | 33.33% | 2.145 |
| 3 | optimized | 83.33% | 77.78% | 22.22% | 2.354 |

## Paired per-question changes

Every question is compared, including previously correct cases. Counts below refer to repeated trials, not additional unique questions.

| Question | Baseline correct | Optimized correct | Fixed trials | Regressed trials |
| --- | ---: | ---: | ---: | ---: |
| a01 | 3/3 | 3/3 | 0 | 0 |
| a02 | 3/3 | 3/3 | 0 | 0 |
| a03 | 3/3 | 3/3 | 0 | 0 |
| a04 | 0/3 | 0/3 | 0 | 0 |
| a05 | 0/3 | 0/3 | 0 | 0 |
| a06 | 0/3 | 3/3 | 3 | 0 |
| a07 | 3/3 | 3/3 | 0 | 0 |
| a08 | 3/3 | 3/3 | 0 | 0 |
| a09 | 3/3 | 3/3 | 0 | 0 |
| a10 | 3/3 | 3/3 | 0 | 0 |
| a11 | 0/3 | 3/3 | 3 | 0 |
| a12 | 0/3 | 0/3 | 0 | 0 |
| a13 | 0/3 | 0/3 | 0 | 0 |
| a14 | 3/3 | 3/3 | 0 | 0 |
| a15 | 3/3 | 3/3 | 0 | 0 |
| a16 | 0/3 | 3/3 | 3 | 0 |
| a17 | 3/3 | 3/3 | 0 | 0 |
| a18 | 3/3 | 3/3 | 0 | 0 |
| u01 | 3/3 | 3/3 | 0 | 0 |
| u02 | 3/3 | 3/3 | 0 | 0 |
| u03 | 3/3 | 3/3 | 0 | 0 |
| u04 | 3/3 | 3/3 | 0 | 0 |
| u05 | 3/3 | 3/3 | 0 | 0 |
| u06 | 3/3 | 3/3 | 0 | 0 |

At least one paired fixed trial: a06, a11, a16.
At least one paired regressed trial: none.
Detailed responses, refusals, errors and adjudication notes are in [paired_comparison.csv](paired_comparison.csv).

## Preserved Phase 2 reference

Phase 2 latest is an earlier, noncontemporaneous run and was not changed.
Overall accuracy: 70.83%; Page Recall@3: 72.22%.
Historical raw model quote validity: **unavailable**; rejected raw generations were not captured. Fresh Phase 3 baseline quote metrics above come from its new captured calls, not invented historical values.

Historically correct Phase 2 questions with at least one optimized incorrect trial: none.

## Interpretation and limits

Page Recall concerns annotated pages; Evidence Recall concerns explicitly reviewed supporting bundles and may require row+period/unit context. Answer accuracy grades the final claim text against the same frozen rubric. These are different outcomes. Unknown evidence labels are N/A, not negatives.
Returned citation validity checks original source identity and contiguous text only, conditional on citations that were returned; it cannot establish semantic entailment and a system that refuses most questions may still score high. Raw quote validity uses parseable captured model quotes; malformed generations and its denominator are separately recorded in summary.json.
Latency includes live query embedding, retrieval, generation and validation, excluding HTTP/UI/indexing; startup costs are recorded per run. Repeated observations on this small development set are not independent new questions and do not establish statistical significance or generalization. Holdout performance must be reported separately.

