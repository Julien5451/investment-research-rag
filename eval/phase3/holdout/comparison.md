# Phase 3 independent-company holdout evaluation

Status: agent-adjudicated complete.
1 paired repetitions of the same 12 frozen questions.
Models: gpt-4o-mini / text-embedding-3-small.

## Recorded intervention

Baseline: dense Top-3; optimized: dense Top-5.
Context character ceiling: 16000 for both. Cosine threshold: 0.35 for both.
The selected variant bundles a larger Top-k evidence allowance and the recorded prompt change. It is not a controlled single-variable retrieval or prompt experiment. Top-5 may supply up to five chunks where Top-3 supplies three.

| Metric | Fresh baseline | Optimized | Difference (optimized − baseline) |
| --- | ---: | ---: | ---: |
| Page Recall@3 | 77.78% | 77.78% | +0.00 pp |
| Page Recall@selected k | 77.78% | 88.89% | +11.11 pp |
| Evidence Recall@3 | N/A | N/A | N/A |
| Evidence Recall@selected k | N/A | N/A | N/A |
| Evidence Recall after filtering @selected k | N/A | N/A | N/A |
| Answerable accuracy | 55.56% | 77.78% | +22.22 pp |
| Overall accuracy | 66.67% | 83.33% | +16.67 pp |
| False refusal rate | 44.44% | 22.22% | -22.22 pp |
| Correct refusal rate | 100.00% | 100.00% | +0.00 pp |
| Returned citation validity (provenance) | 100.00% | 100.00% | +0.00 pp |
| Raw model quote validity | 77.78% | 100.00% | +22.22 pp |
| P50 ask latency (s) | 2.235 | 1.973 | -0.262 |
| P95 ask latency (s) | 3.066 | 4.251 | 1.185 |
| Actual context characters, mean | 2226.167 | 3675.083 | 1448.917 |
| Actual context characters, maximum | 2938.000 | 4840.000 | 1902.000 |

Evidence-label coverage (verified / answerable question-trials):
- baseline: at_3: 0/9; at_k: 0/9; after_filter_at_k: 0/9
- optimized: at_3: 0/9; at_k: 0/9; after_filter_at_k: 0/9

## Per-run variability

| Repeat | Variant | Reviewed accuracy | Answerable accuracy | False refusal | P95 (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | baseline | 66.67% | 55.56% | 44.44% | 3.066 |
| 1 | optimized | 83.33% | 77.78% | 22.22% | 4.251 |

## Paired per-question changes

Every question is compared, including previously correct cases. Counts below refer to repeated trials, not additional unique questions.

| Question | Baseline correct | Optimized correct | Fixed trials | Regressed trials |
| --- | ---: | ---: | ---: | ---: |
| h01 | 0/1 | 0/1 | 0 | 0 |
| h02 | 1/1 | 1/1 | 0 | 0 |
| h03 | 1/1 | 1/1 | 0 | 0 |
| h04 | 0/1 | 1/1 | 1 | 0 |
| h05 | 1/1 | 1/1 | 0 | 0 |
| h06 | 1/1 | 1/1 | 0 | 0 |
| h07 | 0/1 | 1/1 | 1 | 0 |
| h08 | 1/1 | 1/1 | 0 | 0 |
| h09 | 0/1 | 0/1 | 0 | 0 |
| h10 | 1/1 | 1/1 | 0 | 0 |
| h11 | 1/1 | 1/1 | 0 | 0 |
| h12 | 1/1 | 1/1 | 0 | 0 |

At least one paired fixed trial: h04, h07.
At least one paired regressed trial: none.
Detailed responses, refusals, errors and adjudication notes are in [paired_comparison.csv](paired_comparison.csv).


## Interpretation and limits

Page Recall concerns annotated pages; Evidence Recall concerns explicitly reviewed supporting bundles and may require row+period/unit context. Answer accuracy grades the final claim text against the same frozen rubric. These are different outcomes. Unknown evidence labels are N/A, not negatives.
Returned citation validity checks original source identity and contiguous text only, conditional on citations that were returned; it cannot establish semantic entailment and a system that refuses most questions may still score high. Raw quote validity uses parseable captured model quotes; malformed generations and its denominator are separately recorded in summary.json.
Latency includes live query embedding, retrieval, generation and validation, excluding HTTP/UI/indexing; startup costs are recorded per run. Repeated observations on this small independent-company holdout do not establish statistical significance or generalizable gains. These results are separate from the Apple development benchmark; no holdout-driven tuning is performed.

