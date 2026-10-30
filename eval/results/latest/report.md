# Research QA evaluation

Dataset: Apple FY2025 annual-report QA / curated 24-question development benchmark
Mode: live; run status: complete
Chat: gpt-4o-mini; embeddings: text-embedding-3-small
Started: 2026-10-30T03:36:38.923432+08:00

| Metric | Value |
| --- | ---: |
| recall_at_3 | 0.7222 |
| page_hit_at_3 | 0.7222 |
| answer_accuracy | 0.7083 |
| answer_accuracy_method | agent_adjudicated |
| automatic_answer_accuracy | 0.7083 |
| automatic_answerable_accuracy | 0.6111 |
| reviewed_accuracy | 0.7083 |
| reviewed_answerable_accuracy | 0.6111 |
| correct_refusal_rate | 1.0000 |
| false_refusal_rate | 0.3333 |
| abstention_precision | 0.5000 |
| error_rate | 0.0000 |
| ask latency count (seconds except count) | 24 |
| ask latency mean (seconds except count) | 2.0622 |
| ask latency p50 (seconds except count) | 2.0454 |
| ask latency p95 (seconds except count) | 2.7417 |
| retrieval latency count (seconds except count) | 24 |
| retrieval latency mean (seconds except count) | 0.5044 |
| retrieval latency p50 (seconds except count) | 0.4189 |
| retrieval latency p95 (seconds except count) | 0.9416 |
| failed_ask latency count (seconds except count) | 0 |
| failed_ask latency mean (seconds except count) | N/A |
| failed_ask latency p50 (seconds except count) | N/A |
| failed_ask latency p95 (seconds except count) | N/A |

## Counts

{"questions": 24, "answerable": 18, "unanswerable": 6, "errors": 0, "abstained": 12, "false_refusals": 6, "correct_refusals": 6, "abstention_reasons": {"model_insufficient_evidence": 10, "unverifiable_citation": 2}, "reviewed": 24, "review_complete": true}

## Methodology

**recall_at_3**: Macro-average page recall over answerable questions: unique annotated (document_name, physical 1-based page) pairs found in raw top-3 chunks / all annotated page pairs. It does not verify exact supporting passage. Search failures score zero; generation failures retain measured retrieval.

**automatic_answer_accuracy**: Preliminary regex check of final answer only, or explicit abstention for unanswerable questions. It does not establish semantic correctness.

**reviewed_accuracy**: Correct agent-adjudicated answers / all questions, including errors. Published only when every response has an explicit review note. Reviews are agent-assisted checks against gold answers, not independent human labels, and use no additional judge-model API calls.

**refusals**: Correct refusal rate = unanswerable explicit abstentions / all unanswerable; false refusal rate = answerable explicit abstentions / all answerable; abstention precision = correct explicit refusals / all explicit abstentions. Request errors never count as refusals and remain in rate denominators.

**latency**: Sequential single-run wall-clock seconds, no warmup excluded. Ask includes one query embedding/retrieval, generation when invoked, retries and final validation; excludes PDF indexing, HTTP and UI. Retrieval measures the same search inside ask, not a second search. p50/p95 use linear interpolation. Successful and failed ask latencies are reported separately.

**scope**: Small curated development benchmark, not an independent held-out estimate.

## Per-question audit

See `responses.jsonl` for raw retrieved chunks, final responses and timings; `results.csv` for a compact worksheet and `reviews.json` for adjudications.
