# Research QA evaluation

Dataset: Costco FY2025 independent-company holdout / 12 questions
Mode: live; run status: complete
Chat: gpt-4o-mini; embeddings: text-embedding-3-small
Started: 2026-10-30T04:52:49.215659+08:00

| Metric | Value |
| --- | ---: |
| recall_at_3 | 0.7778 |
| page_hit_at_3 | 0.7778 |
| answer_accuracy | 0.6667 |
| answer_accuracy_method | agent_adjudicated |
| automatic_answer_accuracy | 0.6667 |
| automatic_answerable_accuracy | 0.5556 |
| reviewed_accuracy | 0.6667 |
| reviewed_answerable_accuracy | 0.5556 |
| correct_refusal_rate | 1.0000 |
| false_refusal_rate | 0.4444 |
| abstention_precision | 0.4286 |
| error_rate | 0.0000 |
| ask latency count (seconds except count) | 12 |
| ask latency mean (seconds except count) | 2.1869 |
| ask latency p50 (seconds except count) | 2.2346 |
| ask latency p95 (seconds except count) | 3.0660 |
| retrieval latency count (seconds except count) | 12 |
| retrieval latency mean (seconds except count) | 0.4346 |
| retrieval latency p50 (seconds except count) | 0.4247 |
| retrieval latency p95 (seconds except count) | 0.4687 |
| failed_ask latency count (seconds except count) | 0 |
| failed_ask latency mean (seconds except count) | N/A |
| failed_ask latency p50 (seconds except count) | N/A |
| failed_ask latency p95 (seconds except count) | N/A |

## Counts

{"questions": 12, "answerable": 9, "unanswerable": 3, "errors": 0, "abstained": 7, "false_refusals": 4, "correct_refusals": 3, "abstention_reasons": {"model_insufficient_evidence": 5, "unverifiable_citation": 2}, "reviewed": 12, "review_complete": true}

## Methodology

**recall_at_3**: Macro-average page recall over answerable questions: unique annotated (document_name, physical 1-based page) pairs found in raw top-3 chunks / all annotated page pairs. It does not verify exact supporting passage. Search failures score zero; generation failures retain measured retrieval.

**automatic_answer_accuracy**: Preliminary regex check of final answer only, or explicit abstention for unanswerable questions. It does not establish semantic correctness.

**reviewed_accuracy**: Correct agent-adjudicated answers / all questions, including errors. Published only when every response has an explicit review note. Reviews are agent-assisted checks against gold answers, not independent human labels, and use no additional judge-model API calls.

**refusals**: Correct refusal rate = unanswerable explicit abstentions / all unanswerable; false refusal rate = answerable explicit abstentions / all answerable; abstention precision = correct explicit refusals / all explicit abstentions. Request errors never count as refusals and remain in rate denominators.

**latency**: Sequential single-run wall-clock seconds, no warmup excluded. Ask includes one query embedding/retrieval, generation when invoked, retries and final validation; excludes PDF indexing, HTTP and UI. Retrieval measures the same search inside ask, not a second search. p50/p95 use linear interpolation. Successful and failed ask latencies are reported separately.

**scope**: Independent-company holdout, reported separately from development results; small sample and single paired run limit generalization.

## Per-question audit

See `run*/responses.jsonl` for raw retrieved chunks, final responses and timings; `results.csv` for a compact worksheet and `run*/reviews.json` for adjudications.

## Phase 3 extended audit

Evidence metrics use verified labels only; see summary.json for label coverage.
Citation validity is provenance, not semantic correctness.

| Metric | Value |
| --- | ---: |
| page_recall_at_k | 0.7777777777777778 |
| evidence_recall_at_3 | N/A |
| evidence_recall_at_k | N/A |
| evidence_recall_after_filter_at_k | N/A |
| returned_citation_validity | 1.0 |
| raw_model_quote_validity | 0.7777777777777778 |
| citation_validation_refusals | 2 |
