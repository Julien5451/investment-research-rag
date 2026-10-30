# Phase A: evidence-level retrieval diagnostics

This is an offline audit of the saved Phase 2 `latest` traces. It makes no chat or
embedding API calls and does not rewrite `eval/dataset.json` or any Phase 2 result.
The sidecar `evidence_labels.json` is a new Phase 3 annotation layer. It records the
frozen dataset SHA-256 and the original identity, page and exact-text SHA-256 of
all 467 chunks. All chunk text was checked as a continuous span in the original
PDF's sorted PyMuPDF extraction, including original financial table layout.

The labels are **agent-assisted source annotations**, not independent human
labels. The fixed development questions and observed traces were already known.
They are suitable for controlled engineering diagnostics, not a held-out estimate.

## What the three metrics mean

- **Page Recall@k:** fraction of originally annotated document/page pairs found in
  raw top-k, averaged over answerable questions. The Phase 2 labels stay frozen.
- **Evidence Recall@k:** fraction of required evidence groups supported by the
  selected chunks, averaged over questions with verified labels. A group has one
  or more reviewed alternative bundles; every piece of a chosen bundle must be
  present. A table value must include its category, financial period and units.
  The same fact on another verified page is an alternative, not a miss. Unknown
  questions, changed source text/identity, and unreviewed groups require review
  and do not silently contribute a positive label.
- **Answer accuracy:** correctness and completeness of the final answer against
  the frozen rubric. Retrieved support can be present while the model refuses or
  omits part of the question. A correct answer can also be produced from context
  that does not fully establish its period. This metric is separate from recall.

The helper also reports full-question support, relevant chunk identities, minimum
number of source chunks, and whether the *selected* support requires multiple
chunks. Numeric coincidence or matching the right page is never sufficient.

## Reconstructed Phase 2 results

| Measurement | Result |
| --- | ---: |
| Answerable questions with verified evidence labels | 18 / 18 |
| Questions requiring label review on this unchanged corpus | 0 |
| Page Recall@3, original frozen page labels | 13 / 18 = 72.22% |
| Evidence Recall@3, reviewed contextual bundles | 12 / 18 = 66.67% |
| Full-question support in saved top-3 | 12 / 18 |
| Retrieved chunks removed by the 0.35 relevance threshold | 0 |
| Retrieved chunks removed by injection patterns | 0 |

The CSV contains all 72 retrieved chunk rows, including question, rank, exact
chunk ID, document, physical page, cosine score, support state, threshold/injection
filter flags and question-level support before/after filtering. Its companion JSON
contains per-question details and input-file hashes. The unanswerable questions
have no positive-recall denominator.

## Verification of Phase 2 failure hypotheses

| Question | Exact trace finding |
| --- | --- |
| `a04` | The channel-share clause is in `5:3`; no saved top-3 chunk contains it. Retrieval miss confirmed. |
| `a05` | The employee count and as-of date are in `7:2`; no saved top-3 chunk contains them. Retrieval miss confirmed. |
| `a12` | The total margin row with its ordered FY2025/FY2024/FY2023 header is in `27:0`; raw top-3 misses it. |
| `a13` | `27:3` discusses drivers of R&D growth but does not contain the expense amount. Full support exists in `27:2`, or in `32:0`+`32:1`; the correct-page hit is not an evidence hit. |
| `a11` | `27:0` contains the full Services percentage row and FY2025 header. Verified evidence is present. The terminal response reports `unverifiable_citation`, so retrieval is not the immediate missing-evidence problem here. |
| `a16` | Top-3 includes `40:0` (investment-table header and partial categories), `40:6` (FY2024 restricted-cash notes) and `42:0` (derivatives). It lacks the combined-balance narrative `28:1` and lacks the required total row `40:2`/`40:3`. Thus supporting evidence is absent, despite the terminal citation-failure reason. Calling this solely a formatting failure would overstate what was recorded. |
| `a06` | `7:1` contains both first-quarter seasonality and the holiday-demand reason. Both required evidence groups are present. The final answer omits the reason, confirming an answer-completeness failure. |

Chunk suffixes above omit the unchanged document SHA-256 prefix for readability.
The failed **raw model-generated quotes** for historical `a11` and `a16` were not
saved in Phase 2. Their precise mismatch, claimed wording and failing source ID
cannot be recovered from the final abstention response. New diagnostic requests
must be labeled as new observations rather than historical reconstructions.

## Additional findings across all questions

- `a01` has a valid alternative fiscal-year-end statement on `80:1`, so it is an
  evidence hit despite missing original gold page 1. Cover, contents,
  certifications and complete annual-statement date headers were reviewed as
  alternatives.
- `a18` retrieves the diluted-EPS row in `39:3` but not that table's FY2025 header
  in `39:2`. It also retrieves `32:3`, which is a diluted **share-count** row, not
  the EPS row. Under strict contextual labels, full evidence requires
  `39:2`+`39:3`, or `32:0`+`32:2`. The actual answer happens to be correct, but the
  selected chunks do not fully establish the requested financial period.
- Alternative complete tables and statement locations are included for `a09`,
  `a10`, `a13`, `a15`, `a16`, `a17` and `a18`. For example, `a16` can use the
  investment-table total of USD132,420 million with the FY2025 header, or the
  balance-sheet components with their date/unit header. Those alternatives need
  multiple chunks and an explicit unit conversion/rounding step; a header alone
  is never accepted.
- Citation provenance validity is narrower than semantic evidence support. A
  genuine quote can still omit a necessary year/unit context, and a final answer
  can omit a fact that is visible only inside its citation.

## Reproduction and limits

```bash
uv run python scripts/retrieval_diagnostics.py
uv run pytest -q tests/test_evidence.py
```

The evidence matcher checks only reviewed source bundles and exact corpus hashes.
It does not attempt automatic semantic relevance classification or LLM judging.
Labels are conservative and do not exhaust every possible paraphrase, reasoning
path or alternative arithmetic derivation. Changed extraction/chunking or another
company corpus requires new verified labels. A row/header bundle establishes
source availability; it does not guarantee the model will correctly read the
table. The Phase 2 dataset and metrics remain intact for honest comparison.
