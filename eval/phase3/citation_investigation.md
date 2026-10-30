# Phase 3 citation and answer-completeness investigation

This investigation checks Phase 2 hypotheses against the saved retrieval traces,
source PDF extraction and four **new** diagnostic chat requests. It does not
rewrite any Phase 2 result. The old failed raw generations were not saved, so the
exact historical `a11`/`a16` quote strings are **unavailable**. A refusal reason
alone cannot distinguish a paraphrased quote from a source mismatch.

The diagnostic requests replay the exact saved Phase 2 Top-3 text and scores,
after verifying every chunk against the copied corpus in
`eval/phase3/workspace/index/chroma`. They do not perform another retrieval or
query-embedding call. They use the configured `gpt-4o-mini` chat model and are
recorded separately in `citation_diagnostics/`. These requests are diagnostic
observations, not a controlled end-to-end score or historical reconstruction.

## Findings

| Question | Evidence check | New diagnostic observation | Classification |
| --- | --- | --- | --- |
| `a11` | Chunk `:27:0` contains the full gross-margin-percentage table, its 2025/2024/2023 header, and the Services row. Source extraction retains the correct row/column order. | Model claims the correct `75.4%` but quotes a synthesized label, `Services gross margin percentage: 75.4%`. That text is not contiguous in the source and remains invalid after whitespace-only normalization. | Model paraphrasing of a financial table; not a whitespace bug or missing answer row in this saved context. |
| `a16` | Top-3 includes `:40:0`, `:40:6`, and `:42:0`. The first chunk contains the 2025 table header plus only cash and money-market-fund rows. The second concerns 2024 restricted balances; the third concerns derivatives. None supplies the requested combined total. | Model incorrectly reports `33.539 billion`, obtained from only the first two investment categories, and invents a noncontiguous quote by omitting the intervening `Level 1:` block. The existing validator rejects it. | Missing total/context due to retrieval/chunk boundaries, plus an unsupported partial-sum inference and a noncontiguous quote. Treating this as merely citation formatting is incorrect. |
| `a06` | Chunk `:7:1` contains the first-quarter fact and the seasonal holiday-demand explanation in the same sentence. | Original prompt again produces claim text giving only the quarter, while its citation contains the missing reason. | Answer completeness; source evidence is sufficient. |

The financial-instruments page was also visually inspected. Its full 2025 table
has a total fair value of USD 132,420 million; the gold narrative on physical
page 28 rounds the combined balance to USD 132.4 billion. That total row is absent
from the saved Top-3 context. The PDF extraction itself retains this row on the
full page; the failure is not that the PDF omits the figure.

## New diagnostic evidence and latency

| File | Prompt | Chat + validation seconds | Outcome |
| --- | --- | ---: | --- |
| [a11_original_prompt.json](citation_diagnostics/a11_original_prompt.json) | Original Phase 2 | 3.931 | Correct claimed number; synthesized quote rejected |
| [a16_original_prompt.json](citation_diagnostics/a16_original_prompt.json) | Original Phase 2 | 1.804 | Incorrect partial total and noncontiguous quote rejected |
| [a06_original_prompt.json](citation_diagnostics/a06_original_prompt.json) | Original Phase 2 | 2.159 | Incomplete final answer accepted; reason appears only in quote |
| [a06_coverage_prompt.json](citation_diagnostics/a06_coverage_prompt.json) | Small coverage/table-copy appendix | 2.372 | Requested holiday reason appears, but the answer also adds an unsupported causal attribution to product introductions; this is **not counted as a clean improvement** |

These timings include chat and local validation only, with exact saved contexts;
they exclude retrieval, embedding, HTTP and UI. Four calls were already complete
when the instruction to defer additional chat calls arrived; no additional calls
were made after that instruction.

## Narrow intervention proposal

Prefer an initial prompt-only trial, with the output schema and citation validator
unchanged. Add general instructions that:

1. Require every requested part, including reasons, to be explicitly answered in
   claim text; information inside a quote does not substitute for answering.
2. Forbid adding other causes or conclusions beyond the support for those parts.
3. Require a financial-table quote to be a contiguous copied row/passage. Row-label
   interpretation, period and unit explanation belong in claim text, not inserted
   into a quotation.
4. Require complete components and period/unit context for a requested total;
   abstain if only a subset of rows is available. Never infer a combined total
   from an incomplete table excerpt.

The coverage diagnostic shows why this change still needs the planned paired
evaluation: better completeness alone can introduce an unsupported extra claim.
No per-question special cases should be added.

### Citation repair retry

Do **not** enable a quote-only repair retry by default on the current evidence.
For `a16`, a repair prompt could convert a correctly rejected unsupported partial
total into an accepted answer by finding a genuine cash-row quote. Exact quote
membership proves provenance, not that the claimed total follows from the text.
It cannot recover the missing total without better retrieval. The same limitation
means preserving a model's claim while replacing its quote is not automatically a
safe operation.

If a later controlled test enables repair, restrict it to at most one additional
call after `unverifiable_citation`, use only the existing context, preserve the
original claim and its numbers/units/periods, require an exact contiguous source
span and record both attempts and their latency. A failed repair must abstain.
Eligibility should require direct support rather than merely locating the same
number somewhere in a chunk. The current system has no reliable generic semantic
eligibility check, so this investigation recommends the cheaper prompt-level
trial and improved retrieval first, with explicit residual limitations.

## Regression coverage

`tests/test_phase3_qa.py` records the observed table-label and noncontiguous-table
quote failures, rejects a quote assigned to the wrong source, and verifies that
an exact original financial row with unchanged period/units remains accepted.
These tests verify citation mechanics; they do not claim to evaluate whether a
mocked model obeys a prompt or to solve general semantic entailment.
