# Independent-company holdout preparation

This file records dataset preparation only. No chat completion, query embedding,
retrieval experiment or end-to-end holdout evaluation was run while selecting
these questions. The implementation and retrieval settings must be frozen before
the parent task evaluates this dataset. Results must be reported separately from
the Apple development set; do not use holdout outcomes to revise the selected
retrieval parameters, prompts or citation policy.

## Source and fixed dataset

- Company: Costco Wholesale Corporation.
- Report: fiscal 2025 Annual Report, including the Form 10-K for the fiscal year
  ended August 31, 2025.
- Official investor-relations source: <https://investor.costco.com/overview/> and
  <https://investor.costco.com/financials/annual-reports-and-proxy-statements/default.aspx>.
- Public report PDF:
  <https://s201.q4cdn.com/287523651/files/doc_financials/2025/ar/COST-Annual-Report-2025.pdf>.
- Independent primary cross-reference: SEC Form 10-K
  <https://www.sec.gov/Archives/edgar/data/909832/000090983225000101/cost-20250831.htm>.
- Local downloaded file: `eval/pdfs/costco_2025.pdf` (excluded from Git).
- PDF size: 6,247,969 bytes; 76 physical pages, below the existing 20 MiB limit.
- PDF SHA-256:
  `25449d39bb369341789c0060e408805eac93c63e2a3a345dd3fae9b8307e72bc`.
- Question-authoring timestamp (Asia/Shanghai):
  `2026-10-30T04:38:45.127499+08:00`.
- Frozen dataset: [holdout_dataset.json](holdout_dataset.json).
- Dataset SHA-256:
  `95af2280b3b86e8e44032a7f47b0cbd43a578bafe065620f9ced89f444a528ac`.
- Size: 12 questions, including 9 answerable and 3 unanswerable; 2 questions are
  written in Chinese and 10 in English.

The report was selected for public availability, suitable file size and layout
difference, before testing the holdout questions. Its financial tables use dotted
row leaders, dense numeric columns and report-specific membership measures;
this differs from Apple's wide, lightly spaced product/financial tables. Physical
page numbers in the JSON are not the printed page numbers: for example, physical
page 41 carries printed page 35.

## Question coverage and reference checks

| ID | Topic | Physical source page(s) | Reference requirement |
| --- | --- | --- | --- |
| h01 | Total revenue | 41 | USD 275,235 million, including membership fees; do not confuse it with net sales |
| h02 | Net income | 41 | USD 8,099 million |
| h03 | Membership renewal | 12 | 92.3% in U.S./Canada and 89.8% worldwide, assigned to the correct geographies |
| h04 | Membership fee growth explanation | 31 | Both new member sign-ups and membership fee increases |
| h05 | Warehouse openings | 30 | 27 openings, 3 relocations and 24 net new warehouses, with correct labels |
| h06 | Executive member sales penetration | 12 | Approximately 73.6% of worldwide net sales |
| h07 | Membership table | 12 | 81,000 thousand paid members and 145,200 thousand total cardholders |
| h08 | Business model explanation | 10 | High sales/rapid inventory turnover together with operating efficiencies; source examples include volume purchasing, efficient distribution and reduced handling in no-frills self-service warehouses |
| h09 | Workforce retention | 13 | Approximately 94%, for U.S./Canada employees with at least one year of service |
| h10 | Future actual net income | None | Explicit abstention: a 2025 report cannot establish exact actual FY2027 net income |
| h11 | Product-level net profit margin | None | Explicit abstention: standalone rotisserie-chicken net profit margin is not separately disclosed |
| h12 | Another company's subscription members | None | Explicit abstention: Walmart+ active paying membership count is outside this Costco report |

All answerable references were checked against the full relevant PDF pages,
including table headings, year columns, financial units and footnotes. The
numerical figures above were not invented or inferred from model answers.
The JSON stores exact original table rows where possible; sentence excerpts are
verified after whitespace normalization against their specified physical page.

For h07, the report's table is in thousands and its accompanying qualification
states that reported membership counts include active memberships and a small
number of expired memberships not renewed in the preceding 12 months. Correct
unit conversion is acceptable; describing these reported totals as a verified
count of only active members is not supported. The question asks for reported
counts, so repeating this optional footnote is not itself required.

For multipart questions h03, h05 and h07, the final answer must contain every
requested figure with correct labels. For explanatory h04 and h08, a citation
containing a missing explanation does not make an incomplete answer correct.
The provided regex patterns are preliminary checks; final adjudication should
check completeness, units, time period, attribution and supporting source text.

Unanswerable questions intentionally have no fabricated source page or quote.
Full extracted-text searches found zero occurrences of `rotisserie`, `hot dog`
and `Walmart+`. The report does mention 2027 in other contexts (such as future
obligations); that is not evidence of actual FY2027 net income.

## Local verification performed

The following existing preparation command completed successfully in WSL:

```bash
uv run python scripts/prepare_eval.py \
  --dataset eval/phase3/holdout_dataset.json --pdf-dir eval/pdfs
```

It verified the pinned PDF hash, 76-page count and all source excerpts, then
reported `Validated 12 questions: 9 answerable, 3 unanswerable`.
The existing evaluation schema and `_prepare_documents` function also accepted
the dataset and verified every gold excerpt before any provider construction.

Relevant pages were rendered with the bundled Poppler binary and visually
inspected, rather than relying on extracted table order alone. Read-only QA
images are available locally (ignored as temporary files):

- `tmp/pdfs/costco-10.png`: business model and operating structure.
- `tmp/pdfs/costco-12.png`: membership tables, percentages and qualifications.
- `tmp/pdfs/costco-13.png`: workforce retention scope.
- `tmp/pdfs/costco-30.png`: openings and fiscal-year operating highlights.
- `tmp/pdfs/costco-31.png`: membership fee growth explanation.
- `tmp/pdfs/costco-41.png`: financial statement headings, units and year columns.

The PDF's front-matter shareholder letter on physical page 3 has problematic
font/text encoding in extraction. No holdout gold reference uses that letter;
the original PDF is preserved unchanged, and no OCR, reflow or replacement text
has been introduced. This is a corpus limitation to retain in the final report,
not a reason to change retrieval settings after seeing holdout performance.

## Evaluation handoff

Dataset preparation is complete. The parent task should record its frozen code,
prompt and retrieval configuration hashes, then run the holdout once with that
selected implementation. Repeated runs, if performed for model variability,
must retain the same frozen configuration and be labeled as such. Preserve all
per-question responses, citations and review notes, including failures.
