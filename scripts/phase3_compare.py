"""Aggregate completed paired Phase 3 runs offline; never call a model API."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import tempfile
from pathlib import Path

from research_qa.evaluation import validate_reviews, write_summary
from research_qa.qa import match_original_quote


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def distribution(values):
    values = sorted(value for value in values if value is not None)

    def percentile(fraction):
        if not values:
            return None
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "count": len(values),
        "mean": mean(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "min": values[0] if values else None,
        "max": values[-1] if values else None,
    }


def raw_quote_counts(rows):
    if any("model_calls" not in row for row in rows):
        return {
            "available": False,
            "valid": None,
            "exact": None,
            "total": None,
            "generations": None,
            "unparseable_generations": None,
        }
    valid = exact = total = generations = unparseable = 0
    for row in rows:
        for call in row["model_calls"]:
            generations += 1
            try:
                content = json.loads(call["content"])
                sources = {
                    source["source_id"]: source["text"] for source in call["request"]["evidence"]
                }
                claims = content["claims"]
                if not isinstance(claims, list):
                    raise ValueError("Invalid claims")
                quotes = [evidence for claim in claims for evidence in claim["evidence"]]
                if any(
                    not isinstance(item.get("quote"), str)
                    or not isinstance(item.get("source_id"), str)
                    for item in quotes
                ):
                    raise ValueError("Invalid quote object")
            except (ValueError, TypeError, KeyError, AttributeError):
                unparseable += 1
                continue
            for quote in quotes:
                total += 1
                text = sources.get(quote["source_id"])
                if text is not None:
                    valid += match_original_quote(text, quote["quote"]) is not None
                    exact += (
                        8 <= len(quote["quote"]) <= 1000
                        and bool(quote["quote"].strip())
                        and quote["quote"] in text
                    )
    return {
        "available": True,
        "valid": valid,
        "exact": exact,
        "total": total,
        "generations": generations,
        "unparseable_generations": unparseable,
    }


def extended_metrics(rows):
    answerable = [row for row in rows if row["answerable"]]
    metrics = {"page_recall_at_k": mean(row.get("page_recall_at_k") for row in answerable)}
    coverage = {}
    for field, label in (
        ("evidence_at_3", "at_3"),
        ("evidence_at_k", "at_k"),
        ("filtered_evidence_at_k", "after_filter_at_k"),
    ):
        verified = [
            row[field]
            for row in answerable
            if row.get(field) and row[field].get("label_status") == "verified"
        ]
        metrics[f"evidence_recall_{label}"] = mean(item.get("evidence_recall") for item in verified)
        metrics[f"full_evidence_support_{label}"] = mean(
            item.get("full_support") for item in verified
        )
        coverage[label] = {
            "verified_question_trials": len(verified),
            "answerable_question_trials": len(answerable),
            "coverage": ratio(len(verified), len(answerable)),
        }
        if field == "filtered_evidence_at_k":
            metrics["relevant_chunks_surviving_filter_mean"] = mean(
                len(item.get("relevant_chunk_ids", [])) for item in verified
            )
    valid = sum(row.get("valid_citations", 0) for row in rows)
    returned = sum(row.get("returned_citations", 0) for row in rows)
    quote_counts = raw_quote_counts(rows)
    metrics.update(
        {
            "evidence_label_coverage": coverage,
            "returned_citation_validity": ratio(valid, returned),
            "returned_citation_counts": {"valid": valid, "total": returned},
            "raw_model_quote_validity": (
                ratio(quote_counts["valid"], quote_counts["total"])
                if quote_counts["available"]
                else None
            ),
            "raw_model_exact_quote_validity": (
                ratio(quote_counts["exact"], quote_counts["total"])
                if quote_counts["available"]
                else None
            ),
            "raw_model_quote_counts": quote_counts,
            "context_characters": distribution(row.get("context_chars") for row in rows),
            "model_calls_per_question_mean": (
                mean(len(row["model_calls"]) for row in rows) if quote_counts["available"] else None
            ),
            "citation_validation_refusals": sum(
                row.get("response") is not None
                and row["response"].get("reason") == "unverifiable_citation"
                for row in rows
            ),
        }
    )
    return metrics


def load_runs(root: Path, allow_preliminary: bool):
    runs = {"baseline": {}, "optimized": {}}
    pending = []
    for variant in runs:
        for path in sorted((root / variant).glob("run*")):
            match = re.fullmatch(r"run([1-9]\d*)", path.name)
            if not path.is_dir() or not match:
                continue
            number = int(match.group(1))
            metadata = read_json(path / "metadata.json")
            if metadata.get("status") != "complete":
                pending.append(f"{variant}/{path.name}")
                continue
            rows = [
                json.loads(line)
                for line in (path / "responses.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ids = [row["id"] for row in rows]
            if not rows or len(ids) != len(set(ids)):
                raise ValueError("Run must contain nonempty, unique question IDs")
            reviews_path = path / "reviews.json"
            reviews = validate_reviews(
                rows, read_json(reviews_path) if reviews_path.exists() else None
            )
            if len(reviews) != len(rows) and not allow_preliminary:
                raise ValueError(
                    "Final comparison requires a review for every question in every run"
                )
            runs[variant][number] = {
                "path": path,
                "metadata": metadata,
                "rows": rows,
                "reviews": reviews,
            }
    if pending and not allow_preliminary:
        raise ValueError("Final comparison cannot omit incomplete paired runs")
    numbers = set(runs["baseline"]) & set(runs["optimized"])
    if not numbers or set(runs["baseline"]) != set(runs["optimized"]):
        raise ValueError("Completed baseline/optimized repeat IDs must match and be nonempty")
    gold_by_id = None
    modes = set()
    invariant_fields = (
        "dataset_sha256",
        "chat_model",
        "embedding_model",
        "min_relevance_score",
        "max_context_chars",
    )
    reference = None
    for number in sorted(numbers):
        for variant in runs:
            run = runs[variant][number]
            meta = run["metadata"]
            modes.add(meta["mode"])
            current = {key: meta.get(key) for key in invariant_fields}
            if any(value is None for value in current.values()):
                raise ValueError("Run lacks recorded dataset/model/filter/context metadata")
            if reference is not None and current != reference:
                raise ValueError(
                    "Dataset/models/threshold/context limits differ across paired runs"
                )
            reference = current
            gold = {
                row["id"]: {
                    key: row.get(key)
                    for key in (
                        "question",
                        "answerable",
                        "reference_answer",
                        "evidence",
                        "required_patterns",
                    )
                }
                for row in run["rows"]
            }
            if gold_by_id is not None and gold != gold_by_id:
                raise ValueError("Question IDs or frozen gold labels differ across paired runs")
            gold_by_id = gold
        # Each variant's implementation/configuration must be fixed across repeats.
    if len(modes) != 1:
        raise ValueError("Cannot combine live runs with injected test clients")
    for variant, variant_runs in runs.items():
        identities = [
            {
                key: run["metadata"].get(key)
                for key in ("retrieval_method", "top_k", "qa_source_sha256", "prompt_sha256")
            }
            for run in variant_runs.values()
        ]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError(f"{variant} configuration changed between repeats")
    return runs, pending


def enrich_summary(rows, metadata, directory, reviews):
    # The shared writer also writes reviews.json. Use a temporary destination and
    # copy only derived outputs so original per-run review bytes are untouched.
    with tempfile.TemporaryDirectory(prefix="phase3-summary-") as scratch:
        temporary = Path(scratch)
        summary = write_summary(rows, metadata, temporary, reviews or None)
        for name in ("summary.json", "results.csv", "report.md"):
            shutil.copyfile(temporary / name, directory / name)
    if metadata.get("aggregation"):
        report = directory / "report.md"
        text = report.read_text(encoding="utf-8")
        text = text.replace("`responses.jsonl`", "`run*/responses.jsonl`")
        text = text.replace("`reviews.json`", "`run*/reviews.json`")
        report.write_text(text, encoding="utf-8")
    holdout = metadata.get("evaluation_scope") == "holdout"
    if holdout:
        report = directory / "report.md"
        text = report.read_text(encoding="utf-8").replace(
            "Small curated development benchmark, not an independent held-out estimate.",
            "Independent-company holdout, reported separately from development results; "
            "small sample and single paired run limit generalization.",
        )
        report.write_text(text, encoding="utf-8")
    summary["metrics"].update(extended_metrics(rows))
    summary["methodology"].update(
        {
            "evidence_recall": "Macro requirement-group recall over verified answerable labels; "
            "unknown/unreviewed labels are unavailable, with coverage reported. "
            "Bundles may need multiple chunks, including period/unit headers.",
            "citation_validity": "Conditional returned-citation identity and raw-source substring "
            "validity; not semantic entailment or correctness. Empty denominator "
            "is N/A. Raw model quote validity is a separate pre-validation metric.",
            "context": "Filtered retrieved text characters, capped at the configured budget; "
            "includes zero-context/error cases. Top-5 can supply more evidence than Top-3.",
            "repetition": (
                "Holdout comparison is reported separately from development. Repeated trials, "
                "if present, do not create additional independent questions or significance."
                if holdout
                else "Repeated trials on the same development set; not new unique questions, "
                "an independent holdout or a significance test."
            ),
        }
    )
    if holdout:
        summary["methodology"]["scope"] = (
            "Independent-company holdout, not used for retrieval/prompt selection; small "
            "sample and single paired run limit generalization. Missing evidence labels are "
            "unavailable, not inferred from physical-page matches."
        )
    save_json(directory / "summary.json", summary)
    extra = [
        "",
        "## Phase 3 extended audit",
        "",
        "Evidence metrics use verified labels only; see summary.json for label coverage.",
        "Citation validity is provenance, not semantic correctness.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "page_recall_at_k",
        "evidence_recall_at_3",
        "evidence_recall_at_k",
        "evidence_recall_after_filter_at_k",
        "returned_citation_validity",
        "raw_model_quote_validity",
        "citation_validation_refusals",
    ):
        value = summary["metrics"][key]
        extra.append(f"| {key} | {'N/A' if value is None else value} |")
    with (directory / "report.md").open("a", encoding="utf-8") as file:
        file.write("\n".join(extra) + "\n")
    return summary


def paired_rows(runs):
    result = []
    for number in sorted(runs["baseline"]):
        baseline, optimized = (runs[name][number] for name in ("baseline", "optimized"))
        other = {row["id"]: row for row in optimized["rows"]}
        for row in baseline["rows"]:
            question_id = row["id"]
            after = other[question_id]
            before_review = baseline["reviews"].get(question_id, {})
            after_review = optimized["reviews"].get(question_id, {})
            before, selected = before_review.get("correct"), after_review.get("correct")
            outcome = (
                "requires_review"
                if before is None or selected is None
                else "fixed"
                if not before and selected
                else "regressed"
                if before and not selected
                else "both_correct"
                if before
                else "both_incorrect"
            )
            result.append(
                {
                    "repeat": number,
                    "question_id": question_id,
                    "question": row["question"],
                    "answerable": row["answerable"],
                    "baseline_correct": before,
                    "optimized_correct": selected,
                    "outcome": outcome,
                    "baseline_abstained": (row.get("response") or {}).get("abstained"),
                    "optimized_abstained": (after.get("response") or {}).get("abstained"),
                    "baseline_reason": (row.get("response") or {}).get("reason"),
                    "optimized_reason": (after.get("response") or {}).get("reason"),
                    "baseline_error": row.get("error_type"),
                    "optimized_error": after.get("error_type"),
                    "baseline_answer": (row.get("response") or {}).get("answer"),
                    "optimized_answer": (after.get("response") or {}).get("answer"),
                    "baseline_review_note": before_review.get("note"),
                    "optimized_review_note": after_review.get("note"),
                }
            )
    return result


def number(value, percent=False):
    if value is None:
        return "N/A"
    return f"{value:.2%}" if percent else f"{value:.3f}"


def comparison_report(root, summaries, runs, paired, pending):
    baseline, optimized = (summaries[name] for name in ("baseline", "optimized"))
    bmeta, ometa = baseline["run"], optimized["run"]
    final = all(summary["counts"]["review_complete"] for summary in summaries.values())
    holdout = bmeta.get("evaluation_scope") == "holdout"
    lines = [
        "# Phase 3 independent-company holdout evaluation"
        if holdout
        else "# Phase 3 paired development evaluation",
        "",
        f"Status: {'agent-adjudicated complete' if final else 'PRELIMINARY — incomplete reviews'}.",
        f"{len(runs['baseline'])} paired repetitions of the same "
        f"{len(runs['baseline'][min(runs['baseline'])]['rows'])} frozen questions.",
        f"Models: {bmeta['chat_model']} / {bmeta['embedding_model']}.",
        "",
        "## Recorded intervention",
        "",
        f"Baseline: {bmeta['retrieval_method']} Top-{bmeta['top_k']}; "
        f"optimized: {ometa['retrieval_method']} Top-{ometa['top_k']}.",
        f"Context character ceiling: {bmeta['max_context_chars']} for both. "
        f"Cosine threshold: {bmeta['min_relevance_score']} for both.",
        "The selected variant bundles a larger Top-k evidence allowance and the recorded "
        "prompt change. It is not a controlled single-variable retrieval or prompt experiment. "
        "Top-5 may supply up to five chunks where Top-3 supplies three.",
        "",
        "| Metric | Fresh baseline | Optimized | Difference (optimized − baseline) |",
        "| --- | ---: | ---: | ---: |",
    ]
    specs = [
        ("Page Recall@3", "recall_at_3", True),
        ("Page Recall@selected k", "page_recall_at_k", True),
        ("Evidence Recall@3", "evidence_recall_at_3", True),
        ("Evidence Recall@selected k", "evidence_recall_at_k", True),
        ("Evidence Recall after filtering @selected k", "evidence_recall_after_filter_at_k", True),
        ("Answerable accuracy", "reviewed_answerable_accuracy", True),
        ("Overall accuracy", "reviewed_accuracy", True),
        ("False refusal rate", "false_refusal_rate", True),
        ("Correct refusal rate", "correct_refusal_rate", True),
        ("Returned citation validity (provenance)", "returned_citation_validity", True),
        ("Raw model quote validity", "raw_model_quote_validity", True),
    ]
    for label, key, percent in specs:
        before, after = baseline["metrics"][key], optimized["metrics"][key]
        difference = after - before if before is not None and after is not None else None
        delta = "N/A" if difference is None else f"{difference * 100:+.2f} pp"
        lines.append(
            f"| {label} | {number(before, percent)} | {number(after, percent)} | {delta} |"
        )
    for label, stage, metric in (
        ("P50 ask latency (s)", "latency_seconds", "p50"),
        ("P95 ask latency (s)", "latency_seconds", "p95"),
    ):
        before, after = (
            summary["metrics"][stage]["ask"][metric] for summary in (baseline, optimized)
        )
        difference = after - before if before is not None and after is not None else None
        lines.append(f"| {label} | {number(before)} | {number(after)} | {number(difference)} |")
    for label, key in (
        ("Actual context characters, mean", "mean"),
        ("Actual context characters, maximum", "max"),
    ):
        before, after = (
            summary["metrics"]["context_characters"][key] for summary in (baseline, optimized)
        )
        lines.append(
            f"| {label} | {number(before)} | {number(after)} | "
            f"{number(after - before) if before is not None and after is not None else 'N/A'} |"
        )
    lines.extend(["", "Evidence-label coverage (verified / answerable question-trials):"])
    for name, summary in summaries.items():
        coverage = summary["metrics"]["evidence_label_coverage"]
        lines.append(
            f"- {name}: "
            + "; ".join(
                f"{key}: {value['verified_question_trials']}/{value['answerable_question_trials']}"
                for key, value in coverage.items()
            )
        )
    lines.extend(
        [
            "",
            "## Per-run variability",
            "",
            "| Repeat | Variant | Reviewed accuracy | Answerable accuracy | "
            "False refusal | P95 (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for repeat in sorted(runs["baseline"]):
        for name in runs:
            metrics = runs[name][repeat]["summary"]["metrics"]
            lines.append(
                f"| {repeat} | {name} | {number(metrics['reviewed_accuracy'], True)} | "
                f"{number(metrics['reviewed_answerable_accuracy'], True)} | "
                f"{number(metrics['false_refusal_rate'], True)} | "
                f"{number(metrics['latency_seconds']['ask']['p95'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired per-question changes",
            "",
            "Every question is compared, including previously correct cases. "
            "Counts below refer to repeated trials, not additional unique questions.",
            "",
            "| Question | Baseline correct | Optimized correct | Fixed trials | Regressed trials |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for question_id in dict.fromkeys(row["question_id"] for row in paired):
        cases = [row for row in paired if row["question_id"] == question_id]
        b = sum(row["baseline_correct"] is True for row in cases)
        o = sum(row["optimized_correct"] is True for row in cases)
        fixed = sum(row["outcome"] == "fixed" for row in cases)
        regressed = sum(row["outcome"] == "regressed" for row in cases)
        unavailable = any(row["outcome"] == "requires_review" for row in cases)
        lines.append(
            f"| {question_id} | {'N/A' if unavailable else str(b) + '/' + str(len(cases))} | "
            f"{'N/A' if unavailable else str(o) + '/' + str(len(cases))} | "
            f"{'N/A' if unavailable else fixed} | {'N/A' if unavailable else regressed} |"
        )
    fixed_ids = sorted({row["question_id"] for row in paired if row["outcome"] == "fixed"})
    regressed_ids = sorted({row["question_id"] for row in paired if row["outcome"] == "regressed"})
    empty_status = "none" if final else "unavailable (incomplete reviews)"
    lines.extend(
        [
            "",
            f"At least one paired fixed trial: {', '.join(fixed_ids) or empty_status}.",
            f"At least one paired regressed trial: {', '.join(regressed_ids) or empty_status}.",
            "Detailed responses, refusals, errors and adjudication notes are in "
            "[paired_comparison.csv](paired_comparison.csv).",
            "",
        ]
    )
    historical_path = root.parent / "results/latest/summary.json"
    if historical_path.is_file():
        historical = read_json(historical_path)
        metrics = historical["metrics"]
        lines.extend(
            [
                "## Preserved Phase 2 reference",
                "",
                "Phase 2 latest is an earlier, noncontemporaneous run and was not changed.",
                f"Overall accuracy: {number(metrics.get('answer_accuracy'), True)}; "
                f"Page Recall@3: {number(metrics.get('recall_at_3'), True)}.",
                "Historical raw model quote validity: **unavailable**; rejected raw "
                "generations were not captured. Fresh Phase 3 baseline quote metrics above "
                "come from its new captured calls, not invented historical values.",
                "",
            ]
        )
        reviews_path = historical_path.parent / "reviews.json"
        if reviews_path.is_file():
            previous = read_json(reviews_path)
            old_regressions = sorted(
                {
                    row["question_id"]
                    for row in paired
                    if previous.get(row["question_id"], {}).get("correct") is True
                    and row["optimized_correct"] is False
                }
            )
            lines.append(
                "Historically correct Phase 2 questions with at least one optimized "
                f"incorrect trial: {', '.join(old_regressions) or 'none'}."
            )
    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            "Page Recall concerns annotated pages; Evidence Recall concerns explicitly "
            "reviewed supporting bundles and may require row+period/unit context. Answer "
            "accuracy grades the final claim text against the same frozen rubric. These "
            "are different outcomes. Unknown evidence labels are N/A, not negatives.",
            "Returned citation validity checks original source identity and contiguous "
            "text only, conditional on citations that were returned; it cannot establish "
            "semantic entailment and a system that refuses most questions may still score "
            "high. Raw quote validity uses parseable captured model quotes; malformed "
            "generations and its denominator are separately recorded in summary.json.",
            "Latency includes live query embedding, retrieval, generation and validation, "
            "excluding HTTP/UI/indexing; startup costs are recorded per run. Repeated "
            + (
                "observations on this small independent-company holdout do not establish "
                "statistical significance or generalizable gains. These results are separate "
                "from the Apple development benchmark; no holdout-driven tuning is performed."
                if holdout
                else "observations on this small development set are not independent new questions "
                "and do not establish statistical significance or generalization. Holdout "
                "performance must be reported separately."
            ),
            "",
        ]
    )
    if pending:
        lines.append("Excluded incomplete runs (preliminary mode only): " + ", ".join(pending))
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(root: Path, *, allow_preliminary=False, evaluation_scope=None):
    runs, pending = load_runs(root, allow_preliminary)
    evaluation_scope = evaluation_scope or (
        "holdout" if root.name.lower() == "holdout" else "development"
    )
    if evaluation_scope not in {"development", "holdout"}:
        raise ValueError("Evaluation scope must be development or holdout")
    summaries = {}
    for variant, variant_runs in runs.items():
        combined, reviews = [], {}
        for repeat in sorted(variant_runs):
            run = variant_runs[repeat]
            run["summary"] = enrich_summary(
                run["rows"],
                {**run["metadata"], "evaluation_scope": evaluation_scope},
                run["path"],
                run["reviews"],
            )
            for row in run["rows"]:
                unique = f"run{repeat}:{row['id']}"
                combined.append({**row, "id": unique, "question_id": row["id"], "repeat": repeat})
                if row["id"] in run["reviews"]:
                    reviews[unique] = run["reviews"][row["id"]]
        first = variant_runs[min(variant_runs)]["metadata"]
        metadata = {
            **first,
            "repeat": "pooled",
            "paired_repetitions": len(variant_runs),
            "repeat_ids": sorted(variant_runs),
            "unique_questions": len(variant_runs[min(variant_runs)]["rows"]),
            "aggregation": f"Pooled question-trials; same fixed {evaluation_scope} questions",
            "evaluation_scope": evaluation_scope,
        }
        summary = enrich_summary(combined, metadata, root / variant, reviews)
        summary["counts"]["unique_questions"] = metadata["unique_questions"]
        summary["counts"]["paired_repetitions"] = len(variant_runs)
        summary["per_run"] = [
            {
                "repeat": repeat,
                "counts": run["summary"]["counts"],
                "metrics": run["summary"]["metrics"],
            }
            for repeat, run in sorted(variant_runs.items())
        ]
        save_json(root / variant / "summary.json", summary)
        summaries[variant] = summary
    paired = paired_rows(runs)
    with (root / "paired_comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    comparison_report(root, summaries, runs, paired, pending)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/phase3"))
    parser.add_argument("--allow-preliminary", action="store_true")
    parser.add_argument("--scope", choices=("development", "holdout"))
    args = parser.parse_args(argv)
    try:
        summaries = aggregate(
            args.root, allow_preliminary=args.allow_preliminary, evaluation_scope=args.scope
        )
        print(
            json.dumps(
                {
                    name: {
                        "counts": summary["counts"],
                        "accuracy": summary["metrics"]["reviewed_accuracy"],
                    }
                    for name, summary in summaries.items()
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Comparison not generated: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
