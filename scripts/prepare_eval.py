"""Download the pinned public corpus and validate gold evidence before evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pymupdf


def normalize(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("eval/pdfs"))
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    pages = {}
    for entry in dataset["documents"]:
        name = entry["document_name"]
        if Path(name).name != name or "\\" in name or not name.lower().endswith(".pdf"):
            raise ValueError("Manifest PDF filename must be a plain .pdf basename")
        target = args.pdf_dir / name
        if not target.exists():
            request = Request(entry["url"], headers={"User-Agent": "ResearchQA-Benchmark/0.1"})
            with urlopen(request, timeout=90) as response:
                data = response.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise ValueError("Public PDF exceeds the benchmark download limit")
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("Downloaded PDF SHA-256 differs from pinned manifest")
            target.write_bytes(data)
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Hash mismatch for {name}; existing file was preserved")
        with pymupdf.open(target) as pdf:
            if len(pdf) != entry["pages"]:
                raise ValueError(f"Unexpected page count for {name}")
            pages[name] = [normalize(page.get_text("text", sort=True)) for page in pdf]
        print(f"Verified {name}: {len(pages[name])} physical pages, pinned SHA-256 matches")

    seen = set()
    for question in dataset["questions"]:
        if question["id"] in seen:
            raise ValueError("Duplicate question ID")
        seen.add(question["id"])
        if question["answerable"] and not question["evidence"]:
            raise ValueError(f"Missing gold evidence: {question['id']}")
        for evidence in question["evidence"]:
            page = pages[evidence["document_name"]][evidence["page"] - 1]
            if normalize(evidence["quote"]) not in page:
                raise ValueError(f"Gold quote not found on the specified page: {question['id']}")
    answerable = sum(question["answerable"] for question in dataset["questions"])
    print(
        f"Validated {len(seen)} questions: {answerable} answerable, "
        f"{len(seen) - answerable} unanswerable"
    )


if __name__ == "__main__":
    main()
