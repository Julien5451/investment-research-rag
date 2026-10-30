"""Check prospective Git files without initializing, staging or pushing the project."""

import argparse
import json
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import dotenv_values


def audit(root: Path) -> dict:
    """Use an ephemeral Git directory to apply the real project's ignore rules."""
    root = root.resolve()
    with TemporaryDirectory(prefix="research-qa-git-audit-") as temporary:
        git_dir = Path(temporary) / "audit.git"
        subprocess.run(["git", "init", "--bare", "--quiet", str(git_dir)], check=True)
        command = ["git", f"--git-dir={git_dir}", f"--work-tree={root}"]
        result = subprocess.run(
            [*command, "ls-files", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
        )
        paths = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
        probes = [
            ".env",
            ".env.production",
            "private.pdf",
            "data/uploads/personal.pdf",
            "data/chroma/chroma.sqlite3",
            "eval/pdfs/apple_2025_10k.pdf",
            "eval/phase3/workspace/index/chroma/chroma.sqlite3",
            "eval/phase3/optimized/run1/index/chroma/chroma.sqlite3",
            ".streamlit/secrets.toml",
            "credentials.pem",
            "private.key",
            "personal.db",
            "local.sqlite",
            ".release/protected_before.json",
            ".venv/pyvenv.cfg",
            "src/research_qa/__pycache__/main.cpython-311.pyc",
            ".pytest_cache/README.md",
            ".ruff_cache/CACHEDIR.TAG",
            "tmp/diagnose_quote.py",
        ]
        ignored = subprocess.run(
            [*command, "check-ignore", "--no-index", "-z", "--stdin"],
            input=b"\0".join(path.encode() for path in probes) + b"\0",
            capture_output=True,
            check=False,
        )
        ignored_paths = {p.decode() for p in ignored.stdout.split(b"\0") if p}
        frozen_paths = [
            "eval/dataset.json",
            "eval/results/latest/results.csv",
            "eval/phase3/frozen/qa_phase2.py",
            "src/research_qa/qa.py",
            "uv.lock",
        ]
        manifest = json.loads((root / "eval/phase3/frozen/phase2_manifest.json").read_text())
        frozen_paths = sorted(
            set(frozen_paths)
            | set(manifest)
            | {
                path
                for path in paths
                if path.startswith("eval/phase3/") or path.startswith("eval/results/")
            }
            | {
                "src/research_qa/retrieval.py",
                "src/research_qa/index.py",
                "src/research_qa/ingestion.py",
            }
        )
        preserved = []
        for relative in frozen_paths:
            data = (root / relative).read_bytes()
            filtered = subprocess.run(
                [
                    *command,
                    "-c",
                    "core.autocrlf=true",
                    "hash-object",
                    f"--path={relative}",
                    "--stdin",
                ],
                input=data,
                check=True,
                capture_output=True,
                cwd=root,
            ).stdout
            unfiltered = subprocess.run(
                [*command, "hash-object", "--no-filters", "--stdin"],
                input=data,
                check=True,
                capture_output=True,
                cwd=root,
            ).stdout
            if filtered == unfiltered:
                preserved.append(relative)
    keys = [
        value.encode()
        for name, value in dotenv_values(root / ".env").items()
        if "KEY" in name.upper() and value and len(value) >= 12
    ]
    findings = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        if (
            path.suffix.lower() == ".pdf"
            or path.name.startswith(".env")
            and path.name != ".env.example"
        ):
            findings.append({"path": relative, "reason": "private-file-pattern-not-ignored"})
        if any(key in data for key in keys):
            findings.append({"path": relative, "reason": "configured-secret-value-found"})
        if re.search(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b", data):
            findings.append({"path": relative, "reason": "possible-api-key-pattern"})
    for probe in probes:
        if probe not in ignored_paths:
            findings.append({"path": probe, "reason": "required-ignore-rule-missing"})
    for relative in frozen_paths:
        if relative not in preserved:
            findings.append({"path": relative, "reason": "git-newline-conversion-changes-hash"})
    return {
        "publishable_files_checked": len(paths),
        "files": sorted(paths),
        "ignore_probes": len(probes),
        "all_ignore_probes_passed": set(probes) <= ignored_paths,
        "frozen_byte_probes": len(frozen_paths),
        "all_frozen_bytes_preserved_with_autocrlf": len(preserved) == len(frozen_paths),
        "findings": findings,
        "scope": "Prospective files selected by Git ignore rules; configured secrets and "
        "common API-key patterns checked without printing their values. Not a "
        "general PII detector or audit of pre-existing Git history.",
        "git_actions": "No project git init, staging, commit, remote creation or push performed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".release/publishability.json"),
        help="Write a new release audit; historical Phase 3 records are preserved",
    )
    parser.add_argument("--manifest", type=Path, help="Also write the sorted candidate file list")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = audit(root)
    target = root / args.output
    if target.resolve().is_relative_to((root / "eval").resolve()):
        parser.error("Release audits must not overwrite historical evaluation artifacts")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.manifest:
        manifest_target = root / args.manifest
        manifest_target.parent.mkdir(parents=True, exist_ok=True)
        manifest_target.write_text("\n".join(report["files"]) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    if report["findings"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
