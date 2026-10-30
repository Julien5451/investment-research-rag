"""Freeze Phase 2 provenance and copy its index for isolated Phase 3 experiments."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def prepare(root: Path, *, rebuild_index: bool = False, settings=None, embeddings=None) -> dict:
    """Verify public provenance, then reuse/copy or explicitly rebuild an index."""
    phase = root / "eval/phase3"
    frozen = phase / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    manifest_path = frozen / "phase2_manifest.json"
    files = [root / "eval/dataset.json"]
    files.extend(p for p in (root / "eval/results").glob("*/*") if p.is_file())
    manifest = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Phase 2 artifact contents changed since the freeze")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(root / "src/research_qa/qa.py", frozen / "qa_phase2.py")
        shutil.copy2(root / "src/research_qa/config.py", frozen / "config_phase2.py")
        shutil.copy2(root / "eval/dataset.json", frozen / "apple_dataset.json")
    target = phase / "workspace/index"
    if not target.exists():
        original_index = root / "eval/results/latest/index"
        if original_index.exists():
            shutil.copytree(original_index, target)
            mode = "copied"
        elif rebuild_index:
            from phase3_build_index import build_index

            from research_qa.config import Settings

            result = build_index(root, settings or Settings(), embeddings=embeddings)
            return {"frozen_artifacts": len(manifest), **result}
        else:
            raise FileNotFoundError(
                "No local index. Download pinned PDFs with scripts/prepare_eval.py, then run "
                "scripts/phase3_setup.py --rebuild-index (embedding API calls, no chat)."
            )
    else:
        mode = "existing"
    from phase3_build_index import validate_index

    count = validate_index(root, target)
    return {"frozen_artifacts": len(manifest), "mode": mode, "chunks": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Fallback for a fresh clone; "
        "uses pinned PDFs and the configured embedding API, never the chat API",
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                prepare(Path(__file__).resolve().parents[1], rebuild_index=args.rebuild_index),
                indent=2,
            )
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    except Exception as exc:
        # Provider exception messages may contain request bodies or credentials.
        raise SystemExit(
            f"Phase 3 setup failed ({type(exc).__name__}); check pinned corpus "
            "and embedding configuration. Upstream details are suppressed."
        ) from None


if __name__ == "__main__":
    main()
