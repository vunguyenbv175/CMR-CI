#!/usr/bin/env python3
"""Verify the public snapshot byte manifest before running any test."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PUBLIC_MANIFEST.json"
ALLOWED_PREFIXES = ("packages/", "services/workstation/", "services/inference/", "tests/")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems: list[str] = []
    files = doc.get("files")
    if not isinstance(files, list) or not files:
        problems.append("manifest has no files")
        files = []
    if not doc.get("source_sha"):
        problems.append("manifest source_sha is missing")
    if doc.get("schema") != "cmr-public-ci-snapshot-manifest/v1":
        problems.append("unexpected manifest schema")

    seen: set[str] = set()
    for entry in files:
        rel = entry.get("public_path")
        if not isinstance(rel, str) or rel in seen:
            problems.append(f"duplicate/invalid public_path: {rel!r}")
            continue
        seen.add(rel)
        if not rel.startswith(ALLOWED_PREFIXES) or ".." in Path(rel).parts:
            problems.append(f"path outside allowlist: {rel}")
            continue
        path = ROOT / rel
        if not path.is_file():
            problems.append(f"missing exported file: {rel}")
            continue
        actual = sha256(path)
        if actual != entry.get("source_sha256"):
            problems.append(f"hash mismatch: {rel} {actual} != {entry.get('source_sha256')}")
        if path.stat().st_size != entry.get("bytes"):
            problems.append(f"size mismatch: {rel}")

    ci_files = doc.get("ci_only_files", [])
    if not isinstance(ci_files, list) or not ci_files:
        problems.append("manifest has no ci_only_files")
        ci_files = []
    for entry in ci_files:
        rel = entry.get("path")
        path = ROOT / rel if isinstance(rel, str) else ROOT / "__invalid__"
        if not isinstance(rel, str) or not path.is_file():
            problems.append(f"missing CI-only file: {rel!r}")
            continue
        if sha256(path) != entry.get("sha256"):
            problems.append(f"CI-only hash mismatch: {rel}")

    listed = seen | {e.get("path") for e in ci_files if isinstance(e, dict)}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "PUBLIC_MANIFEST.json":
            continue
        if rel not in listed:
            problems.append(f"unlisted file in snapshot: {rel}")

    if problems:
        print("MANIFEST_VERIFY=FAIL")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"MANIFEST_VERIFY=PASS files={len(files)} ci_only={len(ci_files)}")
    print(f"SOURCE_SHA={doc['source_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
