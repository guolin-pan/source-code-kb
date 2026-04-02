#!/usr/bin/env python3
"""JSONL Schema Validator.

Validates JSONL files against the JSONL schema specification.
Zero external dependencies — uses only Python standard library.

Usage:
    python validate_jsonl.py <file1.jsonl> [file2.jsonl ...]

Exit codes:
    0 — All lines passed validation
    1 — One or more validation errors found
    2 — Usage error (no files specified, file not found)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── Schema field definitions ─────────────────────────────────

VALID_DOMAINS = {
    "module-internals",
    "module-interface",
    "end-to-end-flow",
    "system-constraints",
    "data-model",
    "build-deploy",
}

RISK_MARKERS = {"risk", "unverified", "speculative", "uncertain", "inferred", "tentative"}


def validate_record(record: dict, line_no: int) -> list[str]:
    """Validate a single JSONL record against the JSONL schema.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []

    # ── Required string fields ──
    for field in ("id", "content", "domain", "topic", "section", "source", "updated_at"):
        val = record.get(field)
        if val is None:
            errors.append(f"{field}: missing (required)")
        elif not isinstance(val, str):
            errors.append(f"{field}: expected string, got {type(val).__name__}")
        elif not val.strip():
            errors.append(f"{field}: must not be empty")

    # ── content: min length ──
    content = record.get("content", "")
    if isinstance(content, str) and content.strip() and len(content) < 50:
        errors.append(f"content: too short ({len(content)} chars, min 50)")

    # ── domain: enum check ──
    domain = record.get("domain", "")
    if isinstance(domain, str) and domain and domain not in VALID_DOMAINS:
        errors.append(f"domain: '{domain}' not in {sorted(VALID_DOMAINS)}")

    # ── scope: list[string], non-empty ──
    scope = record.get("scope")
    if scope is None:
        errors.append("scope: missing (required)")
    elif not isinstance(scope, list):
        errors.append(f"scope: expected list, got {type(scope).__name__}")
    elif len(scope) == 0:
        errors.append("scope: must not be empty")
    else:
        for i, item in enumerate(scope):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"scope[{i}]: must be non-empty string")

    # ── tags: list[string], >= 2 items ──
    tags = record.get("tags")
    if tags is None:
        errors.append("tags: missing (required)")
    elif not isinstance(tags, list):
        errors.append(f"tags: expected list, got {type(tags).__name__}")
    elif len(tags) < 2:
        errors.append(f"tags: expected >= 2 items, got {len(tags)}")
    else:
        for i, item in enumerate(tags):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"tags[{i}]: must be non-empty string")

    # ── confidence: float, 0.0-1.0 ──
    confidence = record.get("confidence")
    if confidence is None:
        errors.append("confidence: missing (required)")
    elif not isinstance(confidence, (int, float)):
        errors.append(f"confidence: expected float, got {type(confidence).__name__}")
    else:
        conf_val = float(confidence)
        if conf_val < 0.0 or conf_val > 1.0:
            errors.append(f"confidence: {conf_val} out of range [0.0, 1.0]")

    # ── updated_at: ISO 8601 format ──
    updated_at = record.get("updated_at", "")
    if isinstance(updated_at, str) and updated_at.strip():
        iso_pattern = r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
        if not re.match(iso_pattern, updated_at):
            errors.append(f"updated_at: '{updated_at}' is not valid ISO 8601")

    # ── meta: optional, must be dict if present ──
    meta = record.get("meta")
    if meta is not None and not isinstance(meta, dict):
        errors.append(f"meta: expected object or null, got {type(meta).__name__}")

    # ── files: required, non-empty list[string] ──
    files = record.get("files")
    if files is None:
        errors.append("files: missing (required)")
    elif not isinstance(files, list):
        errors.append(f"files: expected list, got {type(files).__name__}")
    elif len(files) == 0:
        errors.append("files: must not be empty")
    else:
        for i, item in enumerate(files):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"files[{i}]: must be non-empty string")

    # ── symbols: required, non-empty list[string] ──
    symbols = record.get("symbols")
    if symbols is None:
        errors.append("symbols: missing (required)")
    elif not isinstance(symbols, list):
        errors.append(f"symbols: expected list, got {type(symbols).__name__}")
    elif len(symbols) == 0:
        errors.append("symbols: must not be empty")
    else:
        for i, item in enumerate(symbols):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"symbols[{i}]: must be non-empty string")

    # ── language: required, non-empty string ──
    language = record.get("language")
    if language is None:
        errors.append("language: missing (required)")
    elif not isinstance(language, str):
        errors.append(f"language: expected string, got {type(language).__name__}")
    elif not language.strip():
        errors.append("language: must not be empty")

    # ── Optional fields: type checks ──
    # component: if present, must be string
    component = record.get("component")
    if component is not None and not isinstance(component, str):
        errors.append(f"component: expected string, got {type(component).__name__}")

    # list[string] optional fields
    for field_name in (
        "call_chains",
        "api_exports",
        "api_imports",
        "ipc_mechanism",
        "messages_send",
        "messages_receive",
        "shared_data",
    ):
        val = record.get(field_name)
        if val is not None:
            if not isinstance(val, list):
                errors.append(f"{field_name}: expected list, got {type(val).__name__}")
            else:
                for i, item in enumerate(val):
                    if not isinstance(item, str):
                        errors.append(f"{field_name}[{i}]: expected string, got {type(item).__name__}")

    return errors


def validate_quality(record: dict) -> list[str]:
    """Validate quality gate rules.

    Returns a list of warning/error messages.
    """
    warnings: list[str] = []
    content = record.get("content", "")

    # ── Code block completeness ──
    backtick_blocks = content.count("```")
    if backtick_blocks % 2 != 0:
        warnings.append("quality: unmatched ``` code blocks (odd count)")

    # ── Mermaid completeness ──
    mermaid_opens = len(re.findall(r"```mermaid", content, re.IGNORECASE))
    if mermaid_opens > 0:
        # Count closing ``` after each ```mermaid
        mermaid_closes = 0
        in_mermaid = False
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("```mermaid"):
                in_mermaid = True
            elif in_mermaid and stripped == "```":
                mermaid_closes += 1
                in_mermaid = False
        if mermaid_closes < mermaid_opens:
            warnings.append(f"quality: {mermaid_opens} mermaid blocks but only {mermaid_closes} closings")

    # ── Low confidence risk marker check ──
    confidence = record.get("confidence", 1.0)
    if isinstance(confidence, (int, float)) and float(confidence) < 0.6:
        content_lower = content.lower()
        has_marker = any(marker in content_lower for marker in RISK_MARKERS)
        if not has_marker:
            warnings.append("quality: confidence < 0.6 but content missing risk marker (risk/unverified/speculative/uncertain/inferred)")

    return warnings


def validate_file(file_path: Path) -> tuple[int, int, int]:
    """Validate a single JSONL file.

    Returns (total_lines, passed, failed).
    """
    print(f"Validating: {file_path.name}")

    total = 0
    passed = 0
    failed = 0
    seen_ids: set[str] = set()

    with open(file_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            # Parse JSON
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {line_no}: FAIL  [JSON parse error]")
                print(f"    - {e}")
                failed += 1
                continue

            if not isinstance(record, dict):
                print(f"  Line {line_no}: FAIL  [expected JSON object]")
                failed += 1
                continue

            # Schema validation
            errors = validate_record(record, line_no)

            # ID uniqueness check
            rec_id = record.get("id", "")
            if rec_id:
                if rec_id in seen_ids:
                    errors.append(f"id: '{rec_id}' is duplicate (already seen in this file)")
                seen_ids.add(rec_id)

            # Quality gate checks
            quality_warnings = validate_quality(record)

            all_issues = errors + quality_warnings
            id_display = record.get("id", f"line-{line_no}")

            if errors:
                print(f"  Line {line_no}: FAIL  [id={id_display}]")
                for err in all_issues:
                    print(f"    - {err}")
                failed += 1
            elif quality_warnings:
                print(f"  Line {line_no}: WARN  [id={id_display}]")
                for w in quality_warnings:
                    print(f"    - {w}")
                passed += 1
            else:
                print(f"  Line {line_no}: PASS  [id={id_display}]")
                passed += 1

    return total, passed, failed


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python validate_jsonl.py <file1.jsonl> [file2.jsonl ...]")
        return 2

    total_all = 0
    passed_all = 0
    failed_all = 0

    for arg in sys.argv[1:]:
        file_path = Path(arg)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 2
        if not file_path.is_file():
            print(f"Error: Not a file: {file_path}")
            return 2

        total, passed, failed = validate_file(file_path)
        total_all += total
        passed_all += passed
        failed_all += failed
        print()

    print(f"Summary: {total_all} lines, {passed_all} passed, {failed_all} failed")
    return 1 if failed_all > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
