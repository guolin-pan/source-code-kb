#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source Code KB Ingest — ingest JSONL data via REST API.

Configuration: ~/.source-code-kb/config.json
  { "api-base-url": "http://0.0.0.0:8765" }

Usage:
  ingest.py --file <JSONL-FILE> [--file <JSONL-FILE2> ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ── Configuration ───────────────────────────────────────────────

CONFIG_PATH = Path.home() / ".source-code-kb" / "config.json"
DEFAULT_BASE_URL = "http://0.0.0.0:8765"
DEFAULT_TIMEOUT = 120

VALID_DOMAINS = {
    "module-internals",
    "module-interface",
    "end-to-end-flow",
    "system-constraints",
    "data-model",
    "build-deploy",
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_base_url() -> str:
    return load_config().get("api-base-url", DEFAULT_BASE_URL).rstrip("/")


# ── HTTP helper ─────────────────────────────────────────────────


def api_post(path: str, body: dict, timeout: int = DEFAULT_TIMEOUT) -> dict | None:
    """POST JSON to the KB API. Returns parsed response or None on error."""
    url = f"{get_base_url()}{path}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except TimeoutError:
        print(f"ERROR: request timed out after {timeout}s — {url}", file=sys.stderr)
        return None
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: HTTP {e.code} from {url}\n{detail}", file=sys.stderr)
        return None
    except URLError as e:
        print(f"ERROR: cannot connect to {url} — {e.reason}", file=sys.stderr)
        return None


# ── Schema validation ───────────────────────────────────────────


def validate_record(record: dict, line_no: int) -> list[str]:
    """Validate a single JSONL record. Returns error list (empty if valid)."""
    errors: list[str] = []

    # Required string fields
    for field in ("content", "domain", "topic", "section", "source", "updated_at"):
        val = record.get(field)
        if val is None:
            errors.append(f"L{line_no} {field}: missing")
        elif not isinstance(val, str):
            errors.append(f"L{line_no} {field}: expected string, got {type(val).__name__}")
        elif not val.strip():
            errors.append(f"L{line_no} {field}: empty")

    content = record.get("content", "")
    if isinstance(content, str) and content.strip() and len(content) < 50:
        errors.append(f"L{line_no} content: too short ({len(content)} chars, min 50)")

    domain = record.get("domain", "")
    if isinstance(domain, str) and domain and domain not in VALID_DOMAINS:
        errors.append(f"L{line_no} domain: '{domain}' not in {sorted(VALID_DOMAINS)}")

    # Required list/string fields: files, symbols, language
    files = record.get("files")
    if files is None:
        errors.append(f"L{line_no} files: missing")
    elif not isinstance(files, list) or len(files) == 0:
        errors.append(f"L{line_no} files: must be non-empty list")

    symbols = record.get("symbols")
    if symbols is None:
        errors.append(f"L{line_no} symbols: missing")
    elif not isinstance(symbols, list) or len(symbols) == 0:
        errors.append(f"L{line_no} symbols: must be non-empty list")

    language = record.get("language")
    if language is None:
        errors.append(f"L{line_no} language: missing")
    elif not isinstance(language, str):
        errors.append(f"L{line_no} language: expected string, got {type(language).__name__}")
    elif not language.strip():
        errors.append(f"L{line_no} language: empty")

    scope = record.get("scope")
    if scope is None:
        errors.append(f"L{line_no} scope: missing")
    elif not isinstance(scope, list) or len(scope) == 0:
        errors.append(f"L{line_no} scope: must be non-empty list")

    tags = record.get("tags")
    if tags is None:
        errors.append(f"L{line_no} tags: missing")
    elif not isinstance(tags, list) or len(tags) < 2:
        errors.append(f"L{line_no} tags: must be list with >= 2 items")

    confidence = record.get("confidence")
    if confidence is None:
        errors.append(f"L{line_no} confidence: missing")
    elif not isinstance(confidence, (int, float)):
        errors.append(f"L{line_no} confidence: expected number")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"L{line_no} confidence: {confidence} out of [0.0, 1.0]")

    updated_at = record.get("updated_at", "")
    if isinstance(updated_at, str) and updated_at.strip():
        iso_re = r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
        if not re.match(iso_re, updated_at):
            errors.append(f"L{line_no} updated_at: not valid ISO 8601")

    # Optional list fields
    for opt_field in ("call_chains", "api_exports", "api_imports",
                      "ipc_mechanism", "messages_send", "messages_receive", "shared_data"):
        val = record.get(opt_field)
        if val is not None and not isinstance(val, list):
            errors.append(f"L{line_no} {opt_field}: expected list, got {type(val).__name__}")

    component = record.get("component")
    if component is not None and not isinstance(component, str):
        errors.append(f"L{line_no} component: expected string, got {type(component).__name__}")

    meta = record.get("meta")
    if meta is not None and not isinstance(meta, dict):
        errors.append(f"L{line_no} meta: expected object, got {type(meta).__name__}")

    return errors


def parse_jsonl_lines(lines: list[str], source_label: str) -> list[dict]:
    """Parse and validate JSONL lines. Returns only valid records."""
    valid: list[dict] = []
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  SKIP {source_label} line {line_no}: invalid JSON — {e}", file=sys.stderr)
            continue

        if not isinstance(record, dict):
            print(f"  SKIP {source_label} line {line_no}: expected object, got {type(record).__name__}", file=sys.stderr)
            continue

        errs = validate_record(record, line_no)
        if errs:
            for err in errs:
                print(f"  SKIP {source_label}: {err}", file=sys.stderr)
            continue

        valid.append(record)
    return valid


# ── Ingest command ──────────────────────────────────────────────


def cmd_ingest(args: argparse.Namespace) -> None:
    files = args.file or []

    if not files:
        print("ERROR: provide at least one --file", file=sys.stderr)
        sys.exit(1)

    all_records: list[dict] = []

    for fpath in files:
        p = Path(fpath)
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        with open(p, encoding="utf-8") as f:
            lines = f.readlines()
        records = parse_jsonl_lines(lines, str(p))
        total_lines = sum(1 for ln in lines if ln.strip())
        print(f"  {p.name}: {len(records)} valid record(s) (of {total_lines} total)")
        all_records.extend(records)

    if not all_records:
        print("No valid records to ingest.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSending {len(all_records)} record(s) to API ...")
    resp = api_post("/api/v1/ingest", {"records": all_records})

    if resp is None:
        print("Ingest failed (timeout or connection error). No records ingested.", file=sys.stderr)
        sys.exit(1)

    print(f"  ingested: {resp.get('ingested', 0)}")
    print(f"  skipped (dup): {resp.get('skipped', 0)}")
    server_errors = resp.get("errors", [])
    if server_errors:
        print("  server-side errors:")
        for e in server_errors:
            print(f"    - {e}")


# ── CLI entry point ─────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ingest.py",
        description="Source Code KB Ingest — ingest JSONL data via REST API",
    )
    parser.add_argument("--file", "-f", action="append", metavar="JSONL_FILE",
                        required=True, help="Path to a JSONL file (repeatable)")

    args = parser.parse_args()
    cmd_ingest(args)


if __name__ == "__main__":
    main()
