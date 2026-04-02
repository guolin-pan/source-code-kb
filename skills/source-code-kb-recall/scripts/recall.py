#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source Code KB Recall — retrieve knowledge chunks via REST API.

Configuration: ~/.source-code-kb/config.json
  { "api-base-url": "http://0.0.0.0:8765" }

Usage:
  recall.py -q <QUERY> [-q <QUERY2> ...] [--filters...] -o <OUTPUT.jsonl>
  recall.py -q <QUERY> --hierarchical --top-topics 3 --top-k-per-topic 5 -o <OUTPUT.jsonl>
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ── Configuration ───────────────────────────────────────────────

CONFIG_PATH = Path.home() / ".source-code-kb" / "config.json"
DEFAULT_BASE_URL = "http://0.0.0.0:8765"
DEFAULT_TIMEOUT = 120


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


# ── Result formatting ───────────────────────────────────────────


def _build_result(query: str, resp: dict) -> dict:
    """Build a JSONL-friendly result dict from one query's API response."""
    chunks = []
    for r in resp.get("results", []):
        meta = r.get("metadata", {})
        chunks.append({
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
            "metadata": meta,
        })
    return {"query": query, "total": resp.get("total", 0), "chunks": chunks}


def _build_hierarchical_result(query: str, resp: dict) -> dict:
    """Build result dict for hierarchical search."""
    chunks = []
    for r in resp.get("results", []):
        meta = r.get("metadata", {})
        chunks.append({
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
            "metadata": meta,
        })
    return {
        "query": query,
        "matched_topics": resp.get("matched_topics", []),
        "total": resp.get("total", 0),
        "chunks": chunks,
    }


# ── Search commands ─────────────────────────────────────────────


def _build_filter(args: argparse.Namespace) -> dict | None:
    """Build a SearchFilter dict from CLI args. Returns None if no filters set."""
    f: dict = {}
    if args.domain:
        f["domain"] = args.domain
    if args.topic:
        f["topic"] = args.topic
    if args.section:
        f["section"] = args.section
    if args.language:
        f["language"] = args.language
    if args.component:
        f["component"] = args.component
    if args.scope:
        f["scope"] = args.scope
    if args.tags:
        f["tags"] = args.tags
    if args.min_confidence is not None:
        f["min_confidence"] = args.min_confidence
    return f if f else None


def cmd_search(args: argparse.Namespace) -> None:
    queries = args.query
    if not queries:
        print("ERROR: provide at least one --query", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    search_filter = _build_filter(args)
    results: list[dict] = []

    for q in queries:
        print(f"  query: {q}")

        if args.hierarchical:
            body: dict = {
                "query": q,
                "collection": args.collection,
                "top_topics": args.top_topics,
                "top_k_per_topic": args.top_k_per_topic,
            }
            resp = api_post("/api/v1/search/hierarchical", body)
            if resp is None:
                print("    → FAILED (timeout or connection error)", file=sys.stderr)
                results.append({"query": q, "total": 0, "chunks": [], "error": "timeout or connection error"})
                continue
            result = _build_hierarchical_result(q, resp)
            results.append(result)
            topics = ", ".join(result.get("matched_topics", []))
            print(f"    → {result['total']} result(s), matched topics: {topics}")
        else:
            top_k = args.top_k
            if args.reranker:
                body = {
                    "query": q,
                    "collection": args.collection,
                    "top_k": top_k * 3,
                    "use_reranker": True,
                    "rerank_top_n": top_k,
                }
            else:
                body = {
                    "query": q,
                    "collection": args.collection,
                    "top_k": top_k,
                    "use_reranker": False,
                }
            if search_filter:
                body["filter"] = search_filter

            resp = api_post("/api/v1/search", body)
            if resp is None:
                print("    → FAILED (timeout or connection error)", file=sys.stderr)
                results.append({"query": q, "total": 0, "chunks": [], "error": "timeout or connection error"})
                continue
            result = _build_result(q, resp)
            results.append(result)
            print(f"    → {result['total']} result(s)")

    # Write JSONL output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} result(s) to {output_path}")


# ── CLI entry point ─────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="recall.py",
        description="Source Code KB Recall — retrieve knowledge chunks via REST API",
    )
    parser.add_argument("--query", "-q", action="append", metavar="QUESTION", required=True,
                        help="Query string (repeatable)")
    parser.add_argument("--output", "-o", required=True, metavar="JSONL_FILE",
                        help="Output JSONL file path")
    parser.add_argument("--collection", "-c", default="default",
                        help="Collection name (default: default)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results per query (default: 10)")
    parser.add_argument("--reranker", action="store_true", default=False,
                        help="Enable cross-encoder reranking")

    # Filter options
    filters = parser.add_argument_group("filters")
    filters.add_argument("--domain", help="Filter by domain (e.g., module-internals)")
    filters.add_argument("--topic", help="Filter by topic name")
    filters.add_argument("--section", help="Filter by section name")
    filters.add_argument("--language", help="Filter by programming language")
    filters.add_argument("--component", help="Filter by component/subsystem")
    filters.add_argument("--scope", action="append", metavar="SCOPE",
                         help="Filter by scope (repeatable)")
    filters.add_argument("--tags", action="append", metavar="TAG",
                         help="Filter by tag (repeatable)")
    filters.add_argument("--min-confidence", type=float, metavar="FLOAT",
                         help="Minimum confidence threshold (0.0-1.0)")

    # Hierarchical search options
    hier = parser.add_argument_group("hierarchical search")
    hier.add_argument("--hierarchical", action="store_true", default=False,
                      help="Use hierarchical search (match topics first, then search within)")
    hier.add_argument("--top-topics", type=int, default=3,
                      help="Number of topics to match (default: 3)")
    hier.add_argument("--top-k-per-topic", type=int, default=5,
                      help="Results per matched topic (default: 5)")

    args = parser.parse_args()
    cmd_search(args)


if __name__ == "__main__":
    main()
