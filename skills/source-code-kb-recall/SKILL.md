---
name: source-code-kb-recall
description: Retrieve knowledge chunks from the Source Code Knowledge Base via REST API. Supports full metadata filtering (domain, topic, section, language, component, scope, tags, confidence) and optional cross-encoder reranking. Designed for iterative, multi-round retrieval to support deep analysis.
---

# Source Code KB Recall

## Quick Reference

| Item | Description |
|------|-------------|
| Purpose | Retrieve knowledge chunks from the KB for analysis |
| API | `POST /api/v1/search` and `POST /api/v1/search/hierarchical` |
| Config | `~/.source-code-kb/config.json` — `{ "api-base-url": "http://host:port" }` |
| Script | `python {this-skill-dir}/scripts/recall.py` |
| Output | JSONL file with query results (one JSON object per line) |

## Configuration

The API base URL is read from `~/.source-code-kb/config.json`:

```json
{
  "api-base-url": "http://0.0.0.0:8765"
}
```

If the file does not exist, defaults to `http://0.0.0.0:8765`.

## Iterative Recall Strategy

**A single recall is often not enough.** Use the following strategy to get comprehensive, accurate results:

### 1. Initial broad recall

Start with a broad query to understand what knowledge is available:

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "kernel boot initialization overview" \
  --top-k 10 --output round1.jsonl
```

### 2. Analyze and refine

Read the results. Identify:
- Which topics and components are covered
- Which symbols and files appear in the results
- What gaps or unanswered aspects remain

### 3. Follow-up targeted recalls

Based on analysis, issue more specific queries with filters to fill gaps:

```bash
# Drill into a specific component found in round 1
python {this-skill-dir}/scripts/recall.py \
  -q "setup_arch memory initialization call chain" \
  --component mm --domain module-internals \
  --output round2.jsonl

# Search for error handling not covered in round 1
python {this-skill-dir}/scripts/recall.py \
  -q "kernel boot error handling panic" \
  --section "Error Handling" \
  --output round3.jsonl
```

### 4. Cross-module exploration

When analysis reveals cross-module dependencies, recall the related modules:

```bash
# Found api_imports referencing the VFS layer — recall that module
python {this-skill-dir}/scripts/recall.py \
  -q "VFS file operations interface API" \
  --domain module-interface \
  --output round4.jsonl
```

### 5. During user interaction

As the user asks follow-up questions or requests deeper analysis, **recall again** with new queries derived from the conversation:

- User asks about a specific function → recall with that symbol name
- User asks about error scenarios → recall with `--section "Error Handling"`
- User asks to compare modules → recall both modules with targeted queries
- User asks about data flow → recall with `--domain end-to-end-flow`

**Do not rely solely on previously retrieved chunks.** Each new question may require fresh retrieval to get the most relevant and up-to-date knowledge.

## Usage

### Basic search

```bash
python {this-skill-dir}/scripts/recall.py \
  --query "kernel start_kernel initialization sequence" \
  --output results.jsonl
```

### Multiple queries in one call

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "pci_scan_bus PCI device enumeration" \
  -q "CFS scheduler load balancing mechanism" \
  --output results.jsonl
```

### With filters

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "memory management initialization flow" \
  --domain module-internals \
  --topic "Memory Subsystem Init" \
  --language c \
  --component mm \
  --min-confidence 0.8 \
  --output results.jsonl
```

### With reranker

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "do_fork process creation call chain" \
  --reranker \
  --top-k 5 \
  --output results.jsonl
```

### Hierarchical search

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "kernel subsystem initialization" \
  --hierarchical \
  --top-topics 3 \
  --top-k-per-topic 5 \
  --output results.jsonl
```

### Specify collection

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "query text" \
  --collection my_collection \
  --output results.jsonl
```

## Available Filters

| Filter | Flag | Type | Description |
|--------|------|------|-------------|
| domain | `--domain` | string | One of: `module-internals`, `module-interface`, `end-to-end-flow`, `system-constraints`, `data-model`, `build-deploy` |
| topic | `--topic` | string | Exact topic name match |
| section | `--section` | string | Exact section name match |
| language | `--language` | string | Programming language (e.g., `c`, `python`, `go`) |
| component | `--component` | string | Component/subsystem name |
| scope | `--scope` | string (repeatable) | Applicable boards/products |
| tags | `--tags` | string (repeatable) | Classification tags |
| min_confidence | `--min-confidence` | float | Minimum confidence threshold (0.0-1.0) |

## Output Format

Each line in the output JSONL file is a JSON object:

```json
{
  "query": "the query text",
  "total": 5,
  "chunks": [
    {
      "content": "chunk text...",
      "score": 0.85,
      "metadata": {
        "domain": "module-internals",
        "topic": "Kernel Boot Sequence",
        "section": "Overview",
        "component": "init",
        "language": "c",
        "files": ["init/main.c"],
        "symbols": ["start_kernel", "setup_arch"],
        "call_chains": ["start_kernel→setup_arch→mm_init"],
        "tags": ["boot", "init"],
        "confidence": 0.95,
        "source": "linux-kernel"
      }
    }
  ]
}
```

For hierarchical search, the output includes matched topics:

```json
{
  "query": "query text",
  "matched_topics": ["Kernel Boot Sequence", "PCI Device Management"],
  "total": 10,
  "chunks": [...]
}
```

## When to Use

- Retrieving knowledge chunks for AI-driven code architecture analysis
- Searching by component, symbol, file path, or domain for targeted retrieval
- Filtering chunks by language, confidence, or tags for precision
- **Iterative multi-round recall**: analyze results, identify gaps, recall again with refined queries
- **During user interaction**: recall fresh data whenever the user asks new questions or requests deeper analysis — do not assume previous results are sufficient
