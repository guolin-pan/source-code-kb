---
name: source-code-kb-recall
description: Retrieve knowledge chunks from the Source Code Knowledge Base via REST API. Uses hybrid fusion retrieval (vector + knowledge graph RRF). Supports metadata filtering, cross-encoder reranking, and hierarchical search. Designed for iterative, multi-round retrieval to support deep analysis.
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

## Retrieval Architecture

The server uses two retrieval paths that are fused into a unified result set. Understanding both paths helps you craft more effective queries.

### Vector retrieval (semantic similarity)

Queries are embedded and matched against chunk embeddings in ChromaDB via cosine similarity. This path is effective for:
- Natural language descriptions ("how does the scheduler handle priority inversion")
- Broad topic exploration ("memory management overview")
- Fuzzy matching when exact symbol names are unknown

Metadata **filters** (`--domain`, `--component`, `--language`, etc.) are applied at this layer to narrow the vector search scope.

### Graph retrieval (structural traversal)

Code entities (symbol names, file paths, component names) are resolved to nodes in the knowledge graph, then a BFS traversal (up to `max_hops`, default 2) discovers related chunks through typed edges: `CALLS`, `EXPORTS_API`, `IMPORTS_API`, `IPC_SENDS`, `IPC_RECEIVES`, `SHARES_DATA`, `BELONGS_TO`, etc. This path is effective for:
- Symbol-centric queries ("start_kernel call chain") — finds callers, callees, and structurally related chunks
- Dependency questions ("what depends on module mm") — follows export/import edges across components
- Cross-component flows ("pci_scan_bus interaction with memory subsystem") — traverses IPC and shared data edges

#### Entity sources (two complementary methods)

Graph retrieval resolves entities from **two sources simultaneously**:

1. **Regex extraction** — automatic pattern matching on the query text detects CamelCase, snake_case identifiers and file path patterns. Effective when the query contains literal code identifiers.
2. **LLM-extracted entities** — during the Rewrite step (CLI/Chat), the LLM infers probable symbol names, file paths, and component names from natural language. This is the primary graph activation path for human-language questions.

The API also accepts **externally provided entities** via the `entities` request field (see API section below). These are merged with regex-extracted entities.

> **Key insight**: Without LLM or external entities, a natural language query like "how is a device registered on the bus" yields **0 graph hits**. With LLM entities (`udrv_bus_add_device`, `base/bus.c`, `bus`), the same query yields **10 graph hits**. Always provide entities when calling the API directly.

### Fusion (RRF merge)

Both retrieval paths run in parallel on the server. Results are merged via **Reciprocal Rank Fusion** (RRF):

```
score(d) = α / (k + rank_vector) + (1 − α) / (k + rank_graph)
```

- `α` = `fusion_alpha` (default 0.6) — vector weight; `1 − α` = graph weight
- `k` = `rrf_k` (default 60)

Result scores are RRF-fused scores, not pure cosine similarity.

> **Hierarchical search** (`--hierarchical`) uses the vector retriever only — it matches topics by their overview chunks first, then retrieves within matched topics. Graph fusion does not apply.

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
# Drill into a specific component — uses vector filter + graph traversal
python {this-skill-dir}/scripts/recall.py \
  -q "setup_arch memory initialization call chain" \
  --component mm --domain module-internals \
  --output round2.jsonl

# Search for error handling — vector similarity dominates (no code entity)
python {this-skill-dir}/scripts/recall.py \
  -q "kernel boot error handling panic" \
  --section "Error Handling" \
  --output round3.jsonl
```

### 4. Cross-module exploration

When results reveal cross-module dependencies (e.g. `api_imports` referencing another subsystem), recall the related module:

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "VFS file operations interface API" \
  --domain module-interface \
  --output round4.jsonl
```

### 5. During user interaction

As the user asks follow-up questions or requests deeper analysis, **recall again** with new queries derived from the conversation:

- User asks about a specific function → recall with that symbol name (triggers graph traversal)
- User asks about error scenarios → recall with `--section "Error Handling"` (vector filter)
- User asks to compare modules → recall both modules with targeted queries
- User asks about data flow → recall with `--domain end-to-end-flow`

**Do not rely solely on previously retrieved chunks.** Each new question may require fresh retrieval.

## Usage

### Basic search (vector + graph fusion)

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "kernel start_kernel initialization sequence" \
  --output results.jsonl
```

### Multiple queries in one call

Each query is sent independently; results are written as separate JSONL lines:

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "pci_scan_bus PCI device enumeration" \
  -q "CFS scheduler load balancing mechanism" \
  --output results.jsonl
```

### With metadata filters

Filters are applied to the vector retrieval path to narrow the search scope. Graph results from structural traversal are merged unfiltered:

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

When `--reranker` is enabled, the server first retrieves `top_k × 3` candidates (vector + graph fused), then a cross-encoder model re-scores and keeps `top_k`:

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "do_fork process creation call chain" \
  --reranker \
  --top-k 5 \
  --output results.jsonl
```

### Hierarchical search (vector-only)

Hierarchical search matches topics by overview chunks first, then retrieves within matched topics. This mode uses the vector retriever only — no graph fusion:

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

Filters are applied to the **vector retrieval** path and also post-applied to graph results. They constrain which chunks participate in search.

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

## Entities (Graph-Enhanced Retrieval)

The `entities` field provides pre-extracted code entities to the graph retriever, dramatically improving graph hit rates for natural-language queries.

```bash
python {this-skill-dir}/scripts/recall.py \
  -q "how does device registration work" \
  --symbol udrv_bus_add_device \
  --symbol drv_probe \
  --file-entity base/bus.c \
  --file-entity base/driver.c \
  --component-entity bus \
  --component-entity base \
  --output results.jsonl
```

Equivalent API call:

```json
{
  "query": "how does device registration work",
  "entities": {
    "symbols": ["udrv_bus_add_device", "drv_probe"],
    "files": ["base/bus.c", "base/driver.c"],
    "components": ["bus", "base"]
  }
}
```

| CLI Flag | API Field | Purpose | Example |
|----------|-----------|---------|---------|
| `--symbol NAME` | `entities.symbols` | Function, class, macro, struct names for graph node lookup | `--symbol udrv_bus_add_device` |
| `--file-entity PATH` | `entities.files` | Source file paths for graph node lookup | `--file-entity base/bus.c` |
| `--component-entity NAME` | `entities.components` | Component/subsystem name for graph node lookup | `--component-entity bus` |

All three flags are **repeatable** — use multiple times to specify multiple entities.

Entities are merged with regex-extracted entities from the query text — duplicates are deduplicated automatically.

## Output Format

Each line in the output JSONL file is a JSON object representing one query's results. The `score` field is an RRF-fused score (higher = more relevant), not a raw cosine similarity:

```json
{
  "query": "start_kernel initialization sequence",
  "total": 5,
  "graph_stats": {
    "vector_hits": 8,
    "graph_hits_raw": 5,
    "graph_hits_filtered": 4,
    "merged_total": 10,
    "graph_contributed": 4,
    "graph_only": 2,
    "graph_boosted": 2,
    "rank_improvements": [
      {"id": "...", "topic": "Kernel Boot", "source": "graph", "fused_rank": 3, "change": "new"},
      {"id": "...", "topic": "Init Flow", "source": "vector+graph", "vector_rank": 7, "fused_rank": 2, "change": "↑5"}
    ]
  },
  "chunks": [
    {
      "content": "start_kernel is the first C function called ...",
      "score": 0.82,
      "metadata": {
        "domain": "module-internals",
        "topic": "Kernel Boot Sequence",
        "section": "Initialization Flow",
        "component": "init",
        "language": "c",
        "files": ["init/main.c"],
        "symbols": ["start_kernel", "setup_arch"],
        "call_chains": ["start_kernel→setup_arch→mm_init"],
        "api_exports": ["start_kernel"],
        "tags": ["boot", "init"],
        "confidence": 0.95,
        "source": "linux-kernel",
        "retrieval_source": "vector+graph"
      }
    }
  ]
}
```

### Graph Stats Fields

The `graph_stats` object is present when the fusion retriever is active (graph data exists):

| Field | Description |
|-------|-------------|
| `vector_hits` | Chunks returned by vector search |
| `graph_hits_raw` | Chunks found by graph BFS before filtering |
| `graph_hits_filtered` | Graph chunks remaining after metadata filter |
| `merged_total` | Final result count after RRF merge |
| `graph_contributed` | Results with graph contribution (`graph_only` + `graph_boosted`) |
| `graph_only` | New results found exclusively via graph (not in vector results) |
| `graph_boosted` | Results present in both vector and graph (rank improved by graph) |
| `rank_improvements` | Per-result rank change details: `change` is `"new"` or `"↑N"` |

Each result's `metadata.retrieval_source` indicates provenance: `"vector"`, `"graph"`, or `"vector+graph"`.
```

For **hierarchical search**, the output additionally includes matched topic names:

```json
{
  "query": "kernel subsystem initialization",
  "matched_topics": ["Kernel Boot Sequence", "PCI Device Management"],
  "total": 10,
  "chunks": [...]
}
```

When a query fails (timeout or connection error), the result line contains an `error` field:

```json
{
  "query": "...",
  "total": 0,
  "chunks": [],
  "error": "timeout or connection error"
}
```

## When to Use

- **Semantic search**: broad natural language queries to explore topics (vector path dominant)
- **Structural search**: queries containing symbol names, file paths, or call chains (graph path dominant)
- **Entity-enhanced search**: pass `entities` for natural language queries to activate graph retrieval without requiring literal code names in the query
- **Filtered search**: narrow by domain, component, language, or confidence (applied to both vector and graph results)
- **Hierarchical search**: topic-level exploration then drill-down (vector-only)
- **Iterative multi-round recall**: analyze results → refine queries → recall again
- **During user interaction**: recall fresh data for each new question — do not assume previous results are sufficient

### Entity-enhanced recall best practice

When calling the API from an agent or script:
1. Parse the user's question to identify or infer likely code entities
2. Pass them in the `entities` field alongside the natural language `query`
3. Check `graph_stats` in the response to verify graph contribution
4. If `graph_only > 0`, the graph added documents that vector search alone missed
