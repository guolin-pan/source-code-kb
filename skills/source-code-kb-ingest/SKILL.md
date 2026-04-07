---
name: source-code-kb-ingest
description: Ingest JSONL knowledge chunks into the Source Code Knowledge Base via REST API. Validates records against the JSONL schema before sending.
---

# Source Code KB Ingest

## Quick Reference

| Item | Description |
|------|-------------|
| Purpose | Ingest JSONL knowledge chunks into the KB vector store |
| API | `POST /api/v1/ingest` |
| Config | `~/.source-code-kb/config.json` — `{ "api-base-url": "http://host:port" }` |
| Script | `python {this-skill-dir}/scripts/ingest.py` |
| Input | JSONL files |

## Configuration

The API base URL is read from `~/.source-code-kb/config.json`:

```json
{
  "api-base-url": "http://0.0.0.0:8765"
}
```

If the file does not exist, defaults to `http://0.0.0.0:8765`.

## Usage

### Ingest from file

```bash
python {this-skill-dir}/scripts/ingest.py --file knowledge.jsonl
```

### Ingest multiple files

```bash
python {this-skill-dir}/scripts/ingest.py \
  -f project_a/knowledge.jsonl \
  -f project_b/knowledge.jsonl
```

## JSONL Schema

Every record must have these required fields:

| Field | Type | Description |
|-------|------|-------------|
| content | string | Knowledge text (min 50 chars) |
| domain | string | One of: `module-internals`, `module-interface`, `end-to-end-flow`, `system-constraints`, `data-model`, `build-deploy` |
| topic | string | Topic name |
| section | string | Section within the topic |
| scope | list[string] | Applicable boards/products |
| tags | list[string] | At least 2 tags |
| confidence | float | 0.0-1.0 |
| source | string | Source identifier |
| updated_at | string | ISO 8601 timestamp |
| files | list[string] | Related source file paths |
| symbols | list[string] | Key symbols (functions, classes, macros) |
| language | string | Programming language |

Optional fields: `id`, `component`, `call_chains`, `api_exports`, `api_imports`, `ipc_mechanism`, `messages_send`, `messages_receive`, `shared_data`, `meta`.

### Graph-Critical Fields

The following fields are used to build an entity-relationship graph for cross-component retrieval. **They are strongly recommended whenever the information exists in the source data** — omitting them degrades graph-based recall:

| Field | Graph Usage |
|-------|-------------|
| `component` | Anchors all relationships to a subsystem — **always set this** |
| `call_chains` | Creates CALLS edges between symbols (e.g., `["A→B→C"]`) |
| `api_exports` | Creates EXPORTS_API edges (component provides these APIs) |
| `api_imports` | Creates IMPORTS_API edges (component consumes these APIs) |
| `ipc_mechanism` | Creates IPC channel nodes and edges |
| `messages_send` / `messages_receive` | Creates message-passing edges between components |
| `shared_data` | Creates shared data structure nodes and access edges |

The ingest pipeline handles both vector indexing and graph construction in a single pass — no separate graph-import step is needed.

> **Entity matching at recall time**: During retrieval, an LLM infers probable code entities (symbols, files, components) from the user's question and matches them against graph nodes built from these fields. Use exact code identifiers — not descriptions — so that graph nodes can be resolved. For example, `"symbols": ["udrv_bus_add_device"]` not `"symbols": ["bus add function"]`.

## Validation

Records are validated client-side before sending to the API:

- All required fields must be present and non-empty
- `domain` must be one of the 6 valid values
- `content` minimum 50 characters
- `tags` must have at least 2 items
- `confidence` must be in [0.0, 1.0]
- `updated_at` must be valid ISO 8601
- Optional list fields must be lists if present
- `component` must be string if present
- `meta` must be dict if present

Invalid records are skipped with error messages. Only valid records are sent to the API.

## When to Use

- Ingesting knowledge extracted by `source-code-kb-analyzer` into the vector store and knowledge graph
- Loading pre-prepared JSONL knowledge files into the KB
- Batch-loading knowledge from multiple projects
