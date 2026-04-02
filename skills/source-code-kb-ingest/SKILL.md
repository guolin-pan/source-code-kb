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

- Ingesting knowledge extracted by `source-code-kb-analyzer` into the vector store
- Loading pre-prepared JSONL knowledge files into the KB
- Batch-loading knowledge from multiple projects
