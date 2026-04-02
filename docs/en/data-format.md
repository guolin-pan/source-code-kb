# Data Format (JSONL Schema)

Each line in a `.jsonl` file is a JSON object with the following fields:

## Fields

| Field        | Type      | Required | Description                                                  |
| ------------ | --------- | -------- | ------------------------------------------------------------ |
| `id`         | string    | Yes      | Unique ID: `{domain}:{topic}:{section}:{hash8}`              |
| `content`    | string    | Yes      | Knowledge text (min 50 chars). This is what gets vectorized. |
| `domain`     | string    | Yes      | Knowledge domain (e.g. `code-analysis`, `networking`)        |
| `topic`      | string    | Yes      | Parent topic name                                            |
| `section`    | string    | Yes      | Section within the topic (e.g. `Overview`, `Internals`)      |
| `scope`      | list[str] | Yes      | Applicability scope (boards, products, versions)             |
| `tags`       | list[str] | Yes      | Keyword tags (at least 2)                                    |
| `confidence` | float     | Yes      | Confidence score 0.0–1.0                                     |
| `source`     | string    | Yes      | Source identifier (project name, file path, URL)             |
| `updated_at` | string    | Yes      | Last update timestamp (ISO 8601)                             |
| `meta`       | object    | No       | Optional extension metadata (stored as JSON string)          |

## Deduplication

Re-ingesting a document with the same `id` is silently skipped.

## Example

```json
{
  "id": "code-analysis:Boot Initialization:Overview:a1b2c3d4",
  "content": "The platform initialization module performs hardware detection, driver loading, and service registration at system startup...",
  "domain": "code-analysis",
  "topic": "Boot Initialization",
  "section": "Overview",
  "scope": ["board-A", "board-B"],
  "tags": ["startup", "init"],
  "confidence": 0.95,
  "source": "platform",
  "updated_at": "2025-01-15T10:00:00Z",
  "meta": {"files": ["platform/main.c"], "symbols": ["plat_init", "hw_detect"]}
}
```

## ID Generation

The `id` field follows the pattern `{domain}:{topic}:{section}:{hash8}`, where `hash8` is the first 8 characters of a content-based hash. If `id` is omitted during API ingestion, it is auto-generated from the record fields.

## Meta Field

The optional `meta` object can store arbitrary extension data. Common usage includes:

- `files` — Source file paths
- `symbols` — Code symbols (functions, classes, variables)
- `call_chains` — Function call chains (e.g. `"main→init→setup"`)
- `api_exports` / `api_imports` — Exported/imported API symbols
- `component` — Component or module name
- `language` — Programming language
