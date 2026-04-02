# REST API

The REST API (`sckb serve`) provides retrieval and ingestion endpoints. Search results are ordered by vector similarity score by default; optional cross-encoder reranking can be enabled per request.

## `GET /health`

```bash
curl http://localhost:8765/health
# → {"status": "ok"}
```

## `GET /api/v1/collections`

List all collections with document counts.

```bash
curl http://localhost:8765/api/v1/collections
# → [{"name": "default", "count": 42}, ...]
```

## `GET /api/v1/collections/{name}/stats`

Get detailed statistics for a collection.

```bash
curl http://localhost:8765/api/v1/collections/default/stats
# → {"name": "default", "exists": true, "count": 42, "sample_metadata_keys": [...]}
```

## `GET /api/v1/collections/{name}/topics`

List all topics in a collection.

```bash
curl http://localhost:8765/api/v1/collections/default/topics
# → {"collection": "default", "topics": ["Alarm Management", "Boot Initialization"], "total": 2}
```

## `POST /api/v1/search` — Semantic Search

Basic search:

```bash
curl -X POST http://localhost:8765/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "alarm report mechanism", "top_k": 5}'
```

With metadata filter:

```bash
curl -X POST http://localhost:8765/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "initialization",
    "collection": "default",
    "top_k": 5,
    "filter": {
      "topic": "Boot Initialization",
      "min_confidence": 0.8
    }
  }'
```

Supported filter fields: `domain`, `topic`, `section`, `scope` (list), `tags` (list), `min_confidence`.

With reranking:

```bash
curl -X POST http://localhost:8765/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "alarm report mechanism", "top_k": 10, "use_reranker": true, "rerank_top_n": 5}'
```

> By default, no reranking is applied (`use_reranker` defaults to `false`). Set `"use_reranker": true` to enable the configured cross-encoder reranker. `rerank_top_n` controls how many results to retain after reranking.

## `POST /api/v1/search/hierarchical` — Hierarchical Search

Finds relevant topics via Overview sections first, then retrieves details within those topics. Useful for broad questions that span multiple topics.

```bash
curl -X POST http://localhost:8765/api/v1/search/hierarchical \
  -H "Content-Type: application/json" \
  -d '{"query": "device management", "top_topics": 3, "top_k_per_topic": 5}'
```

## `POST /api/v1/ingest` — Ingest JSONL Data

Ingest knowledge records into the vector store via API. Accepts either a list of structured records or raw JSONL text.

With inline records:

```bash
curl -X POST http://localhost:8765/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "default",
    "records": [
      {
        "content": "The platform initialization module performs hardware detection...",
        "domain": "code-analysis",
        "topic": "Boot Initialization",
        "section": "Overview",
        "scope": ["board-A"],
        "tags": ["startup", "init"],
        "confidence": 0.95,
        "source": "platform",
        "updated_at": "2025-01-15T10:00:00Z"
      }
    ]
  }'
```

With raw JSONL text:

```bash
curl -X POST http://localhost:8765/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "default",
    "jsonl_content": "{\"content\":\"...\",\"domain\":\"code-analysis\",...}"
  }'
```

Response:

```json
{"ingested": 1, "skipped": 0, "errors": []}
```

> Records that fail schema validation are rejected; specific errors are returned in the `errors` field. Duplicate records (same `id`) are silently skipped and counted in `skipped`.
