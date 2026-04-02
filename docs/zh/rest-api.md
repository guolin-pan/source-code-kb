# REST API

REST API（`sckb serve`）提供检索和数据导入端点。搜索结果默认按向量相似度排序；可按请求启用交叉编码器重排序。

## `GET /health`

```bash
curl http://localhost:8765/health
# → {"status": "ok"}
```

## `GET /api/v1/collections`

列出所有集合及文档数量。

```bash
curl http://localhost:8765/api/v1/collections
# → [{"name": "default", "count": 42}, ...]
```

## `GET /api/v1/collections/{name}/stats`

获取指定集合的详细统计信息。

```bash
curl http://localhost:8765/api/v1/collections/default/stats
# → {"name": "default", "exists": true, "count": 42, "sample_metadata_keys": [...]}
```

## `GET /api/v1/collections/{name}/topics`

列出集合中的所有主题。

```bash
curl http://localhost:8765/api/v1/collections/default/topics
# → {"collection": "default", "topics": ["Alarm Management", "Boot Initialization"], "total": 2}
```

## `POST /api/v1/search` — 语义搜索

基本搜索：

```bash
curl -X POST http://localhost:8765/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "alarm report mechanism", "top_k": 5}'
```

带元数据过滤：

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

支持的过滤字段：`domain`、`topic`、`section`、`scope`（列表）、`tags`（列表）、`min_confidence`。

启用重排序：

```bash
curl -X POST http://localhost:8765/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "alarm report mechanism", "top_k": 10, "use_reranker": true, "rerank_top_n": 5}'
```

> 默认不启用重排序（`use_reranker` 默认为 `false`）。设置 `"use_reranker": true` 启用重排序。`rerank_top_n` 控制重排序后保留的结果数。

## `POST /api/v1/search/hierarchical` — 分层搜索

先通过 Overview 章节找到相关主题，再在这些主题内检索详细内容。适用于跨多个主题的宽泛查询。

```bash
curl -X POST http://localhost:8765/api/v1/search/hierarchical \
  -H "Content-Type: application/json" \
  -d '{"query": "device management", "top_topics": 3, "top_k_per_topic": 5}'
```

## `POST /api/v1/ingest` — 导入 JSONL 数据

通过 API 导入知识记录到向量存储。支持结构化记录列表或 JSONL 原始文本。

结构化记录：

```bash
curl -X POST http://localhost:8765/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "default",
    "records": [
      {
        "content": "The platform initialization module performs hardware detection, driver loading, and service registration at system startup...",
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

响应：

```json
{"ingested": 1, "skipped": 0, "errors": []}
```

> 未通过 Schema 验证的记录会被拒绝，具体错误信息在 `errors` 字段中返回。重复记录（相同 `id`）会被静默跳过，计入 `skipped`。
