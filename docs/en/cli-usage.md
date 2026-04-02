# CLI Usage

The `sckb` CLI provides commands for data ingestion, querying, chatting, serving, and statistics.

## `sckb ingest` — Data Ingestion

Vectorize JSONL data and store it in ChromaDB.

```bash
# Import a single file (uses default collection from config.yaml)
uv run sckb ingest -i data/knowledge.jsonl

# Import an entire directory (recursively scans all .jsonl files)
uv run sckb ingest -i data/

# Specify a collection name
uv run sckb ingest -i data.jsonl -c my_collection
```

## `sckb query` — Single Q&A

Retrieve relevant chunks, apply reranking (if enabled), then generate an answer via LLM.

```bash
# Basic query (simple mode)
uv run sckb query "What is alarm suppression?"

# Agent mode (auto-classify + multi-step reasoning)
uv run sckb query "Compare the alarm and configuration systems" --mode agent

# Specify result count
uv run sckb query "startup process" --top-k 10

# Use a specific collection
uv run sckb query "driver loading" -c my_collection
```

## `sckb chat` — Interactive Multi-turn Chat

Contextual conversation with persistent history within a session. Each turn performs retrieval, optional reranking, and LLM generation.

```bash
# Start with default collection
uv run sckb chat

# Start in agent mode
uv run sckb chat --mode agent

# Use a specific collection
uv run sckb chat -c my_collection
```

### Chat Slash Commands

| Command               | Description                                  |
| --------------------- | -------------------------------------------- |
| `/sources`            | Show sources cited in the last answer        |
| `/filter topic=XXX`   | Set a metadata filter for subsequent queries |
| `/filter clear`       | Clear all active filters                     |
| `/mode simple\|agent` | Switch query mode                            |
| `/clear`              | Clear conversation history                   |
| `/stats`              | Show collection statistics                   |
| `/help`               | Display command help                         |
| `/quit`               | Exit chat                                    |

## `sckb serve` — Start REST API

Starts a lightweight HTTP API for programmatic access.

```bash
# Default: port 8765, all interfaces
uv run sckb serve

# Custom port and bind host
uv run sckb serve --port 9000 --host 127.0.0.1
```

Interactive API docs (Swagger UI): `http://localhost:8765/docs`

See [REST API](rest-api.md) for endpoint details.

## `sckb stats` — Knowledge Base Statistics

```bash
# List all collections with document counts
uv run sckb stats

# Show details for a specific collection
uv run sckb stats -c default
```

## Common Options

| Option                 | Description                 |
| ---------------------- | --------------------------- |
| `--config PATH`        | Override config file path   |
| `-c, --collection`     | Specify collection name     |
| `--top-k N`            | Number of retrieval results |
| `--mode simple\|agent` | Query processing mode       |
| `--help`               | Show command help           |
