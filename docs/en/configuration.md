# Configuration

The system configuration file is `config.yaml` in the project root. Copy `config.yaml.example` to get started:

```bash
cp config.yaml.example config.yaml
```

## Configuration Reference

```yaml
# Embedding model (Ollama)
embedding:
  model: "qwen3-embedding:latest"
  base_url: "http://your-ollama-host:11434"

# Vector store (ChromaDB)
vectorstore:
  type: "chromadb"
  persist_dir: "./data/chromadb"   # Local persistence directory
  collection: "default"            # Default collection name

# LLM (OpenAI API-compatible — used by sckb query / sckb chat only)
llm:
  base_url: "http://your-llm-host:11434/v1"
  model: "your-model-name"
  api_key: "your-api-key"

# Reranker (local CrossEncoder — used by sckb query / sckb chat only)
reranker:
  model: "BAAI/bge-reranker-v2-m3"   # HuggingFace model ID; loaded locally
  base_url: "http://your-ollama-host:11434"
  top_n: 5
  mode: "local"                       # "local" = sentence-transformers CrossEncoder

# Retrieval parameters
retrieval:
  top_k: 10             # Vector search candidate count
  rerank_top_n: 5       # Documents retained after reranking
  use_multiquery: true   # Expand query with LLM-generated variants
  use_reranker: true     # Apply local CrossEncoder reranker (CLI pipeline only)
```

## Field Details

### `embedding`

| Field      | Required | Description                 |
| ---------- | -------- | --------------------------- |
| `model`    | Yes      | Ollama embedding model name |
| `base_url` | Yes      | Ollama service URL          |

### `vectorstore`

| Field         | Required | Default             | Description                 |
| ------------- | -------- | ------------------- | --------------------------- |
| `type`        | No       | `"chromadb"`        | Vector store type           |
| `persist_dir` | No       | `"./data/chromadb"` | Local persistence directory |
| `collection`  | No       | `"default"`         | Default collection name     |

### `llm`

| Field      | Required | Description                               |
| ---------- | -------- | ----------------------------------------- |
| `base_url` | Yes      | OpenAI API-compatible endpoint URL        |
| `model`    | Yes      | Model name                                |
| `api_key`  | Yes      | API key (use `"ollama"` for local Ollama) |

> The LLM is only needed for `sckb query` and `sckb chat`. The REST API (`sckb serve`) does not require it.

### `reranker`

| Field      | Required | Description                                      |
| ---------- | -------- | ------------------------------------------------ |
| `model`    | Yes      | HuggingFace model ID for CrossEncoder            |
| `base_url` | Yes      | Ollama service URL (for remote mode)             |
| `top_n`    | Yes      | Number of results to retain after reranking      |
| `mode`     | Yes      | `"local"` for sentence-transformers CrossEncoder |

### `retrieval`

| Field            | Required | Description                                       |
| ---------------- | -------- | ------------------------------------------------- |
| `top_k`          | Yes      | Number of vector search candidates                |
| `rerank_top_n`   | Yes      | Documents retained after reranking                |
| `use_multiquery` | Yes      | Enable LLM-based query expansion                  |
| `use_reranker`   | Yes      | Enable CrossEncoder reranking (CLI pipeline only) |

## Using a Different LLM

The system supports any OpenAI API-compatible backend:

```yaml
# OpenAI
llm:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"
  api_key: "sk-..."

# Local Ollama
llm:
  base_url: "http://localhost:11434/v1"
  model: "llama3"
  api_key: "ollama"
```

## Switching Embedding Model

```yaml
embedding:
  model: "your-model-name"
  base_url: "http://your-ollama-host:11434"
```

> After switching models, **re-ingest all data**. Different models produce different vector dimensions; the existing ChromaDB data becomes incompatible.

## CLI Config Override

All CLI commands accept `--config /path/to/config.yaml` to override the default location.
