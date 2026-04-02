# Source Code Knowledge Base

A RAG (Retrieval-Augmented Generation) knowledge base system for source code architecture analysis, built on **LangChain + LangGraph + ChromaDB + Ollama**.

Ingest structured knowledge from JSONL files, then query it with vector search, optional reranking, and LLM-generated answers — either through a full interactive CLI pipeline or a lightweight retrieval-only REST API.

## Quick Start

### Prerequisites

- **Python** >= 3.13
- **uv** (recommended Python package manager)
- **Ollama** service with `qwen3-embedding:latest` pulled (for embedding)
- **LLM service** (Ollama or any OpenAI API-compatible endpoint — required only for `sckb query` / `sckb chat`)

### Installation

```bash
cd /path/to/source-code-knowledge-base
cp config.yaml.example config.yaml   # Edit with your service URLs
uv sync
uv run sckb --help
```

### Try It Out

```bash
# 1. Ingest test data
uv run sckb ingest -i tests/test_data.jsonl

# 2. Check knowledge base status
uv run sckb stats

# 3. Single query
uv run sckb query "What initialization steps occur during system startup?"

# 4. Agent mode (auto-classify + multi-step reasoning)
uv run sckb query "Compare the alarm and configuration management systems" --mode agent

# 5. Interactive multi-turn chat
uv run sckb chat

# 6. Start REST API
uv run sckb serve
```

## Documentation

### English

- [Architecture](docs/en/architecture.md) — System overview, layer descriptions, project structure
- [Configuration](docs/en/configuration.md) — config.yaml reference, LLM/embedding setup
- [Data Format](docs/en/data-format.md) — JSONL schema, field descriptions, examples
- [CLI Usage](docs/en/cli-usage.md) — ingest, query, chat, serve, stats commands
- [REST API](docs/en/rest-api.md) — HTTP endpoints with curl examples
- [Agent Mode](docs/en/agent-mode.md) — LangGraph agent, query classification, processing paths
- [FAQ](docs/en/faq.md) — Troubleshooting and common questions

### 中文文档

- [系统架构](docs/zh/architecture.md) — 总体架构、各层说明、项目结构
- [配置说明](docs/zh/configuration.md) — config.yaml 参考、LLM/Embedding 配置
- [数据格式](docs/zh/data-format.md) — JSONL Schema、字段说明、示例
- [CLI 使用指南](docs/zh/cli-usage.md) — ingest、query、chat、serve、stats 命令
- [REST API](docs/zh/rest-api.md) — HTTP 端点及 curl 示例
- [Agent 模式](docs/zh/agent-mode.md) — LangGraph Agent、查询分类、处理路径
- [常见问题](docs/zh/faq.md) — 故障排查与常见问题

## License

MIT License
