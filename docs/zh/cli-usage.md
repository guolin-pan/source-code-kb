# CLI 使用指南

`sckb` CLI 提供数据导入、查询、对话、REST 服务和统计等命令。

## `sckb ingest` — 数据导入

将 JSONL 数据向量化并存入 ChromaDB。

```bash
# Import a single file (uses default collection from config.yaml)
uv run sckb ingest -i data/knowledge.jsonl

# Import an entire directory (recursively scans all .jsonl files)
uv run sckb ingest -i data/

# Specify a collection name
uv run sckb ingest -i data.jsonl -c my_collection
```

## `sckb query` — 单次问答

检索相关文档，应用重排序（如启用），然后通过 LLM 生成答案。

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

## `sckb chat` — 多轮对话

在会话内维护对话历史的上下文对话。每轮执行：检索 → 可选重排序 → LLM 生成。

```bash
# Start with default collection
uv run sckb chat

# Agent mode
uv run sckb chat --mode agent

# Use a specific collection
uv run sckb chat -c my_collection
```

### 对话斜杠命令

| 命令                  | 说明                         |
| --------------------- | ---------------------------- |
| `/sources`            | 显示上次回答引用的来源       |
| `/filter topic=XXX`   | 设置后续查询的元数据过滤条件 |
| `/filter clear`       | 清除所有过滤条件             |
| `/mode simple\|agent` | 切换查询模式                 |
| `/clear`              | 清空对话历史                 |
| `/stats`              | 显示集合统计信息             |
| `/help`               | 显示帮助                     |
| `/quit`               | 退出对话                     |

## `sckb serve` — 启动 REST 服务

启动轻量级 HTTP API，供程序化访问。

```bash
# Default: port 8765, all interfaces
uv run sckb serve

# Custom port and bind host
uv run sckb serve --port 9000 --host 127.0.0.1
```

交互式 API 文档（Swagger UI）：`http://localhost:8765/docs`

详细端点说明请参阅 [REST API](rest-api.md)。

## `sckb stats` — 知识库统计

```bash
# List all collections with document counts
uv run sckb stats

# Show details for a specific collection
uv run sckb stats -c default
```

## 通用选项

| 选项                   | 说明             |
| ---------------------- | ---------------- |
| `--config PATH`        | 覆盖配置文件路径 |
| `-c, --collection`     | 指定集合名称     |
| `--top-k N`            | 检索结果数量     |
| `--mode simple\|agent` | 查询处理模式     |
| `--help`               | 显示命令帮助     |
