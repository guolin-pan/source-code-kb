# 配置说明

系统配置文件为项目根目录下的 `config.yaml`。首次使用时复制模板：

```bash
cp config.yaml.example config.yaml
```

## 配置参考

```yaml
# Embedding 模型（Ollama）
embedding:
  model: "qwen3-embedding:latest"
  base_url: "http://your-ollama-host:11434"

# 向量存储（ChromaDB）
vectorstore:
  type: "chromadb"
  persist_dir: "./data/chromadb"   # 本地持久化目录
  collection: "default"            # 默认集合名称

# LLM（兼容 OpenAI API — 仅 sckb query / sckb chat 使用）
llm:
  base_url: "http://your-llm-host:11434/v1"
  model: "your-model-name"
  api_key: "your-api-key"

# 重排序器（本地 CrossEncoder — 仅 sckb query / sckb chat 使用）
reranker:
  model: "BAAI/bge-reranker-v2-m3"   # HuggingFace 模型 ID；本地加载
  base_url: "http://your-ollama-host:11434"
  top_n: 5
  mode: "local"                       # "local" = sentence-transformers CrossEncoder

# 检索参数
retrieval:
  top_k: 10             # 向量搜索候选数量
  rerank_top_n: 5       # 重排序后保留的文档数
  use_multiquery: true   # 启用基于 LLM 的查询扩展
  use_reranker: true     # 启用 CrossEncoder 重排序（仅 CLI 流程）
```

## 字段说明

### `embedding`

| 字段       | 必填 | 说明                    |
| ---------- | ---- | ----------------------- |
| `model`    | 是   | Ollama embedding 模型名 |
| `base_url` | 是   | Ollama 服务地址         |

### `vectorstore`

| 字段          | 必填 | 默认值              | 说明           |
| ------------- | ---- | ------------------- | -------------- |
| `type`        | 否   | `"chromadb"`        | 向量存储类型   |
| `persist_dir` | 否   | `"./data/chromadb"` | 本地持久化目录 |
| `collection`  | 否   | `"default"`         | 默认集合名称   |

### `llm`

| 字段       | 必填 | 说明                                    |
| ---------- | ---- | --------------------------------------- |
| `base_url` | 是   | 兼容 OpenAI API 的服务端点 URL          |
| `model`    | 是   | 模型名称                                |
| `api_key`  | 是   | API 密钥（本地 Ollama 使用 `"ollama"`） |

> LLM 仅用于 `sckb query` 和 `sckb chat`。REST API（`sckb serve`）不需要 LLM。

### `reranker`

| 字段       | 必填 | 说明                                 |
| ---------- | ---- | ------------------------------------ |
| `model`    | 是   | HuggingFace CrossEncoder 模型 ID     |
| `base_url` | 是   | Ollama 服务地址（remote 模式使用）   |
| `top_n`    | 是   | 重排序后保留的结果数                 |
| `mode`     | 是   | `"local"` 使用 sentence-transformers |

### `retrieval`

| 字段             | 必填 | 说明                               |
| ---------------- | ---- | ---------------------------------- |
| `top_k`          | 是   | 向量搜索候选数量                   |
| `rerank_top_n`   | 是   | 重排序后保留的文档数               |
| `use_multiquery` | 是   | 启用 LLM 查询扩展                  |
| `use_reranker`   | 是   | 启用 CrossEncoder 重排序（仅 CLI） |

## 使用不同的 LLM

系统支持任何兼容 OpenAI API 的后端：

```yaml
# OpenAI
llm:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"
  api_key: "sk-..."

# 本地 Ollama
llm:
  base_url: "http://localhost:11434/v1"
  model: "llama3"
  api_key: "ollama"
```

## 切换 Embedding 模型

```yaml
embedding:
  model: "your-model-name"
  base_url: "http://your-ollama-host:11434"
```

> 切换模型后需**重新导入所有数据**。不同模型产生不同维度的向量，现有 ChromaDB 数据将不兼容。

## CLI 配置覆盖

所有 CLI 命令接受 `--config /path/to/config.yaml` 参数覆盖默认配置路径。
