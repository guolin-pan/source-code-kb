# 数据格式（JSONL Schema）

每个 `.jsonl` 文件的每一行是一个 JSON 对象，包含以下字段：

## 字段

| 字段         | 类型      | 必填 | 说明                                          |
| ------------ | --------- | ---- | --------------------------------------------- |
| `id`         | string    | 是   | 唯一 ID：`{domain}:{topic}:{section}:{hash8}` |
| `content`    | string    | 是   | 知识文本（最少 50 字符），用于向量化          |
| `domain`     | string    | 是   | 知识领域（如 `code-analysis`、`networking`）  |
| `topic`      | string    | 是   | 父主题名称                                    |
| `section`    | string    | 是   | 主题内的章节（如 `Overview`、`Internals`）    |
| `scope`      | list[str] | 是   | 适用范围（板卡、产品、版本）                  |
| `tags`       | list[str] | 是   | 关键词标签（至少 2 个）                       |
| `confidence` | float     | 是   | 置信度分数 0.0–1.0                            |
| `source`     | string    | 是   | 来源标识（项目名、文件路径、URL）             |
| `updated_at` | string    | 是   | 最后更新时间戳（ISO 8601）                    |
| `meta`       | object    | 否   | 可选扩展元数据（存储为 JSON 字符串）          |

## 去重机制

重复导入相同 `id` 的文档时，系统自动跳过。

## 示例

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

## ID 生成规则

`id` 字段遵循 `{domain}:{topic}:{section}:{hash8}` 格式，其中 `hash8` 是基于内容的哈希值的前 8 位。通过 API 导入时若省略 `id`，系统会根据记录字段自动生成。

## Meta 字段

可选的 `meta` 对象可存储任意扩展数据。常用字段包括：

- `files` — 源代码文件路径
- `symbols` — 代码符号（函数、类、变量）
- `call_chains` — 函数调用链（如 `"main→init→setup"`）
- `api_exports` / `api_imports` — 导出/导入的 API 符号
- `component` — 组件或模块名称
- `language` — 编程语言
