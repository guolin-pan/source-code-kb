# Topic Document Template

## Document Structure

Each topic produces a Markdown document with the following sections.
Not all sections are required — include only those relevant to the topic.

## Frontmatter

```markdown
# T{nn}: {Topic Title}

- **Domain**: code-analysis
- **Projects**: {list of involved projects}
- **Key Files**: {top 5-10 most important source files}
- **Config Files**: {related config files}
- **Scope**: {applicable boards/products}
- **Last Updated**: {ISO 8601 date}
```

## Required Sections

### 1. Overview

- What this module or subsystem does (1–3 paragraphs)
- Entry-point functions and their roles
- Which projects contribute to this functionality
- Key data structures

### 2. Internals

- Detailed call chains using notation: `caller → callee → …`
- State machines (prefer Mermaid diagrams where helpful)
- Data-flow descriptions
- Key algorithms and their purpose
- Thread/process model, if applicable

### 3. Configuration

For each configuration key relevant to this topic:

```
Config Key: {full key path}
  File: {config file path}
  Type: {data type}
  Default: {default value}
  Override Chain: {default file} → {board-specific file} → effective
  Effective Value per Board:
    board-A: {value}
    board-B: {value}
  Impact: {what this key controls in the code}
  Code Reference: {function that reads this key}
```

### 4. Variant Matrix

| Aspect        | Board-A | Board-B  | Board-C |
| ------------- | ------- | -------- | ------- |
| Config key X  | value   | value    | value   |
| Feature Y     | enabled | disabled | enabled |
| Init sequence | A→B→C   | A→C      | A→B→C→D |

### 5. Dependencies

- Inter-module dependencies within the same project
- Cross-project dependencies (IPC, shared libraries, RPC)
- Build-time versus runtime dependencies
- Optional versus required dependencies

### 6. Error Handling

- Error-detection mechanisms
- Fallback and recovery behavior
- Logging and alerting
- Impact of failures on dependent modules

## Optional Sections (include when relevant)

### Interface / Inter-Module (include when applicable — CRITICAL for Graph)
- Exported APIs (api_exports): Functions this module provides to others — list every public function with its purpose
- Imported APIs (api_imports): External functions this module consumes — list every cross-module call
- IPC mechanisms: How this module communicates with others (sockets, shared memory, D-Bus, message queues, etc.)
- Messages: What messages/events this module sends and receives — use consistent names matching other modules
- Shared data: Global or shared data structures this module reads/writes — use the actual variable/struct name

> **Why this section matters for graph quality**: The knowledge graph builds cross-component edges from `api_exports`, `api_imports`, `ipc_mechanism`, `messages_send`, `messages_receive`, and `shared_data`. If this section is omitted or vague, the graph cannot link this module to its callers/callees, and queries like "What depends on this module?" will return incomplete results.

### Interface Definitions
- Public API functions with signatures
- IPC message formats
- RPC / protobuf definitions

### Performance Characteristics
- Timing constraints
- Resource-usage patterns
- Optimization notes

### Security Considerations
- Access-control mechanisms
- Input validation
- Sensitive-data handling

## JSONL Chunk Generation Rules

When converting a topic document to JSONL chunks:

1. Each section heading becomes the `section` field value
2. The `topic` field is the topic title (without T{nn} prefix)
3. Set `domain` to the appropriate value: `module-internals`, `module-interface`, `end-to-end-flow`, `system-constraints`, `data-model`, or `build-deploy`
4. Set `scope` to the boards/products listed in frontmatter
5. Generate at least 2 `tags` per chunk from key concepts
6. Set `confidence` based on evidence quality:
   - 1.0: Directly read from source code
   - 0.8-0.9: Inferred from code patterns with high confidence
   - 0.6-0.7: Inferred with moderate confidence
   - 0.3-0.5: Speculative or under-documented
7. Include `files`, `symbols`, `language` (required) and `meta` with `config_keys` when available
8. Each chunk's `content` must be self-contained (readable without other chunks)

### Graph Field Rules (apply to EVERY chunk)

9. **`component`** is REQUIRED — set it to the subsystem name for this topic (e.g., `"mm"`, `"net-driver"`, `"scheduler"`). Use the same identifier across all chunks of the same module.
10. **`call_chains`** — extract every execution path mentioned in the content as a `→`-separated symbol chain. If the content says "A calls B then C", produce `["A→B→C"]`. Pure symbol names only, no file names or parentheses.
11. **`api_exports`** — list every function this module provides to others (public headers, EXPORT_SYMBOL, registered callbacks). Use exact symbol names.
12. **`api_imports`** — list every function this module calls from other modules. Use exact symbol names.
13. **`ipc_mechanism`** — if the module uses any IPC, name the mechanism type (e.g., `"unix_socket"`, `"shared_memory"`, `"dbus"`).
14. **`messages_send`** / **`messages_receive`** — for event-driven systems, list message/event type identifiers.
15. **`shared_data`** — name any shared data structures, caches, or global state accessed by multiple components.
16. **`symbols`** must list ALL function/class/macro/variable names that appear in the `content` — this powers both vector-search boosting and graph node creation.
