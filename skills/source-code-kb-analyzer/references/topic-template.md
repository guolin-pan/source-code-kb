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

### Interface / Inter-Module (include when applicable)
- Exported APIs (api_exports): Functions this module provides to others
- Imported APIs (api_imports): External functions this module consumes
- IPC mechanisms: How this module communicates with others
- Messages: What messages this module sends/receives
- Shared data: Global or shared data this module reads/writes

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
