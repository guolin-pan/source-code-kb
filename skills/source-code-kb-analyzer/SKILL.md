---
name: source-code-kb-analyzer
description: AI-driven multi-project source code and configuration analysis.  Scans codebases (C/C++/Python/Go/Java/etc.) to produce structured knowledge: Topic Markdown documents + JSONL chunks for vector-store ingestion.
---

# Source Code KB Analyzer

## Quick Reference

| Item | Description |
|------|-------------|
| Input | One or more project directories (source code + configuration files) |
| Output | Topic Markdown documents + JSONL chunks |
| Interaction | Topic list confirmation — the only human-in-the-loop step |
| Indexing | Dual-index per project: code index + configuration index |
| Depth | Call chains, configuration override chains, cross-project dependencies |
| Config Awareness | JSON / XML / YAML / board profiles treated as first-class artifacts |

## Workflow

### Phase 1: Multi-Project Scan

For each project directory, launch a subagent to:

1. **Build Code Index**: List all source files, identify entry points, key modules, and public APIs
2. **Build Config Index**: Identify all config files (JSON, XML, YAML, .conf, board profiles), extract key-value structures and inheritance/override relationships
3. **Cross-Project Dependencies**: Detect IPC mechanisms, shared headers, RPC definitions, protobuf schemas

**Scan prompt**: See [references/scan-prompt.md](references/scan-prompt.md)

Each subagent returns a structured scan report (JSON). Merge all reports into a unified project map.

### Phase 2: Topic Discovery + User Confirmation

Based on merged scan reports:

1. AI identifies knowledge topics (functional modules, subsystems, cross-cutting concerns)
2. For each topic, determine:
   - Title and brief description
   - Involved projects and files
   - Related configuration files and keys
   - Estimated sections (Overview, Internals, Configuration, etc.)
3. Present the topic list to the user for confirmation — **this is the only interaction point**
4. The user may add, remove, or merge topics

### Phase 3: Deep Analysis per Topic

For each confirmed topic, run detailed analysis:

1. **Code Call Chain Analysis**: Trace execution paths, identify key functions, data flows
2. **Config Decision Chain Analysis**: For each config key, trace default → override → effective value across board variants
3. **Cross-Reference**: Link code behavior to config-driven decisions
4. **Evidence Collection**: Every conclusion must cite source file + line or config file + key path

**Analysis prompt**: See [references/analysis-prompt.md](references/analysis-prompt.md)

Output per topic:
- Markdown document (human-readable, following topic template)
- JSONL chunks (machine-readable)

### Phase 3.5: Global Correlation Analysis (Optional)

After per-topic deep analysis, this optional phase correlates information across all analyzed modules to discover cross-module dependencies and end-to-end flows.

- **Input**: Module summaries from Phase 3 — specifically `api_exports`, `api_imports`, `messages_send`, `messages_receive` from each topic
- **Analysis**: LLM deduces cross-module dependencies, end-to-end flows, and communication patterns by matching exports to imports and tracing message flows
- **Output**: Chunks with `domain=end-to-end-flow` (cross-module flow descriptions) and `domain=module-interface` (module interaction summaries)
- **Reference**: See [references/correlation-prompt.md](references/correlation-prompt.md)

### Phase 4: Structured Output

Generate output directory:
```
output/
  {project-group}/
    topics/
      T01_topic_name.md
      T02_topic_name.md
      ...
    knowledge.jsonl          # All JSONL chunks combined
    scan_report.json         # Phase 1 scan results
    README.md                # Index of all topics
```

After generating `knowledge.jsonl`, run validation using the bundled script:
```bash
python {this-skill-dir}/scripts/validate_jsonl.py output/{project-group}/knowledge.jsonl
```

## JSONL Schema

Every JSONL chunk must have these 13 required fields + optional fields:

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique ID, format: `{domain}:{topic}:{section}:{hash8}` |
| content | string | Knowledge text (only field vectorized). Min 50 chars |
| domain | string | One of: `module-internals`, `module-interface`, `end-to-end-flow`, `system-constraints`, `data-model`, `build-deploy` |
| topic | string | Topic name (e.g., "Kernel Boot Sequence") |
| section | string | Section within the topic (e.g., "Overview", "Internals", "Configuration") |
| scope | list[string] | Applicable boards/products (e.g., `["board-A", "board-B"]`) |
| tags | list[string] | At least 2 tags (e.g., `["boot", "init", "memory"]`) |
| confidence | float | 0.0-1.0, how confident in this analysis |
| source | string | Source identifier (e.g., project name or file path) |
| updated_at | string | ISO 8601 timestamp |
| files | list[string] | Related source file paths (REQUIRED) |
| symbols | list[string] | Key symbols: function names, class names, macros, global variables (REQUIRED) |
| language | string | Programming language (REQUIRED) |

#### Optional First-Class Fields

| Field | Type | Description |
|-------|------|-------------|
| component | string | Component/subsystem name |
| call_chains | list[string] | Call chains (e.g., `["main→init→setup"]`) |
| api_exports | list[string] | Exported/public APIs |
| api_imports | list[string] | Consumed external APIs |
| ipc_mechanism | list[string] | IPC mechanisms used |
| messages_send | list[string] | Messages sent |
| messages_receive | list[string] | Messages received |
| shared_data | list[string] | Shared data read/written |
| meta | object | Extension pocket field, stored as JSON string |

### meta Field for Code Analysis

The `meta` field is for extension data only. Recommended sub-fields in `meta`:
- `config_keys`: config keys involved
- `variants`: board-specific variations
- `cross_refs`: cross-project references

### Content Writing Convention (MANDATORY — EVERY RULE MUST BE FOLLOWED — EVERY CHUNK MUST SHOW EVIDENCE OF COMPLIANCE)

chunk content serves two equally important reader types. Both must be addressed through section specialization — never try to satisfy both in a single chunk:

- Troubleshooting AI: needs to reason from symptom to root cause and next investigation step
- Knowledge-seeking user: needs to understand background, principles, and architecture

#### Section Responsibilities (MANDATORY)

Each section has a fixed writing orientation. Deviating from this is a violation.

| Section | Writing Orientation | Serves |
|---------|---------------------|--------|
| Overview | Principles, architecture, component relationships, design intent | Background understanding, concept learning |
| Internals | Data flow, call chains, state machines, working mechanisms | Deep understanding, code reading |
| Configuration | Config key meanings, defaults, behavioral impact | Config tuning, parameter understanding |
| Error Handling | Fault symptoms, silent failures, degradation behavior | Troubleshooting, fault isolation |
| Interface | How modules interact (APIs, IPC, messages), what a module exports and imports | Integration understanding, dependency tracing |

#### MUST Include — Foundational Rules (apply to all chunks)

Every chunk MUST demonstrate at least the rules relevant to its section. Omitting applicable rules is a violation.

| Priority | Rule | Required Writing Pattern |
|----------|------|--------------------------|
| P1 | **Fault symptom to root cause mapping** | "When X is observed, it typically means Y because Z" |
| P1 | **Key constraints and boundary conditions** | "Maximum N", "requires X before Y", "does not support Z" |
| P2 | **Component dependency and startup ordering** | "A depends on B being ready; if B is absent, A silently returns empty data" |
| P2 | **Silent failure and degradation behavior** | "On error, the module returns the last cached value instead of an error code; callers cannot distinguish stale from fresh data" |
| P2 | **How configuration drives runtime behavior** | "When config key X is absent, default value N applies, which may cause Y" |
| P3 | **Cross-component interaction and async coupling** | "A notifies B asynchronously via MQ; a brief inconsistency window exists between the event and B's state update" |
| P3 | **Platform or board variant behavioral differences** | "On board type A, path X is taken; on board type B, path Y is taken instead" |

#### MUST Include — High-Value Extended Rules (apply when the information exists in the source)

These rules are not optional when the relevant information is present. Skipping them when applicable is a violation.

| Priority | Rule | Required Writing Pattern |
|----------|------|--------------------------|
| P1 | **Actual error log keywords** | Quote the exact string printed by the code; this enables direct matching against user-pasted logs |
| P1 | **Negative knowledge** | "X does NOT handle Y; Y is handled by Z" — eliminates wrong investigation directions immediately |
| P2 | **Typical pitfalls and common misuse** | "A common mistake is doing X; the correct approach is Y because Z" |
| P2 | **Verification and confirmation methods** | "To confirm whether X is the cause, check Y" |
| P2 | **Term disambiguation** | "In this project, X specifically means ..., which differs from its general usage" |
| P3 | **Design intent** | "X was chosen over Y because Z must be guaranteed in scenario W" |

#### MUST NOT Include

Violating any of these rules requires the chunk to be rewritten before it can be accepted.

| Forbidden | Reason |
|-----------|--------|
| Line numbers (e.g., `func.c:123`) | Source lines change with every commit; references become stale immediately |
| Bare function name lists without explanation | No semantic value; cannot guide reasoning |
| Code-location snapshots ("this function is at ...") | Describes where, not why or what happens |
| Vague module responsibility summaries ("this module handles ...") | Too abstract to match any concrete fault description |
| Emoji or special decorative symbols | Inconsistent tokenization; no semantic value |
| Hardcoded absolute paths | Become invalid when environment changes |

File names belong in the `files` field only. The `content` field must be meaningful without any file or line reference.

Project names should live in the `source` field. Use project names in `content` only when distinguishing between multiple co-existing projects; never as a mandatory prefix on every chunk.

#### Mandatory Self-Check Before Finalizing Each Chunk

Both questions must be answered before a chunk is accepted:

1. If a developer describes a failure symptom, does this chunk help an AI identify a probable cause or the next thing to check?
2. If a user wants to understand how this module works, does this chunk provide meaningful background knowledge?

Each question targets a different section. A chunk needs to answer only the question matching its section. If neither question can be answered yes, the chunk must be rewritten.

---

### Quality Gates (5 Rules)

1. **Semantic Completeness**: Content must be self-contained; code blocks must be properly closed
2. **Scope Declaration**: `scope` must list all applicable boards/products; use `["all"]` only when truly universal
3. **Traceability**: Every factual claim must be traceable to source file or config key (via `files`, `symbols`, and `meta`, not inline line numbers)
4. **Searchability**: `tags` >= 2 items; tags should cover the key concepts in the content
5. **Diagnostic Value**: Content must follow the Content Writing Convention above — principle/mechanism/failure-behavior oriented, never line-number oriented

## Topic Document Structure

See [references/topic-template.md](references/topic-template.md) for the full template.

Key sections per topic:
1. **Overview**: Purpose of the module/subsystem, entry points
2. **Internals**: Call chains, state machines, data flows
3. **Configuration**: Config keys, override chains, effective values per board variant
4. **Variant Matrix**: Board-by-board comparison table
5. **Dependencies**: Cross-project and inter-module dependencies
6. **Error Handling**: Error paths, fallback behavior, logging

## Subagent Strategy

- **Phase 1**: One subagent per project for parallel scanning
- **Phase 3**: One subagent per topic for parallel deep analysis
- Each subagent receives focused instructions and returns structured results
- Main agent merges and validates subagent outputs

## Large Project Handling

For projects with 500+ files:
1. Scan in breadth-first mode (directory structure → key files only)
2. Focus on entry points, public APIs, and config files
3. Deep-dive only into files relevant to confirmed topics
4. Use `file_search` and `grep_search` for targeted lookups instead of reading all files

## Section Splitting Rules

When generating JSONL chunks:
- Each section becomes one chunk (no mechanical splitting)
- If a section exceeds ~2000 chars, split at logical boundaries
- Each chunk must be semantically complete (no dangling references)
- The `section` field identifies the chunk within its topic
