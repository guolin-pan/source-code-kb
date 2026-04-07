# Phase 3 Analysis Prompt

## Per-Topic Deep Analysis Instructions

You are performing deep analysis on a specific topic identified during the scan phase. Your goal is to produce a comprehensive Topic Markdown document and corresponding JSONL chunks.

### Input

You receive:
1. **Topic**: Title, description, and scope
2. **Relevant Files**: List of source files and config files from the scan report
3. **Project Map**: The merged scan report for cross-referencing

### Analysis Steps

#### Step 1: Read and Understand

For each relevant file:
1. Read the full file (or key sections for large files)
2. Identify functions, classes, and data structures related to this topic
3. Note call relationships (who calls whom)
4. Track data flow (input → processing → output)

#### Step 2: Trace Call Chains

Starting from entry points:
1. Follow the execution path step by step
2. Record the complete call chain using notation: `caller() → callee() → ...`
3. Note conditional branches (if/switch) that affect the flow
4. Identify where config values influence behavior

#### Step 3: Analyze Config Impact

For each config key relevant to this topic:
1. Find the default value (in default config file)
2. Find all override files (board-specific, environment-specific)
3. Determine the effective value for each board variant
4. Trace which code reads this config and how it affects behavior
5. Build the override chain: `default.json:key=X → board_A.json:key=Y → effective=Y`

#### Step 4: Build Variant Matrix

If multiple boards/products are involved:
1. Compare behavior across variants
2. Identify what differs (config values, enabled features, code paths)
3. Create a comparison table

#### Step 5: Generate Outputs

##### Markdown Document
Follow the [topic-template.md](topic-template.md) structure:
- Write each section with concrete evidence
- Include code snippets for key logic (keep snippets focused, 5-20 lines)
- Use Mermaid diagrams for state machines and complex flows
- Include the variant matrix table

##### JSONL Chunks
For each section of the document, generate a JSONL chunk:

```json
{
  "id": "{domain}:{topic}:{section}:{hash8}",
  "content": "Section content (self-contained, min 50 chars)",
  "domain": "module-internals",
  "topic": "Topic Name",
  "section": "Section Name",
  "scope": ["x86_64", "arm64"],
  "tags": ["tag1", "tag2", "tag3"],
  "confidence": 0.9,
  "source": "project-name",
  "updated_at": "2025-01-01T00:00:00Z",
  "files": ["init/main.c", "arch/x86/kernel/setup.c"],
  "symbols": ["start_kernel", "setup_arch"],
  "language": "c",
  "component": "init",
  "call_chains": ["start_kernel→setup_arch→mm_init"],
  "api_exports": ["start_kernel", "setup_arch"],
  "api_imports": ["mm_init"],
  "ipc_mechanism": [],
  "messages_send": [],
  "messages_receive": [],
  "shared_data": [],
  "meta": {
    "config_keys": ["CONFIG_SMP"],
    "variants": {"x86_64": "y"},
    "cross_refs": ["mm:memory-init"]
  }
}
```

> **IMPORTANT — Dual-Purpose Output**: Each JSONL chunk serves two systems simultaneously:
> 1. **Vector store**: The `content` field is embedded for semantic search. `symbols`, `files`, and `call_chains` are appended to the embedding text to improve code-query matching.
> 2. **Knowledge graph**: The fields `component`, `call_chains`, `api_exports`, `api_imports`, `ipc_mechanism`, `messages_send`, `messages_receive`, and `shared_data` are parsed to build an entity-relationship graph. The graph enables multi-hop traversal queries like "What depends on module X?" or "Trace the call path from A to B across modules."
>
> **You must populate both groups in a single pass.** There is no separate graph-extraction step — if graph fields are empty despite the content describing relationships, graph-based recall will fail for those relationships.

#### Graph Field Extraction Rules

When writing each JSONL chunk, follow these rules to ensure the knowledge graph is fully populated:

| Rule                                                       | Instruction                                                                                                                                                                                                                         |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Always set `component`**                                 | Every chunk must have a non-empty `component` value identifying the subsystem (e.g., `"mm"`, `"net-driver"`, `"scheduler"`). This is the anchor for all graph relationships.                                                        |
| **Extract `call_chains` from every execution path**        | If the content describes "A calls B which calls C", produce `"call_chains": ["A→B→C"]`. Use pure symbol names — no file names, no parentheses, no line numbers. Use `→` (Unicode arrow) as separator.                               |
| **Extract `api_exports` for public/exported interfaces**   | If this module defines functions callable by other modules (declared in public headers, exported via EXPORT_SYMBOL, registered as callbacks), list them in `api_exports`.                                                           |
| **Extract `api_imports` for consumed external interfaces** | If this module calls functions defined in OTHER modules, list those symbols in `api_imports`. This creates cross-component edges in the graph.                                                                                      |
| **Extract `ipc_mechanism` for communication patterns**     | If the module uses sockets, shared memory, D-Bus, message queues, pipes, RPC, or any IPC, name the mechanism.                                                                                                                       |
| **Extract `messages_send` / `messages_receive`**           | For event-driven or message-passing systems, list the message types or event names. Ensure every `messages_send` in one chunk has a corresponding `messages_receive` in another chunk (when both sender and receiver are analyzed). |
| **Extract `shared_data` for shared state**                 | If the module reads/writes global data structures, shared memory regions, databases, or caches used by multiple components, name them.                                                                                              |
| **`symbols` must be exhaustive**                           | Every function, class, macro, or global variable name mentioned in the `content` should appear in `symbols`. This is critical for both vector-search boosting and graph node creation.                                              |

#### Domain Selection Guide

Choose the `domain` value based on the chunk's focus:

| Domain               | When to Use                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------- |
| `module-internals`   | Internal workings of a single module: call chains, state machines, data flow, algorithms |
| `module-interface`   | How a module interacts with others: exported APIs, consumed APIs, contracts              |
| `end-to-end-flow`    | Cross-module flows spanning multiple components for a user-facing operation              |
| `system-constraints` | Limits, boundary conditions, timing constraints, resource budgets                        |
| `data-model`         | Data structures, database schemas, serialization formats, shared data definitions        |
| `build-deploy`       | Build system, compilation, linking, deployment, packaging                                |

#### Extracting Interface Information (MANDATORY for Graph)

When analyzing source code, **actively extract the following for every chunk** — these fields directly feed the knowledge graph:

- **component**: Identify which subsystem/module this code belongs to. Use a short, consistent identifier (e.g., `"mm"`, `"net-core"`, `"scheduler"`). The same component name must be used across all chunks describing the same subsystem.
- **api_exports**: Functions declared in public headers, exported symbols (`EXPORT_SYMBOL`), service endpoints, callback registrations that other modules can call. Use the exact symbol name.
- **api_imports**: External functions this module calls from other subsystems, library dependencies, service clients. Use the exact symbol name, not descriptions.
- **ipc_mechanism**: Communication patterns such as sockets, shared memory, message queues, RPC, D-Bus, pipes, signals. Name the specific mechanism.
- **messages_send** / **messages_receive**: Message types, event names, signal identifiers. Be consistent with naming across sender and receiver chunks.
- **shared_data**: Named global data structures, shared caches, databases, shared memory regions. Use the variable/structure name, not a description.
- **call_chains**: Every execution path described in the content must be captured as a call chain string using `→` notation (e.g., `"init→setup_device→register_irq"`). Pure symbol names only.

> **Why this matters**: The knowledge graph builds typed edges from these fields. Missing `api_exports` means the graph cannot link callers to callees across modules. Missing `component` means symbols float unanchored. Missing `call_chains` means multi-hop traversal ("trace the boot sequence from start to device probing") returns nothing.

### Analysis Rules

1. **Evidence Required**: Every factual claim must cite a file path or config key
2. **No Speculation Without Marking**: If uncertain, set confidence < 0.6 and include risk markers (e.g., \"speculative\", \"unverified\", \"inferred\") in the content
3. **Config Evidence Format**: `{file}:{key_path}={effective_value}`
4. **Cross-Project Links**: When behavior involves multiple projects, explain the interaction point
5. **Code Snippets**: Include only the most relevant lines; always specify the file path
6. **Self-Contained Chunks**: Each JSONL chunk's content must be readable without other chunks
7. **Mermaid Completeness**: All ```mermaid blocks must have matching closing ```
