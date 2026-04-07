# Phase 1 Scan Prompt

## Per-Project Scan Instructions

You are scanning a software project to build a knowledge index. Your output is a structured scan report.

### Step 1: Directory Structure Inventory

List the top-level directory structure. For each directory:
- Purpose (infer from name and contents)
- Approximate file count
- Primary programming language(s)

### Step 2: Code Index

Identify and catalog:

1. **Entry Points**: main() functions, init routines, service start functions
2. **Public APIs**: Exported functions, header files with public interfaces
3. **Key Modules**: Core business logic files (not utility/helper files)
4. **IPC Endpoints**: Socket handlers, message processors, RPC service definitions
5. **Build System**: Makefile/CMake/build scripts that reveal module relationships

For each item found, record:
```json
{
  "type": "entry_point|public_api|key_module|ipc_endpoint|build_target",
  "file": "relative/path/to/file",
  "symbol": "function_or_class_name",
  "description": "brief description",
  "dependencies": ["other_files_or_modules"],
  "component": "subsystem_name",
  "api_exports": ["exported_func1", "exported_func2"],
  "api_imports": ["consumed_external_func1"],
  "messages_send": ["MSG_TYPE_A", "MSG_TYPE_B"],
  "messages_receive": ["MSG_TYPE_C"],
  "ipc_mechanism": ["socket", "shared_memory"],
  "shared_data": ["global_config_table", "device_registry"],
  "call_chains": ["main→init_subsystem→register_device"]
}
```

> **Graph-Aware Scanning**: The `component`, `api_exports`, `api_imports`, `messages_send`, `messages_receive`, `ipc_mechanism`, `shared_data`, and `call_chains` fields collected during scanning serve two purposes:
> 1. They feed directly into the Global Correlation Analysis (Phase 3.5) for cross-module dependency discovery
> 2. They guide the Deep Analysis phase (Phase 3) to ensure graph-critical relationship data is extracted into every JSONL chunk
>
> **Be specific and consistent**: Use exact symbol names (not descriptions), consistent component identifiers across files, and `→` notation for call chains. Every public header function should appear in `api_exports`; every cross-module function call should appear in `api_imports`.

### Step 3: Config Index

Identify all configuration files and their structure:

1. **Config Files**: JSON, XML, YAML, .conf, .ini, .properties, board profile files
2. **Config Schema**: For each config file, extract the key hierarchy
3. **Override Chains**: Identify default configs and board/product-specific overrides
4. **Config Consumers**: Which source files read these configs

For each config entry, record:
```json
{
  "file": "relative/path/to/config.json",
  "format": "json|xml|yaml|ini|custom",
  "keys": [
    {
      "key_path": "section.subsection.key",
      "type": "string|int|bool|list",
      "default_value": "...",
      "description": "inferred purpose"
    }
  ],
  "overridden_by": ["path/to/board_specific_config.json"],
  "consumers": ["source_file_that_reads_this.c"]
}
```

### Step 4: Cross-Project Dependencies

Look for:
- Shared header files or libraries
- IPC mechanisms (sockets, pipes, shared memory, message queues)
- RPC definitions (protobuf, thrift, gRPC)
- Database access patterns
- External service calls

### Output Format

Return a JSON object:
```json
{
  "project_name": "...",
  "root_path": "...",
  "languages": ["c", "python"],
  "file_count": 150,
  "code_index": [...],
  "config_index": [...],
  "cross_project_deps": [...],
  "suggested_topics": [
    {
      "title": "Topic Title",
      "description": "Brief description",
      "component": "subsystem-name",
      "key_files": ["file1.c", "file2.c"],
      "config_files": ["config.json"],
      "estimated_sections": ["Overview", "Internals", "Configuration"],
      "key_symbols": ["main_func", "init_func"],
      "api_exports": ["public_api_func"],
      "api_imports": ["external_dep_func"],
      "ipc_mechanism": ["socket"],
      "shared_data": ["global_table"]
    }
  ]
}
```

## Scanning Rules

1. **Breadth First**: Scan directory structure before diving into files
2. **Config = First Class**: Config files are as important as source code
3. **Skip Generated**: Ignore build output, generated code, vendored dependencies
4. **Large Files**: For files > 500 lines, read the first 50 lines + grep for key patterns
5. **Board Variants**: Look for directory patterns like `board_*/`, `platform/*/`, `profiles/`
