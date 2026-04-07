# Global Correlation Analysis Prompt

## Purpose

After per-topic deep analysis (Phase 3), this optional phase correlates information across all analyzed modules to discover cross-module dependencies and end-to-end flows that are not visible from any single module's perspective.

## Input

You receive module summaries extracted from Phase 3 analysis. For each module/topic:
- Module name and component
- api_exports: Functions/APIs this module provides
- api_imports: Functions/APIs this module consumes from others
- messages_send: Messages/events this module emits
- messages_receive: Messages/events this module listens for
- shared_data: Global data structures or databases this module reads/writes
- ipc_mechanism: Communication mechanisms used

## Analysis Steps

### Step 1: Build Dependency Graph

For each module's api_imports, find which module's api_exports provides that function. Build a directed dependency graph: Module A -> Module B means A calls B's API.

### Step 2: Trace End-to-End Flows

For key user-facing operations (init, config change, alarm report, etc.):
1. Identify the entry point module
2. Trace through api_imports -> api_exports chains
3. Track message passing (messages_send -> messages_receive)
4. Note shared_data access patterns
5. Document the complete flow as a call chain across modules

### Step 3: Identify Communication Patterns

- Which modules communicate via direct API calls vs. message passing vs. shared data?
- Are there circular dependencies?
- What are the critical paths (modules that many others depend on)?

## Output Format

Generate JSONL chunks with:
- domain: "end-to-end-flow" for cross-module flow descriptions
- domain: "module-interface" for module interaction summaries
- Include all relevant files, symbols from participating modules
- call_chains should show the full cross-module chain

### Graph Field Requirements for Correlation Chunks

Correlation chunks are the **most important source of cross-module graph edges**. The knowledge graph relies on these chunks to connect components that were analyzed independently in Phase 3.

| Field                                | Requirement for Correlation Chunks                                                                                                                                                                       |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `component`                          | Set to the primary subsystem of the flow entry point, or use a synthetic name like `"cross-module"` for flows spanning many subsystems                                                                   |
| `call_chains`                        | **MUST contain the full cross-module call chain** using `→` notation. This is the primary source of cross-component CALLS edges in the graph. E.g., `["netif_receive_skb→ip_rcv→fib_lookup→tcp_v4_rcv"]` |
| `api_exports`                        | List symbols that are entry points *into* a module from the perspective of this flow                                                                                                                     |
| `api_imports`                        | List symbols that this flow *calls into* other modules — this creates cross-module IMPORTS_API edges                                                                                                     |
| `ipc_mechanism`                      | If the cross-module communication uses IPC instead of direct function calls, name the mechanism                                                                                                          |
| `messages_send` / `messages_receive` | For async flows, list the message types that bridge modules                                                                                                                                              |
| `shared_data`                        | If modules interact through shared data structures, name them                                                                                                                                            |
| `symbols`                            | **Must include ALL symbols from ALL participating modules** mentioned in the content                                                                                                                     |
| `files`                              | **Must include files from ALL participating modules**                                                                                                                                                    |

### Example JSONL chunk (end-to-end-flow):
```json
{
  "id": "end-to-end-flow:{topic}:{section}:{hash8}",
  "content": "The network packet receive path spans three subsystems: the network device driver calls netif_receive_skb() which hands the packet to the core networking layer. The IP layer processes the packet via ip_rcv() and ip_rcv_finish(), performing route lookup through fib_lookup(). Finally, the transport layer delivers the payload via tcp_v4_rcv() which queues the data to the appropriate socket receive buffer for userspace consumption.",
  "domain": "end-to-end-flow",
  "topic": "Network Packet Receive Path",
  "section": "Overview",
  "scope": ["all"],
  "tags": ["networking", "cross-module", "flow"],
  "confidence": 0.85,
  "source": "correlation-analysis",
  "updated_at": "2025-01-01T00:00:00Z",
  "files": ["net/core/dev.c", "net/ipv4/ip_input.c", "net/ipv4/tcp_ipv4.c"],
  "symbols": ["netif_receive_skb", "ip_rcv", "ip_rcv_finish", "fib_lookup", "tcp_v4_rcv"],
  "language": "c",
  "component": "net",
  "call_chains": ["netif_receive_skb→ip_rcv→ip_rcv_finish→fib_lookup→tcp_v4_rcv"],
  "api_exports": ["netif_receive_skb"],
  "api_imports": ["tcp_v4_rcv"],
  "ipc_mechanism": [],
  "messages_send": [],
  "messages_receive": [],
  "shared_data": ["sk_buff"],
  "meta": {}
}
```

## Rules

1. Only generate cross-module chunks for relationships actually evidenced in the module summaries
2. Set confidence based on evidence quality (direct API match = 0.9+, inferred from naming = 0.7-0.8)
3. Each chunk must be self-contained and readable without other chunks
