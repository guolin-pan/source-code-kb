"""Knowledge graph node and edge type definitions."""

from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    """Types of entities that can appear as graph nodes."""

    SYMBOL = "symbol"            # Function, variable, or class name
    FILE = "file"                # Source file path
    COMPONENT = "component"      # Logical component / subsystem
    DATA_STRUCTURE = "data_structure"  # Shared data structure
    IPC_CHANNEL = "ipc_channel"  # IPC mechanism or message channel
    CHUNK = "chunk"              # A knowledge-base chunk (document)


class EdgeType(str, Enum):
    """Types of relationships between graph nodes."""

    CALLS = "calls"                  # Symbol A calls symbol B
    EXPORTS_API = "exports_api"      # Component exports an API symbol
    IMPORTS_API = "imports_api"      # Component imports an API symbol
    DEFINED_IN = "defined_in"        # Symbol is defined in a file
    BELONGS_TO = "belongs_to"        # Symbol / file belongs to a component
    CONTAINS_SYMBOL = "contains_symbol"  # Chunk contains / describes a symbol
    CONTAINS_FILE = "contains_file"      # Chunk references a file
    IPC_SENDS = "ipc_sends"          # Component sends via IPC channel
    IPC_RECEIVES = "ipc_receives"    # Component receives via IPC channel
    SHARES_DATA = "shares_data"      # Component accesses shared data structure
