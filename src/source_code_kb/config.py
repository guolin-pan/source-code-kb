"""Configuration loading and management module.

Loads system configuration from a YAML file, supporting Embedding, vector store,
LLM, Reranker, and retrieval parameters.  A configuration file is required;
only VectorStoreConfig provides built-in defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── Sub-module configuration dataclasses ────────────────────────


@dataclass
class EmbeddingConfig:
    """Embedding model configuration.

    Attributes:
        model: Name of the embedding model deployed on Ollama.
        base_url: Base URL of the Ollama service.
    """

    # The Ollama model name used to generate vector embeddings for documents
    # and queries (e.g. "nomic-embed-text", "mxbai-embed-large").
    model: str
    # Points to the Ollama REST API (e.g. "http://localhost:11434").
    base_url: str


@dataclass
class VectorStoreConfig:
    """Vector store configuration.

    Attributes:
        type: Vector store backend type (chromadb).
        persist_dir: ChromaDB persistence directory path.
        collection: Default collection name.
    """

    # Only "chromadb" is supported today; the field exists to allow future
    # backends (e.g. FAISS, Milvus) without a config schema change.
    type: str = "chromadb"
    # Directory where ChromaDB persists its index to disk.  A relative path is
    # resolved from the working directory at runtime.
    persist_dir: str = "./data/chromadb"
    # ChromaDB collection name.  Each ingestion run can target a different
    # collection to isolate projects within the same persist_dir.
    collection: str = "default"


@dataclass
class LLMConfig:
    """Large Language Model (LLM) configuration.

    Uses an OpenAI-compatible API interface, supporting local Ollama,
    remote API endpoints, and other backends.

    Attributes:
        base_url: Base URL of the OpenAI-compatible API.
        model: Model name.
        api_key: API key (not required for Ollama).
    """

    # Endpoint exposing an OpenAI-compatible chat/completions API.
    # For local Ollama this is typically "http://localhost:11434/v1";
    # for cloud providers it is their standard API base URL.
    base_url: str
    model: str
    # api_key may be an empty string or a placeholder (e.g. "ollama") when
    # using a local Ollama instance that does not enforce authentication.
    # For cloud-hosted LLMs (OpenAI, Azure, etc.) a real key is required.
    api_key: str


@dataclass
class RerankerConfig:
    """Reranker model configuration.

    Obtains embeddings via the Ollama /api/embed endpoint,
    then computes cosine similarity for reranking.

    Attributes:
        model: Reranker model name (HuggingFace model ID for local mode,
               Ollama model name for remote mode).
        base_url: Ollama service URL (only used when mode is 'remote').
        top_n: Number of results to retain after reranking.
        mode: 'local' uses sentence-transformers CrossEncoder (recommended),
              'remote' uses Ollama embed API (approximate, not recommended).
    """

    # In "local" mode this is a HuggingFace model ID (e.g.
    # "BAAI/bge-reranker-v2-m3") downloaded and run via sentence-transformers.
    # In "remote" mode it is the Ollama model name used for the /api/embed call.
    model: str
    # Only relevant when mode == "remote"; ignored in local mode because the
    # CrossEncoder runs entirely in-process without an external service.
    base_url: str
    # How many top-scoring documents survive the reranking step and are passed
    # to the LLM context window.  Must be <= RetrievalConfig.top_k.
    top_n: int
    # Selects the reranking strategy:
    #   "local"  - sentence-transformers CrossEncoder (accurate, needs GPU/CPU);
    #   "remote" - Ollama embed endpoint + cosine similarity (faster but less
    #              accurate because generic embeddings are not trained for
    #              cross-encoder relevance scoring).
    mode: str


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration.

    Attributes:
        top_k: Number of candidate documents returned by vector search.
        rerank_top_n: Number of documents retained after reranking.
        use_multiquery: Whether to enable multi-query variants (improves recall).
        use_reranker: Whether to enable reranking (improves precision).
    """

    # Initial candidate pool size fetched from the vector store.  A larger
    # value gives the reranker more material to work with but costs more
    # embedding comparisons.
    top_k: int
    # Final number of documents kept after reranking.  Only meaningful when
    # use_reranker is True; otherwise top_k results go straight to the LLM.
    rerank_top_n: int
    # When True, the retrieval pipeline generates multiple rephrasings of the
    # user query (via the LLM) and merges their vector-search results.  This
    # improves recall at the cost of extra LLM and embedding calls.
    use_multiquery: bool
    # When True, a reranker model re-scores the initial top_k candidates and
    # keeps only rerank_top_n.  Improves precision by filtering out
    # false-positive vector matches.
    use_reranker: bool


@dataclass
class GraphConfig:
    """Knowledge-graph configuration.

    Attributes:
        persist_dir: Directory for graph persistence (pickle).
        max_hops: Maximum BFS traversal depth for graph retrieval.
        fusion_alpha: Weight for vector results in RRF (0-1).
                      Graph weight = 1 - fusion_alpha.
        rrf_k: Constant *k* in the RRF formula (typically 60).
    """

    persist_dir: str = "./data/graph"
    max_hops: int = 2
    fusion_alpha: float = 0.6
    rrf_k: int = 60


@dataclass
class AppConfig:
    """Top-level application configuration aggregating all sub-module configs."""

    embedding: EmbeddingConfig
    llm: LLMConfig
    reranker: RerankerConfig
    retrieval: RetrievalConfig
    # VectorStoreConfig is the only sub-config with built-in defaults, so it
    # can be omitted entirely from the YAML file and still produce a valid
    # AppConfig.  All other sections are required.
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


# ── Configuration loading utilities ─────────────────────────────


def _dict_to_dataclass[T](cls: type[T], data: dict[str, Any]) -> T:
    """Convert a dictionary to a dataclass instance, ignoring unknown fields.

    Args:
        cls: Target dataclass type.
        data: Raw dictionary data.

    Returns:
        An instance of *cls*.
    """
    # Introspect the dataclass to discover which field names it declares.
    # __dataclass_fields__ is a dict[str, Field] added by the @dataclass
    # decorator; extracting .name from each value gives the full set of
    # constructor-accepted keyword arguments.
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}

    # Filter out any keys in the raw YAML dict that are NOT declared fields.
    # This makes the loader forward-compatible: adding a new key to the YAML
    # file won't blow up an older version of the code, and typos are silently
    # ignored rather than raising a TypeError on the constructor call below.
    filtered = {k: v for k, v in data.items() if k in valid_keys}

    # Unpack the filtered dict as keyword arguments.  If a required field
    # (one with no default) is missing from the YAML, Python's normal
    # TypeError ("missing required argument") will propagate — the caller
    # gets a clear error pointing at the missing config key.
    return cls(**filtered)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load a YAML configuration file.

    Search order:
    1. If *config_path* is specified, load that file directly.
    2. Otherwise look for config.yaml / config.yml in the current directory.
    3. If none is found, raise an error (a config file is required).

    Args:
        config_path: Optional path to the configuration file.

    Returns:
        An AppConfig instance.

    Raises:
        FileNotFoundError: When no configuration file can be located.
    """
    # --- Step 1: Auto-detect configuration file when no explicit path given ---
    # Try well-known filenames in the current working directory.  ".yaml" is
    # checked before ".yml" by convention (both are valid YAML extensions).
    if config_path is None:
        candidates = [Path("config.yaml"), Path("config.yml")]
        for c in candidates:
            if c.exists():
                config_path = c
                break

    # --- Step 2: Validate that we actually have a usable file ---
    # Unlike many CLI tools that fall back to hard-coded defaults, this project
    # requires a config file because the embedding/LLM/reranker settings have
    # no sensible universal defaults (model names and URLs vary per deployment).
    if config_path is None or not Path(config_path).exists():
        raise FileNotFoundError(
            "Configuration file not found. Provide --config or place config.yaml in the current directory."
        )

    # --- Step 3: Read and parse the YAML file ---
    # yaml.safe_load() returns None for an empty file, so we coalesce to an
    # empty dict to avoid TypeErrors when calling .get() below.
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # --- Step 4: Build typed sub-configs and assemble the top-level object ---
    # Each YAML top-level key maps to one dataclass.  Missing sections become
    # empty dicts, which is fine for VectorStoreConfig (all fields have
    # defaults) but will raise TypeError for the others.
    return AppConfig(
        embedding=_dict_to_dataclass(EmbeddingConfig, raw.get("embedding", {})),
        vectorstore=_dict_to_dataclass(VectorStoreConfig, raw.get("vectorstore", {})),
        llm=_dict_to_dataclass(LLMConfig, raw.get("llm", {})),
        reranker=_dict_to_dataclass(RerankerConfig, raw.get("reranker", {})),
        retrieval=_dict_to_dataclass(RetrievalConfig, raw.get("retrieval", {})),
        graph=_dict_to_dataclass(GraphConfig, raw.get("graph", {})),
    )
