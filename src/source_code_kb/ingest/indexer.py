"""Indexer — vectorizes documents and stores them in ChromaDB.

Core responsibilities:
1. Create embedding functions (calling a remote Ollama service)
2. Manage the ChromaDB persistent vector store (create/open collections)
3. Batch-ingest documents (automatic deduplication)
4. Provide collection statistics and management functions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb
from langchain_ollama import OllamaEmbeddings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from source_code_kb.config import AppConfig

console = Console()

# Metadata fields stored in ChromaDB (excluding id, content, and vector)
METADATA_KEYS = [
    "domain", "topic", "section", "scope", "tags",
    "confidence", "source", "updated_at", "meta",
    "files", "symbols", "language", "component",
    "call_chains", "api_exports", "api_imports",
    "ipc_mechanism", "messages_send", "messages_receive", "shared_data",
]


def _create_embedding_fn(config: AppConfig) -> OllamaEmbeddings:
    """Create an Ollama embedding function."""
    console.print(
        f"[cyan]Loading embedding model:[/cyan] {config.embedding.model} "
        f"(base_url={config.embedding.base_url})"
    )
    return OllamaEmbeddings(
        model=config.embedding.model,
        base_url=config.embedding.base_url,
    )


def _connect_db(config: AppConfig) -> chromadb.ClientAPI:
    """Connect to a ChromaDB database."""
    return chromadb.PersistentClient(path=config.vectorstore.persist_dir)


def _build_embedding_text(content: str, metadata: dict) -> str:
    """Build enhanced text for embedding by appending code metadata to content.

    The original content is stored in ChromaDB's documents field unchanged.
    Only the embedding vector is generated from this enhanced text, improving
    recall for queries that mention function names, file paths, or call chains.
    """
    # Start with the original content as the primary semantic signal.
    parts = [content]

    # Append selected code-metadata fields so that their tokens influence the
    # embedding vector.  This is a retrieval-augmented trick: when a user query
    # mentions a specific symbol name ("init_scheduler") or file path
    # ("scheduler.c"), the embedding similarity will be higher because those
    # tokens are present in the vector even if the prose content only describes
    # the concept indirectly.
    #
    # Important: the *stored* document in ChromaDB remains the original
    # content (without these appendages), so that what the user sees in search
    # results is clean prose.  The enhancement only affects the vector space.
    for key in ("symbols", "files", "call_chains", "api_exports", "api_imports"):
        vals = metadata.get(key)
        if vals and isinstance(vals, list):
            # Format as "key: val1 val2 val3" — space-delimited so the
            # tokenizer treats each value as a separate token.
            parts.append(f"{key}: {' '.join(str(v) for v in vals)}")

    # Component name is a scalar string, not a list, so handle it separately.
    component = metadata.get("component")
    if component:
        parts.append(f"component: {component}")

    # Join all parts with newlines to keep them as distinct "paragraphs"
    # for the embedding model's tokenizer.
    return "\n".join(parts)


def _flatten_metadata(metadata: dict) -> dict:
    """Flatten metadata for ChromaDB storage.

    ChromaDB only supports str/int/float/bool metadata values.
    Lists (scope, tags) are joined as comma-separated strings.
    """
    flat = {}
    for key, value in metadata.items():
        # ChromaDB stores the document ID in a dedicated "ids" column, so skip
        # the "id" key here to avoid redundancy and potential conflicts.
        if key == "id":
            continue  # id is stored separately in ChromaDB

        # --- ChromaDB type restriction handling ---
        # ChromaDB metadata values must be one of: str, int, float, or bool.
        # Python lists (e.g. scope=["boardA","boardB"], tags=["init","boot"])
        # are not supported natively, so we flatten them to comma-separated
        # strings.  This is a lossy transformation, but the retriever can
        # split on commas to reconstruct the list when needed.
        if isinstance(value, list):
            flat[key] = ",".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            # Already a ChromaDB-compatible scalar type — store as-is.
            flat[key] = value
        elif value is None:
            # ChromaDB does not accept None; coerce to empty string so the
            # key is still present (useful for consistent schema across docs).
            flat[key] = ""
        else:
            # Catch-all for unexpected types (e.g. nested dicts that slipped
            # past validation) — convert to their string representation.
            flat[key] = str(value)
    return flat


def create_vectorstore(
    config: AppConfig, collection_name: str | None = None,
) -> chromadb.Collection:
    """Create or open a ChromaDB collection (does not perform any ingestion).

    Args:
        config: Application configuration.
        collection_name: Collection name (defaults to the value from config).

    Returns:
        A ChromaDB Collection instance.
    """
    client = _connect_db(config)
    name = collection_name or config.vectorstore.collection
    return client.get_or_create_collection(name)


def ingest_documents(
    config: AppConfig,
    documents: list[Document],
    collection_name: str | None = None,
) -> int:
    """Vectorize documents and store them in ChromaDB.

    Ingestion pipeline:
    1. Retrieve existing document IDs (for deduplication)
    2. Filter out duplicate documents (by id)
    3. Generate embedding vectors
    4. Write to the ChromaDB collection

    Args:
        config: Application configuration.
        documents: List of Documents to ingest.
        collection_name: Target collection name.

    Returns:
        Number of newly ingested documents.
    """
    if not documents:
        console.print("[yellow]No documents to ingest.[/yellow]")
        return 0

    client = _connect_db(config)
    name = collection_name or config.vectorstore.collection
    collection = client.get_or_create_collection(name)
    embed_fn = _create_embedding_fn(config)

    # --- Deduplication logic ---
    # Fetch every ID already present in the collection.  This is an O(N) call
    # on the collection size, but it is done once per ingest run rather than
    # once per document.  The IDs are loaded into a set for O(1) membership
    # checks below.
    existing_data = collection.get()
    existing_ids: set[str] = set(existing_data["ids"]) if existing_data["ids"] else set()

    # Walk the incoming documents and keep only those whose ID is not yet in
    # the collection.  Also add each accepted ID to existing_ids immediately so
    # that if the incoming batch itself contains duplicates (same ID appears
    # twice in the JSONL), only the first occurrence is kept.
    new_docs: list[Document] = []
    for doc in documents:
        doc_id = doc.metadata.get("id", "")
        if doc_id and doc_id not in existing_ids:
            new_docs.append(doc)
            existing_ids.add(doc_id)

    if not new_docs:
        console.print("[yellow]All documents already indexed (dedup by id).[/yellow]")
        return 0

    # --- Batch embedding and ChromaDB insertion ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Embedding & indexing {len(new_docs)} documents...", total=None)

        # Build two parallel lists:
        #  - contents: the original prose text stored as ChromaDB "documents"
        #    (what the user will see in search results).
        #  - embedding_texts: content enriched with code metadata (symbols,
        #    files, etc.) that the embedding model converts to vectors.
        # This separation ensures the vector captures code-aware semantics
        # while the stored document remains clean human-readable text.
        contents = [doc.page_content for doc in new_docs]
        embedding_texts = [
            _build_embedding_text(doc.page_content, doc.metadata)
            for doc in new_docs
        ]

        # Send the entire batch to the Ollama embedding service in one call.
        # This is far more efficient than per-document requests because the
        # model can process the batch with a single round-trip.
        vectors = embed_fn.embed_documents(embedding_texts)

        # Prepare the three parallel arrays that ChromaDB.add() expects:
        #   ids       — unique string identifier per document (for dedup)
        #   documents — the raw text stored alongside the vector
        #   embeddings — the dense vector produced by the embedding model
        #   metadatas — flattened key/value pairs for filtering and display
        ids = [doc.metadata.get("id", "") for doc in new_docs]
        metadatas = [_flatten_metadata(doc.metadata) for doc in new_docs]

        # ChromaDB's add() is an upsert-by-ID operation.  Because we already
        # filtered out existing IDs above, every ID here is new, so this is
        # effectively a pure insert.
        collection.add(
            ids=ids,
            documents=contents,
            embeddings=vectors,
            metadatas=metadatas,
        )

    console.print(
        f"[green]✓ Indexed {len(new_docs)} new documents[/green] "
        f"(skipped {len(documents) - len(new_docs)} duplicates)"
    )
    return len(new_docs)


# ── Collection management functions ──────────────────────────────────


def get_collection_stats(config: AppConfig, collection_name: str | None = None) -> dict:
    """Get basic statistics for a collection."""
    client = _connect_db(config)
    name = collection_name or config.vectorstore.collection

    try:
        collection = client.get_collection(name)
        count = collection.count()
        return {
            "exists": True,
            "name": name,
            "count": count,
            "sample_metadata_keys": METADATA_KEYS,
        }
    except Exception:
        return {"exists": False, "name": name, "count": 0}


def list_collections(config: AppConfig) -> list[str]:
    """List all collection names."""
    client = _connect_db(config)
    return [c.name for c in client.list_collections()]


def get_collection_topics(config: AppConfig, collection_name: str | None = None) -> list[str]:
    """Get a list of unique topic values in a collection."""
    client = _connect_db(config)
    name = collection_name or config.vectorstore.collection

    try:
        collection = client.get_collection(name)

        # Fetch only the metadatas column (no documents or embeddings) to
        # minimize memory usage — we only need the "topic" field.
        data = collection.get(include=["metadatas"])

        # Iterate over every document's metadata dict to collect unique topic
        # values.  ChromaDB does not support SQL-style DISTINCT queries, so we
        # must do the deduplication client-side with a set.  The `or []` guard
        # handles the case where the collection is empty (metadatas may be None).
        topics: set[str] = set()
        for m in (data.get("metadatas") or []):
            topic = m.get("topic", "")
            if topic:
                topics.add(topic)

        # Return a sorted list for deterministic output in CLI and tests.
        return sorted(topics)
    except Exception:
        # Collection may not exist yet — return empty rather than crashing.
        return []
