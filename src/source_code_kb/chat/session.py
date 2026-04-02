"""Chat session management — Maintains conversation history and context state.

Manages the following during interactive Q&A:
- Conversation history via LangChain InMemoryChatMessageHistory
- Active metadata filter criteria
- Query mode switching (simple / agent)
- Last retrieval chunks for follow-up reuse (R5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


@dataclass
class ChatSession:
    """Chat session manager.

    Maintains the complete state of an interactive conversation, including
    message history (via LangChain), active filter criteria, query mode,
    and last retrieval state.

    Attributes:
        active_filter: Currently active metadata filter criteria
        mode: Query mode ("simple" for direct retrieval / "agent" for Agent reasoning)
        max_history: Maximum number of history messages (older messages are trimmed automatically)
        last_chunks: Chunks from the most recent retrieval (for follow-up reuse)
        last_queries: Queries used in the most recent retrieval round
    """

    active_filter: dict[str, str] = field(default_factory=dict)
    mode: str = "simple"
    max_history: int = 20
    last_chunks: list[Any] = field(default_factory=list)
    last_queries: list[str] = field(default_factory=list)
    _chat_history: InMemoryChatMessageHistory = field(default_factory=InMemoryChatMessageHistory)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self._chat_history.add_user_message(content)
        self._trim_history()

    def add_assistant_message(self, content: str, sources: list[dict] | None = None) -> None:
        """Add an assistant reply to the history.

        Sources are stored as metadata on the AIMessage for later retrieval.
        """
        msg = AIMessage(content=content)
        if sources:
            # Piggyback on LangChain's additional_kwargs dict to attach source
            # references (file paths, chunk IDs, etc.) directly to the message
            # object.  This avoids a separate side-channel and lets
            # get_last_sources() find them by walking the message list.
            msg.additional_kwargs["sources"] = sources
        # Use add_message (not add_ai_message) so we can pass our pre-built
        # AIMessage that already carries additional_kwargs metadata.
        self._chat_history.add_message(msg)
        self._trim_history()

    def get_history_text(self, last_n: int = 6) -> str:
        """Get recent conversation history text for prompt injection.

        Formatted as "User: xxx\\nAssistant: xxx" text.
        """
        # Slice only the last `last_n` messages to keep the context window
        # small.  A smaller window reduces token cost and avoids confusing the
        # LLM with stale context from much earlier in the conversation.
        messages = self._chat_history.messages[-last_n:]
        parts: list[str] = []
        for msg in messages:
            # Render each message in a simple "Role: content" format that is
            # safe to inject directly into a prompt template.  Using plain
            # text (rather than structured chat objects) avoids prompt-injection
            # edge cases where role markers inside message content could
            # confuse the downstream LLM.
            if isinstance(msg, HumanMessage):
                parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"Assistant: {msg.content}")
        return "\n".join(parts)

    @property
    def history(self) -> list:
        """Access the underlying LangChain message list (for compatibility)."""
        return self._chat_history.messages

    def get_last_sources(self) -> list[dict[str, Any]]:
        """Get the reference source list from the most recent assistant reply."""
        # Walk the history in reverse so we find the *most recent* AI message
        # first.  Not every AI message carries sources (e.g., follow-up answers
        # that reuse prior chunks may omit them), so we keep scanning backward
        # until we find one that has a non-empty sources list.
        for msg in reversed(self._chat_history.messages):
            if isinstance(msg, AIMessage):
                sources = msg.additional_kwargs.get("sources", [])
                if sources:
                    return sources
        return []

    def set_filter(self, key: str, value: str) -> None:
        """Set a metadata filter criterion."""
        self.active_filter[key] = value

    def clear_filter(self) -> None:
        """Clear all active filter criteria."""
        self.active_filter.clear()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._chat_history.clear()

    def save_last_retrieval(self, chunks: list[Any], queries: list[str]) -> None:
        """Save the most recent retrieval state for follow-up reuse (R5)."""
        self.last_chunks = chunks
        self.last_queries = queries

    def is_follow_up(self, user_input: str, config: Any = None) -> bool:
        """Check if user input is a follow-up on the same topic (R5).

        When a config is provided, uses LLM to classify the intent.
        Falls back to keyword heuristic when LLM is unavailable.
        """
        # R5 feature: reuse previously retrieved chunks for follow-up questions
        # instead of performing a brand-new retrieval.  This reduces latency
        # and cost when the user asks successive questions about the same topic.

        # Guard: if there are no prior chunks to reuse, or the conversation is
        # too short to establish context, this cannot be a follow-up.
        if not self.last_chunks or len(self._chat_history.messages) < 2:
            return False

        # Explicit re-retrieval keywords always force a fresh search, even if
        # the LLM might otherwise classify the query as a follow-up.  This
        # gives the user an escape hatch to override follow-up detection.
        re_retrieve_keywords = {"search again", "retrieve", "re-search", "重新搜索", "再搜索"}
        lower_input = user_input.lower()
        for kw in re_retrieve_keywords:
            if kw in lower_input:
                return False

        # LLM-based classification: when a config object is available we can
        # call the LLM to decide whether the new query is a continuation of
        # the current conversation topic.  This is more accurate than simple
        # heuristics, especially for ambiguous or complex questions.
        if config is not None:
            try:
                from source_code_kb.generation.generator import classify_follow_up

                history = self.get_history_text(last_n=4)
                return classify_follow_up(user_input, history, config)
            except Exception:
                pass  # fall through to heuristic

        # Heuristic fallback: if the LLM is unavailable (no config, import
        # error, or runtime failure), conservatively assume follow-up.  This
        # avoids unnecessary retrievals but may occasionally reuse stale
        # chunks—an acceptable trade-off when no LLM is at hand.
        return True

    def _trim_history(self) -> None:
        """Trim history messages to stay within the max_history limit."""
        messages = self._chat_history.messages
        if len(messages) > self.max_history:
            # FIFO trimming: keep only the most recent max_history messages by
            # slicing off the oldest entries from the front of the list.  This
            # bounds memory usage in long-running sessions while preserving the
            # most relevant recent context for the LLM.
            self._chat_history.messages = messages[-self.max_history:]
