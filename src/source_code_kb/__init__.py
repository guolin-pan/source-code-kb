"""Source Code Knowledge Base (SCKB) — RAG knowledge base for source code architecture analysis.

Root package for the source code knowledge base system.
Built on LangChain + LangGraph + ChromaDB + Ollama.
"""

# ChromaDB internally uses sqlite3 and requires SQLite >= 3.35.0 for features
# like RETURNING and math functions.  Many system-bundled Python builds ship an
# older SQLite (especially on RHEL / CentOS / Amazon Linux), which causes
# ChromaDB to crash at import time.  The pysqlite3-binary wheel bundles a
# recent SQLite and exposes it as the "pysqlite3" package.
#
# The three lines below perform a module-level monkey-patch so that every
# subsequent "import sqlite3" in this process (including inside ChromaDB and
# its dependencies) transparently receives pysqlite3 instead.
import sys

# Dynamically import pysqlite3 by name string.  Using __import__() rather than
# a plain "import pysqlite3" avoids adding the name to this module's namespace
# and makes the intent — a one-shot side-effect import — explicit.
__import__("pysqlite3")

# Swap the entry in sys.modules: remove "pysqlite3" and place it under the
# "sqlite3" key.  After this, any module that does "import sqlite3" will get
# the pysqlite3 module object.  The .pop() ensures the old "pysqlite3" key is
# cleaned up so only the canonical "sqlite3" name remains in the module cache.
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

__version__ = "0.1.0"
