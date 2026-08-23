"""rag: Obsidian vault の取り込み・パース・分割・索引 (L3)。

``llmkit`` (L2) が再エクスポートする公開シンボルだけを使う。``llmkit`` の
サブモジュール (``llmkit.client`` 等) を直接 import しないことで、層の境界を
散文の主張ではなく機械検証できる状態に保つ (``harness`` と同じ扱い)。

加えてこの層には固有の制約が 2 つある。

- **推論ランタイムに直接触れない** (D-25)。埋め込みの HTTP 呼び出しは
  ``llmkit.embeddings`` にあり、``rag/`` には HTTP メソッド呼び出しも
  エンドポイントのパス文字列も存在しない。
- **vault に触れるのは :mod:`rag.vault` だけ** (D-30)。そのモジュールには
  書き込み API が 1 つも無い。

``__all__`` は各サブモジュールが宣言する ``__all__`` の和集合であり、
``llmkit/__init__.py`` / ``harness/__init__.py`` と同じ方針で維持する
(``cli`` は L4 の入口であって公開 API ではないため含めない)。
"""

from rag.chunker import (
    Chunk,
    chunk_note,
    estimate_tokens,
    render_embed_text,
)
from rag.indexer import (
    CHUNKS_FILENAME,
    INDEX_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    IndexManifest,
    IndexPlan,
    IndexResult,
    ManifestEmbedding,
    ManifestNote,
    ManifestTotals,
    build_index,
    chunks_path,
    index_fingerprint,
    load_manifest,
    manifest_path,
    plan_index,
    write_manifest,
)
from rag.parser import (
    FrontmatterValue,
    Heading,
    ParsedNote,
    iter_headings,
    parse_note,
)
from rag.settings import (
    DEFAULT_EXCLUDE_GLOBS,
    DEFAULT_INCLUDE_GLOBS,
    ChunkSettings,
    EmbedSettings,
    RagSettings,
    load_settings,
)
from rag.store import (
    ChunkRecord,
    InMemoryVectorStore,
    JsonlVectorStore,
    VectorStore,
)
from rag.vault import (
    VaultFile,
    iter_vault_files,
    read_note_bytes,
    read_note_text,
)

__all__ = [
    "CHUNKS_FILENAME",
    "DEFAULT_EXCLUDE_GLOBS",
    "DEFAULT_INCLUDE_GLOBS",
    "INDEX_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "Chunk",
    "ChunkRecord",
    "ChunkSettings",
    "EmbedSettings",
    "FrontmatterValue",
    "Heading",
    "InMemoryVectorStore",
    "IndexManifest",
    "IndexPlan",
    "IndexResult",
    "JsonlVectorStore",
    "ManifestEmbedding",
    "ManifestNote",
    "ManifestTotals",
    "ParsedNote",
    "RagSettings",
    "VaultFile",
    "VectorStore",
    "build_index",
    "chunk_note",
    "chunks_path",
    "estimate_tokens",
    "index_fingerprint",
    "iter_headings",
    "iter_vault_files",
    "load_manifest",
    "load_settings",
    "manifest_path",
    "parse_note",
    "plan_index",
    "read_note_bytes",
    "read_note_text",
    "render_embed_text",
    "write_manifest",
]
