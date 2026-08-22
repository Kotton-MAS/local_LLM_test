"""llmkit: 設定駆動の OpenAI 互換推論クライアント層 (L2)。

L3 (上位アプリ) はこのモジュールが再エクスポートする公開シンボルのみを import する。
推論ランタイム固有の型 (httpx / 生 JSON) は公開 API に露出させない。

``__all__`` は各サブモジュールが自ら宣言している ``__all__`` の和集合であり、
サブモジュール側で新しい公開シンボルを追加すると自動的にここへも反映される
(``tests/test_layout.py::test_public_api_matches_the_union_of_submodule_all``
が両者の一致を機械的に固定する)。
"""

from llmkit.bootstrap import BootstrapResult, bootstrap
from llmkit.catalog import (
    MODEL_CATALOG,
    ModelRole,
    ModelSpec,
    get_model_spec,
    known_model_ids,
    resolve_model_spec,
)
from llmkit.client import (
    ChatClient,
    ChatMessage,
    ChatResult,
    ChatRole,
    OpenAICompatibleClient,
    TokenUsage,
)
from llmkit.config import (
    AppConfig,
    GenerationParams,
    ProfileConfig,
    RuntimeConfig,
    RuntimeKind,
    VramConfig,
    load_config,
)
from llmkit.errors import (
    ConfigError,
    ContextLengthError,
    LlmkitError,
    ModelNotFoundError,
    OutOfMemoryError,
    RuntimeUnavailableError,
    UpstreamError,
    VramBudgetExceededError,
)
from llmkit.manifest import (
    DEFAULT_OUTPUT_DIR,
    SCHEMA_VERSION,
    ManifestGeneration,
    ManifestModelEntry,
    ManifestProfile,
    ManifestRuntime,
    ManifestVram,
    RunManifest,
    build_manifest,
    compute_config_sha256,
    manifest_filename,
    write_manifest,
)
from llmkit.vram import (
    ResolvedProfile,
    VramEstimate,
    check_budget,
    estimate_profile,
    estimate_resolved_profile,
    resolve_profile,
)

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "MODEL_CATALOG",
    "SCHEMA_VERSION",
    "AppConfig",
    "BootstrapResult",
    "ChatClient",
    "ChatMessage",
    "ChatResult",
    "ChatRole",
    "ConfigError",
    "ContextLengthError",
    "GenerationParams",
    "LlmkitError",
    "ManifestGeneration",
    "ManifestModelEntry",
    "ManifestProfile",
    "ManifestRuntime",
    "ManifestVram",
    "ModelNotFoundError",
    "ModelRole",
    "ModelSpec",
    "OpenAICompatibleClient",
    "OutOfMemoryError",
    "ProfileConfig",
    "ResolvedProfile",
    "RunManifest",
    "RuntimeConfig",
    "RuntimeKind",
    "RuntimeUnavailableError",
    "TokenUsage",
    "UpstreamError",
    "VramBudgetExceededError",
    "VramConfig",
    "VramEstimate",
    "bootstrap",
    "build_manifest",
    "check_budget",
    "compute_config_sha256",
    "estimate_profile",
    "estimate_resolved_profile",
    "get_model_spec",
    "known_model_ids",
    "load_config",
    "manifest_filename",
    "resolve_model_spec",
    "resolve_profile",
    "write_manifest",
]
