"""llmkit: 設定駆動の推論クライアント層 (L2)。

送出するワイヤプロトコルは ``runtime.kind`` で選ぶ (Ollama ネイティブ ``/api/chat``
または OpenAI 互換 ``/chat/completions``、D-10)。層の共通境界はワイヤプロトコル
ではなく :class:`ChatClient` Protocol である。

L3 (上位アプリ) はこのモジュールが再エクスポートする公開シンボルのみを import する。
推論ランタイム固有の型 (httpx / 生 JSON) は公開 API に露出させない。

``__all__`` は各サブモジュールが自ら宣言している ``__all__`` の和集合であり、
サブモジュール側で新しい公開シンボルを追加すると自動的にここへも反映される
(``tests/test_layout.py::test_public_api_matches_the_union_of_submodule_all``
が両者の一致を機械的に固定する)。
"""

from llmkit.bootstrap import BootstrapResult, bootstrap, bootstrap_from_config
from llmkit.catalog import (
    MODEL_CATALOG,
    ModelRole,
    ModelSpec,
    ServingRuntime,
    get_model_spec,
    known_model_ids,
    resolve_model_spec,
)
from llmkit.client import (
    ApiStyle,
    ChatClient,
    ChatMessage,
    ChatResult,
    ChatRole,
    ChatTimings,
    OllamaNativeClient,
    OpenAICompatibleClient,
    TokenUsage,
    api_style_for,
    create_chat_client,
    endpoint_url_for,
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
    "ApiStyle",
    "AppConfig",
    "BootstrapResult",
    "ChatClient",
    "ChatMessage",
    "ChatResult",
    "ChatRole",
    "ChatTimings",
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
    "OllamaNativeClient",
    "OpenAICompatibleClient",
    "OutOfMemoryError",
    "ProfileConfig",
    "ResolvedProfile",
    "RunManifest",
    "RuntimeConfig",
    "RuntimeKind",
    "RuntimeUnavailableError",
    "ServingRuntime",
    "TokenUsage",
    "UpstreamError",
    "VramBudgetExceededError",
    "VramConfig",
    "VramEstimate",
    "api_style_for",
    "bootstrap",
    "bootstrap_from_config",
    "build_manifest",
    "check_budget",
    "compute_config_sha256",
    "create_chat_client",
    "endpoint_url_for",
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
