"""llmkit の例外階層。

すべての例外は「何が起きたか (message)」と「どうすれば直るか (remediation)」を
分けて保持する。``str(exc)`` は両方を含むため、CLI がそのまま人間に見せられる。

セキュリティ原則 (CLAUDE.md): 例外メッセージに api_key・トークンの値を載せない。
環境変数「名」は載せてよい。
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "ConfigError",
    "ContextLengthError",
    "LlmkitError",
    "ModelNotFoundError",
    "OutOfMemoryError",
    "RuntimeUnavailableError",
    "UpstreamError",
    "VramBudgetExceededError",
]


class LlmkitError(Exception):
    """llmkit が送出するすべての例外の基底。"""

    def __init__(self, message: str, *, remediation: str) -> None:
        super().__init__(message)
        self.message = message
        self.remediation = remediation

    def __str__(self) -> str:
        if not self.remediation:
            return self.message
        return f"{self.message} / 対処: {self.remediation}"


class ConfigError(LlmkitError):
    """設定ファイルの読み込み・検証に失敗した。"""


class VramBudgetExceededError(LlmkitError):
    """プロファイルの想定 VRAM 使用量が予算 (GiB) を超えた。

    内訳 (モデル別重み・KV・オーバーヘッド・合計・予算・超過量) を属性として保持する。
    単位はすべて GiB (D-03)。
    """

    def __init__(
        self,
        message: str,
        *,
        remediation: str,
        profile_name: str,
        weights_gib: Mapping[str, float],
        kv_cache_gib: float,
        runtime_overhead_gib: float,
        total_gib: float,
        budget_gib: float,
        excess_gib: float,
    ) -> None:
        super().__init__(message, remediation=remediation)
        self.profile_name = profile_name
        self.weights_gib = dict(weights_gib)
        self.kv_cache_gib = kv_cache_gib
        self.runtime_overhead_gib = runtime_overhead_gib
        self.total_gib = total_gib
        self.budget_gib = budget_gib
        self.excess_gib = excess_gib


class RuntimeUnavailableError(LlmkitError):
    """推論ランタイムに接続できない (未起動・URL 誤り等)。"""


class ModelNotFoundError(LlmkitError):
    """要求したモデルがランタイム側に存在しない。"""


class OutOfMemoryError(LlmkitError):
    """ランタイム側で VRAM が枯渇した (組み込みの OutOfMemoryError とは別物)。"""


class ContextLengthError(LlmkitError):
    """要求したコンテキスト長がモデルの上限を超えた。"""


class UpstreamError(LlmkitError):
    """ランタイムが想定外の応答を返した。

    レスポンスボディ全文はメッセージに載せない (内部実装詳細の漏洩防止)。
    """
