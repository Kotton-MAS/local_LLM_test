"""設定層。TOML を読み、pydantic で検証して :class:`AppConfig` を返す。

- 形式は TOML (stdlib ``tomllib``)。YAML 依存を持ち込まない。
- スキーマは **pydantic dataclass** で定義する。``pydantic.BaseModel`` を継承すると
  mypy の ``disallow_any_explicit = true`` が ``[explicit-any]`` を出すため
  (pyproject の mypy 設定は緩めない、という制約が優先される)。
- api_key は設定ファイルに書かない。``runtime.api_key_env`` (環境変数「名」) で
  参照し、値は :class:`pydantic.SecretStr` で保持する (D-05)。
- 検証エラーは必ず :class:`llmkit.errors.ConfigError` に翻訳し、
  メッセージに該当キー名を含める。
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tomllib
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import (
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass

from llmkit.errors import ConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "AppConfig",
    "GenerationParams",
    "ProfileConfig",
    "RuntimeConfig",
    "RuntimeKind",
    "VramConfig",
    "load_config",
]

RuntimeKind = Literal["ollama", "openai_compatible"]

_FORBID_EXTRA = ConfigDict(extra="forbid")

NonEmptyStr = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


@dataclass(frozen=True, config=_FORBID_EXTRA)
class RuntimeConfig:
    """推論ランタイムの接続情報。

    Attributes:
        base_url: **実行マニフェスト (outputs/runs/*.json)・INFO ログ・CLI
            標準出力へ verbatim (そのままの文字列) で記録される。**
            認証情報を URL のどの位置にも含めないこと。userinfo
            (``user:pass@``) とクエリ・フラグメント (``?...`` / ``#...``)
            は入口 (:meth:`_validate_base_url`) で拒否するが、path 中に
            埋め込まれた秘密 (例: ゲートウェイ URL
            ``https://host/<token>/v1``) は機械的に判別できないため
            防げない (F-2-007, CWE-532)。値は環境変数に置き
            ``runtime.api_key_env`` で参照すること。
    """

    base_url: str
    kind: RuntimeKind = "ollama"
    api_key_env: NonEmptyStr = "LLMKIT_API_KEY"
    timeout_s: PositiveFloat = 120.0
    is_local: bool = True

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            msg = "http:// または https:// で始まる URL を指定してください"
            raise ValueError(msg)
        if parsed.username or parsed.password:
            # D-05 が塞いだ「設定ファイルに秘密を書く」経路が base_url の
            # userinfo (user:pass@host) 経由で復活しないようにする (CWE-532)。
            # 通した場合、実行マニフェスト・INFO ログ・CLI 標準出力・
            # RuntimeUnavailableError/UpstreamError のメッセージの4経路に
            # 平文で複製される。
            msg = (
                "base_url に認証情報 (user:pass@) を含めないでください。"
                "値は環境変数に置き runtime.api_key_env で参照してください"
            )
            raise ValueError(msg)
        if parsed.query or parsed.fragment:
            # F-2-007 (CWE-532): クエリ・フラグメント経由の秘密混入
            # (例 'https://host/v1?key=SECRET', 'https://host/v1#SECRET')
            # を入口で拒否する。base_url は chat 等のパスを連結して使うため、
            # クエリ・フラグメントが付くと連結後に不正な URL になる
            # (正当性の観点でも不要)。path 中の秘密は機械判別できないため
            # ここでは弾けない (上記 Attributes の注意書き参照)。
            msg = (
                "base_url にクエリ文字列 (?...) やフラグメント (#...) を"
                "含めないでください。パスを連結して使うため不正な URL になります"
            )
            raise ValueError(msg)
        return value.rstrip("/")


@dataclass(frozen=True, config=_FORBID_EXTRA)
class GenerationParams:
    """生成パラメータ。全項目が実行マニフェストに記録される。"""

    model: NonEmptyStr
    context_tokens: PositiveInt
    temperature: Annotated[float, Field(ge=0.0, le=2.0)]
    top_p: Annotated[float, Field(gt=0.0, le=1.0)]
    max_output_tokens: PositiveInt
    seed: int = 0


@dataclass(frozen=True, config=_FORBID_EXTRA)
class VramConfig:
    """VRAM 予算とプロファイル選択。単位はすべて GiB (D-03)。"""

    budget_gib: PositiveFloat
    runtime_overhead_gib: NonNegativeFloat
    active_profile: NonEmptyStr


@dataclass(frozen=True, config=_FORBID_EXTRA)
class ProfileConfig:
    """1 プロファイルに同居させるモデルの構成。"""

    generation: NonEmptyStr
    embedding: str | None = None
    reranker: str | None = None

    def model_ids(self) -> tuple[str, ...]:
        """このプロファイルが同時に GPU へ載せるモデル ID を宣言順に返す。"""
        ids = (self.generation, self.embedding, self.reranker)
        return tuple(model_id for model_id in ids if model_id)


@dataclass(frozen=True, config=_FORBID_EXTRA)
class AppConfig:
    """アプリケーション設定のルート。

    ``api_key`` は TOML からではなく ``runtime.api_key_env`` が指す環境変数から
    解決される。``SecretStr`` のため repr / str に平文が出ない。
    """

    runtime: RuntimeConfig
    generation: GenerationParams
    vram: VramConfig
    profiles: dict[str, ProfileConfig]
    api_key: SecretStr = SecretStr("")

    @model_validator(mode="after")
    def _validate_active_profile(self) -> AppConfig:
        if self.vram.active_profile not in self.profiles:
            known = ", ".join(sorted(self.profiles)) or "(なし)"
            msg = (
                f"vram.active_profile '{self.vram.active_profile}' に対応する "
                f"[profiles.*] がありません。定義済み: {known}"
            )
            raise ValueError(msg)
        return self

    def active_profile(self) -> ProfileConfig:
        """``vram.active_profile`` が指すプロファイル定義を返す。"""
        return self.profiles[self.vram.active_profile]


_APP_CONFIG_ADAPTER: TypeAdapter[AppConfig] = TypeAdapter(AppConfig)


def _format_validation_error(exc: ValidationError) -> str:
    """pydantic の ValidationError を「キー名: 理由」の一覧に整形する。"""
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(ルート)"
        lines.append(f"{location}: {error['msg']}")
    return "; ".join(lines)


def _read_toml(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"設定ファイルを読めません: {path}"
        raise ConfigError(
            msg, remediation=f"パスが正しいか確認してください ({exc.strerror})"
        ) from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = f"設定ファイルの TOML 構文が不正です: {path} ({exc})"
        raise ConfigError(msg, remediation="TOML の構文を修正してください") from exc


def _raise_inline_secret(location: str) -> None:
    msg = f"{location} を設定ファイルに書くことはできません"
    raise ConfigError(
        msg,
        remediation=(
            "値は環境変数に置き、runtime.api_key_env にその環境変数名を指定してください"
        ),
    )


def _reject_inline_secrets(raw: object) -> None:
    """設定ファイル本体に秘密情報を書く経路を塞ぐ (D-05)。"""
    if not isinstance(raw, dict):
        return
    if "api_key" in raw:
        _raise_inline_secret("api_key")
    runtime_section = raw.get("runtime")
    if isinstance(runtime_section, dict) and "api_key" in runtime_section:
        _raise_inline_secret("runtime.api_key")


def _resolve_api_key(runtime: RuntimeConfig) -> SecretStr:
    """環境変数から api_key を解決する。値はメッセージに出さない。"""
    value = os.environ.get(runtime.api_key_env, "")
    if not value and not runtime.is_local:
        msg = (
            f"環境変数 {runtime.api_key_env} が未設定のため api_key を解決できません "
            f"(runtime.is_local = false)"
        )
        raise ConfigError(
            msg,
            remediation=(
                f"export {runtime.api_key_env}=<APIキー> を設定するか、"
                f"runtime.is_local = true にしてください"
            ),
        )
    return SecretStr(value)


def load_config(path: Path) -> AppConfig:
    """TOML 設定ファイルを読み込んで検証済みの :class:`AppConfig` を返す。

    Raises:
        ConfigError: 読み込み・構文・スキーマ・api_key 解決のいずれかに失敗した場合。
    """
    raw = _read_toml(path)
    _reject_inline_secrets(raw)

    try:
        config = _APP_CONFIG_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        msg = f"設定エラー ({path}): {_format_validation_error(exc)}"
        raise ConfigError(
            msg, remediation="該当キーの値・型・綴りを確認してください"
        ) from exc

    resolved = dataclasses.replace(config, api_key=_resolve_api_key(config.runtime))
    logger.debug(
        "設定を読み込みました: path=%s profile=%s model=%s",
        path,
        resolved.vram.active_profile,
        resolved.generation.model,
    )
    return resolved
