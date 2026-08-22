"""推論クライアント抽象と OpenAI 互換実装 (L2)。

- 上位層 (L3) は :class:`ChatClient` Protocol と :class:`ChatMessage` /
  :class:`ChatResult` にのみ依存する。``httpx`` の型や生の JSON dict は
  公開 API に露出させない。
- ``httpx.Client`` はコンストラクタで注入できる。テストは
  ``httpx.MockTransport`` を注入することで実 HTTP を 1 バイトも出さない (D-02)。
- レスポンスは pydantic dataclass + ``TypeAdapter`` で厳格にパースし、必須
  フィールドの欠損は :class:`llmkit.errors.UpstreamError` にする (D-07)。
  ``pydantic.BaseModel`` は使わない (mypy ``disallow_any_explicit`` と両立しない)。
- 例外メッセージに api_key の値・レスポンスボディ全文を載せない
  (CLAUDE.md セキュリティ原則)。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol, runtime_checkable

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as pydantic_dataclass

from llmkit.catalog import MODEL_CATALOG, resolve_model_spec
from llmkit.config import AppConfig
from llmkit.errors import (
    ContextLengthError,
    ModelNotFoundError,
    OutOfMemoryError,
    RuntimeUnavailableError,
    UpstreamError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ChatClient",
    "ChatMessage",
    "ChatResult",
    "ChatRole",
    "OpenAICompatibleClient",
    "TokenUsage",
]

ChatRole = Literal["system", "user", "assistant"]

_CHAT_COMPLETIONS_PATH = "/chat/completions"

# 応答本文から原因を推定するためのマーカー。突き合わせにのみ使い、
# 本文そのものは例外メッセージに載せない。
_MODEL_NOT_FOUND_MARKERS = (
    "model not found",
    "no such model",
    "try pulling it first",
)
_OUT_OF_MEMORY_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda error",
    "cudamalloc",
    "insufficient memory",
)
_CONTEXT_LENGTH_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "context window",
    "num_ctx",
    "token limit",
)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """チャット 1 発話。"""

    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """1 回の生成で消費したトークン数。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ChatResult:
    """生成結果と、Phase 2 の比較ハーネスがそのまま使える速度指標。"""

    text: str
    model: str
    finish_reason: str
    usage: TokenUsage
    latency_s: float

    @property
    def tokens_per_second(self) -> float:
        """生成トークン数 / レイテンシ (秒)。

        ``latency_s`` が 0 以下のときは 0.0 を返す (計測不能を 0 で表し、
        ZeroDivisionError を上位に伝播させない)。
        """
        if self.latency_s <= 0.0:
            return 0.0
        return self.usage.completion_tokens / self.latency_s


@runtime_checkable
class ChatClient(Protocol):
    """L3 が依存する唯一の推論インタフェース。"""

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResult:
        """発話列を送って 1 応答を得る。"""
        ...


_RESPONSE_CONFIG = ConfigDict(extra="ignore")


@pydantic_dataclass(frozen=True, config=_RESPONSE_CONFIG)
class _ResponseMessage:
    content: str


@pydantic_dataclass(frozen=True, config=_RESPONSE_CONFIG)
class _ResponseChoice:
    message: _ResponseMessage
    finish_reason: str


@pydantic_dataclass(frozen=True, config=_RESPONSE_CONFIG)
class _ResponseUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@pydantic_dataclass(frozen=True, config=_RESPONSE_CONFIG)
class _ChatCompletion:
    model: str
    choices: list[_ResponseChoice]
    usage: _ResponseUsage


_COMPLETION_ADAPTER: TypeAdapter[_ChatCompletion] = TypeAdapter(_ChatCompletion)


def _contains_marker(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _format_validation_error(exc: ValidationError) -> str:
    """欠損箇所を「キー名: 理由」に整形する。入力値 (= 本文) は含めない。"""
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(ルート)"
        lines.append(f"{location}: {error['msg']}")
    return "; ".join(lines)


class OpenAICompatibleClient:
    """OpenAI 互換 ``/chat/completions`` を叩く同期クライアント。

    Ollama も外部 API も同じ実装で扱う (ネイティブ ``/api/chat`` は実装しない)。
    ``context_tokens`` は Ollama 拡張の ``options.num_ctx`` としてボディに載せる。
    """

    def __init__(
        self, config: AppConfig, *, http_client: httpx.Client | None = None
    ) -> None:
        """Args:
        config: 読み込み済み設定。送出パラメータの唯一の出典。
        http_client: 注入する ``httpx.Client``。省略時は設定の
            ``timeout_s`` から生成する (テストでは必ず MockTransport を注入する)。

        Raises:
            ConfigError: ``runtime.is_local=true`` かつ ``generation.model`` が
                カタログに無い場合。``is_local=false`` の場合はカタログ未登録
                でも passthrough で解決される (``resolve_model_spec`` 参照)。
        """
        self._config = config
        self._spec = resolve_model_spec(
            config.generation.model, is_local=config.runtime.is_local
        )
        self._owns_http_client = http_client is None
        self._http = (
            http_client
            if http_client is not None
            else httpx.Client(timeout=config.runtime.timeout_s)
        )

    @property
    def served_name(self) -> str:
        """推論ランタイムへ送るモデル実名 (``ModelSpec.served_name``)。"""
        return self._spec.served_name

    @property
    def endpoint_url(self) -> str:
        """リクエスト先の完全 URL。"""
        return f"{self._config.runtime.base_url}{_CHAT_COMPLETIONS_PATH}"

    def close(self) -> None:
        """自前で生成した ``httpx.Client`` だけを閉じる (注入されたものは閉じない)。"""
        if self._owns_http_client:
            self._http.close()

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResult:
        """発話列を送って 1 応答を得る。

        Raises:
            RuntimeUnavailableError: ランタイムに接続できない場合。
            ModelNotFoundError: ランタイム側にモデルが無い場合。
            OutOfMemoryError: ランタイム側で VRAM が枯渇した場合。
            ContextLengthError: 要求コンテキスト長が上限を超えた場合。
            UpstreamError: その他の 4xx/5xx、または応答の解析に失敗した場合。
        """
        body = self._build_request_body(messages)
        logger.debug(
            "推論リクエストを送信します: url=%s model=%s context_tokens=%d",
            self.endpoint_url,
            self.served_name,
            self._config.generation.context_tokens,
        )
        started = time.perf_counter()
        response = self._send(body)
        latency_s = time.perf_counter() - started

        self._raise_for_error_status(response)
        completion = self._parse_completion(response)
        choice = completion.choices[0]
        return ChatResult(
            text=choice.message.content,
            model=completion.model,
            finish_reason=choice.finish_reason,
            usage=TokenUsage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
            ),
            latency_s=latency_s,
        )

    def _build_request_body(self, messages: Sequence[ChatMessage]) -> dict[str, object]:
        """設定値をリクエストボディへ配線する (仕様書 §4 T3 / 有効性観点 E1-E3)。"""
        generation = self._config.generation
        return {
            "model": self._spec.served_name,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_output_tokens,
            "seed": generation.seed,
            # Ollama ネイティブ拡張。互換エンドポイントが無視しても害はない。
            "options": {"num_ctx": generation.context_tokens},
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self._config.api_key.get_secret_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _send(self, body: dict[str, object]) -> httpx.Response:
        try:
            return self._http.post(
                self.endpoint_url, json=body, headers=self._headers()
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            msg = (
                f"推論ランタイムに接続できません (base_url="
                f"{self._config.runtime.base_url})。"
                f"ランタイムが起動していない可能性があります"
            )
            raise RuntimeUnavailableError(
                msg,
                remediation=(
                    "`ollama serve` でランタイムを起動するか、"
                    "runtime.base_url が正しいか確認してください"
                ),
            ) from exc
        except httpx.HTTPError as exc:
            msg = (
                f"推論ランタイムとの通信に失敗しました "
                f"(url={self.endpoint_url}, 種別={type(exc).__name__})"
            )
            raise UpstreamError(
                msg,
                remediation=(
                    "runtime.timeout_s とネットワーク経路、"
                    "ランタイム側のログを確認してください"
                ),
            ) from exc

    def _raise_for_error_status(self, response: httpx.Response) -> None:
        """HTTP ステータスと本文マーカーを llmkit の例外へ翻訳する。

        本文は原因推定にのみ使い、メッセージには一切載せない。
        """
        status = response.status_code
        if status < httpx.codes.BAD_REQUEST:
            return
        body = response.text.lower()
        generation = self._config.generation

        if status == httpx.codes.NOT_FOUND or _contains_marker(
            body, _MODEL_NOT_FOUND_MARKERS
        ):
            msg = (
                f"モデル '{self._spec.served_name}' が推論ランタイムに"
                f"見つかりません (HTTP {status})"
            )
            raise ModelNotFoundError(
                msg,
                remediation=(
                    f"`ollama pull {self._spec.served_name}` を実行するか、"
                    f"generation.model ('{self._spec.model_id}') を見直してください"
                ),
            )

        if _contains_marker(body, _OUT_OF_MEMORY_MARKERS):
            msg = (
                f"推論ランタイムで VRAM が不足しました (プロファイル "
                f"'{self._config.vram.active_profile}', context_tokens="
                f"{generation.context_tokens}, HTTP {status})"
            )
            raise OutOfMemoryError(
                msg,
                remediation=(
                    "より小さいプロファイル (埋め込み・リランカーを外す、"
                    "または小さい生成モデル) へ切り替えるか、"
                    "generation.context_tokens を減らしてください"
                ),
            )

        if status == httpx.codes.BAD_REQUEST and _contains_marker(
            body, _CONTEXT_LENGTH_MARKERS
        ):
            # F-2-002: max_context_tokens はカタログ登録済みモデルの実値。
            # resolve_model_spec の passthrough (is_local=false かつ未登録)
            # では _PASSTHROUGH_MAX_CONTEXT_TOKENS という placeholder が
            # 入っているため、それを「モデルの上限」として出すと自己矛盾した
            # メッセージになる (例: 上限 1048576 を超えたと言いつつ実際の
            # 要求は 16384)。カタログ由来かどうかで文言を分ける。
            if self._spec.model_id in MODEL_CATALOG:
                msg = (
                    f"要求したコンテキスト長 context_tokens="
                    f"{generation.context_tokens} がモデル "
                    f"'{self._spec.served_name}' の上限 max_context_tokens="
                    f"{self._spec.max_context_tokens} を超えています (HTTP {status})"
                )
            else:
                msg = (
                    f"要求したコンテキスト長 context_tokens="
                    f"{generation.context_tokens} でモデル "
                    f"'{self._spec.served_name}' がコンテキスト長超過を"
                    f"報告しました (HTTP {status})。llmkit/catalog.py に未登録の"
                    "モデルのため上限は不明です。ランタイム/API 提供元のドキュメ"
                    "ントで上限を確認してください"
                )
            raise ContextLengthError(
                msg,
                remediation=(
                    "generation.context_tokens を減らすか、"
                    "より長いコンテキストを扱えるモデルへ切り替えてください"
                ),
            )

        msg = (
            f"推論ランタイムが HTTP {status} を返しました "
            f"(url={self.endpoint_url}, model={self._spec.served_name})"
        )
        raise UpstreamError(
            msg,
            remediation=(
                "ランタイム側のログでエラー内容を確認してください "
                "(応答本文は秘匿のためメッセージに含めていません)"
            ),
        )

    def _parse_completion(self, response: httpx.Response) -> _ChatCompletion:
        """応答 JSON を厳格にパースする (D-07)。"""
        try:
            payload: object = json.loads(response.text)
        except json.JSONDecodeError as exc:
            msg = (
                f"推論ランタイムの応答が JSON ではありません "
                f"(url={self.endpoint_url}, HTTP {response.status_code})"
            )
            raise UpstreamError(
                msg,
                remediation=(
                    "runtime.base_url が OpenAI 互換エンドポイント "
                    "(末尾 /v1) を指しているか確認してください"
                ),
            ) from exc

        try:
            completion = _COMPLETION_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            msg = (
                f"推論ランタイムの応答に必須フィールドがありません "
                f"(url={self.endpoint_url}): {_format_validation_error(exc)}"
            )
            raise UpstreamError(
                msg,
                remediation=(
                    "ランタイムが OpenAI 互換の chat completion 形式を"
                    "返しているか確認してください"
                ),
            ) from exc

        if not completion.choices:
            msg = (
                f"推論ランタイムの応答の choices が空です "
                f"(url={self.endpoint_url}, model={self._spec.served_name})"
            )
            raise UpstreamError(
                msg,
                remediation=(
                    "プロンプトとランタイム側のログを確認してください "
                    "(生成が打ち切られた可能性があります)"
                ),
            )
        return completion
