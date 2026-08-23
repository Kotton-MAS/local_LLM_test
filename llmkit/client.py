"""推論クライアント抽象と 2 経路の HTTP 実装 (L2)。

- 上位層 (L3) は :class:`ChatClient` Protocol と :class:`ChatMessage` /
  :class:`ChatResult` にのみ依存する。``httpx`` の型や生の JSON dict は
  公開 API に露出させない。
- 経路は 2 つ。``runtime.kind = "ollama"`` はネイティブ ``/api/chat``
  (:class:`OllamaNativeClient`)、``"openai_compatible"`` は OpenAI 互換
  ``/chat/completions`` (:class:`OpenAICompatibleClient`)。**分岐は
  :func:`create_chat_client` の 1 か所だけ**に置く (D-10)。HTTP 送信・接続
  エラーの翻訳・ステータス/本文からの例外翻訳・api_key ヘッダ・``httpx.Client``
  の所有権は chat 非依存の基底 :class:`_HttpEndpointClient` に集約し、
  :class:`_HttpChatClient` はチャット固有の流れ、具象はリクエストボディの
  組み立てと応答パースだけを持つ。埋め込み
  (:mod:`llmkit.embeddings`) も同じ基底を継承し、**例外翻訳表を 1 実装に
  保つ** (D-25)。
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
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import (
    ClassVar,
    Literal,
    NoReturn,
    Protocol,
    Self,
    assert_never,
    runtime_checkable,
)

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as pydantic_dataclass

from llmkit.catalog import MODEL_CATALOG, ModelSpec, resolve_model_spec
from llmkit.config import AppConfig, RuntimeKind
from llmkit.errors import (
    ContextLengthError,
    ModelNotFoundError,
    OutOfMemoryError,
    RuntimeUnavailableError,
    UpstreamError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ApiStyle",
    "ChatClient",
    "ChatMessage",
    "ChatResult",
    "ChatRole",
    "ChatTimings",
    "OllamaNativeClient",
    "OpenAICompatibleClient",
    "TokenUsage",
    "api_style_for",
    "create_chat_client",
    "endpoint_url_for",
]

ChatRole = Literal["system", "user", "assistant"]

#: 送出するワイヤプロトコル。``runtime.kind`` から :func:`api_style_for` で導出する。
ApiStyle = Literal["ollama_native", "openai_compatible"]

_CHAT_COMPLETIONS_PATH = "/chat/completions"
_NATIVE_CHAT_PATH = "/api/chat"
_OPENAI_VERSION_SEGMENT = "/v1"

_NANOSECONDS_PER_SECOND = 1e9

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


def api_style_for(kind: RuntimeKind) -> ApiStyle:
    """``runtime.kind`` を送出するワイヤプロトコルへ写す (D-10)。

    ``kind = "ollama"`` はネイティブ ``/api/chat`` を使う。Phase 0 実測で
    OpenAI 互換経路が ``options.num_ctx`` を無視することを確認しているため。
    """
    if kind == "ollama":
        return "ollama_native"
    return "openai_compatible"


def endpoint_url_for(base_url: str, style: ApiStyle) -> str:
    """``base_url`` と API スタイルからリクエスト先の完全 URL を導出する (D-11)。

    ``base_url`` は OpenAI 互換の ``/v1`` を指す前提で configs に固定されている。
    ネイティブ経路では末尾の ``/v1`` を **1 セグメントだけ**取り除いて
    ``/api/chat`` を連結する (``http://localhost:11434/v1`` →
    ``http://localhost:11434/api/chat``)。ネイティブ URL 用の設定キーは追加しない。

    HTTP を 1 バイトも出さない純関数。``style`` が増えたときに追加漏れを
    mypy strict が検出できるよう ``match`` + ``assert_never`` で網羅する
    (F-4-002: 未知 style が無警告でネイティブ経路に落ちていた反省)。
    """
    root = base_url.rstrip("/")
    match style:
        case "openai_compatible":
            return f"{root}{_CHAT_COMPLETIONS_PATH}"
        case "ollama_native":
            if root.endswith(_OPENAI_VERSION_SEGMENT):
                root = root[: -len(_OPENAI_VERSION_SEGMENT)]
            return f"{root}{_NATIVE_CHAT_PATH}"
        case _:
            assert_never(style)


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


def _tokens_per_second(count: int | None, seconds: float | None) -> float | None:
    """欠測 (``None``) と計測不能 (0 秒以下) を ``None`` として区別する。"""
    if count is None or seconds is None or seconds <= 0.0:
        return None
    return count / seconds


@dataclass(frozen=True, slots=True)
class ChatTimings:
    """prompt / eval を分離した計測値 (D-13)。

    欠測は ``None`` で表し 0 で埋めない。Ollama はプロンプトキャッシュヒット時に
    ``prompt_eval_*`` を返さないため、0 で埋めると「未計測」と「実際に 0 トークン」
    が区別できず、Phase 2 の比較が静かに壊れる (D-07 と同じ趣旨)。
    OpenAI 互換経路はこの情報を返さないため :attr:`ChatResult.timings` は常に
    ``None`` になる。

    ``eval_seconds`` (この分母) は Ollama 内部の純粋な生成時間のみを計測する。
    これは :attr:`ChatResult.latency_s` (HTTP 往復全体。モデルの再ロードや
    キュー待ちを含み得る) とは計測区間が異なる。実測でコールド時 (モデル再
    ロードを伴う 1 回目) に :attr:`eval_tokens_per_second` が
    :attr:`ChatResult.tokens_per_second` の約 1.8 倍になることを確認済み
    (F-4-004)。``timings`` が非 ``None`` のときは、この
    :attr:`eval_tokens_per_second` の方を『生成速度』の代表値として優先する
    こと。``ChatResult.tokens_per_second`` はあくまで壁時計ベースの参考値。
    """

    prompt_eval_count: int | None
    prompt_eval_seconds: float | None
    eval_count: int | None
    eval_seconds: float | None

    @property
    def prompt_tokens_per_second(self) -> float | None:
        """プロンプト処理速度 (tokens/s)。欠測・0 秒なら ``None`` (0.0 ではない)。"""
        return _tokens_per_second(self.prompt_eval_count, self.prompt_eval_seconds)

    @property
    def eval_tokens_per_second(self) -> float | None:
        """生成速度 (tokens/s)。欠測・0 秒なら ``None`` (0.0 ではない)。"""
        return _tokens_per_second(self.eval_count, self.eval_seconds)


@dataclass(frozen=True, slots=True)
class ChatResult:
    """生成結果と、Phase 2 の比較ハーネスがそのまま使える速度指標。

    ``timings`` はネイティブ ``/api/chat`` 応答が返す prompt / eval 分離の
    計測値 (D-13)。OpenAI 互換経路は返さないため ``None`` のまま。
    """

    text: str
    model: str
    finish_reason: str
    usage: TokenUsage
    latency_s: float
    timings: ChatTimings | None = None

    @property
    def tokens_per_second(self) -> float:
        """生成トークン数 / レイテンシ (秒)。壁時計ベースの参考値。

        ``latency_s`` は HTTP リクエスト送信から応答受信までの往復全体を
        計測するため、モデルの再ロードやランタイム側のキュー待ちを含み
        得る。``timings`` (:class:`ChatTimings`) が非 ``None`` の場合は、
        Ollama 内部の純粋な生成時間のみを分母にする
        ``timings.eval_tokens_per_second`` の方を『生成速度』の代表値として
        優先すること。実測でコールド時に両者が最大 1.8 倍乖離することを
        確認している (F-4-004)。

        ``latency_s`` が 0 以下、または ``usage.completion_tokens`` が
        (``eval_count`` 欠測などにより) 0 のときは 0.0 を返す。この 0.0 は
        「計測した結果 0 だった」ではなく「この値からは分からない」を表す
        欠測のシグナルであり、実測値として比較・記録してはならない
        (ZeroDivisionError を上位に伝播させないための表現でもある)。
        """
        if self.latency_s <= 0.0:
            return 0.0
        return self.usage.completion_tokens / self.latency_s

    @property
    def measured_tokens_per_second(self) -> float | None:
        """欠測を ``None`` で表す、``tokens_per_second`` の加算的な代替 (F-5-001)。

        ``tokens_per_second`` は互換のため 0.0 を「欠測」に流用しているが、
        呼び出し側がそれを「実測 0」と取り違える余地を型では塞げない
        (実際に CLI の ``doctor`` がその取り違えを起こした)。この property は
        優先順位を型で固定する:

        1. ``timings.eval_tokens_per_second`` が非 ``None`` ならそれを返す
           (Ollama 内部の純粋な生成時間が分母。最も信頼できる)。
        2. それが無く、かつ ``latency_s > 0`` **かつ**
           ``usage.completion_tokens > 0`` のときだけ壁時計ベースの値
           (``tokens_per_second`` と同じ計算) を返す。
        3. どちらでもなければ ``None`` (計測不能。0.0 ではない)。

        新規コードはこちらを使うこと。``tokens_per_second`` は既存呼び出し元
        との互換のためだけに残す。
        """
        if self.timings is not None:
            eval_rate = self.timings.eval_tokens_per_second
            if eval_rate is not None:
                return eval_rate
        if self.latency_s > 0.0 and self.usage.completion_tokens > 0:
            return self.usage.completion_tokens / self.latency_s
        return None


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


@pydantic_dataclass(frozen=True, config=_RESPONSE_CONFIG)
class _NativeMessage:
    content: str


@pydantic_dataclass(frozen=True, config=_RESPONSE_CONFIG)
class _NativeChatResponse:
    """Ollama ネイティブ ``/api/chat`` の非ストリーミング応答。

    必須は ``model`` / ``message.content`` / ``done`` の 3 つだけ。
    ``done_reason`` と計測値はプロンプトキャッシュヒット時などに返らないため
    任意にする (必須にすると正常な生成が UpstreamError になる)。
    """

    model: str
    message: _NativeMessage
    done: bool
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


_COMPLETION_ADAPTER: TypeAdapter[_ChatCompletion] = TypeAdapter(_ChatCompletion)
_NATIVE_ADAPTER: TypeAdapter[_NativeChatResponse] = TypeAdapter(_NativeChatResponse)


def _contains_marker(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _format_validation_error(exc: ValidationError) -> str:
    """欠損箇所を「キー名: 理由」に整形する。入力値 (= 本文) は含めない。"""
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(ルート)"
        lines.append(f"{location}: {error['msg']}")
    return "; ".join(lines)


def _seconds_from_nanoseconds(nanoseconds: int | None) -> float | None:
    """ネイティブ応答の ns を秒へ。欠測は欠測のまま返す (0 で埋めない)。"""
    if nanoseconds is None:
        return None
    return nanoseconds / _NANOSECONDS_PER_SECOND


def _embedded_error_marker(body_text: str) -> str | None:
    """本文トップレベルに非空の ``error`` があればその文字列を返す。

    Ollama は HTTP 200 のままエラーを本文に載せることがある。戻り値は原因推定の
    マーカーとしてのみ使い、例外メッセージには載せない。
    """
    try:
        payload: object = json.loads(body_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error: object = payload.get("error")
    if not error:
        return None
    return str(error)


class _HttpEndpointClient(ABC):
    """HTTP 経由で推論ランタイムの 1 エンドポイントを叩くクライアントの基底。

    chat に依存しない部分だけをここに集約する: ``httpx.Client`` の所有権と
    close、api_key ヘッダ、POST 送信と接続エラーの翻訳、ステータス/本文からの
    例外翻訳表、JSON デコードとスキーマ違反の翻訳。

    チャット (:class:`_HttpChatClient`) と埋め込み
    (:class:`llmkit.embeddings.OpenAIEmbeddingClient`) が **同じ翻訳表の唯一の
    実装** を共有するための基底であり、翻訳表を 2 か所に複製しないことが目的
    (D-25)。エンドポイント URL (:attr:`endpoint_url`) とリクエスト固有の文脈
    (:meth:`_request_context`) だけをサブクラスが決める。
    """

    def __init__(
        self,
        config: AppConfig,
        spec: ModelSpec,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Args:
        config: 読み込み済み設定。送出パラメータの唯一の出典。
        spec: このクライアントが叩くモデルの解決済みメタデータ。解決方法
            (どの設定キーを見るか) はサブクラスが決める。
        http_client: 注入する ``httpx.Client``。省略時は設定の
            ``timeout_s`` から生成する (テストでは必ず MockTransport を注入する)。
        """
        self._config = config
        self._spec = spec
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
    @abstractmethod
    def endpoint_url(self) -> str:
        """リクエスト先の完全 URL。"""

    @abstractmethod
    def _request_context(self) -> str:
        """例外メッセージに載せる『何をどれだけ要求したか』の記述。

        :class:`~llmkit.errors.OutOfMemoryError` と
        :class:`~llmkit.errors.ContextLengthError` のメッセージだけが使う。
        チャットは ``context_tokens=<値>`` を返し、既存のメッセージ文面を
        そのまま再現する。基底に既定値を置かない (置くと、新しい経路が
        黙ってチャットの文言を名乗る)。
        """

    @abstractmethod
    def _model_setting_reference(self) -> str:
        """モデル ID の出典となる設定キーの記述 (対処メッセージ用)。

        :class:`~llmkit.errors.ModelNotFoundError` の対処だけが使う。
        チャットのモデルは ``generation.model`` が出典だが、埋め込みは
        ``profiles.<プロファイル名>.embedding`` が唯一の出典であり (D-27)、
        基底が前者を名乗ると「存在しない設定キーを直せ」と案内することに
        なる。実際に ``generation.model`` を書き換えると、埋め込みは直らない
        まま生成モデルだけが壊れる。基底に既定値を置かない。
        """

    def close(self) -> None:
        """自前で生成した ``httpx.Client`` だけを閉じる (注入されたものは閉じない)。"""
        if self._owns_http_client:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

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
        """エラーステータスを llmkit の例外へ翻訳する。

        本文は原因推定にのみ使い、メッセージには一切載せない。
        """
        if response.status_code < httpx.codes.BAD_REQUEST:
            return
        self._raise_translated_error(response.status_code, response.text)

    def _raise_if_body_reports_error(self, response: httpx.Response) -> None:
        """HTTP 200 のまま本文に載ったエラーも同じ翻訳表へ流す。

        Ollama はネイティブ ``/api/chat`` でも OpenAI 互換エンドポイントでも
        200 + ``{"error": ...}`` を返すことがある。ステータスだけを見ると
        「スキーマ違反」に化けて原因が消える。
        """
        marker = _embedded_error_marker(response.text)
        if marker is not None:
            self._raise_translated_error(
                response.status_code, marker, from_body_error=True
            )

    def _raise_translated_error(
        self, status: int, body_text: str, *, from_body_error: bool = False
    ) -> NoReturn:
        """翻訳表。全経路が共有する唯一の実装。

        OpenAI のエラースキーマはパースせず、HTTP ステータスと本文の小文字部分
        一致だけで判定する。そのためネイティブ形式の本文にもそのまま成立する。
        本体は判定順序 (model_not_found → out_of_memory → context_length →
        generic) を並べるだけの薄い関数にし、各判定は専用メソッドへ切り出す
        (F-4-005: 4 種の判定が 1 関数に同居して 100 行に伸びていた)。

        Args:
            status: 応答の HTTP ステータス。
            body_text: 原因推定に使う文字列 (本文全体、またはネイティブ応答の
                トップレベル ``error`` の値)。メッセージには載せない。
            from_body_error: HTTP 200 のまま本文にエラーが載っていた場合に真。
                ステータスで門番している分岐 (コンテキスト長超過) をこの経路でも
                通すために使う。
        """
        body = body_text.lower()
        self._raise_if_model_not_found(status, body)
        self._raise_if_out_of_memory(status, body)
        self._raise_if_context_length_exceeded(
            status, body, from_body_error=from_body_error
        )
        self._raise_generic_upstream_error(status, from_body_error=from_body_error)

    def _raise_if_model_not_found(self, status: int, body: str) -> None:
        """マーカー不一致なら何もせず戻り、次の判定へ進む (fallthrough)。"""
        if status != httpx.codes.NOT_FOUND and not _contains_marker(
            body, _MODEL_NOT_FOUND_MARKERS
        ):
            return
        msg = (
            f"モデル '{self._spec.served_name}' が推論ランタイムに"
            f"見つかりません (HTTP {status})"
        )
        raise ModelNotFoundError(
            msg,
            remediation=(
                f"`ollama pull {self._spec.served_name}` を実行するか、"
                f"{self._model_setting_reference()} を見直してください"
            ),
        )

    def _raise_if_out_of_memory(self, status: int, body: str) -> None:
        if not _contains_marker(body, _OUT_OF_MEMORY_MARKERS):
            return
        msg = (
            f"推論ランタイムで VRAM が不足しました (プロファイル "
            f"'{self._config.vram.active_profile}', {self._request_context()}"
            f", HTTP {status})"
        )
        raise OutOfMemoryError(
            msg,
            remediation=(
                "より小さいプロファイル (埋め込み・リランカーを外す、"
                "または小さい生成モデル) へ切り替えるか、"
                "generation.context_tokens を減らしてください"
            ),
        )

    def _raise_if_context_length_exceeded(
        self, status: int, body: str, *, from_body_error: bool
    ) -> None:
        """判定条件と判定順序は全経路で共有し、文面だけをフックに委ねる。"""
        if not (
            (status == httpx.codes.BAD_REQUEST or from_body_error)
            and _contains_marker(body, _CONTEXT_LENGTH_MARKERS)
        ):
            return
        raise ContextLengthError(
            self._context_length_message(status),
            remediation=self._context_length_remediation(),
        )

    @abstractmethod
    def _context_length_subject(self) -> str:
        """コンテキスト長超過メッセージの主語。経路ごとに何が長すぎたかが違う。

        チャットは「要求したコンテキスト長 context_tokens=16384 」を返し、
        埋め込みは「埋め込み入力」を返す。**判定 (カタログ登録済みか
        passthrough か) は基底に残し、主語だけを差し替える** のが要点で、
        メッセージ全体をフックにすると F-2-002 の分岐が新しい経路から
        黙って抜け落ちる (実際に F-9-001 として発生した: 埋め込み経路が
        placeholder の 1048576 を「モデルの上限」として断言していた)。

        戻り値は直後の「がモデル…」「でモデル…」に続くため、値で終わる
        経路 (チャット) は末尾に空白を含める。基底に既定値を置かない
        (置くと、新しい経路が黙ってチャットの主語を名乗る)。
        """

    def _context_length_message(self, status: int) -> str:
        """コンテキスト長超過のメッセージ。主語だけが経路ごとに違う。"""
        # F-2-002: max_context_tokens はカタログ登録済みモデルの実値。
        # resolve_model_spec の passthrough (is_local=false かつ未登録)
        # では _PASSTHROUGH_MAX_CONTEXT_TOKENS という placeholder が
        # 入っているため、それを「モデルの上限」として出すと自己矛盾した
        # メッセージになる (例: 上限 1048576 を超えたと言いつつ実際の
        # 要求は 16384)。カタログ由来かどうかで文言を分ける。
        if self._spec.model_id in MODEL_CATALOG:
            return (
                f"{self._context_length_subject()}がモデル "
                f"'{self._spec.served_name}' の上限 max_context_tokens="
                f"{self._spec.max_context_tokens} を超えています (HTTP {status})"
            )
        return (
            f"{self._context_length_subject()}でモデル "
            f"'{self._spec.served_name}' がコンテキスト長超過を"
            f"報告しました (HTTP {status})。llmkit/catalog.py に未登録の"
            "モデルのため上限は不明です。ランタイム/API 提供元のドキュメ"
            "ントで上限を確認してください"
        )

    @abstractmethod
    def _context_length_remediation(self) -> str:
        """コンテキスト長超過の対処。経路ごとに直す設定キーが違う。

        チャットは ``generation.context_tokens`` を、埋め込みはチャンクの
        上限トークン数を案内する。基底に既定値を置くと、この基底へ新しく
        載る経路 (リランカー) が上書きを忘れたまま気づかれずに済んでしまい、
        チャット向けの対処文面を誤って名乗る (round-10 レビュー: 埋め込みの
        ``_context_length_message`` が F-9-001 として一度これと同型の欠陥を
        起こしている。対処メッセージ側でも同じ型の欠陥が起き得るため、
        ``ABC`` と mypy が上書き漏れを機械的に検出できるよう abstract にする)。
        """

    def _raise_generic_upstream_error(
        self, status: int, *, from_body_error: bool
    ) -> NoReturn:
        """他の判定に該当しなかった場合のフォールバック。必ず送出する。"""
        if from_body_error:
            msg = (
                f"推論ランタイムが応答本文でエラーを報告しました "
                f"(HTTP {status}, url={self.endpoint_url}, "
                f"model={self._spec.served_name})"
            )
        else:
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

    def _decode_json(self, response: httpx.Response, *, remediation: str) -> object:
        """応答本文を JSON として読む。失敗は経路ごとの対処つき UpstreamError。"""
        try:
            payload: object = json.loads(response.text)
        except json.JSONDecodeError as exc:
            msg = (
                f"推論ランタイムの応答が JSON ではありません "
                f"(url={self.endpoint_url}, HTTP {response.status_code})"
            )
            raise UpstreamError(msg, remediation=remediation) from exc
        return payload

    def _raise_schema_error(
        self, exc: ValidationError, *, remediation: str
    ) -> NoReturn:
        """必須フィールド欠損を UpstreamError へ (入力値は載せない, D-07)。"""
        msg = (
            f"推論ランタイムの応答に必須フィールドがありません "
            f"(url={self.endpoint_url}): {_format_validation_error(exc)}"
        )
        raise UpstreamError(msg, remediation=remediation) from exc


class _HttpChatClient(_HttpEndpointClient):
    """HTTP 経由の推論クライアントに共通する実装。

    HTTP 送信・例外翻訳・``httpx.Client`` の所有権は基底
    :class:`_HttpEndpointClient` が持つ。ここに残るのはチャット固有の流れ
    (:meth:`chat`) と、チャットのエンドポイント導出・要求文脈だけ。具象が持つ
    のはリクエストボディの組み立て (:meth:`_build_request_body`) と応答の
    パース (:meth:`_build_result`) だけ。``runtime.kind`` の分岐はここではなく
    :func:`create_chat_client` の 1 か所だけに置く (D-10)。
    """

    #: この実装が話すワイヤプロトコル。``endpoint_url`` の導出に使う。
    _api_style: ClassVar[ApiStyle]

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
        super().__init__(
            config,
            resolve_model_spec(
                config.generation.model, is_local=config.runtime.is_local
            ),
            http_client=http_client,
        )

    @property
    def endpoint_url(self) -> str:
        """リクエスト先の完全 URL。"""
        return endpoint_url_for(self._config.runtime.base_url, self._api_style)

    def _request_context(self) -> str:
        """チャットが要求したコンテキスト長 (例外メッセージ用)。"""
        return f"context_tokens={self._config.generation.context_tokens}"

    def _context_length_subject(self) -> str:
        """チャットが長すぎたのは「要求したコンテキスト長」。

        末尾の空白は、基底の文型が直後に「がモデル…」を続けるため。
        値 (``context_tokens=16384``) で終わる主語なので区切りが要る。
        """
        return f"要求したコンテキスト長 {self._request_context()} "

    def _model_setting_reference(self) -> str:
        """チャットのモデル ID の出典は ``generation.model`` の 1 か所。"""
        return f"generation.model ('{self._spec.model_id}')"

    def _context_length_remediation(self) -> str:
        """チャットで直すのは ``generation.context_tokens``。文面は不変。"""
        return (
            "generation.context_tokens を減らすか、"
            "より長いコンテキストを扱えるモデルへ切り替えてください"
        )

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
        return self._build_result(response, latency_s=latency_s)

    @abstractmethod
    def _build_request_body(self, messages: Sequence[ChatMessage]) -> dict[str, object]:
        """設定値をリクエストボディへ配線する (仕様書 §5 有効性観点 E1-E3 / E10)。"""

    @abstractmethod
    def _build_result(
        self, response: httpx.Response, *, latency_s: float
    ) -> ChatResult:
        """成功応答を :class:`ChatResult` へ変換する (D-07 に従い厳格にパース)。"""


class OpenAICompatibleClient(_HttpChatClient):
    """OpenAI 互換 ``/chat/completions`` を叩く同期クライアント。

    外部 API も、OpenAI 互換エンドポイントを持つローカルランタイム
    (``runtime.kind = "openai_compatible"``) も同じ実装で扱う。
    ``context_tokens`` は Ollama 拡張の ``options.num_ctx`` としてボディに載せる
    (互換エンドポイントは実機では無視する。ネイティブ経路は
    :class:`OllamaNativeClient`)。
    """

    _api_style: ClassVar[ApiStyle] = "openai_compatible"

    def _build_request_body(self, messages: Sequence[ChatMessage]) -> dict[str, object]:
        """設定値をリクエストボディへ配線する (仕様書 §5 有効性観点 E1-E3)。"""
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

    def _build_result(
        self, response: httpx.Response, *, latency_s: float
    ) -> ChatResult:
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

    def _parse_completion(self, response: httpx.Response) -> _ChatCompletion:
        """応答 JSON を厳格にパースする (D-07)。"""
        payload = self._decode_json(
            response,
            remediation=(
                "runtime.base_url が OpenAI 互換エンドポイント "
                "(末尾 /v1) を指しているか確認してください"
            ),
        )

        try:
            completion = _COMPLETION_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            self._raise_schema_error(
                exc,
                remediation=(
                    "ランタイムが OpenAI 互換の chat completion 形式を"
                    "返しているか確認してください"
                ),
            )

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


class OllamaNativeClient(_HttpChatClient):
    """Ollama ネイティブ ``/api/chat`` を叩く同期クライアント (D-10)。

    OpenAI 互換経路と違い ``options.num_ctx`` が実機に反映される (Phase 0 実測)。
    ``stream`` は必ず ``false`` を送る (省略すると NDJSON が返り、1 応答として
    パースできない)。公開 API に露出するのはクラス名だけで、ランタイム固有の
    型は返さない。
    """

    _api_style: ClassVar[ApiStyle] = "ollama_native"

    def _build_request_body(self, messages: Sequence[ChatMessage]) -> dict[str, object]:
        """設定値をネイティブのボディへ配線する (仕様書 §5 有効性観点 E10)。"""
        generation = self._config.generation
        return {
            "model": self._spec.served_name,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            # 省略すると NDJSON ストリームが返るため必須。
            "stream": False,
            "options": {
                "num_ctx": generation.context_tokens,
                "temperature": generation.temperature,
                "top_p": generation.top_p,
                "num_predict": generation.max_output_tokens,
                "seed": generation.seed,
            },
        }

    def _raise_for_error_status(self, response: httpx.Response) -> None:
        """エラーステータスに加え、HTTP 200 + 本文 ``error`` も翻訳表へ流す。"""
        super()._raise_for_error_status(response)
        self._raise_if_body_reports_error(response)

    def _build_result(
        self, response: httpx.Response, *, latency_s: float
    ) -> ChatResult:
        parsed = self._parse_response(response)
        if not parsed.done:
            msg = (
                f"推論ランタイムが未完了の応答 (done=false) を返しました "
                f"(url={self.endpoint_url}, model={self._spec.served_name})"
            )
            raise UpstreamError(
                msg,
                remediation=(
                    "リクエストに stream=false が載っているか、"
                    "ランタイム側のログで生成が打ち切られていないか確認してください"
                ),
            )

        # TokenUsage は非 Optional のため欠測は 0 とする。欠測かどうかの識別は
        # ChatTimings 側の None が担う (D-13)。
        prompt_tokens = parsed.prompt_eval_count or 0
        completion_tokens = parsed.eval_count or 0
        return ChatResult(
            text=parsed.message.content,
            model=parsed.model,
            finish_reason=parsed.done_reason or "stop",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_s=latency_s,
            timings=ChatTimings(
                prompt_eval_count=parsed.prompt_eval_count,
                prompt_eval_seconds=_seconds_from_nanoseconds(
                    parsed.prompt_eval_duration
                ),
                eval_count=parsed.eval_count,
                eval_seconds=_seconds_from_nanoseconds(parsed.eval_duration),
            ),
        )

    def _parse_response(self, response: httpx.Response) -> _NativeChatResponse:
        """応答 JSON を厳格にパースする (D-07)。必須は 3 フィールドだけ。"""
        payload = self._decode_json(
            response,
            remediation=(
                "リクエストに stream=false が載っているか、runtime.base_url が"
                "ネイティブ API (/api/chat) を導出できる値か確認してください"
            ),
        )

        try:
            return _NATIVE_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            self._raise_schema_error(
                exc,
                remediation=(
                    "ランタイムが Ollama ネイティブ /api/chat の応答形式を"
                    "返しているか確認してください"
                ),
            )


def create_chat_client(
    config: AppConfig, *, http_client: httpx.Client | None = None
) -> ChatClient:
    """``runtime.kind`` に対応する推論クライアントを生成する (D-10)。

    **``runtime.kind`` を見る分岐はここ 1 か所だけ**。具象クラスは自分がどの
    経路かを知っているが、どの経路を使うかは決めない。

    Args:
        config: 読み込み済み設定。
        http_client: 注入する ``httpx.Client`` (テストでは MockTransport)。

    Returns:
        ``kind = "ollama"`` なら :class:`OllamaNativeClient`、
        ``"openai_compatible"`` なら :class:`OpenAICompatibleClient`。
    """
    style = api_style_for(config.runtime.kind)
    # ``style`` が増えたときに追加漏れを mypy strict が検出できるよう
    # ``match`` + ``assert_never`` で網羅する。既定を持たせると
    # endpoint_url_for と向きが逆の fallthrough になりかねない (F-4-002)。
    match style:
        case "ollama_native":
            return OllamaNativeClient(config, http_client=http_client)
        case "openai_compatible":
            return OpenAICompatibleClient(config, http_client=http_client)
        case _:
            assert_never(style)
