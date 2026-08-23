"""埋め込みクライアント (L2)。

- 上位層 (L3) は :class:`EmbeddingClient` Protocol と :class:`EmbeddingBatch`
  にのみ依存する。``httpx`` の型や生の JSON dict は公開 API に露出させない。
  RAG 側 (L3) に HTTP を持たせない (D-25): 持たせると
  :class:`~llmkit.client._HttpEndpointClient` の例外翻訳表を複製することになり、
  「応答本文を例外メッセージに載せない」というセキュリティ原則が 2 か所に分岐する。
- 送信先は ``runtime.base_url`` から ``/embeddings`` を導出する
  (:func:`embeddings_url_for`)。**``runtime.kind`` で分岐しない**。D-10 が
  OpenAI 互換経路を退けた理由 (``options.num_ctx`` が無視される) は埋め込みには
  存在せず、Phase 0 で Ollama の ``/v1/embeddings`` が 768 次元を返すことを
  実測済みのため。新しい設定キーも追加しない (D-11 と同じ導出方針)。
- 埋め込みモデルは ``config.active_profile().embedding`` **だけ**を出典とする
  (D-27)。``generation.model`` は参照しない。
- レスポンスは pydantic dataclass + ``TypeAdapter`` で厳格にパースし、件数・
  順序・次元の不整合は :class:`llmkit.errors.UpstreamError` にする (D-07)。
  ``pydantic.BaseModel`` は使わない (D-08)。
- 例外メッセージに api_key の値・レスポンスボディ全文を載せない
  (CLAUDE.md セキュリティ原則)。埋め込みの入力テキスト (= ノート本文) も
  ログ・例外に 1 文字も出さない。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as pydantic_dataclass

from llmkit.catalog import ModelRole, ModelSpec, resolve_model_spec
from llmkit.client import _HttpEndpointClient
from llmkit.config import AppConfig
from llmkit.errors import ConfigError, UpstreamError

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingBatch",
    "EmbeddingClient",
    "OpenAIEmbeddingClient",
    "create_embedding_client",
    "embeddings_url_for",
]

_EMBEDDINGS_PATH = "/embeddings"

#: ``ModelSpec.role`` に要求する値。カタログ側の Literal をそのまま使う。
_EMBEDDING_ROLE: ModelRole = "embedding"


def embeddings_url_for(base_url: str) -> str:
    """``base_url`` から埋め込みエンドポイントの完全 URL を導出する。

    ``runtime.kind`` では分岐しない。Ollama にはネイティブの埋め込み API
    (``/api/embed``) もあるが、D-10 が OpenAI 互換経路を退けた理由
    (``options.num_ctx`` が無視される) は埋め込みには存在せず、Phase 0 で
    ``/v1/embeddings`` の 768 次元応答を実測しているため、経路を 1 本に保つ。

    HTTP を 1 バイトも出さない純関数。

    Args:
        base_url: ``runtime.base_url`` (OpenAI 互換の ``/v1`` を指す前提)。

    Returns:
        ``base_url`` の末尾スラッシュを落として ``/embeddings`` を連結した URL。
    """
    return f"{base_url.rstrip('/')}{_EMBEDDINGS_PATH}"


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """1 回の埋め込み要求の結果。

    Attributes:
        vectors: 入力と**同じ順序**に整列済みのベクトル列。
        model: ランタイムが名乗ったモデル名 (設定値ではなく応答由来)。
        dimensions: 全ベクトルに共通の次元数。
        latency_s: HTTP 往復の所要時間 (秒)。
    """

    vectors: tuple[tuple[float, ...], ...]
    model: str
    dimensions: int
    latency_s: float


@runtime_checkable
class EmbeddingClient(Protocol):
    """L3 が依存する唯一の埋め込みインタフェース。"""

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """テキスト列を送ってベクトル列を得る (順序は入力と一致する)。"""
        ...


_RESPONSE_CONFIG = ConfigDict(extra="ignore")


@pydantic_dataclass(frozen=True, config=_RESPONSE_CONFIG)
class _EmbeddingItem:
    index: int
    embedding: list[float]


@pydantic_dataclass(frozen=True, config=_RESPONSE_CONFIG)
class _EmbeddingResponse:
    """OpenAI 互換 ``/embeddings`` の応答。

    必須は ``data`` (``index`` / ``embedding``) と ``model`` の 2 つ。
    ``object`` / ``usage`` は使わないため無視する。
    """

    data: list[_EmbeddingItem]
    model: str


_EMBEDDING_ADAPTER: TypeAdapter[_EmbeddingResponse] = TypeAdapter(_EmbeddingResponse)

_PARSE_REMEDIATION = (
    "runtime.base_url が OpenAI 互換エンドポイント (末尾 /v1) を指しているか、"
    "指定したモデルが埋め込みモデルとしてランタイムに登録されているか"
    "確認してください"
)


def _resolve_embedding_spec(config: AppConfig) -> ModelSpec:
    """アクティブプロファイルの ``embedding`` を :class:`ModelSpec` へ解決する。

    出典はここ 1 か所だけ (D-27)。``generation.model`` は参照しない。

    Raises:
        ConfigError: プロファイルに ``embedding`` が無い場合、または解決した
            モデルの ``role`` が ``"embedding"`` でない場合。
    """
    profile_name = config.vram.active_profile
    model_id = config.active_profile().embedding
    if model_id is None:
        msg = (
            f"プロファイル '{profile_name}' に埋め込みモデルが設定されていません "
            f"(profiles.{profile_name}.embedding が未設定)"
        )
        raise ConfigError(
            msg,
            remediation=(
                f"設定ファイルの [profiles.{profile_name}] に "
                f'embedding = "ruri-v3-310m" を追加するか、embedding を持つ'
                f"プロファイルへ vram.active_profile を切り替えてください"
            ),
        )

    spec = resolve_model_spec(
        model_id, is_local=config.runtime.is_local, role=_EMBEDDING_ROLE
    )
    if spec.role != _EMBEDDING_ROLE:
        msg = (
            f"profiles.{profile_name}.embedding に指定されたモデル '{model_id}' の"
            f"役割は '{spec.role}' であり、埋め込みには使えません"
        )
        raise ConfigError(
            msg,
            remediation=(
                "llmkit/catalog.py で role='embedding' のモデル "
                "(例 ruri-v3-310m) を指定してください"
            ),
        )
    return spec


def _reject_unembeddable_texts(texts: tuple[str, ...]) -> None:
    """空の入力を早期に弾く。テキストの中身は例外に載せない。

    Raises:
        ConfigError: 入力が 0 件の場合、または空文字列を含む場合。
    """
    if not texts:
        msg = "埋め込み対象のテキストが 1 件もありません"
        raise ConfigError(
            msg,
            remediation=(
                "呼び出し側で空のバッチを送らないでください "
                "(空ノートは 0 チャンクとして扱い、要求そのものを行わない)"
            ),
        )
    empty_positions = [
        position for position, text in enumerate(texts) if not text.strip()
    ]
    if empty_positions:
        msg = (
            f"埋め込み対象に空文字列 (または空白のみ) のテキストが含まれています "
            f"(位置={empty_positions})"
        )
        raise ConfigError(
            msg,
            remediation=(
                "空・空白のみのテキストは埋め込み要求から除外してください "
                "(ランタイムは次元 0 のベクトルやエラーを返し、索引が静かに壊れます)"
            ),
        )


class OpenAIEmbeddingClient(_HttpEndpointClient):
    """OpenAI 互換 ``/embeddings`` を叩く同期クライアント。

    例外翻訳表・api_key ヘッダ・``httpx.Client`` の所有権は基底
    :class:`~llmkit.client._HttpEndpointClient` と共有する (D-25)。ここが持つのは
    エンドポイント導出・リクエストボディの組み立て・応答の整合性検査だけ。
    """

    def __init__(
        self, config: AppConfig, *, http_client: httpx.Client | None = None
    ) -> None:
        """Args:
        config: 読み込み済み設定。埋め込みモデルの出典は
            ``profiles[vram.active_profile].embedding`` のみ (D-27)。
        http_client: 注入する ``httpx.Client`` (テストでは MockTransport)。

        Raises:
            ConfigError: プロファイルに ``embedding`` が無い、role が
                ``"embedding"`` でない、または ``is_local=true`` で
                カタログ未登録の場合。
        """
        super().__init__(
            config, _resolve_embedding_spec(config), http_client=http_client
        )

    @property
    def endpoint_url(self) -> str:
        """リクエスト先の完全 URL (``runtime.kind`` に依存しない)。"""
        return embeddings_url_for(self._config.runtime.base_url)

    def _request_context(self) -> str:
        """例外メッセージ用の要求文脈。入力テキストは 1 文字も載せない。"""
        return f"embedding={self._spec.model_id}"

    def _context_length_subject(self) -> str:
        """埋め込みで長すぎたのは入力テキストそのもの。

        チャットの主語 (``要求したコンテキスト長 context_tokens=N``) を
        そのまま流用すると「要求したコンテキスト長 embedding=ruri-v3-310m」
        という非文になる。``embedding=...`` はモデル名でありコンテキスト長
        ではない。入力の中身も長さもメッセージに載せない (D-07)。
        """
        return "埋め込み入力"

    def _model_setting_reference(self) -> str:
        """埋め込みモデル ID の出典は有効プロファイルの ``embedding`` (D-27)。

        ``generation.model`` を名乗ってはならない。そこを書き換えても
        埋め込みは直らず、生成モデルだけが壊れる。
        """
        profile = self._config.vram.active_profile
        return f"profiles.{profile}.embedding ('{self._spec.model_id}')"

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """テキスト列を埋め込む。

        Args:
            texts: 埋め込むテキスト。空文字列・空白のみを含めてはならない。

        Returns:
            入力と同じ順序に整列済みの :class:`EmbeddingBatch`。

        Raises:
            ConfigError: 入力が 0 件、または空文字列を含む場合。
            RuntimeUnavailableError: ランタイムに接続できない場合。
            ModelNotFoundError: ランタイム側にモデルが無い場合。
            OutOfMemoryError: ランタイム側で VRAM が枯渇した場合。
            ContextLengthError: 入力がモデルのコンテキスト長を超えた場合。
            UpstreamError: その他の 4xx/5xx、応答の解析に失敗した場合、または
                件数・``index``・次元が入力と整合しない場合。
        """
        payload = tuple(texts)
        _reject_unembeddable_texts(payload)

        body = self._build_request_body(payload)
        logger.debug(
            "埋め込みリクエストを送信します: url=%s model=%s texts=%d",
            self.endpoint_url,
            self.served_name,
            len(payload),
        )
        started = time.perf_counter()
        response = self._send(body)
        latency_s = time.perf_counter() - started

        self._raise_for_error_status(response)
        return self._build_batch(response, expected=len(payload), latency_s=latency_s)

    def _raise_for_error_status(self, response: httpx.Response) -> None:
        """エラーステータスに加え、HTTP 200 + 本文 ``error`` も翻訳表へ流す。"""
        super()._raise_for_error_status(response)
        self._raise_if_body_reports_error(response)

    def _context_length_remediation(self) -> str:
        """埋め込みで直すのは ``generation.context_tokens`` ではない。"""
        return (
            "チャンクの上限トークン数を下げて分割し直すか、"
            "より長いコンテキストを扱える埋め込みモデルへ切り替えてください"
        )

    def _build_request_body(self, texts: tuple[str, ...]) -> dict[str, object]:
        """設定値をリクエストボディへ配線する (仕様書 §5 有効性観点 E27)。"""
        return {"model": self._spec.served_name, "input": list(texts)}

    def _build_batch(
        self, response: httpx.Response, *, expected: int, latency_s: float
    ) -> EmbeddingBatch:
        parsed = self._parse_response(response)
        items = self._ordered_items(parsed.data, expected=expected)
        dimensions = self._common_dimensions(items)
        return EmbeddingBatch(
            vectors=tuple(tuple(item.embedding) for item in items),
            model=parsed.model,
            dimensions=dimensions,
            latency_s=latency_s,
        )

    def _parse_response(self, response: httpx.Response) -> _EmbeddingResponse:
        """応答 JSON を厳格にパースする (D-07)。"""
        payload = self._decode_json(response, remediation=_PARSE_REMEDIATION)
        try:
            return _EMBEDDING_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            self._raise_schema_error(exc, remediation=_PARSE_REMEDIATION)

    def _ordered_items(
        self, items: Sequence[_EmbeddingItem], *, expected: int
    ) -> tuple[_EmbeddingItem, ...]:
        """``index`` で昇順に整列し、入力と 1 対 1 で対応することを確かめる。

        件数が合わないまま返すと、上位層は「n 番目のチャンクのベクトル」を
        取り違えたまま索引を作り、例外もテスト失敗も出さずに検索結果だけが
        壊れる。応答本文はメッセージに載せない (D-07)。
        """
        if len(items) != expected:
            msg = (
                f"埋め込み応答の件数が要求と一致しません "
                f"(要求={expected}, 応答={len(items)}, "
                f"url={self.endpoint_url}, model={self._spec.served_name})"
            )
            raise UpstreamError(
                msg,
                remediation=(
                    "ランタイム側のログを確認してください "
                    "(応答本文は秘匿のためメッセージに含めていません)"
                ),
            )

        ordered = tuple(sorted(items, key=lambda item: item.index))
        actual_indices = tuple(item.index for item in ordered)
        if actual_indices != tuple(range(expected)):
            msg = (
                f"埋め込み応答の index が 0..{expected - 1} の並びになっていません "
                f"(url={self.endpoint_url}, model={self._spec.served_name})"
            )
            raise UpstreamError(
                msg,
                remediation=(
                    "ランタイムが OpenAI 互換の embeddings 形式 "
                    "(data[].index が入力位置) を返しているか確認してください"
                ),
            )
        return ordered

    def _common_dimensions(self, items: Sequence[_EmbeddingItem]) -> int:
        """全ベクトルに共通の次元数を返す。不揃い・0 次元は UpstreamError。

        次元が混ざったベクトルをそのまま永続化すると、コサイン類似度が計算
        できないか、静かに無意味な値を返す。ここで止める。
        """
        dimensions = len(items[0].embedding)
        mismatched = sorted(
            {len(item.embedding) for item in items} - {dimensions},
        )
        if mismatched:
            msg = (
                f"埋め込み応答のベクトル次元が揃っていません "
                f"(先頭={dimensions}, 他={mismatched}, "
                f"url={self.endpoint_url}, model={self._spec.served_name})"
            )
            raise UpstreamError(
                msg,
                remediation=(
                    "同一モデルの応答か、ランタイム側で切り詰めが"
                    "起きていないか確認してください"
                ),
            )
        if dimensions == 0:
            msg = (
                f"埋め込み応答のベクトルが空です (次元 0, "
                f"url={self.endpoint_url}, model={self._spec.served_name})"
            )
            raise UpstreamError(
                msg,
                remediation=(
                    "指定したモデルが埋め込みモデルとしてランタイムに"
                    "登録されているか確認してください"
                ),
            )
        return dimensions


def create_embedding_client(
    config: AppConfig, *, http_client: httpx.Client | None = None
) -> EmbeddingClient:
    """埋め込みクライアントを生成する。

    ``runtime.kind`` では分岐しない (:func:`embeddings_url_for` 参照)。経路が
    1 本であることを型と実装の両方で示すため、``create_chat_client`` のような
    ``match`` は置かない。

    Args:
        config: 読み込み済み設定。
        http_client: 注入する ``httpx.Client`` (テストでは MockTransport)。

    Returns:
        :class:`OpenAIEmbeddingClient`。

    Raises:
        ConfigError: 埋め込みモデルの解決に失敗した場合
            (:func:`_resolve_embedding_spec` 参照)。
    """
    return OpenAIEmbeddingClient(config, http_client=http_client)
