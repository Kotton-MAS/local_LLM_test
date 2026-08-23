"""埋め込みクライアント (L2) の検証 (仕様書 §4 T1)。

実 HTTP は 1 バイトも発行しない (D-02)。すべて ``httpx.MockTransport``
(``conftest.RecordingTransport``) で完結する。

このファイルが固定するのは 3 点:

1. **例外翻訳表を chat と共有していること** — 404 / 接続不能 / VRAM 不足 /
   HTTP 200 + 本文 error のいずれも、``llmkit/client.py`` の
   ``_HttpEndpointClient`` にある唯一の実装を通る (D-25)。基底の分岐を 1 つ
   潰すと ``test_client_errors.py`` とこのファイルの**両方**が落ちる。
2. **埋め込みモデルの出典が 1 か所であること** — ``profiles[active].embedding``
   だけがリクエストの ``model`` を決める (D-27 / E27)。
3. **応答の整合性** — 件数・``index`` の並び・次元が入力と整合しないまま
   通さない。通すと索引が例外もテスト失敗も出さずに壊れる (D-07)。
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
import pytest
from conftest import DEFAULT_CONFIG, ConfigWriter, RecordingTransport

from llmkit.config import AppConfig, load_config
from llmkit.embeddings import (
    EmbeddingBatch,
    EmbeddingClient,
    OpenAIEmbeddingClient,
    create_embedding_client,
    embeddings_url_for,
)
from llmkit.errors import (
    ConfigError,
    ContextLengthError,
    LlmkitError,
    ModelNotFoundError,
    OutOfMemoryError,
    RuntimeUnavailableError,
    UpstreamError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_MODULE = REPO_ROOT / "llmkit" / "embeddings.py"

#: configs/default.toml の profiles.rag_default.embedding が指すモデルの served_name。
SERVED_NAME = "hf.co/Targoyle/ruri-v3-310m-GGUF"

#: Phase 0 実測 (docs/phase0-vram-measurements.md) の次元数。
DIMENSIONS = 768

API_KEY = "sk-test-do-not-leak-0123456789"

#: 応答本文に混ぜる「例外メッセージに出てはいけない」文字列。
SECRET_DETAIL = "INTERNAL-STACKTRACE-DO-NOT-LEAK"

#: 埋め込む側の秘密 (ノート本文相当)。ログ・例外に 1 文字も出てはいけない。
SECRET_NOTE = "社外秘のノート本文-DO-NOT-LEAK"

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------
# 補助
# --------------------------------------------------------------------------


def vector(seed: int, *, dimensions: int = DIMENSIONS) -> list[float]:
    """入力位置ごとに区別できる決定論的なベクトル。"""
    return [float(seed)] * dimensions


def embedding_payload(
    *,
    count: int,
    dimensions: int = DIMENSIONS,
    indices: Sequence[int] | None = None,
    model: str = SERVED_NAME,
) -> dict[str, object]:
    """OpenAI 互換 ``/embeddings`` の正常応答を組み立てる。"""
    order = tuple(indices) if indices is not None else tuple(range(count))
    return {
        "object": "list",
        "model": model,
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": vector(index, dimensions=dimensions),
            }
            for index in order
        ],
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }


def responding_json(payload: dict[str, object], *, status: int = 200) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def responding_text(status: int, body: str) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return handler


def raising(exc: Exception) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def run_embed(
    config: AppConfig, handler: Handler, texts: Sequence[str]
) -> tuple[EmbeddingBatch, RecordingTransport]:
    """MockTransport 経由で 1 回だけ埋め込みを実行する。"""
    transport = RecordingTransport(handler)
    with transport.client() as http_client:
        client = create_embedding_client(config, http_client=http_client)
        return client.embed(texts), transport


@pytest.fixture
def local_config() -> AppConfig:
    """configs/default.toml (kind=ollama / is_local=true / embedding あり)。"""
    return load_config(DEFAULT_CONFIG)


# --------------------------------------------------------------------------
# (a) 正常系: 件数・次元・順序
# --------------------------------------------------------------------------


def test_three_texts_return_three_vectors_in_input_order(
    local_config: AppConfig,
) -> None:
    """受け入れ基準 (a): 3 件のテキスト → 3 ベクトル・次元 768・入力順。"""
    batch, transport = run_embed(
        local_config,
        responding_json(embedding_payload(count=3)),
        ["いち", "に", "さん"],
    )

    assert batch.dimensions == DIMENSIONS
    assert len(batch.vectors) == 3
    assert all(len(row) == DIMENSIONS for row in batch.vectors)
    assert batch.vectors == tuple(tuple(vector(index)) for index in range(3))
    assert batch.model == SERVED_NAME
    assert batch.latency_s >= 0.0
    assert transport.call_count == 1


def test_request_body_carries_every_text_in_order(local_config: AppConfig) -> None:
    """``input`` は入力順そのままで、1 リクエストにまとまる。"""
    _, transport = run_embed(
        local_config,
        responding_json(embedding_payload(count=3)),
        ["いち", "に", "さん"],
    )

    request = transport.requests[0]
    body = request.read().decode("utf-8")
    assert '"input"' in body
    assert body.index("いち") < body.index("に") < body.index("さん")
    assert request.url.path.endswith("/v1/embeddings")


def test_the_client_satisfies_the_embedding_client_protocol(
    local_config: AppConfig,
) -> None:
    """L3 は Protocol にだけ依存すれば足りる。"""
    transport = RecordingTransport()
    with transport.client() as http_client:
        client = create_embedding_client(local_config, http_client=http_client)
        assert isinstance(client, EmbeddingClient)
        assert isinstance(client, OpenAIEmbeddingClient)


# --------------------------------------------------------------------------
# (b) 件数不一致 / (c) index の整列 / 次元
# --------------------------------------------------------------------------


def test_short_data_raises_upstream_error_without_the_body(
    local_config: AppConfig,
) -> None:
    """受け入れ基準 (b): ``data`` が 2 件しか返らない → UpstreamError。

    件数が合わないまま返すと、上位層は「n 番目のチャンクのベクトル」を
    取り違えたまま索引を作り、例外もテスト失敗も出さずに検索結果だけが壊れる。
    """
    payload = embedding_payload(count=2)
    payload["warning"] = SECRET_DETAIL

    with pytest.raises(UpstreamError) as excinfo:
        run_embed(local_config, responding_json(payload), ["いち", "に", "さん"])

    message = str(excinfo.value)
    assert "要求=3" in message
    assert "応答=2" in message
    assert SECRET_DETAIL not in message


def test_reversed_index_is_sorted_back_into_input_order(
    local_config: AppConfig,
) -> None:
    """受け入れ基準 (c): ``index`` が逆順で返っても入力順に整列される。"""
    batch, _ = run_embed(
        local_config,
        responding_json(embedding_payload(count=3, indices=(2, 1, 0))),
        ["いち", "に", "さん"],
    )

    assert batch.vectors == tuple(tuple(vector(index)) for index in range(3))


def test_duplicated_index_raises_upstream_error(local_config: AppConfig) -> None:
    """件数が合っていても ``index`` が 0..n-1 の並びでなければ対応付けできない。"""
    with pytest.raises(UpstreamError) as excinfo:
        run_embed(
            local_config,
            responding_json(embedding_payload(count=3, indices=(0, 0, 1))),
            ["いち", "に", "さん"],
        )

    assert "index" in str(excinfo.value)


def test_mixed_dimensions_raise_upstream_error(local_config: AppConfig) -> None:
    """次元が揃わないベクトルを索引に入れるとコサイン類似度が無意味になる。"""
    payload = embedding_payload(count=2)
    data = payload["data"]
    assert isinstance(data, list)
    entry = data[1]
    assert isinstance(entry, dict)
    entry["embedding"] = vector(1, dimensions=DIMENSIONS - 1)

    with pytest.raises(UpstreamError) as excinfo:
        run_embed(local_config, responding_json(payload), ["いち", "に"])

    message = str(excinfo.value)
    assert "次元" in message
    assert str(DIMENSIONS) in message


def test_empty_vector_raises_upstream_error(local_config: AppConfig) -> None:
    """次元 0 のベクトルは「埋め込めた」ことにしない。"""
    payload = embedding_payload(count=1, dimensions=0)

    with pytest.raises(UpstreamError) as excinfo:
        run_embed(local_config, responding_json(payload), ["いち"])

    assert "次元 0" in str(excinfo.value)


def test_malformed_response_raises_upstream_error(local_config: AppConfig) -> None:
    """D-07: 必須フィールド欠損は UpstreamError (入力値は載せない)。"""
    with pytest.raises(UpstreamError) as excinfo:
        run_embed(
            local_config,
            responding_json({"object": "list", "model": SERVED_NAME}),
            ["いち"],
        )

    message = str(excinfo.value)
    assert "必須フィールド" in message
    assert "data" in message


def test_non_json_response_raises_upstream_error(local_config: AppConfig) -> None:
    with pytest.raises(UpstreamError) as excinfo:
        run_embed(local_config, responding_text(200, "<html>error</html>"), ["いち"])

    assert "JSON ではありません" in str(excinfo.value)


# --------------------------------------------------------------------------
# (d)(e)(f) 例外翻訳表の共有 (D-25)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (404, '{"error": {"message": "not found"}}'),
        (500, '{"error": "model not found, try pulling it first"}'),
    ],
    ids=["HTTP 404", "本文に model not found"],
)
def test_missing_model_maps_to_model_not_found(
    local_config: AppConfig, status: int, body: str
) -> None:
    """受け入れ基準 (d): 404 → ModelNotFoundError (served_name を含む)。

    ここが落ちるとき ``test_client_errors.py`` も落ちるなら、翻訳表が chat と
    共有されている証拠になる (D-25)。
    """
    with pytest.raises(ModelNotFoundError) as excinfo:
        run_embed(local_config, responding_text(status, body), ["いち"])

    message = str(excinfo.value)
    assert SERVED_NAME in message
    assert f"ollama pull {SERVED_NAME}" in message


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("timed out"),
    ],
    ids=["ConnectError", "ConnectTimeout"],
)
def test_connect_failure_maps_to_runtime_unavailable(
    local_config: AppConfig, exc: Exception
) -> None:
    """受け入れ基準 (e): 接続不能 → RuntimeUnavailableError。"""
    with pytest.raises(RuntimeUnavailableError) as excinfo:
        run_embed(local_config, raising(exc), ["いち"])

    message = str(excinfo.value)
    assert "http://localhost:11434/v1" in message
    assert "ollama serve" in message


def test_body_error_on_http_200_uses_the_shared_translation_table(
    local_config: AppConfig,
) -> None:
    """受け入れ基準 (f): 200 + 本文 error も既存の翻訳表に従う。

    ステータスだけを見るとスキーマ違反 (UpstreamError) に化けて原因が消える。
    """
    with pytest.raises(ModelNotFoundError):
        run_embed(
            local_config,
            responding_text(200, '{"error": "model not found, try pulling it first"}'),
            ["いち"],
        )


def test_out_of_memory_body_maps_to_out_of_memory_error(
    local_config: AppConfig,
) -> None:
    """VRAM 不足も chat と同じ分岐を通る。文面は埋め込み側の文脈を名乗る。"""
    with pytest.raises(OutOfMemoryError) as excinfo:
        run_embed(
            local_config,
            responding_text(500, '{"error": "CUDA error: out of memory"}'),
            ["いち"],
        )

    message = str(excinfo.value)
    assert "rag_default" in message
    assert "embedding=ruri-v3-310m" in message
    # generation.context_tokens は埋め込み要求と無関係。混入させない。
    assert "context_tokens=16384" not in message


def test_context_length_message_does_not_claim_a_generation_setting(
    local_config: AppConfig,
) -> None:
    """埋め込みは ``generation.context_tokens`` を要求していない。

    知らない要求量を知っているふりで書くと、対処の方向 (チャンク上限を下げる)
    が読み手に伝わらない。
    """
    with pytest.raises(ContextLengthError) as excinfo:
        run_embed(
            local_config,
            responding_text(400, '{"error": "maximum context length exceeded"}'),
            ["いち"],
        )

    message = str(excinfo.value)
    assert SERVED_NAME in message
    assert "max_context_tokens=8192" in message
    assert "context_tokens=16384" not in message
    assert "チャンク" in message


def test_context_length_message_does_not_invent_a_limit_for_passthrough_models(
    tmp_config: ConfigWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-9-001 回帰: is_local=false + カタログ未登録の埋め込みモデルでは、

    存在しない上限を断言しない。カタログ登録済みかどうかの分岐
    (F-2-002, ``client.py`` の ``_context_length_message``) は chat と埋め込みの
    両経路で共有される 1 実装であるべきで、埋め込み側だけが上書きして分岐を
    欠落させてはならない。
    """
    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)
    config = load_config(
        tmp_config(
            {
                "is_local = true": "is_local = false",
                'embedding = "ruri-v3-310m"': 'embedding = "text-embedding-3-small"',
            }
        )
    )

    with pytest.raises(ContextLengthError) as excinfo:
        run_embed(
            config,
            responding_text(400, '{"error": "maximum context length exceeded"}'),
            ["いち"],
        )

    message = str(excinfo.value)
    assert "text-embedding-3-small" in message
    assert "1048576" not in message
    assert "上限は不明" in message


def test_other_status_maps_to_upstream_error_without_body(
    local_config: AppConfig,
) -> None:
    with pytest.raises(UpstreamError) as excinfo:
        run_embed(
            local_config,
            responding_text(503, f'{{"error": "{SECRET_DETAIL}"}}'),
            ["いち"],
        )

    message = str(excinfo.value)
    assert "503" in message
    assert SECRET_DETAIL not in message


def test_other_httpx_errors_map_to_upstream_error(local_config: AppConfig) -> None:
    """httpx 固有の例外を L2 の公開 API に漏らさない。"""
    with pytest.raises(UpstreamError) as excinfo:
        run_embed(local_config, raising(httpx.ReadTimeout("read timed out")), ["いち"])

    assert not isinstance(excinfo.value, httpx.HTTPError)
    assert "ReadTimeout" in str(excinfo.value)


FAILURE_HANDLERS: dict[str, Handler] = {
    "接続不可": raising(httpx.ConnectError("connection refused")),
    "モデル不在": responding_text(404, '{"error": "not found"}'),
    "VRAM 不足": responding_text(500, '{"error": "CUDA error: out of memory"}'),
    "コンテキスト超過": responding_text(400, '{"error": "context length exceeded"}'),
    "その他 5xx": responding_text(503, '{"error": "unavailable"}'),
    "件数不一致": responding_json(embedding_payload(count=1)),
}


@pytest.mark.parametrize(
    "handler", list(FAILURE_HANDLERS.values()), ids=list(FAILURE_HANDLERS)
)
def test_error_messages_never_contain_the_input_text(
    local_config: AppConfig, handler: Handler
) -> None:
    """どの失敗経路でもノート本文が例外に漏れない (CLAUDE.md ログ出力ルール)。"""
    with pytest.raises(LlmkitError) as excinfo:
        run_embed(local_config, handler, [SECRET_NOTE, "に"])

    assert SECRET_NOTE not in str(excinfo.value)
    assert SECRET_NOTE not in repr(excinfo.value)
    assert SECRET_NOTE not in excinfo.value.remediation


def test_error_messages_never_contain_the_api_key(
    tmp_config: ConfigWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)
    config = load_config(tmp_config({"is_local = true": "is_local = false"}))
    assert config.api_key.get_secret_value() == API_KEY

    with pytest.raises(LlmkitError) as excinfo:
        run_embed(config, responding_text(503, '{"error": "unavailable"}'), ["いち"])

    assert API_KEY not in str(excinfo.value)
    assert API_KEY not in repr(excinfo.value)


# --------------------------------------------------------------------------
# エンドポイント導出
# --------------------------------------------------------------------------


def test_embeddings_url_is_derived_from_the_base_url() -> None:
    assert embeddings_url_for("http://h:11434/v1") == "http://h:11434/v1/embeddings"
    assert embeddings_url_for("http://h:11434/v1/") == "http://h:11434/v1/embeddings"
    assert (
        embeddings_url_for("https://api.example.com/v1")
        == "https://api.example.com/v1/embeddings"
    )


@pytest.mark.parametrize(
    "edits",
    [{}, {'kind = "ollama"': 'kind = "openai_compatible"'}],
    ids=["ollama", "openai_compatible"],
)
def test_embeddings_url_does_not_depend_on_runtime_kind(
    tmp_config: ConfigWriter, edits: dict[str, str]
) -> None:
    """``runtime.kind`` を変えても送信先が変わらない。

    chat は D-10 で経路が 2 本に分かれるが、埋め込みにその根拠
    (``options.num_ctx`` が無視される) は無く、経路は 1 本に保つ。
    """
    config = load_config(tmp_config(edits))

    _, transport = run_embed(
        config, responding_json(embedding_payload(count=1)), ["いち"]
    )

    assert str(transport.requests[0].url) == "http://localhost:11434/v1/embeddings"


# --------------------------------------------------------------------------
# E27: 埋め込みモデルの出典は profiles[active].embedding だけ (D-27)
# --------------------------------------------------------------------------


def request_model(config: AppConfig) -> str:
    """1 回埋め込んで、リクエストボディの ``model`` を取り出す。"""
    _, transport = run_embed(
        config, responding_json(embedding_payload(count=1)), ["いち"]
    )
    body = transport.requests[0].read().decode("utf-8")
    match = re.search(r'"model"\s*:\s*"([^"]+)"', body)
    assert match is not None, body
    return match.group(1)


def test_embedding_model_comes_from_the_active_profile_only(
    tmp_config: ConfigWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E27 / D-27 guard: リクエストの ``model`` を決めるのはただ 1 か所。

    掃引するのは ``profiles[vram.active_profile].embedding``。``generation.model``
    を振っても埋め込みリクエストは変わらない。出典が 2 か所あると、片方だけを
    変えた実行が「別のモデルで埋め込んだ索引」を静かに作る (D-19 と同型)。
    """
    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)
    external = {"is_local = true": "is_local = false"}

    baseline = load_config(tmp_config(external, name="baseline.toml"))
    assert request_model(baseline) == SERVED_NAME

    # generation.model を振っても埋め込みリクエストは動かない。
    other_generation = load_config(
        tmp_config(
            {**external, 'model = "qwen3-14b"': 'model = "qwen3-8b"'},
            name="other_generation.toml",
        )
    )
    assert request_model(other_generation) == SERVED_NAME

    # profiles[active].embedding を振ると動く。
    other_embedding = load_config(
        tmp_config(
            {
                **external,
                'embedding = "ruri-v3-310m"': 'embedding = "text-embedding-3-small"',
            },
            name="other_embedding.toml",
        )
    )
    assert request_model(other_embedding) == "text-embedding-3-small"


def test_profile_without_an_embedding_model_raises_config_error(
    tmp_config: ConfigWriter,
) -> None:
    """``profiles.long_context`` には embedding が無い。"""
    config = load_config(
        tmp_config(
            {'active_profile = "rag_default"': 'active_profile = "long_context"'}
        )
    )
    transport = RecordingTransport()

    with transport.client() as http_client, pytest.raises(ConfigError) as excinfo:
        create_embedding_client(config, http_client=http_client)

    message = str(excinfo.value)
    assert "embedding" in message
    assert "long_context" in message
    assert "ruri-v3-310m" in excinfo.value.remediation
    assert transport.call_count == 0


def test_a_non_embedding_role_model_is_rejected(tmp_config: ConfigWriter) -> None:
    """役割 (role) の検査。生成モデルを embedding に書いても通さない。"""
    config = load_config(
        tmp_config({'embedding = "ruri-v3-310m"': 'embedding = "qwen3-14b"'})
    )
    transport = RecordingTransport()

    with transport.client() as http_client, pytest.raises(ConfigError) as excinfo:
        create_embedding_client(config, http_client=http_client)

    message = str(excinfo.value)
    assert "qwen3-14b" in message
    assert "generation" in message
    assert transport.call_count == 0


# --------------------------------------------------------------------------
# 入力の検証・所有権・隔離
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "texts",
    [["いち", "", "さん"], ["いち", "   \n ", "さん"]],
    ids=["空文字列", "空白のみ"],
)
def test_empty_text_raises_config_error_without_sending(
    local_config: AppConfig, texts: list[str]
) -> None:
    """空文字列を送るとランタイムの挙動が分かれ、索引が静かに壊れる。"""
    transport = RecordingTransport()
    with transport.client() as http_client:
        client = create_embedding_client(local_config, http_client=http_client)
        with pytest.raises(ConfigError) as excinfo:
            client.embed(texts)

    assert "位置=[1]" in str(excinfo.value)
    assert transport.call_count == 0


def test_empty_batch_raises_config_error_without_sending(
    local_config: AppConfig,
) -> None:
    transport = RecordingTransport()
    with transport.client() as http_client:
        client = create_embedding_client(local_config, http_client=http_client)
        with pytest.raises(ConfigError):
            client.embed([])

    assert transport.call_count == 0


def test_create_embedding_client_sends_nothing_at_construction(
    local_config: AppConfig,
) -> None:
    transport = RecordingTransport()
    with transport.client() as http_client:
        create_embedding_client(local_config, http_client=http_client)

    assert transport.call_count == 0


def test_injected_http_client_is_not_closed(local_config: AppConfig) -> None:
    """注入された ``httpx.Client`` の所有権は呼び出し側に残る。"""
    transport = RecordingTransport()
    with transport.client() as http_client:
        client = create_embedding_client(local_config, http_client=http_client)
        assert isinstance(client, OpenAIEmbeddingClient)
        with client:
            pass
        assert not http_client.is_closed


def test_api_key_header_is_sent_only_when_configured(
    local_config: AppConfig, tmp_config: ConfigWriter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-05: api_key が空なら Authorization を付けない。"""
    _, local_transport = run_embed(
        local_config, responding_json(embedding_payload(count=1)), ["いち"]
    )
    assert "Authorization" not in local_transport.requests[0].headers

    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)
    external = load_config(tmp_config({"is_local = true": "is_local = false"}))
    _, external_transport = run_embed(
        external, responding_json(embedding_payload(count=1)), ["いち"]
    )
    assert (
        external_transport.requests[0].headers["Authorization"] == f"Bearer {API_KEY}"
    )


def test_embeddings_module_never_touches_the_gpu_or_spawns_processes() -> None:
    """D-01 / D-23 を L2 の新モジュールでも維持する (AST。全文一致ではない)。

    docstring には ``nvidia-smi`` を「参照しない」という説明を書く余地があるため、
    実行されるコード (import と非 docstring の文字列リテラル) だけを見る。
    """
    tree = ast.parse(EMBEDDINGS_MODULE.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    forbidden = frozenset({"subprocess", "GPUtil", "pynvml", "nvidia_smi"})

    offences: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offences += [
                f"import {alias.name}"
                for alias in node.names
                if alias.name.split(".")[0] in forbidden
            ]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.split(".")[0] in forbidden:
                offences.append(f"from {node.module} import ...")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and re.search(r"nvidia-smi|nvidia_smi", node.value)
        ):
            offences.append(f"literal {node.value!r}")

    assert not offences, (
        f"llmkit/embeddings.py が GPU / 外部プロセスに触れている: {offences}"
    )


def test_model_not_found_points_at_the_setting_that_actually_holds_the_id(
    tmp_config: ConfigWriter,
) -> None:
    """404 の対処が案内する設定キーが、実際にそのモデル ID を持つこと。

    埋め込みモデル ID の出典は ``profiles.<名前>.embedding`` の 1 か所である
    (D-27)。基底が ``generation.model`` を名乗ると、実在しない設定キーを
    直せと案内することになる。実際にその案内へ従うと、埋め込みは直らない
    まま生成モデルだけが壊れる。

    この検査は「案内先の設定キーに、メッセージが名乗るモデル ID が本当に
    書かれているか」を設定ファイルから読み直して確かめる。文言だけを固定
    すると、設定側の構造が変わったときに気づけない。
    """
    config = load_config(tmp_config())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    client = create_embedding_client(
        config, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ModelNotFoundError) as excinfo:
        client.embed(["これは埋め込み対象のテキストです。"])

    remediation = str(excinfo.value)

    # 生成側の設定キーを名乗ってはならない。
    assert "generation.model" not in remediation, (
        f"埋め込みの対処が generation.model を名乗っている: {remediation}"
    )

    # 案内された設定キーを設定オブジェクトから引き直し、同じ ID を持つこと。
    profile_name = config.vram.active_profile
    expected_key = f"profiles.{profile_name}.embedding"
    assert expected_key in remediation, (
        f"対処が {expected_key} を案内していない: {remediation}"
    )
    assert config.active_profile().embedding == config.profiles[profile_name].embedding
    embedding_id = config.active_profile().embedding
    assert embedding_id is not None
    assert f"('{embedding_id}')" in remediation, (
        f"案内された設定キーの実際の値 {embedding_id!r} が対処に現れない: {remediation}"
    )


def test_context_length_message_reads_as_a_sentence_on_both_paths(
    tmp_config: ConfigWriter,
) -> None:
    """コンテキスト長超過の文面が、埋め込み経路でも日本語として成立すること。

    F-9-001 の修正で埋め込み側の上書きを外した結果、チャット用の主語
    ``要求したコンテキスト長 {要求量}`` に埋め込みの値が流し込まれ、
    ``要求したコンテキスト長 embedding=ruri-v3-310m`` という非文になった。
    ``embedding=...`` はモデル名でありコンテキスト長ではない。

    主語だけをフックにする形へ直したので、両経路の文面を固定する。
    「偽の数値を出さない」(F-9-001) と「文として成立する」は別の要求で、
    前者だけを検査すると後者が黙って壊れる。
    """
    config = load_config(tmp_config())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "context length"}})

    client = create_embedding_client(
        config, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ContextLengthError) as excinfo:
        client.embed(["埋め込む文章"])

    message = str(excinfo.value)

    # チャット用の主語が流れ込んでいないこと。
    assert "要求したコンテキスト長" not in message, (
        f"埋め込みがチャットの主語を名乗っている: {message}"
    )
    # モデル ID がコンテキスト長として提示されていないこと。
    assert "コンテキスト長 embedding=" not in message, (
        f"モデル名をコンテキスト長として提示している: {message}"
    )
    assert message.startswith("埋め込み入力"), (
        f"埋め込みの主語で始まっていない: {message}"
    )
    # 基底の分岐 (登録済み) が効いており、実値の上限が出ること。
    assert "max_context_tokens=8192" in message, message
