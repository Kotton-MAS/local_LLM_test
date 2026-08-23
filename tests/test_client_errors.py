"""エラー翻訳表 (仕様書 §4 T3) の検証。

表の各行に 1 テスト以上を対応させ、**送出される例外型**と
**メッセージ内の必須要素**の両方をアサートする。
実 HTTP は発行しない (D-02)。すべて ``httpx.MockTransport`` で完結する。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from llmkit.client import ChatMessage, OpenAICompatibleClient
from llmkit.config import AppConfig, load_config
from llmkit.errors import (
    ContextLengthError,
    LlmkitError,
    ModelNotFoundError,
    OutOfMemoryError,
    RuntimeUnavailableError,
    UpstreamError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"
EXTERNAL_CONFIG = REPO_ROOT / "configs" / "external_openai.toml"

API_KEY = "sk-test-do-not-leak-0123456789"

Handler = Callable[[httpx.Request], httpx.Response]


def run_chat(config: AppConfig, handler: Handler) -> None:
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        client.chat([ChatMessage(role="user", content="こんにちは")])


def responding(status: int, body: str) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return handler


def raising(exc: Exception) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("timed out"),
    ],
    ids=["ConnectError", "ConnectTimeout"],
)
def test_connect_failure_maps_to_runtime_unavailable(exc: Exception) -> None:
    with pytest.raises(RuntimeUnavailableError) as excinfo:
        run_chat(load_config(DEFAULT_CONFIG), raising(exc))

    message = str(excinfo.value)
    assert "http://localhost:11434/v1" in message
    assert "ollama serve" in message
    assert "起動していない可能性" in message


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("read timed out"),
        httpx.WriteTimeout("write timed out"),
        httpx.PoolTimeout("pool timed out"),
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.ProtocolError("malformed response"),
    ],
    ids=[
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "ProtocolError",
    ],
)
def test_other_httpx_errors_map_to_upstream_error(exc: Exception) -> None:
    """F-1-030: ConnectError/ConnectTimeout 以外の httpx.HTTPError は UpstreamError。

    翻訳しないと httpx 固有の例外が公開 API に漏れる
    (ハード制約「推論ランタイム固有の型を L2 の公開 API に露出させない」)。
    """
    with pytest.raises(UpstreamError) as excinfo:
        run_chat(load_config(DEFAULT_CONFIG), raising(exc))

    message = str(excinfo.value)
    assert type(exc).__name__ in message
    assert "http://localhost:11434/v1/chat/completions" in message
    # httpx 固有の例外そのものが上位に漏れていないこと
    assert not isinstance(excinfo.value, httpx.HTTPError)


def test_other_httpx_error_chains_the_original_exception() -> None:
    """``raise ... from exc`` で例外連鎖が保たれていること (慣習)。"""
    original = httpx.ReadTimeout("read timed out")

    with pytest.raises(UpstreamError) as excinfo:
        run_chat(load_config(DEFAULT_CONFIG), raising(original))

    assert excinfo.value.__cause__ is original


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (404, '{"error": {"message": "not found"}}'),
        (500, '{"error": "model not found, try pulling it first"}'),
    ],
    ids=["HTTP 404", "本文に model not found"],
)
def test_missing_model_maps_to_model_not_found(status: int, body: str) -> None:
    with pytest.raises(ModelNotFoundError) as excinfo:
        run_chat(load_config(DEFAULT_CONFIG), responding(status, body))

    message = str(excinfo.value)
    assert "qwen3:14b-q4_K_M" in message
    assert "ollama pull qwen3:14b-q4_K_M" in message


@pytest.mark.parametrize(
    "body",
    [
        '{"error": "CUDA error: out of memory"}',
        '{"error": "cudaMalloc failed: insufficient memory"}',
    ],
    ids=["out of memory", "cudaMalloc"],
)
def test_out_of_memory_body_maps_to_out_of_memory_error(body: str) -> None:
    with pytest.raises(OutOfMemoryError) as excinfo:
        run_chat(load_config(DEFAULT_CONFIG), responding(500, body))

    message = str(excinfo.value)
    assert "rag_default" in message
    assert "16384" in message
    assert "より小さいプロファイル" in message


def test_context_length_400_maps_to_context_length_error() -> None:
    base = load_config(DEFAULT_CONFIG)
    # qwen3-14b の max_context_tokens は 32768 (Phase 0 実測で較正)。要求値と上限を
    # 別の値にして「どちらの数値もメッセージに出ている」ことを区別できるようにする。
    config = dataclasses.replace(
        base, generation=dataclasses.replace(base.generation, context_tokens=65536)
    )

    with pytest.raises(ContextLengthError) as excinfo:
        run_chat(
            config,
            responding(400, '{"error": "maximum context length exceeded"}'),
        )

    message = str(excinfo.value)
    assert "65536" in message
    assert "32768" in message


def test_context_length_400_for_passthrough_model_does_not_claim_a_fake_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-2-002: passthrough (カタログ未登録) では偽の上限値を主張しない。

    resolve_model_spec の passthrough は max_context_tokens に
    ``_PASSTHROUGH_MAX_CONTEXT_TOKENS`` という placeholder を入れる。これを
    「モデルの上限」として出すと『要求 16384 が上限 1048576 を超えた』という
    自己矛盾したメッセージになる (実測で確認済みの退行)。カタログ未登録の
    場合はその placeholder の数値をメッセージに出さないことを固定する。
    """
    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)
    base = load_config(EXTERNAL_CONFIG)
    config = dataclasses.replace(
        base, generation=dataclasses.replace(base.generation, model="gpt-4o-mini")
    )

    with pytest.raises(ContextLengthError) as excinfo:
        run_chat(
            config,
            responding(400, '{"error": "maximum context length exceeded"}'),
        )

    message = str(excinfo.value)
    assert "gpt-4o-mini" in message
    assert "16384" in message
    assert "1048576" not in message
    assert "上限は不明" in message


def test_context_length_remediation_is_abstract_on_the_shared_base() -> None:
    """round-10 レビュー: 対処メッセージの上書き忘れを ABC/mypy で検出できること。

    ``_context_length_message`` は F-9-001 の教訓でフックになっているが、
    対処 (remediation) を返す ``_context_length_remediation`` は基底に
    チャット向けの既定文面が残ったままで abstract ではなかった。埋め込みは
    上書きしているため今は症状が出ないが、次にこの基底へ載る経路 (リランカー)
    が上書きを忘れると同じ型の欠陥が対処メッセージ側で再発する。
    ``__abstractmethods__`` に含まれ、上書きを忘れたサブクラスがインスタンス化
    できないことを固定する。
    """
    from llmkit.catalog import resolve_model_spec
    from llmkit.client import _HttpEndpointClient

    assert "_context_length_remediation" in _HttpEndpointClient.__abstractmethods__

    class _ForgetsToOverrideRemediation(_HttpEndpointClient):
        @property
        def endpoint_url(self) -> str:
            return "http://example.invalid/v1/rerank"

        def _request_context(self) -> str:
            return "pairs=1"

        def _model_setting_reference(self) -> str:
            return "profiles.default.reranker"

        def _context_length_subject(self) -> str:
            return "リランク対象"

        # _context_length_remediation を意図的に上書きしない。

    config = load_config(DEFAULT_CONFIG)
    spec = resolve_model_spec(config.generation.model, is_local=config.runtime.is_local)
    with pytest.raises(TypeError, match="_context_length_remediation"):
        _ForgetsToOverrideRemediation(config, spec)  # type: ignore[abstract]


def test_other_status_maps_to_upstream_error_without_body() -> None:
    secret_detail = "INTERNAL-STACKTRACE-DO-NOT-LEAK"

    with pytest.raises(UpstreamError) as excinfo:
        run_chat(
            load_config(DEFAULT_CONFIG),
            responding(503, f'{{"error": "{secret_detail}"}}'),
        )

    message = str(excinfo.value)
    assert "503" in message
    assert secret_detail not in message


FAILURE_HANDLERS = {
    "接続不可": raising(httpx.ConnectError("connection refused")),
    "モデル不在": responding(404, '{"error": "not found"}'),
    "VRAM 不足": responding(500, '{"error": "CUDA error: out of memory"}'),
    "コンテキスト超過": responding(400, '{"error": "context length exceeded"}'),
    "その他 5xx": responding(503, '{"error": "unavailable"}'),
    "応答が不正": responding(200, '{"choices": []}'),
}


@pytest.mark.parametrize(
    "handler", list(FAILURE_HANDLERS.values()), ids=list(FAILURE_HANDLERS)
)
def test_error_messages_never_contain_api_key(
    handler: Handler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """どの失敗経路でも api_key の平文が例外に漏れない。"""
    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)
    config = load_config(EXTERNAL_CONFIG)
    assert config.api_key.get_secret_value() == API_KEY

    with pytest.raises(LlmkitError) as excinfo:
        run_chat(config, handler)

    assert API_KEY not in str(excinfo.value)
    assert API_KEY not in repr(excinfo.value)
    assert API_KEY not in excinfo.value.remediation
