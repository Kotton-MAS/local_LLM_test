"""推論クライアント (llmkit/client.py) の正常系と応答パースのテスト。

このファイルは 1 つの guard_test を含む:

- ``test_malformed_response_raises_upstream_error`` … D-07
  (応答は厳格にパースし、必須フィールドの欠損は UpstreamError にする)

実 HTTP は発行しない (D-02)。すべて ``httpx.MockTransport`` で完結する。
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import httpx
import pytest
from conftest import SUCCESS_PAYLOAD

from llmkit.client import (
    ChatClient,
    ChatMessage,
    ChatResult,
    OpenAICompatibleClient,
    TokenUsage,
)
from llmkit.config import AppConfig, load_config
from llmkit.errors import ConfigError, UpstreamError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"
EXTERNAL_CONFIG = REPO_ROOT / "configs" / "external_openai.toml"

API_KEY = "sk-test-do-not-leak-0123456789"


def payload_without(*paths: tuple[str, ...]) -> dict[str, object]:
    """正常応答から指定パスのキーを削って壊した応答を作る。"""
    broken: dict[str, object] = copy.deepcopy(SUCCESS_PAYLOAD)
    for path in paths:
        current: object = broken
        for key in path[:-1]:
            assert isinstance(current, dict)
            nested: object = current[key]
            if isinstance(nested, list):
                nested = nested[0]
            current = nested
        assert isinstance(current, dict)
        del current[path[-1]]
    return broken


def chat_with(
    config: AppConfig,
    *,
    payload: object = SUCCESS_PAYLOAD,
    status: int = 200,
    text: str | None = None,
) -> tuple[ChatResult, httpx.Request]:
    """MockTransport 経由で 1 回 chat を実行し、結果と捕捉リクエストを返す。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        result = client.chat([ChatMessage(role="user", content="こんにちは")])

    assert len(captured) == 1
    return result, captured[0]


def test_chat_returns_parsed_result() -> None:
    result, _ = chat_with(load_config(DEFAULT_CONFIG))

    assert result.text == "テスト応答"
    assert result.model == "qwen3:14b-q4_K_M"
    assert result.finish_reason == "stop"
    assert result.usage == TokenUsage(
        prompt_tokens=11, completion_tokens=7, total_tokens=18
    )
    assert result.latency_s >= 0.0


def test_request_targets_chat_completions_endpoint() -> None:
    _, request = chat_with(load_config(DEFAULT_CONFIG))

    assert str(request.url) == "http://localhost:11434/v1/chat/completions"
    assert request.method == "POST"


def test_tokens_per_second_uses_completion_tokens_and_latency() -> None:
    result = ChatResult(
        text="x",
        model="qwen3:14b-q4_K_M",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=60, total_tokens=160),
        latency_s=2.0,
    )

    assert result.tokens_per_second == 30.0


def test_tokens_per_second_is_zero_when_latency_is_zero() -> None:
    """計測不能 (latency_s == 0) でも ZeroDivisionError にしない。"""
    result = ChatResult(
        text="x",
        model="qwen3:14b-q4_K_M",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=7, total_tokens=8),
        latency_s=0.0,
    )

    assert result.tokens_per_second == 0.0


MALFORMED_PAYLOADS = {
    "choices 欠損": payload_without(("choices",)),
    "choices が空": {**SUCCESS_PAYLOAD, "choices": []},
    "message 欠損": payload_without(("choices", "message")),
    "content 欠損": payload_without(("choices", "message", "content")),
    "finish_reason 欠損": payload_without(("choices", "finish_reason")),
    "usage 欠損": payload_without(("usage",)),
    "completion_tokens 欠損": payload_without(("usage", "completion_tokens")),
    "model 欠損": payload_without(("model",)),
    "content の型不一致": {
        **SUCCESS_PAYLOAD,
        "choices": [
            {"message": {"content": {"parts": ["x"]}}, "finish_reason": "stop"}
        ],
    },
}


@pytest.mark.parametrize(
    "payload", list(MALFORMED_PAYLOADS.values()), ids=list(MALFORMED_PAYLOADS)
)
def test_malformed_response_raises_upstream_error(payload: object) -> None:
    """D-07 guard: 必須フィールドが欠けた応答は UpstreamError で早期に落とす。"""
    with pytest.raises(UpstreamError) as excinfo:
        chat_with(load_config(DEFAULT_CONFIG), payload=payload)

    assert "http://localhost:11434/v1/chat/completions" in str(excinfo.value)


def test_non_json_response_raises_upstream_error() -> None:
    with pytest.raises(UpstreamError) as excinfo:
        chat_with(load_config(DEFAULT_CONFIG), text="<html>not json</html>")

    assert "JSON" in str(excinfo.value)


def test_local_and_remote_configs_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    """E6: base_url / api_key の違いがリクエストにそのまま現れる。"""
    monkeypatch.delenv("LLMKIT_API_KEY", raising=False)
    local = load_config(DEFAULT_CONFIG)
    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)
    remote = load_config(EXTERNAL_CONFIG)

    _, local_request = chat_with(local)
    _, remote_request = chat_with(remote)

    assert str(local_request.url) == "http://localhost:11434/v1/chat/completions"
    assert str(remote_request.url) == "https://api.example.com/v1/chat/completions"
    assert "authorization" not in local_request.headers
    assert remote_request.headers["authorization"] == f"Bearer {API_KEY}"


def test_changing_model_id_changes_served_name() -> None:
    """受け入れ条件1: 呼び出しコードを変えず config の model だけで切り替わる。"""
    base = load_config(DEFAULT_CONFIG)
    smaller = dataclasses.replace(
        base, generation=dataclasses.replace(base.generation, model="qwen3-8b")
    )

    _, base_request = chat_with(base)
    _, smaller_request = chat_with(smaller)

    assert json.loads(base_request.content)["model"] == "qwen3:14b-q4_K_M"
    assert json.loads(smaller_request.content)["model"] == "qwen3:8b-q4_K_M"


def test_unregistered_model_id_is_sent_as_is_for_remote_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-1-001: is_local=false なら未登録モデル名でも ConfigError にならない。

    要件書 L144「外部 API のモデルへ設定変更のみで切り替えられる」の再現。
    ``configs/external_openai.toml`` の generation.model をカタログ未登録の
    名前 (例: gpt-4o-mini) に差し替えても送出できる。
    """
    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)
    remote = load_config(EXTERNAL_CONFIG)
    remote = dataclasses.replace(
        remote, generation=dataclasses.replace(remote.generation, model="gpt-4o-mini")
    )

    result, request = chat_with(remote)

    assert json.loads(request.content)["model"] == "gpt-4o-mini"
    assert result.text == "テスト応答"


def test_unregistered_model_id_still_raises_config_error_for_local_runtime() -> None:
    """is_local=true の場合は passthrough を許さず、従来どおり ConfigError。"""
    local = load_config(DEFAULT_CONFIG)
    local = dataclasses.replace(
        local, generation=dataclasses.replace(local.generation, model="gpt-4o-mini")
    )

    with pytest.raises(ConfigError) as excinfo:
        # コンストラクタは http_client に触れる前にモデル解決で例外を送出する
        # ため、http_client は省略してよい (実 HTTP は発行しない, D-02)。
        OpenAICompatibleClient(local)

    assert "gpt-4o-mini" in str(excinfo.value)


def test_client_satisfies_chat_client_protocol() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=SUCCESS_PAYLOAD)
        )
    ) as http_client:
        client: ChatClient = OpenAICompatibleClient(
            load_config(DEFAULT_CONFIG), http_client=http_client
        )

        assert isinstance(client, ChatClient)


def test_client_with_no_injected_http_client_creates_and_owns_one() -> None:
    """F-1-031: ``http_client`` を省略した自己所有モードのコンストラクタ分岐。

    実 HTTP は 1 バイトも発行しない (D-02)。生成した ``httpx.Client`` を
    ``close()`` が実際に閉じることまでを検証する (close() 内の分岐も含む)。
    """
    client = OpenAICompatibleClient(load_config(DEFAULT_CONFIG))

    assert isinstance(client._http, httpx.Client)
    assert client._owns_http_client is True
    assert client._http.is_closed is False

    client.close()

    assert client._http.is_closed is True


def test_client_with_no_injected_http_client_uses_configured_timeout() -> None:
    """自己所有モードで生成される httpx.Client が runtime.timeout_s を使う。"""
    config = load_config(DEFAULT_CONFIG)

    client = OpenAICompatibleClient(config)
    try:
        assert client._http.timeout.connect == config.runtime.timeout_s
    finally:
        client.close()


def test_injected_http_client_is_not_closed_by_client_close() -> None:
    """注入された httpx.Client の所有権は呼び出し側にある。"""
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=SUCCESS_PAYLOAD)
        )
    )
    with OpenAICompatibleClient(
        load_config(DEFAULT_CONFIG), http_client=http_client
    ) as client:
        assert client.served_name == "qwen3:14b-q4_K_M"

    assert http_client.is_closed is False
    http_client.close()
