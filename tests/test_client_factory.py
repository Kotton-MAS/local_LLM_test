"""``runtime.kind`` からクライアント実装を選ぶファクトリの検証 (D-10 / E9)。

このファイルは 1 つの guard_test を含む:

- ``test_runtime_kind_selects_the_client_implementation`` … D-10
  (runtime.kind がクライアント実装を選ぶ。分岐は create_chat_client の 1 か所)

``kind`` が「マニフェストに記録されるだけの飾り」に戻ったら (F-2-003 の再発)
このファイルが落ちる。実 HTTP は発行しない (D-02)。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import httpx
import pytest
from conftest import ConfigWriter, RecordingTransport

from llmkit.client import (
    ApiStyle,
    ChatClient,
    ChatMessage,
    OllamaNativeClient,
    OpenAICompatibleClient,
    api_style_for,
    create_chat_client,
)
from llmkit.config import RuntimeKind, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"

#: default.toml の 1 行。ここだけを書き換えて経路を反転させる。
LOCAL_KIND_LINE = 'kind = "ollama"'
COMPAT_KIND_LINE = 'kind = "openai_compatible"'


def chat_once(config_path: Path) -> tuple[ChatClient, httpx.Request]:
    """設定ファイルからクライアントを作り、1 回だけ chat して結果を返す。"""
    transport = RecordingTransport()
    with transport.client() as http_client:
        client = create_chat_client(load_config(config_path), http_client=http_client)
        result = client.chat([ChatMessage(role="user", content="こんにちは")])

    assert result.text == "テスト応答"
    assert transport.call_count == 1
    return client, transport.requests[0]


def body_of(request: httpx.Request) -> dict[str, object]:
    payload: object = json.loads(request.content)
    assert isinstance(payload, dict)
    return {str(key): value for key, value in payload.items()}


# --------------------------------------------------------------------------
# guard_test (D-10)
# --------------------------------------------------------------------------


def test_runtime_kind_selects_the_client_implementation(
    tmp_config: ConfigWriter,
) -> None:
    """E9 guard: 設定 1 行の違いだけで実装・送信先・ボディ形状が反転する。"""
    native_path = tmp_config(name="native.toml")
    compat_path = tmp_config({LOCAL_KIND_LINE: COMPAT_KIND_LINE}, name="compat.toml")

    # 2 つの設定の差が「kind の 1 行」だけであることを先に固定する。
    # 他の行も違っていたら、この後の反転が kind 由来だと言えない。
    differences = [
        (before, after)
        for before, after in zip(
            native_path.read_text(encoding="utf-8").splitlines(),
            compat_path.read_text(encoding="utf-8").splitlines(),
            strict=True,
        )
        if before != after
    ]
    assert differences == [(LOCAL_KIND_LINE, COMPAT_KIND_LINE)]

    native_client, native_request = chat_once(native_path)
    compat_client, compat_request = chat_once(compat_path)

    # 実装
    assert isinstance(native_client, OllamaNativeClient)
    assert isinstance(compat_client, OpenAICompatibleClient)

    # 送信先
    assert str(native_request.url) == "http://localhost:11434/api/chat"
    assert str(compat_request.url) == "http://localhost:11434/v1/chat/completions"

    # ボディ形状 (ネイティブは stream/options、互換は max_tokens をトップレベルに)
    native_body = body_of(native_request)
    compat_body = body_of(compat_request)
    assert native_body["stream"] is False
    assert native_body["options"] == {
        "num_ctx": 16384,
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 1024,
        "seed": 0,
    }
    assert "stream" not in compat_body
    assert compat_body["max_tokens"] == 1024
    assert compat_body["options"] == {"num_ctx": 16384}


# --------------------------------------------------------------------------
# api_style_for
# --------------------------------------------------------------------------


EXPECTED_API_STYLES: dict[RuntimeKind, ApiStyle] = {
    "ollama": "ollama_native",
    "openai_compatible": "openai_compatible",
}


def test_api_style_for_covers_every_runtime_kind() -> None:
    """RuntimeKind に値が増えたら、この表を更新するまで落ちる。"""
    assert set(EXPECTED_API_STYLES) == set(get_args(RuntimeKind))


#: ApiStyle に値が増えたら更新するまで落ちる (F-4-002)。RuntimeKind 版の
#: ``test_api_style_for_covers_every_runtime_kind`` と同型の網羅ガード。
#: ``endpoint_url_for`` / ``create_chat_client`` 自体の追加漏れは
#: ``match ... case _: assert_never(style)`` により mypy strict が検出する
#: (このテストはその前提となる値集合を固定するもの)。
EXPECTED_API_STYLE_VALUES: frozenset[ApiStyle] = frozenset(
    {"ollama_native", "openai_compatible"}
)


def test_api_style_covers_every_defined_value() -> None:
    """ApiStyle の値集合がここに列挙した値と過不足なく一致すること。"""
    assert set(get_args(ApiStyle)) == EXPECTED_API_STYLE_VALUES


@pytest.mark.parametrize(
    ("kind", "expected"),
    list(EXPECTED_API_STYLES.items()),
    ids=list(EXPECTED_API_STYLES),
)
def test_api_style_for_maps_runtime_kind(kind: RuntimeKind, expected: ApiStyle) -> None:
    assert api_style_for(kind) == expected


# --------------------------------------------------------------------------
# ファクトリの振る舞い
# --------------------------------------------------------------------------


def test_create_chat_client_returns_a_chat_client_protocol_instance() -> None:
    transport = RecordingTransport()
    with transport.client() as http_client:
        client = create_chat_client(
            load_config(DEFAULT_CONFIG), http_client=http_client
        )

        assert isinstance(client, ChatClient)


def test_create_chat_client_sends_nothing_at_construction() -> None:
    """生成だけでは HTTP を 1 バイトも出さない (受け入れ条件3 と同じ作法)。"""
    transport = RecordingTransport()
    with transport.client() as http_client:
        create_chat_client(load_config(DEFAULT_CONFIG), http_client=http_client)

    assert transport.call_count == 0


def test_create_chat_client_keeps_the_injected_http_client_open() -> None:
    """注入された httpx.Client の所有権は呼び出し側にある (基底で共有)。"""
    transport = RecordingTransport()
    http_client = transport.client()
    client = create_chat_client(load_config(DEFAULT_CONFIG), http_client=http_client)

    assert isinstance(client, OllamaNativeClient)
    client.close()

    assert http_client.is_closed is False
    http_client.close()
