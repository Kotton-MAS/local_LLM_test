"""生成パラメータがリクエストボディへ配線されていることの検証。

仕様書 §5 有効性観点 E1-E3 の中核。``model`` / ``context_tokens`` /
``temperature`` / ``top_p`` / ``max_output_tokens`` / ``seed`` の 6 項目すべてに
ついて「設定値を変えるとボディの対応フィールドが変わる」ことを固定する。
1 項目でも配線が外れれば該当ケースが落ちる。

実 HTTP は発行しない (D-02)。すべて ``httpx.MockTransport`` で完結する。
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from conftest import SUCCESS_PAYLOAD

from llmkit.client import ChatMessage, OpenAICompatibleClient
from llmkit.config import AppConfig, GenerationParams, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


def send_chat(config: AppConfig) -> httpx.Request:
    """MockTransport 経由で 1 回 chat を実行し、捕捉したリクエストを返す。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=SUCCESS_PAYLOAD)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        client.chat([ChatMessage(role="user", content="こんにちは")])

    assert len(captured) == 1
    return captured[0]


def request_body(request: httpx.Request) -> dict[str, object]:
    payload: object = json.loads(request.content)
    assert isinstance(payload, dict)
    return {str(key): value for key, value in payload.items()}


def body_value(body: dict[str, object], path: tuple[str, ...]) -> object:
    """ドット区切り相当のパスでボディの値を取り出す。"""
    current: object = body
    for key in path:
        assert isinstance(current, dict), f"'{key}' の親が dict ではありません"
        current = current[key]
    return current


GenerationVariant = Callable[[GenerationParams], GenerationParams]


def with_generation(config: AppConfig, variant: GenerationVariant) -> AppConfig:
    return dataclasses.replace(config, generation=variant(config.generation))


@dataclasses.dataclass(frozen=True)
class WiringCase:
    """1 パラメータの掃引ケース。

    ``variant_a`` / ``variant_b`` で設定値を 2 通りに振り、``path`` が指す
    リクエストボディの値が ``expected_a`` / ``expected_b`` に追随することを見る。
    """

    field: str
    path: tuple[str, ...]
    variant_a: GenerationVariant
    variant_b: GenerationVariant
    expected_a: object
    expected_b: object


WIRING_CASES = (
    # E1: model は ModelSpec.served_name に解決されてから送出される
    WiringCase(
        field="model",
        path=("model",),
        variant_a=lambda g: dataclasses.replace(g, model="qwen3-14b"),
        variant_b=lambda g: dataclasses.replace(g, model="qwen3-8b"),
        expected_a="qwen3:14b-q4_K_M",
        expected_b="qwen3:8b-q4_K_M",
    ),
    # E2: context_tokens は Ollama 拡張の options.num_ctx に載る
    WiringCase(
        field="context_tokens",
        path=("options", "num_ctx"),
        variant_a=lambda g: dataclasses.replace(g, context_tokens=4096),
        variant_b=lambda g: dataclasses.replace(g, context_tokens=16384),
        expected_a=4096,
        expected_b=16384,
    ),
    # E3: 生成パラメータはトップレベルに載る
    WiringCase(
        field="temperature",
        path=("temperature",),
        variant_a=lambda g: dataclasses.replace(g, temperature=0.1),
        variant_b=lambda g: dataclasses.replace(g, temperature=1.3),
        expected_a=0.1,
        expected_b=1.3,
    ),
    WiringCase(
        field="top_p",
        path=("top_p",),
        variant_a=lambda g: dataclasses.replace(g, top_p=0.5),
        variant_b=lambda g: dataclasses.replace(g, top_p=0.95),
        expected_a=0.5,
        expected_b=0.95,
    ),
    WiringCase(
        field="max_output_tokens",
        path=("max_tokens",),
        variant_a=lambda g: dataclasses.replace(g, max_output_tokens=128),
        variant_b=lambda g: dataclasses.replace(g, max_output_tokens=2048),
        expected_a=128,
        expected_b=2048,
    ),
    WiringCase(
        field="seed",
        path=("seed",),
        variant_a=lambda g: dataclasses.replace(g, seed=7),
        variant_b=lambda g: dataclasses.replace(g, seed=12345),
        expected_a=7,
        expected_b=12345,
    ),
)


@pytest.mark.parametrize(
    "case", WIRING_CASES, ids=[case.field for case in WIRING_CASES]
)
def test_generation_params_reach_request_body(case: WiringCase) -> None:
    """6 項目すべてが「設定を変えるとボディが変わる」ことを満たす (E1-E3)。"""
    base = load_config(DEFAULT_CONFIG)

    config_a = with_generation(base, case.variant_a)
    config_b = with_generation(base, case.variant_b)
    body_a = request_body(send_chat(config_a))
    body_b = request_body(send_chat(config_b))

    assert case.expected_a != case.expected_b, "掃引ケースが同値では配線を検出できない"
    assert body_value(body_a, case.path) == case.expected_a
    assert body_value(body_b, case.path) == case.expected_b


def test_wiring_cases_cover_all_generation_fields() -> None:
    """掃引ケースが GenerationParams の全フィールドを網羅していること。"""
    generation = load_config(DEFAULT_CONFIG).generation
    generation_fields = {field.name for field in dataclasses.fields(generation)}
    assert {case.field for case in WIRING_CASES} == generation_fields


def test_request_body_contains_exactly_the_expected_keys() -> None:
    body = request_body(send_chat(load_config(DEFAULT_CONFIG)))

    assert set(body) == {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "options",
    }


def test_messages_are_serialized_in_order() -> None:
    config = load_config(DEFAULT_CONFIG)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=SUCCESS_PAYLOAD)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        client.chat(
            [
                ChatMessage(role="system", content="あなたは日本語アシスタントです"),
                ChatMessage(role="user", content="こんにちは"),
            ]
        )

    body = request_body(captured[0])
    assert body["messages"] == [
        {"role": "system", "content": "あなたは日本語アシスタントです"},
        {"role": "user", "content": "こんにちは"},
    ]
