"""ネイティブ ``/api/chat`` クライアント (OllamaNativeClient) の検証。

このファイルは 2 つの guard_test を含む:

- ``test_native_endpoint_url_is_derived_from_base_url`` … D-11
  (エンドポイントは base_url の末尾 /v1 を 1 セグメントだけ除去して導出する。
  ネイティブ URL 用の設定キーは追加しない)
- ``test_native_result_exposes_prompt_and_eval_timings`` … D-13
  (prompt/eval を分離した計測値を持ち、欠測は 0 ではなく None で表す)

エラー翻訳は OpenAI 互換経路と**同じ表**を共有していることまで見る
(片方だけ直る/壊れることを防ぐ)。実 HTTP は発行しない (D-02)。すべて
``httpx.MockTransport`` で完結する。
"""

from __future__ import annotations

import dataclasses
import io
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from conftest import NATIVE_SUCCESS_PAYLOAD, SUCCESS_PAYLOAD

from llmkit.cli import main as cli_main
from llmkit.client import (
    ApiStyle,
    ChatClient,
    ChatMessage,
    ChatResult,
    ChatTimings,
    OllamaNativeClient,
    OpenAICompatibleClient,
    endpoint_url_for,
)
from llmkit.config import AppConfig, GenerationParams, load_config
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

NATIVE_URL = "http://localhost:11434/api/chat"
API_KEY = "sk-test-do-not-leak-0123456789"

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------
# ヘルパ
# --------------------------------------------------------------------------


def responding_json(payload: object, status: int = 200) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def responding(status: int, body: str) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return handler


def raising(exc: Exception) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def run_native_chat(
    config: AppConfig, handler: Handler | None = None
) -> tuple[ChatResult, httpx.Request]:
    """MockTransport 経由でネイティブ chat を 1 回実行する。"""
    captured: list[httpx.Request] = []
    inner = handler if handler is not None else responding_json(NATIVE_SUCCESS_PAYLOAD)

    def record(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return inner(request)

    with httpx.Client(transport=httpx.MockTransport(record)) as http_client:
        client = OllamaNativeClient(config, http_client=http_client)
        result = client.chat([ChatMessage(role="user", content="こんにちは")])

    assert len(captured) == 1
    return result, captured[0]


def request_body(request: httpx.Request) -> dict[str, object]:
    payload: object = json.loads(request.content)
    assert isinstance(payload, dict)
    return {str(key): value for key, value in payload.items()}


def body_value(body: dict[str, object], path: tuple[str, ...]) -> object:
    current: object = body
    for key in path:
        assert isinstance(current, dict), f"'{key}' の親が dict ではありません"
        current = current[key]
    return current


def payload_without(*keys: str) -> dict[str, object]:
    return {
        key: value for key, value in NATIVE_SUCCESS_PAYLOAD.items() if key not in keys
    }


# --------------------------------------------------------------------------
# guard_test (D-11): エンドポイント導出
# --------------------------------------------------------------------------


ENDPOINT_CASES: tuple[tuple[str, ApiStyle, str], ...] = (
    # (base_url, style, 期待 URL)
    ("http://localhost:11434/v1", "ollama_native", NATIVE_URL),
    ("http://localhost:11434/v1/", "ollama_native", NATIVE_URL),
    ("http://localhost:11434", "ollama_native", NATIVE_URL),
    # /v1 は 1 セグメントだけ除去する (前段のゲートウェイ path を食べない)
    (
        "http://gateway.example.com/v1/v1",
        "ollama_native",
        "http://gateway.example.com/v1/api/chat",
    ),
    (
        "http://gateway.example.com/ollama/v1",
        "ollama_native",
        "http://gateway.example.com/ollama/api/chat",
    ),
    # /v1 で終わらない path は削らない
    (
        "http://gateway.example.com/v1beta",
        "ollama_native",
        "http://gateway.example.com/v1beta/api/chat",
    ),
    # 互換経路は現状と同一 (連結するだけ)
    (
        "http://localhost:11434/v1",
        "openai_compatible",
        "http://localhost:11434/v1/chat/completions",
    ),
    (
        "https://api.example.com/v1",
        "openai_compatible",
        "https://api.example.com/v1/chat/completions",
    ),
)


@pytest.mark.parametrize(
    ("base_url", "style", "expected"),
    ENDPOINT_CASES,
    ids=[f"{case[1]}:{case[0]}" for case in ENDPOINT_CASES],
)
def test_native_endpoint_url_is_derived_from_base_url(
    base_url: str, style: ApiStyle, expected: str
) -> None:
    """D-11 guard: base_url から導出する純関数。HTTP を 1 バイトも出さない。

    ``endpoint_url_for`` は httpx にもソケットにも触れない。設定キーを増やさず
    ``runtime.base_url`` だけから 2 経路の URL が決まることを固定する。
    """
    assert endpoint_url_for(base_url, style) == expected


def test_native_client_endpoint_url_uses_the_derived_url_without_sending() -> None:
    """クライアントの ``endpoint_url`` が導出関数と一致する (別実装を持たない)。

    リクエストは 1 件も発行しない (トランスポートの記録が空であることで示す)。
    """
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        requests.append(request)
        return httpx.Response(200, json=NATIVE_SUCCESS_PAYLOAD)

    config = load_config(DEFAULT_CONFIG)
    with httpx.Client(transport=httpx.MockTransport(record)) as http_client:
        client = OllamaNativeClient(config, http_client=http_client)

        assert client.endpoint_url == endpoint_url_for(
            config.runtime.base_url, "ollama_native"
        )
        assert client.endpoint_url == NATIVE_URL

    assert requests == []


def test_native_request_targets_the_native_chat_endpoint() -> None:
    _, request = run_native_chat(load_config(DEFAULT_CONFIG))

    assert str(request.url) == NATIVE_URL
    assert request.method == "POST"


# --------------------------------------------------------------------------
# リクエストボディ (E10)
# --------------------------------------------------------------------------


def test_native_request_body_contains_exactly_the_expected_keys() -> None:
    """トップレベル 4 キー厳密。``stream`` の省略は NDJSON を招くため必須。"""
    _, request = run_native_chat(load_config(DEFAULT_CONFIG))
    body = request_body(request)

    assert set(body) == {"model", "messages", "stream", "options"}
    assert body["stream"] is False

    options = body["options"]
    assert isinstance(options, dict)
    assert set(options) == {
        "num_ctx",
        "temperature",
        "top_p",
        "num_predict",
        "seed",
    }


def test_native_messages_are_serialized_in_order() -> None:
    config = load_config(DEFAULT_CONFIG)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=NATIVE_SUCCESS_PAYLOAD)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OllamaNativeClient(config, http_client=http_client)
        client.chat(
            [
                ChatMessage(role="system", content="あなたは日本語アシスタントです"),
                ChatMessage(role="user", content="こんにちは"),
            ]
        )

    assert request_body(captured[0])["messages"] == [
        {"role": "system", "content": "あなたは日本語アシスタントです"},
        {"role": "user", "content": "こんにちは"},
    ]


GenerationVariant = Callable[[GenerationParams], GenerationParams]


def with_generation(config: AppConfig, variant: GenerationVariant) -> AppConfig:
    return dataclasses.replace(config, generation=variant(config.generation))


@dataclasses.dataclass(frozen=True)
class NativeWiringCase:
    """1 パラメータの掃引ケース (互換経路と同じ 6 項目・別のボディパス)。"""

    field: str
    path: tuple[str, ...]
    variant_a: GenerationVariant
    variant_b: GenerationVariant
    expected_a: object
    expected_b: object


NATIVE_WIRING_CASES = (
    NativeWiringCase(
        field="model",
        path=("model",),
        variant_a=lambda g: dataclasses.replace(g, model="qwen3-14b"),
        variant_b=lambda g: dataclasses.replace(g, model="qwen3-8b"),
        expected_a="qwen3:14b-q4_K_M",
        expected_b="qwen3:8b-q4_K_M",
    ),
    # ネイティブ経路では options.num_ctx が実機に反映される (Phase 0 実測)
    NativeWiringCase(
        field="context_tokens",
        path=("options", "num_ctx"),
        variant_a=lambda g: dataclasses.replace(g, context_tokens=4096),
        variant_b=lambda g: dataclasses.replace(g, context_tokens=16384),
        expected_a=4096,
        expected_b=16384,
    ),
    NativeWiringCase(
        field="temperature",
        path=("options", "temperature"),
        variant_a=lambda g: dataclasses.replace(g, temperature=0.1),
        variant_b=lambda g: dataclasses.replace(g, temperature=1.3),
        expected_a=0.1,
        expected_b=1.3,
    ),
    NativeWiringCase(
        field="top_p",
        path=("options", "top_p"),
        variant_a=lambda g: dataclasses.replace(g, top_p=0.5),
        variant_b=lambda g: dataclasses.replace(g, top_p=0.95),
        expected_a=0.5,
        expected_b=0.95,
    ),
    # max_output_tokens は互換経路の max_tokens ではなく options.num_predict
    NativeWiringCase(
        field="max_output_tokens",
        path=("options", "num_predict"),
        variant_a=lambda g: dataclasses.replace(g, max_output_tokens=128),
        variant_b=lambda g: dataclasses.replace(g, max_output_tokens=2048),
        expected_a=128,
        expected_b=2048,
    ),
    NativeWiringCase(
        field="seed",
        path=("options", "seed"),
        variant_a=lambda g: dataclasses.replace(g, seed=7),
        variant_b=lambda g: dataclasses.replace(g, seed=12345),
        expected_a=7,
        expected_b=12345,
    ),
)


@pytest.mark.parametrize(
    "case", NATIVE_WIRING_CASES, ids=[case.field for case in NATIVE_WIRING_CASES]
)
def test_native_generation_params_reach_request_body(case: NativeWiringCase) -> None:
    """E10: 6 項目すべてが「設定を変えるとネイティブボディが変わる」ことを満たす。"""
    base = load_config(DEFAULT_CONFIG)

    _, request_a = run_native_chat(with_generation(base, case.variant_a))
    _, request_b = run_native_chat(with_generation(base, case.variant_b))

    assert case.expected_a != case.expected_b, "掃引ケースが同値では配線を検出できない"
    assert body_value(request_body(request_a), case.path) == case.expected_a
    assert body_value(request_body(request_b), case.path) == case.expected_b


def test_native_wiring_cases_cover_all_generation_fields() -> None:
    """掃引ケースが GenerationParams の全フィールドを網羅していること。"""
    generation = load_config(DEFAULT_CONFIG).generation
    generation_fields = {field.name for field in dataclasses.fields(generation)}

    assert {case.field for case in NATIVE_WIRING_CASES} == generation_fields


def test_native_request_sends_api_key_header_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLMKIT_API_KEY", raising=False)
    _, local_request = run_native_chat(load_config(DEFAULT_CONFIG))
    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)
    _, remote_request = run_native_chat(load_config(EXTERNAL_CONFIG))

    assert "authorization" not in local_request.headers
    assert remote_request.headers["authorization"] == f"Bearer {API_KEY}"


# --------------------------------------------------------------------------
# 応答パース
# --------------------------------------------------------------------------


def test_native_chat_returns_parsed_result() -> None:
    result, _ = run_native_chat(load_config(DEFAULT_CONFIG))

    assert result.text == "テスト応答"
    assert result.model == "qwen3:14b-q4_K_M"
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 18
    assert result.latency_s >= 0.0


def test_native_finish_reason_falls_back_to_stop_when_done_reason_is_absent() -> None:
    result, _ = run_native_chat(
        load_config(DEFAULT_CONFIG),
        responding_json(payload_without("done_reason")),
    )

    assert result.finish_reason == "stop"


def test_native_client_satisfies_chat_client_protocol() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=NATIVE_SUCCESS_PAYLOAD)
        )
    ) as http_client:
        client: ChatClient = OllamaNativeClient(
            load_config(DEFAULT_CONFIG), http_client=http_client
        )

        assert isinstance(client, ChatClient)


MALFORMED_NATIVE_PAYLOADS = {
    "model 欠損": payload_without("model"),
    "message 欠損": payload_without("message"),
    "done 欠損": payload_without("done"),
    "content 欠損": {**NATIVE_SUCCESS_PAYLOAD, "message": {"role": "assistant"}},
    "content の型不一致": {
        **NATIVE_SUCCESS_PAYLOAD,
        "message": {"content": {"parts": ["x"]}},
    },
    "done の型不一致": {**NATIVE_SUCCESS_PAYLOAD, "done": {"finished": True}},
    "eval_count の型不一致": {**NATIVE_SUCCESS_PAYLOAD, "eval_count": "たくさん"},
}


@pytest.mark.parametrize(
    "payload",
    list(MALFORMED_NATIVE_PAYLOADS.values()),
    ids=list(MALFORMED_NATIVE_PAYLOADS),
)
def test_native_malformed_response_raises_upstream_error(payload: object) -> None:
    """D-07 と同じ趣旨: 必須フィールドの欠損・型不一致は UpstreamError。"""
    with pytest.raises(UpstreamError) as excinfo:
        run_native_chat(load_config(DEFAULT_CONFIG), responding_json(payload))

    assert NATIVE_URL in str(excinfo.value)


OPTIONAL_NATIVE_FIELDS = (
    "done_reason",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


@pytest.mark.parametrize("field", OPTIONAL_NATIVE_FIELDS)
def test_native_optional_fields_may_be_absent(field: str) -> None:
    """プロンプトキャッシュヒット等で計測値が返らなくても正常終了する。

    必須にすると正常な生成が UpstreamError になる (仕様書 §4 T3)。
    """
    result, _ = run_native_chat(
        load_config(DEFAULT_CONFIG), responding_json(payload_without(field))
    )

    assert result.text == "テスト応答"


def test_native_done_false_raises_upstream_error() -> None:
    payload = {**NATIVE_SUCCESS_PAYLOAD, "done": False}

    with pytest.raises(UpstreamError) as excinfo:
        run_native_chat(load_config(DEFAULT_CONFIG), responding_json(payload))

    assert "done=false" in str(excinfo.value)


def test_native_non_json_response_raises_upstream_error() -> None:
    """NDJSON ストリームを受け取ったときに何を疑えばよいかを対処に書く。"""
    with pytest.raises(UpstreamError) as excinfo:
        run_native_chat(
            load_config(DEFAULT_CONFIG), responding(200, '{"a":1}\n{"b":2}\n')
        )

    assert "JSON" in str(excinfo.value)
    assert "stream=false" in excinfo.value.remediation


# --------------------------------------------------------------------------
# guard_test (D-13): prompt/eval 分離の計測値
# --------------------------------------------------------------------------


def test_native_result_exposes_prompt_and_eval_timings() -> None:
    """D-13 guard: prompt/eval の t/s を返し、欠測は 0.0 ではなく None にする。"""
    payload = {
        **NATIVE_SUCCESS_PAYLOAD,
        "prompt_eval_count": 11,
        "prompt_eval_duration": 550_000_000,
        "eval_count": 64,
        "eval_duration": 1_000_000_000,
    }

    result, _ = run_native_chat(load_config(DEFAULT_CONFIG), responding_json(payload))

    timings = result.timings
    assert timings is not None
    assert timings.prompt_eval_count == 11
    assert timings.prompt_eval_seconds == pytest.approx(0.55)
    assert timings.eval_count == 64
    assert timings.eval_seconds == pytest.approx(1.0)
    assert timings.prompt_tokens_per_second == pytest.approx(20.0)
    assert timings.eval_tokens_per_second == pytest.approx(64.0)

    # プロンプトキャッシュヒットで prompt_eval_* が返らない応答。
    # UpstreamError にせず、欠測を None として明示する (0 で埋めない)。
    cached = {
        key: value
        for key, value in payload.items()
        if key not in {"prompt_eval_count", "prompt_eval_duration"}
    }

    cached_result, _ = run_native_chat(
        load_config(DEFAULT_CONFIG), responding_json(cached)
    )

    cached_timings = cached_result.timings
    assert cached_timings is not None
    assert cached_timings.prompt_eval_count is None
    assert cached_timings.prompt_eval_seconds is None
    assert cached_timings.prompt_tokens_per_second is None
    assert cached_timings.eval_tokens_per_second == pytest.approx(64.0)


def test_native_eval_count_absence_is_not_reported_as_a_measured_zero(
    tmp_path: Path,
) -> None:
    """F-4-001 guard: eval_count 欠測時の 0.0 を『実測値』として扱わない契約。

    ``eval_count`` を返さない正常応答 (UpstreamError にはならない) では
    ``usage.completion_tokens == 0`` になり、``ChatResult.tokens_per_second``
    もその副作用で 0.0 を返す。この 0.0 は「計測した結果 0 だった」ではなく
    「timings.eval_tokens_per_second is None (欠測)」を伴う欠測のシグナルで
    あることを固定する。D-13 の guard_test は timings 側しか見ておらず、
    ``ChatResult.tokens_per_second`` 側の食い違いはこのテストが担う。
    """
    payload = payload_without("eval_count")

    result, _ = run_native_chat(load_config(DEFAULT_CONFIG), responding_json(payload))

    assert result.usage.completion_tokens == 0
    assert result.tokens_per_second == 0.0
    assert result.timings is not None
    assert result.timings.eval_count is None
    assert result.timings.eval_tokens_per_second is None
    assert result.measured_tokens_per_second is None

    # 消費側 (CLI doctor) でも同じ契約を固定する (F-5-001): 0.0 t/s が
    # 実測値であるかのように表示されてはならない。
    stdout = io.StringIO()
    stderr = io.StringIO()
    with httpx.Client(transport=httpx.MockTransport(responding_json(payload))) as http:
        code = cli_main(
            [
                "doctor",
                "--config",
                str(DEFAULT_CONFIG),
                "--output-dir",
                str(tmp_path / "runs"),
            ],
            http_client=http,
            stdout=stdout,
            stderr=stderr,
        )
    combined = stdout.getvalue() + stderr.getvalue()
    assert code == 0, combined
    assert "0.0 t/s" not in combined
    assert "eval_count 欠測" in combined


@pytest.mark.parametrize(
    ("count", "seconds"),
    [(None, 1.0), (10, None), (10, 0.0), (None, None), (10, -1.0)],
    ids=["count 欠測", "seconds 欠測", "0 秒", "両方欠測", "負の秒"],
)
def test_chat_timings_reports_none_for_missing_or_unmeasurable_values(
    count: int | None, seconds: float | None
) -> None:
    """欠測・計測不能を None で返す (0.0 は「実測 0」と区別できないため使わない)。"""
    timings = ChatTimings(
        prompt_eval_count=count,
        prompt_eval_seconds=seconds,
        eval_count=count,
        eval_seconds=seconds,
    )

    assert timings.prompt_tokens_per_second is None
    assert timings.eval_tokens_per_second is None


def test_openai_compatible_result_has_no_timings() -> None:
    """互換経路は prompt/eval を返さないため timings は常に None (D-13)。"""
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=SUCCESS_PAYLOAD)
        )
    ) as http_client:
        client = OpenAICompatibleClient(
            load_config(DEFAULT_CONFIG), http_client=http_client
        )
        result = client.chat([ChatMessage(role="user", content="こんにちは")])

    assert result.timings is None


# --------------------------------------------------------------------------
# エラー翻訳 (互換経路と同じ表を共有していること)
# --------------------------------------------------------------------------


ERROR_CASES: dict[str, tuple[Handler, type[LlmkitError]]] = {
    "接続不可": (
        raising(httpx.ConnectError("connection refused")),
        RuntimeUnavailableError,
    ),
    "モデル不在": (responding(404, '{"error": "not found"}'), ModelNotFoundError),
    "VRAM 不足": (
        responding(500, '{"error": "CUDA error: out of memory"}'),
        OutOfMemoryError,
    ),
    "コンテキスト超過": (
        responding(400, '{"error": "context length exceeded"}'),
        ContextLengthError,
    ),
    "その他 5xx": (responding(503, '{"error": "unavailable"}'), UpstreamError),
}


def run_compat_chat(config: AppConfig, handler: Handler) -> None:
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        client.chat([ChatMessage(role="user", content="こんにちは")])


@pytest.mark.parametrize(
    ("handler", "expected"), list(ERROR_CASES.values()), ids=list(ERROR_CASES)
)
def test_native_error_translation_matches_the_compat_path(
    handler: Handler, expected: type[LlmkitError]
) -> None:
    """5 経路すべてで互換経路と同じ例外型が出る (翻訳表は 1 つしかない)。"""
    config = load_config(DEFAULT_CONFIG)

    with pytest.raises(LlmkitError) as native_info:
        run_native_chat(config, handler)
    with pytest.raises(LlmkitError) as compat_info:
        run_compat_chat(config, handler)

    assert type(native_info.value) is expected
    assert type(native_info.value) is type(compat_info.value)


EMBEDDED_ERROR_CASES: dict[str, tuple[str, type[LlmkitError]]] = {
    "モデル不在": (
        '{"error": "model not found, try pulling it first"}',
        ModelNotFoundError,
    ),
    "VRAM 不足": (
        '{"error": "cudaMalloc failed: insufficient memory"}',
        OutOfMemoryError,
    ),
    "コンテキスト超過": (
        '{"error": "maximum context length exceeded"}',
        ContextLengthError,
    ),
    "その他": ('{"error": "unexpected failure"}', UpstreamError),
}


@pytest.mark.parametrize(
    ("body", "expected"),
    list(EMBEDDED_ERROR_CASES.values()),
    ids=list(EMBEDDED_ERROR_CASES),
)
def test_native_http_200_with_error_body_is_translated(
    body: str, expected: type[LlmkitError]
) -> None:
    """HTTP 200 のまま本文にエラーが載る場合も同じ翻訳表に流す。

    ここで拾わないと「必須フィールドが無い」という誤った UpstreamError になり、
    原因 (モデル不在・VRAM 不足) が失われる。
    """
    with pytest.raises(LlmkitError) as excinfo:
        run_native_chat(load_config(DEFAULT_CONFIG), responding(200, body))

    assert type(excinfo.value) is expected


def test_native_empty_error_field_does_not_break_a_successful_response() -> None:
    """``error`` が空文字なら正常応答として扱う (「非空」の境界)。"""
    payload = {**NATIVE_SUCCESS_PAYLOAD, "error": ""}

    result, _ = run_native_chat(load_config(DEFAULT_CONFIG), responding_json(payload))

    assert result.text == "テスト応答"


def test_native_connect_failure_mentions_base_url_and_ollama_serve() -> None:
    """メッセージに出すのは endpoint_url ではなく runtime.base_url。

    利用者が直せるのは設定値 (base_url) であって導出後の URL ではない。
    """
    with pytest.raises(RuntimeUnavailableError) as excinfo:
        run_native_chat(
            load_config(DEFAULT_CONFIG), raising(httpx.ConnectError("refused"))
        )

    message = str(excinfo.value)
    assert "http://localhost:11434/v1" in message
    assert "ollama serve" in message
    assert "起動していない可能性" in message


def test_native_upstream_error_does_not_contain_the_response_body() -> None:
    secret_detail = "INTERNAL-STACKTRACE-DO-NOT-LEAK"

    with pytest.raises(UpstreamError) as excinfo:
        run_native_chat(
            load_config(DEFAULT_CONFIG),
            responding(503, f'{{"error": "{secret_detail}"}}'),
        )

    message = str(excinfo.value)
    assert "503" in message
    assert secret_detail not in message


def test_native_200_error_body_is_not_echoed_in_the_message() -> None:
    secret_detail = "INTERNAL-STACKTRACE-DO-NOT-LEAK"

    with pytest.raises(UpstreamError) as excinfo:
        run_native_chat(
            load_config(DEFAULT_CONFIG),
            responding(200, f'{{"error": "{secret_detail}"}}'),
        )

    assert secret_detail not in str(excinfo.value)
    assert "200" in str(excinfo.value)


NATIVE_FAILURE_HANDLERS: dict[str, Handler] = {
    name: handler for name, (handler, _) in ERROR_CASES.items()
}
NATIVE_FAILURE_HANDLERS["応答が不正"] = responding(200, "{}")
NATIVE_FAILURE_HANDLERS["本文にエラー (HTTP 200)"] = responding(
    200, '{"error": "unexpected failure"}'
)


@pytest.mark.parametrize(
    "handler",
    list(NATIVE_FAILURE_HANDLERS.values()),
    ids=list(NATIVE_FAILURE_HANDLERS),
)
def test_native_error_messages_never_contain_api_key(
    handler: Handler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ネイティブ経路のどの失敗でも api_key の平文が例外に漏れない。"""
    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)
    config = load_config(EXTERNAL_CONFIG)
    assert config.api_key.get_secret_value() == API_KEY

    with pytest.raises(LlmkitError) as excinfo:
        run_native_chat(config, handler)

    assert API_KEY not in str(excinfo.value)
    assert API_KEY not in repr(excinfo.value)
    assert API_KEY not in excinfo.value.remediation
