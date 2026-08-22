"""要件書 Phase 1 受け入れ条件の機械検証。

``docs/localllmrequirements.md`` L294-L298 の 5 条件と、このモジュールの 5 つの
テスト関数を **1 対 1** に対応させる。各 docstring の先頭に対応行を書く。
条件が増えたらテストも増やす (``test_l298_...`` が対応関係そのものを検査する)。

実 HTTP は発行しない (D-02)。マニフェストの書き出し先は必ず ``tmp_path``。
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
from pathlib import Path

import httpx
import pytest
from conftest import RecordingTransport, write_config_variant

from llmkit.bootstrap import bootstrap
from llmkit.client import ChatMessage, create_chat_client
from llmkit.config import load_config
from llmkit.errors import RuntimeUnavailableError, VramBudgetExceededError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"
REQUIREMENTS = REPO_ROOT / "docs" / "localllmrequirements.md"

#: 要件書の行番号 -> 対応するテスト関数名。
ACCEPTANCE_MAP = {
    294: "test_l294_model_change_alone_switches_the_target",
    295: "test_l295_profile_estimate_is_logged_at_startup",
    296: "test_l296_oversized_profile_warns_and_stops",
    297: "test_l297_runtime_unavailable_error_identifies_the_cause",
    298: "test_l298_automated_tests_exist_for_every_condition",
}


def requirement_line(number: int) -> str:
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    return lines[number - 1]


def test_l294_model_change_alone_switches_the_target() -> None:
    """要件書 L294: 設定ファイルのモデル名変更のみで推論先が切り替わる。

    呼び出しコードは 1 行も変えず、``generation.model`` だけを差し替えると
    送出される ``model`` (= ModelSpec.served_name) が変わる。

    クライアントは ``create_chat_client`` で作る。bootstrap / CLI が実際に通る
    経路 (既定の ``kind = "ollama"`` ならネイティブ ``/api/chat``) で測るため。
    """
    base = load_config(DEFAULT_CONFIG)
    switched = dataclasses.replace(
        base, generation=dataclasses.replace(base.generation, model="qwen3-8b")
    )
    sent: list[str] = []

    for config in (base, switched):
        recorder = RecordingTransport()
        with recorder.client() as http_client:
            # ここから下は 2 回とも完全に同一のコード。
            client = create_chat_client(config, http_client=http_client)
            client.chat([ChatMessage(role="user", content="こんにちは")])
        payload: object = json.loads(recorder.requests[0].content)
        assert isinstance(payload, dict)
        sent.append(str(payload["model"]))

    assert sent == ["qwen3:14b-q4_K_M", "qwen3:8b-q4_K_M"]


def test_l295_profile_estimate_is_logged_at_startup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """要件書 L295: VRAM プロファイルを指定でき、起動時に想定使用量がログに出る。"""
    recorder = RecordingTransport()

    with (
        caplog.at_level(logging.INFO, logger="llmkit"),
        recorder.client() as http_client,
    ):
        result = bootstrap(
            DEFAULT_CONFIG,
            profile_name="long_context",
            http_client=http_client,
            output_dir=tmp_path / "runs",
        )

    info_log = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    )

    assert result.profile.name == "long_context"
    assert "long_context" in info_log
    assert f"{result.estimate.total_gib:.2f}" in info_log
    assert f"{result.estimate.budget_gib:.2f}" in info_log


def test_l296_oversized_profile_warns_and_stops(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """要件書 L296: 予算を超えるプロファイルは起動時に警告して停止する。

    (a) WARNING 以上のログ / (b) VramBudgetExceededError / (c) HTTP 0 回。
    """
    config_path = write_config_variant(
        tmp_path,
        {
            'active_profile = "rag_default"': 'active_profile = "oversized"',
            "context_tokens = 16384": "context_tokens = 131072",
        },
    )
    recorder = RecordingTransport()

    with (
        caplog.at_level(logging.INFO, logger="llmkit"),
        recorder.client() as http_client,
        pytest.raises(VramBudgetExceededError),
    ):
        bootstrap(config_path, http_client=http_client, output_dir=tmp_path / "runs")

    assert any(record.levelno >= logging.WARNING for record in caplog.records)
    assert recorder.call_count == 0


def test_l297_runtime_unavailable_error_identifies_the_cause(tmp_path: Path) -> None:
    """要件書 L297: ランタイム未起動時に原因が特定できるエラーメッセージが出る。"""

    def refuse(request: httpx.Request) -> httpx.Response:
        message = "connection refused"
        raise httpx.ConnectError(message)

    recorder = RecordingTransport(refuse)

    with (
        recorder.client() as http_client,
        pytest.raises(RuntimeUnavailableError) as excinfo,
    ):
        result = bootstrap(
            DEFAULT_CONFIG, http_client=http_client, output_dir=tmp_path / "runs"
        )
        result.client.chat([ChatMessage(role="user", content="こんにちは")])

    message = str(excinfo.value)
    assert "http://localhost:11434/v1" in message
    assert "ollama serve" in message
    assert "起動していない可能性" in message


def test_l298_automated_tests_exist_for_every_condition() -> None:
    """要件書 L298: 上記に対する自動テストが存在しパスする。

    このテスト自身が「条件とテストの 1 対 1 対応」を検査する。実行そのものは
    ``make ci`` (lock-check / lint / fmt-check / type / test) が担保する。
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }

    # 1. 受け入れ条件は 5 個、テストも 5 個で、名前が一致する
    assert len(ACCEPTANCE_MAP) == 5
    assert defined == set(ACCEPTANCE_MAP.values())

    # 2. 各テストの docstring が対応する要件書の行を引用している
    for number, test_name in ACCEPTANCE_MAP.items():
        node = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == test_name
        )
        docstring = ast.get_docstring(node) or ""
        assert f"L{number}" in docstring, test_name

    # 3. 参照している行が要件書の Phase 1 受け入れ条件のままである
    for number in ACCEPTANCE_MAP:
        assert requirement_line(number).startswith("- [ ] "), number

    # 4. make ci がテストを含む単一の検証コマンドであり続けている
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "ci: lock-check lint fmt-check type test" in makefile
