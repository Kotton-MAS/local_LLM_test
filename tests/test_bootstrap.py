"""起動シーケンス (llmkit/bootstrap.py) と CLI (llmkit/cli.py) のテスト。

このファイルは 1 つの guard_test を含む:

- ``test_oversized_profile_warns_and_aborts_without_http`` … D-04
  (VRAM 予算超過は WARNING を出したうえで例外で停止し、HTTP を 1 回も発行しない)

実 HTTP は発行しない (D-02)。すべて ``httpx.MockTransport`` で完結し、
マニフェストの書き出し先は必ず ``tmp_path``。
"""

from __future__ import annotations

import ast
import io
import json
import logging
from pathlib import Path

import httpx
import pytest
from conftest import RecordingTransport, write_config_variant

from llmkit.bootstrap import bootstrap
from llmkit.cli import main as cli_main
from llmkit.client import ChatMessage
from llmkit.errors import ConfigError, VramBudgetExceededError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


def counting_transport(requests: list[httpx.Request]) -> httpx.MockTransport:
    """成功応答を返し、発行されたリクエストを ``requests`` に記録する。"""
    return RecordingTransport(requests=requests).transport


def refusing_transport(requests: list[httpx.Request]) -> httpx.MockTransport:
    """Ollama 未起動相当。接続を試みた事実だけを記録して ConnectError にする。"""

    def refuse(request: httpx.Request) -> httpx.Response:
        message = "connection refused"
        raise httpx.ConnectError(message)

    return RecordingTransport(refuse, requests=requests).transport


# --------------------------------------------------------------------------
# 受け入れ条件2 (要件書 L295): 起動時に想定使用量がログ出力される
# --------------------------------------------------------------------------


def test_bootstrap_logs_estimated_vram_usage(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """INFO ログに「プロファイル名・合計 GiB・予算値」の 3 つが出る。"""
    requests: list[httpx.Request] = []

    with (
        caplog.at_level(logging.INFO, logger="llmkit"),
        httpx.Client(transport=counting_transport(requests)) as http_client,
    ):
        bootstrap(DEFAULT_CONFIG, http_client=http_client, output_dir=tmp_path / "runs")

    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    combined = "\n".join(info_messages)

    assert info_messages, "INFO ログが 1 件も出ていない"
    assert "rag_default" in combined
    assert "12.63" in combined
    assert "14.00" in combined


def test_bootstrap_log_follows_the_selected_profile(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """プロファイルを指定すると、ログの内容がその見積りに追随する。"""
    requests: list[httpx.Request] = []

    with (
        caplog.at_level(logging.INFO, logger="llmkit"),
        httpx.Client(transport=counting_transport(requests)) as http_client,
    ):
        bootstrap(
            DEFAULT_CONFIG,
            profile_name="lightweight",
            http_client=http_client,
            output_dir=tmp_path / "runs",
        )

    assert "lightweight" in caplog.text
    # qwen3-8b 4.18 + KV 0.1425 * 16 + overhead 0.8 = 7.26
    assert "7.26" in caplog.text


# --------------------------------------------------------------------------
# 受け入れ条件3 (要件書 L296) / guard_test (D-04)
# --------------------------------------------------------------------------


def test_oversized_profile_warns_and_aborts_without_http(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """D-04 guard: 予算超過で (a) WARNING (b) 例外 (c) HTTP 0 回 の 3 つを満たす。

    較正後は構成3 (oversized) を 131,072 トークンで使うと見積り 16.74 GiB となり、
    **既定の ``vram.budget_gib`` (14.0) のまま**超過する。予算を下げる回避は使わない。
    """
    config_path = write_config_variant(
        tmp_path,
        {
            'active_profile = "rag_default"': 'active_profile = "oversized"',
            "context_tokens = 16384": "context_tokens = 131072",
        },
    )
    requests: list[httpx.Request] = []

    with (
        caplog.at_level(logging.INFO, logger="llmkit"),
        httpx.Client(transport=counting_transport(requests)) as http_client,
        pytest.raises(VramBudgetExceededError) as excinfo,
    ):
        bootstrap(config_path, http_client=http_client, output_dir=tmp_path / "runs")

    # (a) WARNING 以上のログが出る
    warnings = [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert warnings, "予算超過なのに WARNING 以上のログが出ていない"
    assert "oversized" in "\n".join(record.getMessage() for record in warnings)

    # (b) VramBudgetExceededError が送出され、内訳を保持している
    error = excinfo.value
    assert error.profile_name == "oversized"
    assert error.budget_gib == pytest.approx(14.0)
    assert error.total_gib > error.budget_gib
    assert error.excess_gib == pytest.approx(error.total_gib - error.budget_gib)

    # (c) httpx の呼び出しが 0 回
    assert requests == []

    # 停止した以上、クライアントもマニフェストも作られない
    assert not (tmp_path / "runs").exists()


def test_lowering_the_budget_alone_flips_bootstrap_from_ok_to_abort(
    tmp_path: Path,
) -> None:
    """E4 の起動シーケンス版: budget_gib だけで起動可否が反転する。"""
    requests: list[httpx.Request] = []
    ok_path = write_config_variant(
        tmp_path,
        {
            "budget_gib = 14.0": "budget_gib = 17.0",
            'active_profile = "rag_default"': 'active_profile = "oversized"',
            "context_tokens = 16384": "context_tokens = 131072",
        },
        name="ok.toml",
    )
    ng_path = write_config_variant(
        tmp_path,
        {
            'active_profile = "rag_default"': 'active_profile = "oversized"',
            "context_tokens = 16384": "context_tokens = 131072",
        },
        name="ng.toml",
    )

    with httpx.Client(transport=counting_transport(requests)) as http_client:
        bootstrap(ok_path, http_client=http_client, output_dir=tmp_path / "runs")

        with pytest.raises(VramBudgetExceededError):
            bootstrap(ng_path, http_client=http_client, output_dir=tmp_path / "runs")

    assert requests == []


def test_remote_runtime_skips_the_budget_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """is_local = false なら予算超過でも停止しない。

    外部 API はローカル VRAM を使わないため。
    """
    monkeypatch.setenv("LLMKIT_API_KEY", "sk-test-do-not-leak-0123456789")
    config_path = write_config_variant(
        tmp_path,
        {
            "budget_gib = 14.0": "budget_gib = 1.0",
            "is_local = true": "is_local = false",
        },
    )
    requests: list[httpx.Request] = []

    with httpx.Client(transport=counting_transport(requests)) as http_client:
        result = bootstrap(
            config_path, http_client=http_client, output_dir=tmp_path / "runs"
        )

    assert result.estimate.within_budget is False
    assert result.manifest_path is not None


# --------------------------------------------------------------------------
# 正常系・再現性
# --------------------------------------------------------------------------


def test_bootstrap_returns_usable_client_and_writes_manifest(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    with httpx.Client(transport=counting_transport(requests)) as http_client:
        result = bootstrap(
            DEFAULT_CONFIG, http_client=http_client, output_dir=tmp_path / "runs"
        )
        assert requests == [], "クライアント生成だけで HTTP を発行してはならない"
        chat_result = result.client.chat([ChatMessage(role="user", content="やあ")])

    assert chat_result.text == "テスト応答"
    assert len(requests) == 1
    assert result.manifest_path is not None
    assert result.manifest_path.exists()
    assert result.endpoint_url == "http://localhost:11434/api/chat"
    assert result.served_name == "qwen3:14b-q4_K_M"


def test_bootstrap_can_skip_writing_the_manifest_file(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    with httpx.Client(transport=counting_transport(requests)) as http_client:
        result = bootstrap(
            DEFAULT_CONFIG,
            http_client=http_client,
            output_dir=tmp_path / "runs",
            write_manifest_file=False,
        )

    assert result.manifest_path is None
    assert not (tmp_path / "runs").exists()
    assert result.manifest.config_sha256


def test_config_sha256_is_reproducible_and_content_sensitive(tmp_path: Path) -> None:
    """再現性: 同一設定なら config_sha256 が一致し、1 文字変えると変わる。"""
    requests: list[httpx.Request] = []
    changed = write_config_variant(
        tmp_path, {"temperature = 0.7": "temperature = 0.8"}, name="changed.toml"
    )

    with httpx.Client(transport=counting_transport(requests)) as http_client:
        first = bootstrap(
            DEFAULT_CONFIG, http_client=http_client, output_dir=tmp_path / "runs"
        )
        second = bootstrap(
            DEFAULT_CONFIG, http_client=http_client, output_dir=tmp_path / "runs"
        )
        third = bootstrap(
            changed, http_client=http_client, output_dir=tmp_path / "runs"
        )

    assert first.manifest.config_sha256 == second.manifest.config_sha256
    assert third.manifest.config_sha256 != first.manifest.config_sha256
    assert first.manifest.run_id != second.manifest.run_id


def test_unknown_profile_raises_config_error(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    with (
        httpx.Client(transport=counting_transport(requests)) as http_client,
        pytest.raises(ConfigError) as excinfo,
    ):
        bootstrap(
            DEFAULT_CONFIG,
            profile_name="does_not_exist",
            http_client=http_client,
            output_dir=tmp_path / "runs",
        )

    assert "does_not_exist" in str(excinfo.value)
    assert requests == []


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run_cli(argv: list[str], transport: httpx.MockTransport) -> tuple[int, str, str]:
    """CLI を 1 回実行し、(終了コード, stdout, stderr) を返す。"""
    stdout = io.StringIO()
    stderr = io.StringIO()
    with httpx.Client(transport=transport) as http_client:
        code = cli_main(argv, http_client=http_client, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def test_cli_doctor_reports_runtime_unavailable_with_exit_code_1(
    tmp_path: Path,
) -> None:
    """Ollama 未起動相当 (ConnectError) で終了コード 1、原因と対処が出る。"""
    requests: list[httpx.Request] = []

    code, out, err = run_cli(
        [
            "doctor",
            "--config",
            str(DEFAULT_CONFIG),
            "--output-dir",
            str(tmp_path / "runs"),
        ],
        refusing_transport(requests),
    )

    combined = out + err
    assert code == 1
    assert "ollama serve" in combined
    assert "http://localhost:11434/v1" in combined
    assert len(requests) == 1


def test_cli_doctor_succeeds_against_a_reachable_runtime(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    output_dir = tmp_path / "runs"

    code, out, err = run_cli(
        ["doctor", "--config", str(DEFAULT_CONFIG), "--output-dir", str(output_dir)],
        counting_transport(requests),
    )

    assert code == 0, err
    assert "rag_default" in out
    assert "12.63" in out
    assert "qwen3:14b-q4_K_M" in out
    assert len(requests) == 1
    assert len(list(output_dir.glob("*.json"))) == 1


def test_cli_chat_writes_the_response_text(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    code, out, _ = run_cli(
        [
            "chat",
            "こんにちは",
            "--config",
            str(DEFAULT_CONFIG),
            "--output-dir",
            str(tmp_path / "runs"),
        ],
        counting_transport(requests),
    )

    assert code == 0
    assert "テスト応答" in out
    assert len(requests) == 1
    payload: object = json.loads(requests[0].content)
    assert isinstance(payload, dict)
    assert payload["messages"] == [{"role": "user", "content": "こんにちは"}]


def test_cli_doctor_exits_1_and_makes_no_request_when_budget_exceeded(
    tmp_path: Path,
) -> None:
    config_path = write_config_variant(
        tmp_path,
        {
            'active_profile = "rag_default"': 'active_profile = "oversized"',
            "context_tokens = 16384": "context_tokens = 131072",
        },
    )
    requests: list[httpx.Request] = []

    code, out, err = run_cli(
        [
            "doctor",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
        ],
        counting_transport(requests),
    )

    assert code == 1
    assert requests == []
    assert "oversized" in out + err


def test_llmkit_modules_never_call_print() -> None:
    """人間向け出力は print() ではなくストリームへの書き込みで行う (ruff T20)。"""
    for module_path in sorted((REPO_ROOT / "llmkit").glob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert not calls, f"{module_path.name} が print() を呼んでいる"
