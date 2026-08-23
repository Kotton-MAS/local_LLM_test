"""比較ハーネスの CLI (harness/cli.py) のテスト。

このファイルが固定する性質は 3 つある。

1. **``--dry-run`` は HTTP を 1 バイトも出さずに ``run_fingerprint`` を出す。**
   実行前に「この条件で回す」を提示できることが、比較を回す前の唯一の防御。
2. **絞り込み (``--models`` / ``--limit``) は入力そのものを変える**ため、
   ``run_fingerprint`` はフル実行と一致しない。一致してしまうと「1 問だけ
   回した結果」と「8 問回した結果」が同じ再現条件を名乗る。
3. **絞り込みの失敗は :class:`llmkit.ConfigError` として出る** (D-18 の
   例外階層)。空の ``[[models]]`` をそのまま組み立てると、利用者に pydantic
   の内部エラー (llmkit の例外階層の外) がそのまま表示される。

書き出し先は必ず ``tmp_path`` に注入する。テストはリポジトリの ``results/``
に 1 バイトも書かない。実 HTTP も発行しない (D-02)。
"""

from __future__ import annotations

import ast
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from conftest import (
    DEFAULT_CONFIG,
    NATIVE_SUCCESS_PAYLOAD,
    FakeProbe,
    Handler,
    RecordingTransport,
)

from harness.cli import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SUITE_PATH,
    EXIT_ERROR,
    EXIT_OK,
    main,
)
from harness.gpu import GpuMemory
from harness.report import (
    RECORDS_FILENAME,
    REPORT_FILENAME,
    RUN_JSON_FILENAME,
)
from harness.runner import MANIFESTS_DIRNAME
from harness.suite import load_suite

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "harness"

SUITE_TEXT = """\
[suite]
id = "cli_test"
description = "CLI のテスト用スイート"
warmup_runs = 1

[[models]]
model_id = "qwen3-8b"
profile = "lightweight"

[[models]]
model_id = "gpt-oss-20b"
profile = "long_context"

[[prompts]]
id = "summarize"
text = "次の文章を三行で要約してください。"

[[prompts]]
id = "translate"
text = "次の文を英語に訳してください。"
"""

MODEL_COUNT = 2
PROMPT_COUNT = 2
FIXED_MOMENT = datetime(2026, 8, 23, 12, 34, 56, tzinfo=UTC)
TIMESTAMP = "20260823T123456Z"

#: 出荷するスイートの前提 (Q-1 / Q-3 で承認された内容)。
SHIPPED_PROMPT_COUNT = 8
SHIPPED_MODEL_IDS = ["qwen3-14b", "gpt-oss-20b", "qwen3-8b"]
SHIPPED_MAX_OUTPUT_TOKENS = 512


class Invocation:
    """CLI を 1 回呼んだ結果 (終了コード・標準出力・発行した HTTP)。"""

    def __init__(
        self, code: int, stdout: str, stderr: str, transport: RecordingTransport
    ) -> None:
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.transport = transport

    @property
    def http_calls(self) -> int:
        return self.transport.call_count

    def fingerprint(self) -> str:
        line = next(
            line
            for line in self.stdout.splitlines()
            if line.startswith("run_fingerprint")
        )
        return line.split(":", 1)[1].strip()


def write_suite(directory: Path, text: str = SUITE_TEXT) -> Path:
    destination = directory / "suite.toml"
    destination.write_text(text, encoding="utf-8")
    return destination


def invoke(
    tmp_path: Path,
    *arguments: str,
    handler: Handler | None = None,
    probe: FakeProbe | None = None,
    suite_path: Path | None = None,
    results_root: Path | None = None,
) -> Invocation:
    """``python -m harness.cli run ...`` と同じ経路を実 HTTP なしで呼ぶ。"""
    suite = suite_path if suite_path is not None else write_suite(tmp_path)
    transport = RecordingTransport(handler)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with transport.client() as http_client:
        code = main(
            [
                "run",
                "--suite",
                str(suite),
                "--config",
                str(DEFAULT_CONFIG),
                *arguments,
            ],
            http_client=http_client,
            probe=probe if probe is not None else FakeProbe(),
            results_root=results_root
            if results_root is not None
            else tmp_path / "results",
            clock=lambda: FIXED_MOMENT,
            stdout=stdout,
            stderr=stderr,
        )
    return Invocation(code, stdout.getvalue(), stderr.getvalue(), transport)


def output_dir(tmp_path: Path, fingerprint: str, suite_id: str = "cli_test") -> Path:
    return tmp_path / "results" / suite_id / f"{TIMESTAMP}-{fingerprint[:12]}"


# --------------------------------------------------------------------------
# --dry-run (HTTP 0 回)
# --------------------------------------------------------------------------


def test_dry_run_prints_the_fingerprint_without_issuing_any_http(
    tmp_path: Path,
) -> None:
    """★ 受け入れ条件: ランタイム無しで exit 0、HTTP 0 回。"""
    invocation = invoke(tmp_path, "--dry-run")

    assert invocation.code == EXIT_OK
    assert invocation.http_calls == 0
    assert len(invocation.fingerprint()) == 64
    assert "dry-run" in invocation.stdout


def test_dry_run_writes_nothing_to_the_results_directory(tmp_path: Path) -> None:
    invoke(tmp_path, "--dry-run")

    assert not (tmp_path / "results").exists()


def test_dry_run_reports_the_budget_verdict_for_every_model(tmp_path: Path) -> None:
    invocation = invoke(tmp_path, "--dry-run")

    assert invocation.stdout.count("予算内") == MODEL_COUNT
    assert "予定リクエスト数: 6" in invocation.stdout


def test_dry_run_never_starts_a_vram_probe(tmp_path: Path) -> None:
    """計画だけならプロセスも起動しない (性能観点: プロセス生成 0 回)。"""
    probe = FakeProbe()

    invoke(tmp_path, "--dry-run", probe=probe)

    assert probe.call_count == 0


# --------------------------------------------------------------------------
# 本実行
# --------------------------------------------------------------------------


def test_a_run_writes_the_three_files_and_the_manifests(tmp_path: Path) -> None:
    invocation = invoke(tmp_path)

    assert invocation.code == EXIT_OK
    directory = output_dir(tmp_path, invocation.fingerprint())
    assert (directory / REPORT_FILENAME).is_file()
    assert (directory / RECORDS_FILENAME).is_file()
    assert (directory / RUN_JSON_FILENAME).is_file()
    assert len(list((directory / MANIFESTS_DIRNAME).glob("*.json"))) == MODEL_COUNT
    assert str(directory) in invocation.stdout


def test_the_run_json_agrees_with_the_printed_fingerprint(tmp_path: Path) -> None:
    invocation = invoke(tmp_path)

    directory = output_dir(tmp_path, invocation.fingerprint())
    payload = json.loads((directory / RUN_JSON_FILENAME).read_text(encoding="utf-8"))

    assert payload["run_fingerprint"] == invocation.fingerprint()
    assert payload["started_at_utc"] == "2026-08-23T12:34:56Z"


def test_a_run_issues_one_request_per_warmup_and_prompt(tmp_path: Path) -> None:
    invocation = invoke(tmp_path)

    assert invocation.http_calls == MODEL_COUNT * (1 + PROMPT_COUNT)


def test_a_run_where_every_model_fails_exits_one(tmp_path: Path) -> None:
    """全ケース失敗のときだけ exit 1 (§3 ソフト制約)。"""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    invocation = invoke(tmp_path, handler=handle)

    assert invocation.code == EXIT_ERROR
    assert "1 件も測定できませんでした" in invocation.stdout


def test_one_failing_model_still_exits_zero(tmp_path: Path) -> None:
    """1 モデルだけの失敗は記録して完走する (打ち切らない)。"""

    def handle(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "qwen3:8b-q4_K_M":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(200, json=NATIVE_SUCCESS_PAYLOAD)

    invocation = invoke(tmp_path, handler=handle)

    assert invocation.code == EXIT_OK


def test_the_probe_readings_reach_the_run_json(tmp_path: Path) -> None:
    probe = FakeProbe(
        [
            GpuMemory(name="gpu", used_mib=600, total_mib=16384),
            GpuMemory(name="gpu", used_mib=5720, total_mib=16384),
        ]
    )

    invocation = invoke(tmp_path, probe=probe)

    directory = output_dir(tmp_path, invocation.fingerprint())
    payload = json.loads((directory / RUN_JSON_FILENAME).read_text(encoding="utf-8"))

    assert payload["models"][0]["vram_used_mib"] == 5720


# --------------------------------------------------------------------------
# --models
# --------------------------------------------------------------------------


def test_models_filter_narrows_the_run_and_changes_the_fingerprint(
    tmp_path: Path,
) -> None:
    """★ 受け入れ条件: 対象が 1 件になり、フル実行と別の再現条件になる。"""
    full = invoke(tmp_path, "--dry-run")

    filtered = invoke(tmp_path, "--dry-run", "--models", "qwen3-8b")

    assert filtered.stdout.count("[0]") == 1
    assert "[1]" not in filtered.stdout
    assert "qwen3-8b" in filtered.stdout
    assert filtered.fingerprint() != full.fingerprint()


def test_models_filter_keeps_the_suite_declaration_order(tmp_path: Path) -> None:
    invocation = invoke(tmp_path, "--dry-run", "--models", "gpt-oss-20b,qwen3-8b")

    lines = [line for line in invocation.stdout.splitlines() if line.startswith("  [")]

    assert [line.split()[1] for line in lines] == ["qwen3-8b", "gpt-oss-20b"]


def test_an_unknown_model_name_is_reported_as_a_config_error(tmp_path: Path) -> None:
    """★ 受け入れ条件: メッセージに ``--models`` と指定値が含まれる。"""
    invocation = invoke(tmp_path, "--dry-run", "--models", "qwen3-99b")

    assert invocation.code == EXIT_ERROR
    assert "--models" in invocation.stderr
    assert "qwen3-99b" in invocation.stderr
    assert invocation.http_calls == 0


def test_an_empty_models_filter_never_leaks_a_pydantic_error(tmp_path: Path) -> None:
    """★ 絞り込み結果 0 件は llmkit の例外階層の中で失敗する (§9 決定28)。

    ``dataclasses.replace(suite, models=())`` をそのまま呼ぶと
    ``pydantic_core.ValidationError`` が外に出る。CLI の利用者に見せるのは
    ``ConfigError`` の「対処方法つきメッセージ」でなければならない。
    """
    invocation = invoke(tmp_path, "--dry-run", "--models", ",")

    assert invocation.code == EXIT_ERROR
    assert invocation.stderr.startswith("エラー: ")
    assert "--models" in invocation.stderr
    assert "対処" in invocation.stderr
    assert "ValidationError" not in invocation.stderr
    assert "pydantic" not in invocation.stderr


# --------------------------------------------------------------------------
# --limit
# --------------------------------------------------------------------------


def test_limit_reduces_the_prompts_and_changes_the_fingerprint(
    tmp_path: Path,
) -> None:
    full = invoke(tmp_path, "--dry-run")

    limited = invoke(tmp_path, "--dry-run", "--limit", "1")

    assert "プロンプト      : 1 問" in limited.stdout
    assert limited.fingerprint() != full.fingerprint()


def test_limit_reduces_the_number_of_requests(tmp_path: Path) -> None:
    invocation = invoke(tmp_path, "--limit", "1")

    assert invocation.http_calls == MODEL_COUNT * (1 + 1)


def test_limit_below_one_is_rejected_with_a_config_error(tmp_path: Path) -> None:
    invocation = invoke(tmp_path, "--dry-run", "--limit", "0")

    assert invocation.code == EXIT_ERROR
    assert "--limit" in invocation.stderr
    assert "ValidationError" not in invocation.stderr


def test_limit_above_the_prompt_count_keeps_every_prompt(tmp_path: Path) -> None:
    full = invoke(tmp_path, "--dry-run")

    generous = invoke(tmp_path, "--dry-run", "--limit", "99")

    assert generous.fingerprint() == full.fingerprint()


# --------------------------------------------------------------------------
# 入力の失敗
# --------------------------------------------------------------------------


def test_a_missing_suite_file_exits_one_with_a_remediation(tmp_path: Path) -> None:
    invocation = invoke(tmp_path, "--dry-run", suite_path=tmp_path / "missing.toml")

    assert invocation.code == EXIT_ERROR
    assert "対処" in invocation.stderr
    assert invocation.stdout == ""


# --------------------------------------------------------------------------
# 出荷するスイート (suites/ja_basic.toml)
# --------------------------------------------------------------------------


def test_the_default_paths_point_at_files_that_exist() -> None:
    assert (REPO_ROOT / DEFAULT_SUITE_PATH).is_file()
    assert (REPO_ROOT / DEFAULT_CONFIG_PATH).is_file()


def test_the_shipped_suite_declares_eight_japanese_prompts_and_three_models() -> None:
    """承認済みの内容 (8 問 / 3 モデル / max_output_tokens=512) を固定する。"""
    suite = load_suite(REPO_ROOT / DEFAULT_SUITE_PATH)

    assert suite.suite.id == "ja_basic"
    assert len(suite.prompts) == SHIPPED_PROMPT_COUNT
    assert [case.model_id for case in suite.models] == SHIPPED_MODEL_IDS
    assert all(
        case.max_output_tokens == SHIPPED_MAX_OUTPUT_TOKENS for case in suite.models
    )
    assert len(set(suite.prompt_ids)) == SHIPPED_PROMPT_COUNT


def test_every_shipped_prompt_is_written_in_japanese() -> None:
    """L305 の前提: 日本語のプロンプトであること (CJK を含む)。"""
    suite = load_suite(REPO_ROOT / DEFAULT_SUITE_PATH)

    for prompt in suite.prompts:
        assert any("぀" <= char <= "ヿ" for char in prompt.text), prompt.id


def test_the_shipped_suite_carries_no_credential_like_strings() -> None:
    """``results/`` にコミットされるため、機密らしき文字列を持ち込まない (D-24)。"""
    text = (REPO_ROOT / DEFAULT_SUITE_PATH).read_text(encoding="utf-8")

    for forbidden in ("sk-", "Bearer ", "password", "api_key", "token="):
        assert forbidden not in text, forbidden


# --------------------------------------------------------------------------
# コミット済み results/ 自体の機密走査 (F-8-004: D-24 の掃引はスイート入力
# だけを見ており、出力側 (manifests/*.json・run.json・report.md・
# records.jsonl) 自体は走査していなかった)
# --------------------------------------------------------------------------

RESULTS_ROOT = REPO_ROOT / "results"

#: ``"api_key"`` は正規表現で JSON キーとして厳密に見る。単純な部分一致だと
#: D-05 が意図的に残す ``"api_key_env"`` (環境変数「名」) を誤検出するため。
_FORBIDDEN_RESULTS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-"),
    re.compile(r"Bearer "),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r'"api_key"\s*:'),
    re.compile(r"token="),
    # URL の userinfo (https://user:pass@host/...)。
    re.compile(r"https?://[^/\s]+:[^/@\s]+@"),
    # ホストの絶対パス (ユーザー名を含みうる)。--config に絶対パスを渡した
    # 実行が run.json / manifest に環境依存パスを残す経路も同時に塞ぐ。
    re.compile(r"/home/|/Users/"),
)


def test_committed_results_carry_no_credential_like_strings() -> None:
    """★ D-24 guard 拡張: ``results/`` 配下の全ファイル自体を機密走査する。

    出荷スイートだけを検査していた既存 guard (``D-24``) は入力側しか見て
    おらず、``results/`` にコミットされた成果物 (出力側) は 1 度も走査されて
    いなかった (F-8-004)。base_url のパスに埋め込まれた秘密・絶対パスは
    config.py の検証をすり抜けうるため、出力側も機械的に検査する。
    """
    if not RESULTS_ROOT.exists():
        pytest.skip("results/ が無いチェックアウトではスキップ")

    files = [path for path in RESULTS_ROOT.rglob("*") if path.is_file()]
    assert files, "results/ にコミット済みファイルがありません"

    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_RESULTS_PATTERNS:
            assert not pattern.search(text), f"{path}: {pattern.pattern}"


# --------------------------------------------------------------------------
# print() 禁止 (§9 決定29)
# --------------------------------------------------------------------------


def test_no_harness_module_calls_print() -> None:
    """``harness/`` に ``print()`` の呼び出しが 1 つも無い。

    仕様書 §4 T4 の ``rg -n "print\\(" harness/`` は ``run_fingerprint(`` /
    ``def fingerprint(`` に部分一致して常にヒットするため、単語ではなく
    **呼び出し** を AST で見る。
    """
    offenders: list[str] = []
    for module_path in sorted(HARNESS_DIR.glob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{module_path.name}:{node.lineno}")

    assert not offenders, f"print() を呼んでいる箇所: {offenders}"


def test_the_print_guard_detects_a_real_violation() -> None:
    """上の検査が「何も検出しないだけ」でないことを確かめる (変異検証)。"""
    tree = ast.parse('print("x")\nrun_fingerprint()\n')
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert calls == ["print", "run_fingerprint"]


def test_the_cli_writes_only_to_the_injected_streams(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """出力は注入されたストリームだけに行き、実 stdout を汚さない。"""
    invocation = invoke(tmp_path, "--dry-run")

    captured = capsys.readouterr()
    assert invocation.stdout != ""
    assert invocation.stderr == ""
    assert captured.out == ""
    assert captured.err == ""
