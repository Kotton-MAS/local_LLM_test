"""VRAM 実測プローブ (harness/gpu.py) のテスト。

**実プロセスを 1 つも起動しない。** ``NvidiaSmiProbe`` はコマンド実行関数を
コンストラクタで注入できるため、正常系も失敗系も偽の実行関数で決定論的に測れる
(CI には GPU も ``nvidia-smi`` も無い)。

このファイルが固定する性質は 2 つ:

1. 実測の失敗 (未インストール / 非 0 終了 / パース失敗 / タイムアウト) は
   **例外を外に出さず ``None`` を返す**。実測は比較の付加情報であり、取得できない
   ことが比較実行そのものを落とす理由にならない。
2. GPU に触るコードが ``harness/gpu.py`` の 1 ファイルに隔離されている
   (``llmkit`` は静的テーブルだけで見積もる、D-01)。
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from conftest import DEFAULT_CONFIG, RecordingTransport

from harness.gpu import (
    DEFAULT_NVIDIA_SMI_COMMAND,
    DEFAULT_TIMEOUT_S,
    CommandResult,
    GpuMemory,
    NvidiaSmiProbe,
    VramProbe,
)
from harness.report import MISSING_CELL, render_report
from harness.runner import plan_run, run_suite
from harness.suite import load_suite
from llmkit import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
LLMKIT_DIR = REPO_ROOT / "llmkit"
HARNESS_DIR = REPO_ROOT / "harness"

VALID_CSV = "NVIDIA GeForce RTX 5070 Ti, 596, 16376\n"


class FakeRunner:
    """コマンド実行を差し替える偽の実行関数。プロセスは起動しない。

    ``result`` を返すか ``error`` を送出するかのどちらか。呼ばれた引数を記録する。
    """

    def __init__(
        self,
        *,
        result: CommandResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(self, command: Sequence[str], *, timeout_s: float) -> CommandResult:
        self.calls.append((tuple(command), timeout_s))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def completed(stdout: str, *, returncode: int = 0) -> CommandResult:
    """``subprocess.run`` の戻り値と同じ型の偽の結果 (プロセスは起動しない)。

    偽の型を自作せず ``CompletedProcess`` を使うことで、``CommandResult``
    Protocol が実物と構造的に一致していることも同時に確かめている。
    """
    return subprocess.CompletedProcess(
        args=list(DEFAULT_NVIDIA_SMI_COMMAND),
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def probe_with(runner: FakeRunner) -> NvidiaSmiProbe:
    return NvidiaSmiProbe(runner=runner)


# --------------------------------------------------------------------------
# 5 ケース: (a) 正常 (b) 未インストール (c) 非 0 終了 (d) 壊れた CSV (e) タイムアウト
# --------------------------------------------------------------------------


def test_a_successful_read_returns_gpu_memory() -> None:
    runner = FakeRunner(result=completed(VALID_CSV))

    memory = probe_with(runner).read()

    assert memory == GpuMemory(
        name="NVIDIA GeForce RTX 5070 Ti", used_mib=596, total_mib=16376
    )
    assert runner.calls == [(DEFAULT_NVIDIA_SMI_COMMAND, DEFAULT_TIMEOUT_S)]


FAILURE_CASES: tuple[tuple[str, FakeRunner], ...] = (
    (
        "b_not_installed",
        FakeRunner(error=FileNotFoundError(2, "No such file or directory")),
    ),
    ("c_nonzero_exit", FakeRunner(result=completed("", returncode=9))),
    ("d_broken_csv", FakeRunner(result=completed("これはCSVではない\n"))),
    (
        "e_timeout",
        FakeRunner(
            error=subprocess.TimeoutExpired(
                cmd=list(DEFAULT_NVIDIA_SMI_COMMAND), timeout=DEFAULT_TIMEOUT_S
            )
        ),
    ),
)


@pytest.mark.parametrize(
    ("label", "runner"),
    FAILURE_CASES,
    ids=[label for label, _ in FAILURE_CASES],
)
def test_failures_return_none_without_raising(label: str, runner: FakeRunner) -> None:
    """(b)-(e) いずれも例外が外に出ず ``None`` になる。"""
    assert probe_with(runner).read() is None


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "\n",
        "NVIDIA GeForce RTX 5070 Ti, 596\n",
        "NVIDIA GeForce RTX 5070 Ti, 596, 16376, 42\n",
        "NVIDIA GeForce RTX 5070 Ti, N/A, 16376\n",
        ", 596, 16376\n",
        "[Insufficient Permissions]\n",
    ],
    ids=[
        "empty",
        "blank_line",
        "too_few_columns",
        "too_many_columns",
        "non_numeric_memory",
        "empty_name",
        "driver_message",
    ],
)
def test_unparsable_output_returns_none(stdout: str) -> None:
    assert probe_with(FakeRunner(result=completed(stdout))).read() is None


# --------------------------------------------------------------------------
# 記録する内容とログ
# --------------------------------------------------------------------------


def test_the_command_never_asks_for_the_process_list() -> None:
    """記録するのは GPU 名と used/total のみ。実行中プロセスの一覧は取らない。

    ``--query-compute-apps`` はユーザー名・コマンドラインを返すため、比較結果に
    第三者の情報が混入する経路になる (CLAUDE.md のログ出力ルール)。
    """
    joined = " ".join(DEFAULT_NVIDIA_SMI_COMMAND)

    assert "--query-compute-apps" not in joined
    assert "--query-gpu=name,memory.used,memory.total" in DEFAULT_NVIDIA_SMI_COMMAND
    assert pytest.approx(5.0) == DEFAULT_TIMEOUT_S


def test_the_command_and_timeout_are_injectable() -> None:
    runner = FakeRunner(result=completed(VALID_CSV))
    probe = NvidiaSmiProbe(
        command=("fake-smi", "--query-gpu=name,memory.used,memory.total"),
        runner=runner,
        timeout_s=0.5,
    )

    assert probe.read() is not None
    assert runner.calls == [
        (("fake-smi", "--query-gpu=name,memory.used,memory.total"), 0.5)
    ]


def test_repeated_failures_warn_only_once(caplog: pytest.LogCaptureFixture) -> None:
    """モデルとプロンプトの組ごとに読むため、警告は 1 実行につき 1 回だけにする。"""
    probe = probe_with(FakeRunner(error=FileNotFoundError()))

    with caplog.at_level(logging.DEBUG, logger="harness"):
        for _ in range(5):
            assert probe.read() is None

    warnings = [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    debugs = [record for record in caplog.records if record.levelno == logging.DEBUG]

    assert len(warnings) == 1
    assert len(debugs) == 5


def test_failure_logs_do_not_leak_the_raw_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """パース失敗時に生出力を記録しない (ドライバのメッセージ経由の情報混入防止)。"""
    secret_looking = "Unable to determine the device handle for GPU /home/someone: x"

    with caplog.at_level(logging.DEBUG, logger="harness"):
        assert probe_with(FakeRunner(result=completed(secret_looking))).read() is None

    assert secret_looking not in caplog.text
    assert "/home/someone" not in caplog.text


def test_first_gpu_is_used_when_several_are_present() -> None:
    runner = FakeRunner(result=completed("GPU Zero, 100, 16376\nGPU One, 200, 24564\n"))

    memory = probe_with(runner).read()

    assert memory == GpuMemory(name="GPU Zero", used_mib=100, total_mib=16376)


def test_nvidia_smi_probe_satisfies_the_vram_probe_protocol() -> None:
    probe: VramProbe = probe_with(FakeRunner(result=completed(VALID_CSV)))

    assert probe.read() is not None


# --------------------------------------------------------------------------
# 隔離 (D-01 を L2 側で維持するための境界)
# --------------------------------------------------------------------------

#: 「GPU に触る」ことを表す import 先。llmkit にはこのいずれも現れてはならない。
_FORBIDDEN_IMPORTS = frozenset({"subprocess", "GPUtil", "pynvml", "nvidia_smi"})

#: コード中の文字列リテラルに現れる外部コマンド名。
_FORBIDDEN_LITERAL = re.compile(r"nvidia-smi|nvidia_smi")

#: harness 側は「gpu.py の中だけ」を素の全文一致で見る (仕様書 T2 受け入れ基準)。
_GPU_ACCESS_PATTERN = re.compile(r"nvidia-smi|subprocess|GPUtil|pynvml")


def docstring_constants(tree: ast.Module) -> set[int]:
    """モジュール・クラス・関数の docstring ノードの id 集合を返す。

    ``llmkit/vram.py`` と ``llmkit/catalog.py`` の docstring には「nvidia-smi を
    **参照しない**」という D-01 の説明が元から書かれている。全文一致で検査すると
    この説明文自体が違反として検出されてしまう (逆に、コメントに書いた違反は
    検出されない)。したがって llmkit 側は AST を見て、**実行されるコード**に
    GPU への経路が無いことを測る。
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def gpu_access_offences(source: str) -> list[str]:
    """ソース中の「GPU / 外部プロセスへの経路」を列挙する (docstring は除く)。"""
    tree = ast.parse(source)
    docstrings = docstring_constants(tree)
    offences: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offences += [
                f"import {alias.name}"
                for alias in node.names
                if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS
            ]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.split(".")[0] in _FORBIDDEN_IMPORTS:
                offences.append(f"from {node.module} import ...")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and _FORBIDDEN_LITERAL.search(node.value)
        ):
            offences.append(f"literal {node.value!r}")
    return offences


def test_llmkit_never_touches_the_gpu_or_spawns_processes() -> None:
    """D-01: ``llmkit/`` の実行コードに GPU / 外部プロセスへの経路が 1 つも無い。

    ``harness/gpu.py`` が実測を担うようになっても、見積り側 (L2) は静的テーブル
    だけを出典にし続ける。
    """
    offenders = {
        module_path.name: gpu_access_offences(module_path.read_text(encoding="utf-8"))
        for module_path in sorted(LLMKIT_DIR.glob("*.py"))
    }
    violations = {name: found for name, found in offenders.items() if found}

    assert not violations, f"llmkit が GPU / 外部プロセスに触れている: {violations}"


def test_gpu_access_is_isolated_to_the_harness_gpu_module() -> None:
    """``harness/`` 側で GPU に触るのは gpu.py だけ (全文一致で検査)。"""
    offenders = [
        module_path.name
        for module_path in sorted(HARNESS_DIR.glob("*.py"))
        if module_path.name != "gpu.py"
        and _GPU_ACCESS_PATTERN.search(module_path.read_text(encoding="utf-8"))
    ]

    assert not offenders, f"gpu.py 以外が GPU / 外部プロセスに触れている: {offenders}"


def test_the_llmkit_guard_detects_a_real_violation() -> None:
    """上の検査が「何も検出しないだけ」でないことを確かめる (変異検証)。"""
    assert gpu_access_offences("import subprocess\n") == ["import subprocess"]
    assert gpu_access_offences('CMD = "nvidia-smi"\n') == ["literal 'nvidia-smi'"]
    assert gpu_access_offences('"""nvidia-smi を参照しない。"""\n') == []


# --------------------------------------------------------------------------
# E22: 実測値がレポートまで届く (D-23 guard)
# --------------------------------------------------------------------------

PROBE_SUITE = """\
[suite]
id = "probe_test"
description = "VRAM 実測プローブの掃引用スイート"
warmup_runs = 1

[[models]]
model_id = "qwen3-8b"
profile = "lightweight"

[[prompts]]
id = "summarize"
text = "次の文章を三行で要約してください。"
"""

VRAM_INCREMENT_COLUMN = "VRAM 実測増分 GiB"


class ScriptedProbe:
    """決められた値を順に返す :class:`VramProbe` (実プロセスを起動しない)。"""

    def __init__(self, readings: Sequence[GpuMemory | None]) -> None:
        self._readings = list(readings)
        self.call_count = 0

    def read(self) -> GpuMemory | None:
        self.call_count += 1
        index = min(self.call_count - 1, len(self._readings) - 1)
        return self._readings[index]


def run_with_probe(
    tmp_path: Path, probe: ScriptedProbe
) -> tuple[str, list[int | None]]:
    """1 モデル 1 プロンプトを実行し、レポート本文と ``vram_used_mib`` を返す。"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    suite_path = tmp_path / "suite.toml"
    suite_path.write_text(PROBE_SUITE, encoding="utf-8")
    plan = plan_run(
        load_suite(suite_path), suite_path, load_config(DEFAULT_CONFIG), DEFAULT_CONFIG
    )
    transport = RecordingTransport()
    with transport.client() as http_client:
        result = run_suite(
            plan,
            probe=probe,
            http_client=http_client,
            results_dir=tmp_path / "out",
        )
    return render_report(result), [record.vram_used_mib for record in result.records]


def increment_cell(report: str) -> str:
    """比較表から「VRAM 実測増分 GiB」のセルを取り出す。"""
    lines = report.splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.startswith("| model_id |")
    )
    header = [cell.strip() for cell in lines[start].strip("|").split("|")]
    row = [cell.strip() for cell in lines[start + 2].strip("|").split("|")]
    return row[header.index(VRAM_INCREMENT_COLUMN)]


def test_vram_probe_result_is_visible_in_the_report(tmp_path: Path) -> None:
    """★ E22 / D-23 guard: プローブの戻り値がレポートと記録を動かす。

    プローブを差し替えると (a) ``vram_used_mib`` (b) 比較表の実測列 が変わり、
    読めない環境 (``None``) では ``—`` と ``null`` になる。実測列が固定値や 0 に
    なっている実装ではここが落ちる。
    """
    idle = GpuMemory(name="gpu", used_mib=600, total_mib=16384)

    low_report, low_used = run_with_probe(
        tmp_path / "low",
        ScriptedProbe([idle, GpuMemory(name="gpu", used_mib=5720, total_mib=16384)]),
    )
    high_report, high_used = run_with_probe(
        tmp_path / "high",
        ScriptedProbe([idle, GpuMemory(name="gpu", used_mib=13400, total_mib=16384)]),
    )
    missing_report, missing_used = run_with_probe(
        tmp_path / "missing", ScriptedProbe([None])
    )

    assert set(low_used) == {5720}
    assert set(high_used) == {13400}
    assert set(missing_used) == {None}

    assert increment_cell(low_report) == f"{(5720 - 600) / 1024:.2f}"
    assert increment_cell(high_report) == f"{(13400 - 600) / 1024:.2f}"
    assert increment_cell(low_report) != increment_cell(high_report)
    assert increment_cell(missing_report) == MISSING_CELL
    assert increment_cell(missing_report) != "0.00"
