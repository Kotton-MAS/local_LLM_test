"""要件書 Phase 2 受け入れ条件の機械検証。

``docs/localllmrequirements.md`` L302-L305 の 4 条件と、このモジュールの 4 つの
テスト関数を **1 対 1** に対応させる。各 docstring の先頭に対応行を書き、
各テストは自分が対応する行が受け入れ条件のままであること (``- [`` で始まる)
を最初に確かめる。条件が増えたらテストも増やす。

L302-L304 は ``httpx.MockTransport`` で決定論的に測り、書き出し先は必ず
``tmp_path`` に注入する (リポジトリの ``results/`` へは 1 バイトも書かない)。
L305 だけは**コミット済み成果物の静的検査**であり、実行を伴わない。

実 HTTP は 1 バイトも発行せず (D-02)、実プロセスも起動しない (D-23)。
"""

from __future__ import annotations

import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
from conftest import (
    DEFAULT_CONFIG,
    NATIVE_SUCCESS_PAYLOAD,
    FakeProbe,
    Handler,
    RecordingTransport,
)

from harness.cli import DEFAULT_SUITE_PATH, EXIT_OK, main
from harness.gpu import GpuMemory
from harness.records import RECORD_KEYS
from harness.report import (
    MISSING_CELL,
    RECORDS_FILENAME,
    REPORT_FILENAME,
    RUN_JSON_FILENAME,
)
from harness.runner import MANIFESTS_DIRNAME

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "docs" / "localllmrequirements.md"
RESULTS_ROOT = REPO_ROOT / "results"
SHIPPED_SUITE = REPO_ROOT / DEFAULT_SUITE_PATH

#: 要件書の行番号 -> 対応するテスト関数名。
ACCEPTANCE_MAP = {
    302: "test_l302_suite_and_model_list_produce_a_comparison_file",
    303: "test_l303_output_contains_speeds_and_context_length",
    304: "test_l304_rerunning_the_same_input_reproduces_the_conditions",
    305: "test_l305_japanese_comparison_results_are_committed",
}

#: 出荷スイート (承認済みの Q-1 / Q-3) の前提。
SHIPPED_MODEL_IDS = ["qwen3-14b", "gpt-oss-20b", "qwen3-8b"]
SHIPPED_PROMPT_COUNT = 8

#: 受け入れ条件2 が名指ししている 3 指標の列名。
SPEED_COLUMN = "生成速度 t/s (中央値)"
PROMPT_SPEED_COLUMN = "プロンプト処理速度 t/s (中央値)"
CONTEXT_COLUMN = "設定コンテキスト長"

FIRST_MOMENT = datetime(2026, 8, 23, 12, 34, 56, tzinfo=UTC)
SECOND_MOMENT = datetime(2026, 8, 23, 12, 39, 7, tzinfo=UTC)

#: ひらがな・カタカナ・CJK 統合漢字。日本語プロンプトの判定に使う。
CJK_PATTERN = re.compile("[\u3040-\u30ff\u4e00-\u9fff]")


def requirement_line(number: int) -> str:
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    return lines[number - 1]


def assert_is_acceptance_line(number: int) -> None:
    """参照している行が受け入れ条件のままであること。

    ``- [x]`` (完了) でも落ちないよう、チェック状態ではなく**チェックボックス
    そのもの**を見る (Phase 2 の完了と同時にこのテストが落ちるのは無意味)。
    """
    assert requirement_line(number).startswith("- ["), number


# --------------------------------------------------------------------------
# 実行ヘルパ (実 HTTP なし・実プロセスなし・results/ へ書かない)
# --------------------------------------------------------------------------


DEFAULT_READINGS = (
    GpuMemory(name="fake-gpu", used_mib=847, total_mib=16376),
    GpuMemory(name="fake-gpu", used_mib=12245, total_mib=16376),
    GpuMemory(name="fake-gpu", used_mib=13650, total_mib=16376),
    GpuMemory(name="fake-gpu", used_mib=8280, total_mib=16376),
)


class Invocation:
    """CLI を 1 回呼んだ結果。"""

    def __init__(self, code: int, stdout: str, transport: RecordingTransport) -> None:
        self.code = code
        self.stdout = stdout
        self.transport = transport

    @property
    def http_calls(self) -> int:
        return self.transport.call_count

    @property
    def directory(self) -> Path:
        line = next(
            line for line in self.stdout.splitlines() if line.startswith("出力先")
        )
        return Path(line.split(":", 1)[1].strip())


def run_cli(
    results_root: Path,
    *,
    handler: Handler | None = None,
    moment: datetime = FIRST_MOMENT,
) -> Invocation:
    """出荷スイートと設定ファイルを与えて ``harness.cli`` を 1 回回す。

    「プロンプト集と対象モデルリストを与える」経路そのもの (CLI) を通す。
    入力は実際に出荷する ``suites/ja_basic.toml`` / ``configs/default.toml``。
    """
    transport = RecordingTransport(handler)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with transport.client() as http_client:
        code = main(
            [
                "run",
                "--suite",
                str(SHIPPED_SUITE),
                "--config",
                str(DEFAULT_CONFIG),
            ],
            http_client=http_client,
            probe=FakeProbe(DEFAULT_READINGS),
            results_root=results_root,
            clock=lambda: moment,
            stdout=stdout,
            stderr=stderr,
        )
    return Invocation(code, stdout.getvalue(), transport)


def varying_handler(offset: int = 0) -> Handler:
    """呼ばれるたびに応答本文・トークン数・所要時間が変わるハンドラ。

    「再現される」の検査が恒真にならないようにするために使う (応答と速度が
    毎回同じなら、フィンガープリントが何を入力にしていても一致してしまう)。
    ``offset`` は実行と実行の間で応答をずらすためのもの。
    """
    counter = {"calls": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        index = counter["calls"] + offset
        payload = dict(NATIVE_SUCCESS_PAYLOAD)
        payload["message"] = {"role": "assistant", "content": f"応答 {index} 回目"}
        payload["eval_count"] = 20 + index
        payload["eval_duration"] = 900_000_000 + index * 3_000_000
        payload["total_duration"] = 1_500_000_000 + index * 5_000_000
        return httpx.Response(200, json=payload)

    return handle


# --------------------------------------------------------------------------
# 成果物の読み出し
# --------------------------------------------------------------------------


def load_json_object(path: Path) -> dict[str, object]:
    parsed: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def load_records(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    text = (directory / RECORDS_FILENAME).read_text(encoding="utf-8")
    for line in text.splitlines():
        parsed: object = json.loads(line)
        assert isinstance(parsed, dict)
        rows.append(parsed)
    return rows


def comparison_table(report: str) -> tuple[list[str], list[list[str]]]:
    """モデル比較表のヘッダ行と本体行を取り出す。"""
    lines = report.splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.startswith("| model_id |")
    )
    header = [cell.strip() for cell in lines[start].strip("|").split("|")]
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return header, rows


def cell(header: list[str], row: list[str], column: str) -> str:
    return row[header.index(column)]


def prompt_blocks(report: str) -> list[str]:
    """``report.md`` の「プロンプト:」直後の引用ブロックを取り出す。"""
    blocks: list[str] = []
    lines = report.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "プロンプト:":
            continue
        collected: list[str] = []
        for candidate in lines[index + 1 :]:
            if candidate.startswith(">"):
                collected.append(candidate.lstrip("> "))
            elif collected:
                break
        blocks.append("\n".join(collected))
    return blocks


def committed_run_directories() -> list[Path]:
    """``results/<suite_id>/<ts>-<fp12>/`` のうち 3 ファイルが揃ったもの。"""
    return sorted(
        path.parent
        for path in RESULTS_ROOT.glob(f"*/*/{RUN_JSON_FILENAME}")
        if (path.parent / REPORT_FILENAME).is_file()
        and (path.parent / RECORDS_FILENAME).is_file()
    )


# --------------------------------------------------------------------------
# 受け入れ条件
# --------------------------------------------------------------------------


def test_l302_suite_and_model_list_produce_a_comparison_file(tmp_path: Path) -> None:
    """要件書 L302: プロンプト集と対象モデルリストから比較結果ファイルが生成される。

    入力は出荷スイート (8 問 * 3 モデル) と ``configs/default.toml``。出力の
    3 ファイル (``report.md`` / ``records.jsonl`` / ``run.json``) とモデルごとの
    実行マニフェストが 1 回の実行で揃うことを見る。
    """
    assert_is_acceptance_line(302)

    invocation = run_cli(tmp_path / "results")

    assert invocation.code == EXIT_OK
    directory = invocation.directory
    assert directory.parent.name == "ja_basic"
    for filename in (REPORT_FILENAME, RECORDS_FILENAME, RUN_JSON_FILENAME):
        assert (directory / filename).is_file(), filename
    manifests = sorted((directory / MANIFESTS_DIRNAME).glob("*.json"))
    assert len(manifests) == len(SHIPPED_MODEL_IDS)

    # 入力 (プロンプト集とモデルリスト) が結果ファイルに漏れなく現れている。
    records = load_records(directory)
    measured = [row for row in records if row["phase"] == "measure"]
    assert [row["model_id"] for row in records if row["phase"] == "warmup"] == (
        SHIPPED_MODEL_IDS
    )
    assert len(measured) == len(SHIPPED_MODEL_IDS) * SHIPPED_PROMPT_COUNT
    assert all(set(row) == set(RECORD_KEYS) for row in records)
    assert invocation.http_calls == len(records)


def test_l303_output_contains_speeds_and_context_length(tmp_path: Path) -> None:
    """要件書 L303: 生成速度・プロンプト処理速度・設定コンテキスト長が出力に含まれる。

    列が存在するだけでは足りない (空欄でも「含まれる」と言えてしまう) ため、
    3 モデルすべての行で値が入っていること・欠測記号 ``—`` でないことまで見る。
    """
    assert_is_acceptance_line(303)

    invocation = run_cli(tmp_path / "results")
    report = (invocation.directory / REPORT_FILENAME).read_text(encoding="utf-8")

    header, rows = comparison_table(report)

    for required in (SPEED_COLUMN, PROMPT_SPEED_COLUMN, CONTEXT_COLUMN):
        assert required in header, required
    assert [cell(header, row, "model_id") for row in rows] == SHIPPED_MODEL_IDS

    for row in rows:
        generation = cell(header, row, SPEED_COLUMN)
        prompt_speed = cell(header, row, PROMPT_SPEED_COLUMN)
        context = cell(header, row, CONTEXT_COLUMN)
        assert generation != MISSING_CELL
        assert prompt_speed != MISSING_CELL
        assert float(generation) > 0.0
        assert float(prompt_speed) > 0.0
        assert int(context) == 16384
        assert cell(header, row, "速度出典") == "eval"
        assert cell(header, row, "測定数 n/N") == (
            f"{SHIPPED_PROMPT_COUNT}/{SHIPPED_PROMPT_COUNT}"
        )

    # 構造化された出典 (run.json) にも同じ 3 指標が残っている。
    payload = load_json_object(invocation.directory / RUN_JSON_FILENAME)
    models = payload["models"]
    assert isinstance(models, list)
    for model in models:
        assert isinstance(model, dict)
        generation_summary = model["generation_tokens_per_second"]
        prompt_summary = model["prompt_tokens_per_second"]
        assert isinstance(generation_summary, dict)
        assert isinstance(prompt_summary, dict)
        assert isinstance(generation_summary["median"], float)
        assert isinstance(prompt_summary["median"], float)
        assert model["context_tokens"] == 16384


def test_l304_rerunning_the_same_input_reproduces_the_conditions(
    tmp_path: Path,
) -> None:
    """要件書 L304: 同じ入力での再実行時、設定条件が完全に再現される。

    応答本文・トークン数・レイテンシ・実行時刻・``run_id`` はすべて 2 回で
    異なる状態にしたうえで、``run_fingerprint`` と ``run.json`` の再現条件
    セクション (``reproduction``) が完全一致することを見る (D-20)。
    """
    assert_is_acceptance_line(304)

    first = run_cli(
        tmp_path / "results", handler=varying_handler(), moment=FIRST_MOMENT
    )
    second = run_cli(
        tmp_path / "results", handler=varying_handler(offset=100), moment=SECOND_MOMENT
    )

    assert first.directory != second.directory
    first_json = load_json_object(first.directory / RUN_JSON_FILENAME)
    second_json = load_json_object(second.directory / RUN_JSON_FILENAME)

    # 再現されるもの
    assert first_json["run_fingerprint"] == second_json["run_fingerprint"]
    assert first_json["reproduction"] == second_json["reproduction"]
    assert first_json["suite"] == second_json["suite"]
    assert first_json["config"] == second_json["config"]
    fingerprint = str(first_json["run_fingerprint"])
    assert first.directory.name == f"20260823T123456Z-{fingerprint[:12]}"
    assert second.directory.name == f"20260823T123907Z-{fingerprint[:12]}"

    # 再現されないもの (この差があってなお上が一致していることが要点)
    assert first_json["run_id"] != second_json["run_id"]
    assert first_json["started_at_utc"] != second_json["started_at_utc"]
    first_records = load_records(first.directory)
    second_records = load_records(second.directory)
    assert [row["response_text"] for row in first_records] != (
        [row["response_text"] for row in second_records]
    )
    assert [row["latency_s"] for row in first_records] != (
        [row["latency_s"] for row in second_records]
    )


def test_l305_japanese_comparison_results_are_committed() -> None:
    """要件書 L305: 日本語プロンプトで 3 モデル以上を比較した結果が残っている。

    唯一、実行を伴わない静的検査。**コミット済みの成果物**を読み、3 モデル
    以上・日本語プロンプト・実測値 (速度と VRAM) が揃った実行が 1 つ以上
    あることを確かめる。成果物を消す / 実測列が ``null`` の実行だけになると落ちる。
    """
    assert_is_acceptance_line(305)

    directories = committed_run_directories()
    assert directories, "results/ にコミット済みの比較結果がありません"

    qualifying: list[Path] = []
    for directory in directories:
        records = load_records(directory)
        measured = [
            row for row in records if row["phase"] == "measure" and row["error"] is None
        ]
        model_ids = {row["model_id"] for row in measured}
        if len(model_ids) < 3:
            continue

        report = (directory / REPORT_FILENAME).read_text(encoding="utf-8")
        japanese_prompts = [
            block for block in prompt_blocks(report) if CJK_PATTERN.search(block)
        ]
        if not japanese_prompts:
            continue

        # 3 モデルそれぞれに、速度と VRAM の実測が入った計測レコードがある。
        complete = {
            str(row["model_id"])
            for row in measured
            if isinstance(row["generation_tokens_per_second"], float)
            and row["generation_tokens_per_second"] > 0.0
            and isinstance(row["vram_used_mib"], int)
            and row["vram_used_mib"] > 0
            and row["generation_tokens_per_second_source"] is not None
        }
        if len(complete) < 3:
            continue

        assert all(set(row) == set(RECORD_KEYS) for row in records)
        qualifying.append(directory)

    assert qualifying, (
        "3 モデル以上・日本語プロンプト・実測値つきの比較結果が "
        f"results/ にありません (走査した実行: {[str(d) for d in directories]})"
    )
