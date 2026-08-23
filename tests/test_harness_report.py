"""比較結果の書き出し (harness/report.py) のテスト。

このファイルが固定する性質は 3 つある。

1. **欠測セルは ``—`` で、0 が現れない** (D-21)。速度が 1 件も測れなかった
   モデルの行に ``0.0`` や ``0`` を書くと、「実測 0 トークン/秒」と「未計測」が
   表の上で区別できなくなる。
2. **``run.json`` だけでフィンガープリントを再計算でき、記録値と一致する**
   (D-20)。レポートが自分自身を検証できる状態を保つ。
3. **``records.jsonl`` の各行のキー集合が ``RECORD_KEYS`` と完全一致する**。
   Markdown は先頭 400 文字しか持たないため、全文の唯一の出典が JSONL になる。

実 HTTP は 1 バイトも発行しない (D-02)。書き出し先は必ず ``tmp_path`` で、
リポジトリの ``results/`` には 1 バイトも書かない。
"""

from __future__ import annotations

import json
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

from harness.gpu import GpuMemory
from harness.records import RECORD_KEYS
from harness.report import (
    COMPARISON_TABLE_HEADER,
    MISSING_CELL,
    RECORDS_FILENAME,
    REPORT_FILENAME,
    RESPONSE_PREVIEW_CHARS,
    RUN_JSON_FILENAME,
    build_run_json,
    render_records_jsonl,
    render_report,
    run_output_dir,
    write_run_outputs,
)
from harness.runner import (
    MANIFESTS_DIRNAME,
    RunResult,
    fingerprint_digest,
    plan_run,
    run_suite,
)
from harness.suite import load_suite
from llmkit import ConfigError, load_config

SUITE_TEXT = """\
[suite]
id = "report_test"
description = "レポート書き出しのテスト用スイート"
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
system = "あなたは日本語の要約アシスタントです。"
tags = ["要約"]

[[prompts]]
id = "translate"
text = "次の文を英語に訳してください。"
"""

MODEL_IDS = ["qwen3-8b", "gpt-oss-20b"]
PROMPT_COUNT = 2
FIXED_MOMENT = datetime(2026, 8, 23, 12, 34, 56, tzinfo=UTC)


def execute(
    tmp_path: Path,
    *,
    handler: Handler | None = None,
    probe: FakeProbe | None = None,
    results_dir: Path | None = None,
) -> RunResult:
    """スイートを 1 回実行して :class:`RunResult` を返す (HTTP はモック)。"""
    suite_path = tmp_path / "suite.toml"
    suite_path.write_text(SUITE_TEXT, encoding="utf-8")
    plan = plan_run(
        load_suite(suite_path), suite_path, load_config(DEFAULT_CONFIG), DEFAULT_CONFIG
    )
    transport = RecordingTransport(handler)
    with transport.client() as http_client:
        return run_suite(
            plan,
            probe=probe if probe is not None else FakeProbe(),
            http_client=http_client,
            results_dir=results_dir if results_dir is not None else tmp_path / "out",
            clock=lambda: FIXED_MOMENT,
        )


def unmeasurable_handler() -> Handler:
    """速度を 1 つも算出できない応答 (生成トークン 0)。"""

    def handle(request: httpx.Request) -> httpx.Response:
        payload = dict(NATIVE_SUCCESS_PAYLOAD)
        payload["eval_count"] = 0
        payload["eval_duration"] = 0
        payload["prompt_eval_count"] = 0
        payload["prompt_eval_duration"] = 0
        return httpx.Response(200, json=payload)

    return handle


def responding_with(text: str) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        payload = dict(NATIVE_SUCCESS_PAYLOAD)
        payload["message"] = {"role": "assistant", "content": text}
        return httpx.Response(200, json=payload)

    return handle


def not_found_handler() -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    return handle


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


def load_json_object(path: Path) -> dict[str, object]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


# --------------------------------------------------------------------------
# モデル比較表 (受け入れ条件2)
# --------------------------------------------------------------------------


def test_the_table_header_names_the_three_required_metrics(tmp_path: Path) -> None:
    """★ 受け入れ条件2: ヘッダ行に 3 つの指標名が現れる。"""
    report = render_report(execute(tmp_path))

    header_line = next(
        line for line in report.splitlines() if line.startswith("| model_id |")
    )

    for required in ("生成速度", "プロンプト処理速度", "設定コンテキスト長"):
        assert required in header_line, required


def test_the_table_has_one_row_per_model_in_declaration_order(tmp_path: Path) -> None:
    report = render_report(execute(tmp_path))

    header, rows = comparison_table(report)

    assert header == list(COMPARISON_TABLE_HEADER)
    assert len(rows) == len(MODEL_IDS)
    assert [cell(header, row, "model_id") for row in rows] == MODEL_IDS


def test_measured_speeds_appear_in_the_table(tmp_path: Path) -> None:
    """前提: 測れているときは数値が入る (欠測テストが恒真でないことの確認)。"""
    header, rows = comparison_table(render_report(execute(tmp_path)))

    for row in rows:
        assert cell(header, row, "生成速度 t/s (中央値)") != MISSING_CELL
        assert cell(header, row, "速度出典") == "eval"
        assert cell(header, row, "測定数 n/N") == f"{PROMPT_COUNT}/{PROMPT_COUNT}"


def test_the_context_length_column_shows_the_effective_value(tmp_path: Path) -> None:
    """E18 の表側: 実効コンテキスト長が「設定コンテキスト長」列に出る。"""
    header, rows = comparison_table(render_report(execute(tmp_path)))

    assert {cell(header, row, "設定コンテキスト長") for row in rows} == {"16384"}


# --------------------------------------------------------------------------
# 欠測 (D-21): — であり 0 ではない
# --------------------------------------------------------------------------

_MISSING_COLUMNS = (
    "生成速度 t/s (中央値)",
    "速度出典",
    "プロンプト処理速度 t/s (中央値)",
    "壁時計速度 t/s (中央値)",
    "VRAM 実測増分 GiB",
)


def test_unmeasurable_cells_are_em_dashes_and_never_zero(tmp_path: Path) -> None:
    """★ D-21 guard (レポート側): 欠測セルに 0 / 0.0 を書かない。

    「レポート全体に ``0`` が無いこと」は検査できない。``測定数 n/N`` は
    ``0/2`` が正しい表示であり、``10.06`` のような正当な数値も部分文字列
    ``0.0`` を含む。したがって**欠測セルそのもの**を見る。
    """
    result = execute(tmp_path, handler=unmeasurable_handler())

    header, rows = comparison_table(render_report(result))

    for row in rows:
        for column in _MISSING_COLUMNS:
            assert cell(header, row, column) == MISSING_CELL, column
        assert cell(header, row, "測定数 n/N") == f"0/{PROMPT_COUNT}"
        assert "0.0" not in "|".join(
            cell(header, row, column) for column in _MISSING_COLUMNS
        )


def test_unmeasurable_speeds_stay_null_in_the_run_json(tmp_path: Path) -> None:
    result = execute(tmp_path, handler=unmeasurable_handler())

    payload = build_run_json(result, tmp_path / "out")
    models = payload["models"]
    assert isinstance(models, list)

    for model in models:
        assert isinstance(model, dict)
        assert model["generation_tokens_per_second"] == {
            "median": None,
            "n": 0,
            "total": PROMPT_COUNT,
        }
        assert model["generation_tokens_per_second_source"] is None
        assert model["vram_used_mib"] is None
        assert model["vram_increment_gib"] is None


# --------------------------------------------------------------------------
# records.jsonl
# --------------------------------------------------------------------------


def test_every_jsonl_line_has_exactly_the_record_schema_keys(tmp_path: Path) -> None:
    """★ 受け入れ条件: 各行のキー集合が ``RECORD_KEYS`` と完全一致する。"""
    result = execute(tmp_path)

    lines = render_records_jsonl(result.records).splitlines()

    assert len(lines) == len(result.records)
    assert len(result.records) == len(MODEL_IDS) * (1 + PROMPT_COUNT)
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        assert set(parsed) == set(RECORD_KEYS)


def test_the_jsonl_is_ordered_by_sequence_index_and_keeps_the_warmups(
    tmp_path: Path,
) -> None:
    result = execute(tmp_path)

    rows = [
        json.loads(line) for line in render_records_jsonl(result.records).splitlines()
    ]

    assert [row["sequence_index"] for row in rows] == list(range(len(rows)))
    assert [row["phase"] for row in rows].count("warmup") == len(MODEL_IDS)


def test_the_full_response_lives_in_the_jsonl_not_in_the_markdown(
    tmp_path: Path,
) -> None:
    """Markdown は先頭 400 文字だけ。全文の出典は JSONL 1 か所に保つ。"""
    long_text = "応答" * 600
    result = execute(tmp_path, handler=responding_with(long_text))

    report = render_report(result)
    jsonl = render_records_jsonl(result.records)

    assert long_text[:RESPONSE_PREVIEW_CHARS] in report
    assert long_text[: RESPONSE_PREVIEW_CHARS + 1] not in report
    assert str(RESPONSE_PREVIEW_CHARS) in report, "先頭何文字かを明示している"
    assert all(
        json.loads(line)["response_text"] == long_text for line in jsonl.splitlines()
    )


def test_short_responses_are_not_truncated(tmp_path: Path) -> None:
    result = execute(tmp_path, handler=responding_with("短い応答"))

    assert "短い応答" in render_report(result)


# --------------------------------------------------------------------------
# run.json (D-20)
# --------------------------------------------------------------------------


def test_the_run_json_recomputes_its_own_fingerprint(tmp_path: Path) -> None:
    """★ 受け入れ条件: ``run.json`` から再計算した値が記録値と一致する。"""
    result = execute(tmp_path)

    paths = write_run_outputs(result, tmp_path / "out")
    payload = load_json_object(paths.run_json)
    reproduction = payload["reproduction"]
    assert isinstance(reproduction, dict)

    assert payload["run_fingerprint"] == result.fingerprint
    assert fingerprint_digest(reproduction) == result.fingerprint


def test_the_run_json_survives_a_json_round_trip(tmp_path: Path) -> None:
    """ファイル経由 (str 化) でも再計算が一致する = 桁落ちしていない。"""
    result = execute(tmp_path)
    paths = write_run_outputs(result, tmp_path / "out")

    reloaded = load_json_object(paths.run_json)
    again = json.loads(json.dumps(reloaded, ensure_ascii=False))
    reproduction = again["reproduction"]

    assert fingerprint_digest(reproduction) == again["run_fingerprint"]


def test_the_run_json_carries_the_aggregates(tmp_path: Path) -> None:
    result = execute(
        tmp_path,
        probe=FakeProbe(
            [
                GpuMemory(name="gpu", used_mib=600, total_mib=16384),
                GpuMemory(name="gpu", used_mib=5720, total_mib=16384),
                GpuMemory(name="gpu", used_mib=13500, total_mib=16384),
            ]
        ),
    )

    payload = build_run_json(result, tmp_path / "out")
    models = payload["models"]
    assert isinstance(models, list)
    counts = payload["counts"]

    assert [model["model_id"] for model in models] == MODEL_IDS
    assert models[0]["profile_name"] == "lightweight"
    assert models[0]["vram_used_mib"] == 5720
    assert models[0]["succeeded"] == PROMPT_COUNT
    assert counts == {
        "records": len(result.records),
        "warmup": len(MODEL_IDS),
        "measure": len(MODEL_IDS) * PROMPT_COUNT,
        "error": 0,
    }


def test_the_run_json_records_the_shared_idle_baseline(tmp_path: Path) -> None:
    """★ アイドル基準は ``run.json`` に 1 個だけ記録され、全モデル共通。"""
    result = execute(
        tmp_path,
        probe=FakeProbe(
            [
                GpuMemory(name="gpu", used_mib=844, total_mib=16384),
                GpuMemory(name="gpu", used_mib=12246, total_mib=16384),
                GpuMemory(name="gpu", used_mib=13650, total_mib=16384),
            ]
        ),
    )

    payload = build_run_json(result, tmp_path / "out")

    assert payload["vram_idle_mib"] == 844
    models = payload["models"]
    assert isinstance(models, list)
    assert models[0]["vram_increment_gib"] == pytest.approx((12246 - 844) / 1024)
    assert models[1]["vram_increment_gib"] == pytest.approx((13650 - 844) / 1024)


def test_the_run_json_idle_is_null_when_no_gpu_probe_is_available(
    tmp_path: Path,
) -> None:
    """CI (nvidia-smi 無し) でも ``vram_idle_mib`` は null のまま完走する。"""
    result = execute(tmp_path)

    payload = build_run_json(result, tmp_path / "out")

    assert payload["vram_idle_mib"] is None


def test_the_run_json_never_contains_an_api_key_value(tmp_path: Path) -> None:
    """秘密は書かない。環境変数「名」だけが残る (D-05)。"""
    result = execute(tmp_path)

    paths = write_run_outputs(result, tmp_path / "out")
    text = paths.run_json.read_text(encoding="utf-8")

    assert '"api_key_env"' in text
    assert '"api_key"' not in text
    assert "Bearer" not in text


# --------------------------------------------------------------------------
# 書き出し先
# --------------------------------------------------------------------------


def test_the_output_directory_is_named_by_the_timestamp_and_fingerprint(
    tmp_path: Path,
) -> None:
    result = execute(tmp_path)

    directory = run_output_dir(
        tmp_path / "results",
        suite_id=result.plan.suite_id,
        moment=FIXED_MOMENT,
        fingerprint=result.fingerprint,
    )

    assert directory.parent.name == "report_test"
    assert directory.name == f"20260823T123456Z-{result.fingerprint[:12]}"


def test_writing_produces_the_three_files_next_to_the_manifests(
    tmp_path: Path,
) -> None:
    result = execute(tmp_path)
    output_dir = tmp_path / "out"

    paths = write_run_outputs(result, output_dir)

    assert paths.report == output_dir / REPORT_FILENAME
    assert paths.records == output_dir / RECORDS_FILENAME
    assert paths.run_json == output_dir / RUN_JSON_FILENAME
    assert all(path.is_file() for path in (paths.report, paths.records, paths.run_json))
    assert len(list((output_dir / MANIFESTS_DIRNAME).glob("*.json"))) == len(MODEL_IDS)


def test_the_report_records_the_reproduction_conditions(tmp_path: Path) -> None:
    result = execute(tmp_path)

    report = render_report(result)

    assert result.fingerprint in report
    assert result.run_id in report
    assert result.plan.suite_sha256 in report
    assert result.plan.config_sha256 in report
    assert result.started_at_utc in report


# --------------------------------------------------------------------------
# 失敗したケース
# --------------------------------------------------------------------------


def test_failures_are_shown_as_failures_not_as_empty_responses(
    tmp_path: Path,
) -> None:
    result = execute(tmp_path, handler=not_found_handler())

    report = render_report(result)
    header, rows = comparison_table(report)

    assert "ModelNotFoundError" in report
    for row in rows:
        assert cell(header, row, "生成速度 t/s (中央値)") == MISSING_CELL
        assert cell(header, row, "測定数 n/N") == f"0/{PROMPT_COUNT}"


def test_response_sections_cover_every_prompt_and_every_model(tmp_path: Path) -> None:
    result = execute(tmp_path)

    report = render_report(result)

    for prompt_id in result.plan.suite.prompt_ids:
        assert f"### {prompt_id}" in report
    assert report.count("#### qwen3-8b") == PROMPT_COUNT
    assert report.count("#### gpt-oss-20b") == PROMPT_COUNT


def test_the_prompt_text_and_system_message_are_shown(tmp_path: Path) -> None:
    result = execute(tmp_path)

    report = render_report(result)

    assert "次の文章を三行で要約してください。" in report
    assert "あなたは日本語の要約アシスタントです。" in report
    assert "タグ: 要約" in report


def test_a_pipe_in_the_response_does_not_break_the_table(tmp_path: Path) -> None:
    """応答本文は表のセルに入れない (入れると列がずれる)。"""
    result = execute(tmp_path, handler=responding_with("a | b | c"))

    header, rows = comparison_table(render_report(result))

    assert len(rows) == len(MODEL_IDS)
    assert all(len(row) == len(header) for row in rows)


def test_writing_into_an_unwritable_location_raises_config_error(
    tmp_path: Path,
) -> None:
    """出力先に書けない場合も llmkit の例外階層の中で失敗する。"""
    blocker = tmp_path / "blocked"
    blocker.write_text("ファイルなのでディレクトリを作れない", encoding="utf-8")
    # results_dir を書けない場所に固定しておく (write_run_outputs は省略時
    # result.results_dir を使うため、ここで output_dir を渡す必要が無い)。
    result = execute(tmp_path, results_dir=blocker / "out")

    with pytest.raises(ConfigError, match="比較結果を書き出せません"):
        write_run_outputs(result)


def test_write_run_outputs_rejects_a_mismatched_output_dir(tmp_path: Path) -> None:
    """★ F-8-001 guard: output_dir と result.results_dir が食い違うと ConfigError。

    以前は :func:`harness.report._relative_to` の絶対パス fallback が黙って
    ホストの絶対パスを ``run.json`` に書いていた。値の出所を 1 つにし、
    食い違いを早期に検出する。
    """
    result = execute(tmp_path)

    with pytest.raises(ConfigError, match="results_dir と一致しません"):
        write_run_outputs(result, tmp_path / "somewhere-else")

    with pytest.raises(ConfigError, match="results_dir と一致しません"):
        build_run_json(result, tmp_path / "somewhere-else")
