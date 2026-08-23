"""比較ランナー (harness/runner.py) のテスト。

このファイルの中心は 2 つある。

1. **``run_fingerprint`` は ``config_sha256`` では代替できない** (D-20)。
   ハーネスは設定を in-memory で上書きするため、``config_sha256`` はベース
   設定ファイルの同一性しか表さない。掃引テストは**スイートファイルも設定
   ファイルも 1 バイトも変えずに**実効値だけを振り、それでも
   ``run_fingerprint`` が変わることを固定する。ファイルを書き換えて振ると
   ハッシュが変わるのは当たり前なので、何も測れていないテストになる。
2. **モデル ID は 2 か所へ同時に届く** (D-19)。リクエストの ``model`` と、
   VRAM 見積り・実行マニフェストの ``profile.models[]`` の両方が変わること
   を 1 つのテストで押さえる。

実 HTTP は 1 バイトも発行しない (D-02)。``RecordingTransport`` の
``call_count`` がそのまま「HTTP を何回出したか」の証拠になる。
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import httpx
import pytest
from conftest import (
    DEFAULT_CONFIG,
    NATIVE_SUCCESS_PAYLOAD,
    ConfigWriter,
    FakeProbe,
    Handler,
    RecordingTransport,
)

from harness.gpu import GpuMemory
from harness.records import RunRecord
from harness.report import render_report
from harness.runner import RunPlan, RunResult, plan_run, run_suite
from harness.suite import ComparisonSuite, ModelCase, PromptSpec, load_suite
from llmkit import AppConfig, load_config

SUITE_TEXT = """\
[suite]
id = "runner_test"
description = "ランナーのテスト用スイート"
warmup_runs = 1

[[models]]
model_id = "qwen3-8b"
profile = "lightweight"

[[models]]
model_id = "gpt-oss-20b"
profile = "long_context"

[[prompts]]
id = "summarize"
text = "次の文章を3行で要約してください。"
system = "あなたは日本語の要約アシスタントです。"

[[prompts]]
id = "translate"
text = "次の文を英語に訳してください。"
"""

PROMPT_COUNT = 2
MODEL_COUNT = 2

#: 比較表で実効コンテキスト長を示す列 (仕様書 §4 T4 の列名)。
CONTEXT_LENGTH_COLUMN = "設定コンテキスト長"


# --------------------------------------------------------------------------
# 共通のヘルパ
# --------------------------------------------------------------------------


def write_suite(directory: Path, text: str = SUITE_TEXT) -> Path:
    destination = directory / "suite.toml"
    destination.write_text(text, encoding="utf-8")
    return destination


def build_plan(
    tmp_path: Path,
    *,
    suite: ComparisonSuite | None = None,
    base_config: AppConfig | None = None,
    config_path: Path = DEFAULT_CONFIG,
) -> RunPlan:
    """同じスイートファイル・同じ設定ファイルから実行計画を作る。

    ``suite`` / ``base_config`` を渡すと、**ファイルは変えずに**実効値だけを
    差し替えた計画になる (ハーネスが実行時にやっていることと同じ)。
    """
    suite_path = write_suite(tmp_path)
    loaded = suite if suite is not None else load_suite(suite_path)
    config = base_config if base_config is not None else load_config(config_path)
    return plan_run(loaded, suite_path, config, config_path)


def varying_handler() -> Handler:
    """呼ばれるたびに内容とトークン数が変わるネイティブ応答を返す。"""
    counter = itertools.count()

    def handle(request: httpx.Request) -> httpx.Response:
        index = next(counter)
        payload = dict(NATIVE_SUCCESS_PAYLOAD)
        payload["message"] = {"role": "assistant", "content": f"応答 {index}"}
        payload["eval_count"] = 30 + index
        payload["eval_duration"] = 1_000_000_000 + index * 1_000_000
        return httpx.Response(200, json=payload)

    return handle


def execute(
    plan: RunPlan,
    transport: RecordingTransport,
    tmp_path: Path,
    *,
    probe: FakeProbe | None = None,
    name: str = "results",
) -> RunResult:
    with transport.client() as http_client:
        return run_suite(
            plan,
            probe=probe if probe is not None else FakeProbe(),
            http_client=http_client,
            results_dir=tmp_path / name,
        )


def sent_models(transport: RecordingTransport) -> list[str]:
    return [json.loads(request.content)["model"] for request in transport.requests]


def request_bodies(transport: RecordingTransport) -> list[dict[str, object]]:
    return [json.loads(request.content) for request in transport.requests]


def measured(records: Sequence[RunRecord], case_index: int) -> list[RunRecord]:
    return [
        record
        for record in records
        if record.case_index == case_index and record.phase == "measure"
    ]


def replace_first_case(suite: ComparisonSuite, case: ModelCase) -> ComparisonSuite:
    return dataclasses.replace(suite, models=(case, *suite.models[1:]))


def sweep_variants(
    suite: ComparisonSuite, config: AppConfig
) -> Iterator[tuple[str, ComparisonSuite, AppConfig]]:
    """再現条件を 1 つずつ振る (仕様書 §4 T3 の 13 要素)。

    **どの要素もファイルには書き戻さない**。スイートと設定のハッシュを固定
    したまま実効値だけを振ることで、``run_fingerprint`` が実効値を見ている
    ことを測る (ファイルを書き換えると suite_sha256 が変わり、何も測れない)。
    """
    head = suite.models[0]
    yield (
        "model_id",
        replace_first_case(suite, dataclasses.replace(head, model_id="qwen3-14b")),
        config,
    )
    yield (
        "context_tokens",
        replace_first_case(suite, dataclasses.replace(head, context_tokens=8192)),
        config,
    )
    yield (
        "temperature",
        replace_first_case(suite, dataclasses.replace(head, temperature=0.1)),
        config,
    )
    yield (
        "top_p",
        replace_first_case(suite, dataclasses.replace(head, top_p=0.5)),
        config,
    )
    yield (
        "max_output_tokens",
        replace_first_case(suite, dataclasses.replace(head, max_output_tokens=256)),
        config,
    )
    yield "seed", replace_first_case(suite, dataclasses.replace(head, seed=42)), config
    yield (
        "profile",
        replace_first_case(suite, dataclasses.replace(head, profile="long_context")),
        config,
    )
    yield (
        "warmup_runs",
        dataclasses.replace(
            suite, suite=dataclasses.replace(suite.suite, warmup_runs=3)
        ),
        config,
    )
    yield (
        "prompt_text",
        dataclasses.replace(
            suite,
            prompts=(
                dataclasses.replace(suite.prompts[0], text="別の依頼文です。"),
                *suite.prompts[1:],
            ),
        ),
        config,
    )
    yield (
        "prompt_added",
        dataclasses.replace(
            suite,
            prompts=(*suite.prompts, PromptSpec(id="extra", text="追加の依頼文です。")),
        ),
        config,
    )
    yield (
        "runtime.base_url",
        suite,
        dataclasses.replace(
            config,
            runtime=dataclasses.replace(
                config.runtime, base_url="http://localhost:11435/v1"
            ),
        ),
    )
    yield (
        "runtime.kind",
        suite,
        dataclasses.replace(
            config,
            runtime=dataclasses.replace(config.runtime, kind="openai_compatible"),
        ),
    )
    yield (
        "vram.budget_gib",
        suite,
        dataclasses.replace(
            config, vram=dataclasses.replace(config.vram, budget_gib=15.0)
        ),
    )


# --------------------------------------------------------------------------
# 計画 (HTTP を出さない)
# --------------------------------------------------------------------------


def test_planning_resolves_every_case_without_any_runtime(tmp_path: Path) -> None:
    """``--dry-run`` の実体。ランタイムもプロセスも要らない。"""
    plan = build_plan(tmp_path)

    assert [case.model_id for case in plan.cases] == ["qwen3-8b", "gpt-oss-20b"]
    assert [case.profile_name for case in plan.cases] == ["lightweight", "long_context"]
    assert all(case.starts_without_budget_abort for case in plan.cases)
    assert len(plan.fingerprint) == 64
    assert plan.request_count == MODEL_COUNT * (1 + PROMPT_COUNT)


def test_planning_applies_the_case_to_both_model_id_locations(tmp_path: Path) -> None:
    """計画段階ですでに 2 か所が一致している (D-19)。"""
    plan = build_plan(tmp_path)

    for case in plan.cases:
        assert case.config.generation.model == case.model_id
        assert case.profile.generation.model_id == case.model_id
        assert set(case.estimate.weights_gib) == {case.model_id}


# --------------------------------------------------------------------------
# 再現条件 (D-20)
# --------------------------------------------------------------------------


def test_the_same_input_reproduces_the_fingerprint_across_two_runs(
    tmp_path: Path,
) -> None:
    """★ 応答内容もレイテンシも毎回違うのに、再現条件は一致する。"""
    transport = RecordingTransport(varying_handler())

    first = execute(build_plan(tmp_path), transport, tmp_path, name="first")
    second = execute(build_plan(tmp_path), transport, tmp_path, name="second")

    assert first.fingerprint == second.fingerprint
    assert first.run_id != second.run_id, "run_id は実行ごとに変わる"

    first_texts = [record.response_text for record in first.records]
    second_texts = [record.response_text for record in second.records]
    assert first_texts != second_texts, "前提: 応答内容は毎回違う"
    assert [record.latency_s for record in first.records] != [
        record.latency_s for record in second.records
    ], "前提: レイテンシは毎回違う"


def test_same_input_reproduces_the_fingerprint_while_config_sha256_alone_does_not(
    tmp_path: Path,
) -> None:
    """★ D-20 guard: ``config_sha256`` は実効値の同一性を表さない。

    モデルだけが違う 2 つの実行計画は、**同じベース設定ファイル**から作られる
    ため ``config_sha256`` が一致する。これで再現性を判定すると「20B の結果と
    14B の結果は同じ条件で得られた」と読めてしまう。
    """
    suite = load_suite(write_suite(tmp_path))
    other = replace_first_case(
        suite, dataclasses.replace(suite.models[0], model_id="qwen3-14b")
    )

    baseline = build_plan(tmp_path)
    varied = build_plan(tmp_path, suite=other)

    assert varied.config_sha256 == baseline.config_sha256
    assert varied.suite_sha256 == baseline.suite_sha256
    assert varied.fingerprint != baseline.fingerprint


def test_every_reproduction_field_changes_the_fingerprint(tmp_path: Path) -> None:
    """★ E23 guard: 13 要素のどれを振っても ``run_fingerprint`` が変わる。"""
    suite = load_suite(write_suite(tmp_path))
    config = load_config(DEFAULT_CONFIG)
    baseline = build_plan(tmp_path)
    seen: dict[str, str] = {baseline.fingerprint: "baseline"}

    for label, varied_suite, varied_config in sweep_variants(suite, config):
        plan = build_plan(tmp_path, suite=varied_suite, base_config=varied_config)

        assert plan.suite_sha256 == baseline.suite_sha256, label
        assert plan.config_sha256 == baseline.config_sha256, label
        assert plan.fingerprint != baseline.fingerprint, label
        assert plan.fingerprint not in seen, (
            f"{label} が {seen.get(plan.fingerprint)} と衝突"
        )
        seen[plan.fingerprint] = label

    assert len(seen) == 14, "13 要素 + baseline がすべて別のハッシュになる"


def test_the_fingerprint_ignores_the_response_and_the_clock(tmp_path: Path) -> None:
    """応答・速度・実行時刻はフィンガープリントに入らない。"""
    plan = build_plan(tmp_path)
    slow = RecordingTransport(varying_handler())

    result = execute(plan, slow, tmp_path)

    assert result.fingerprint == plan.fingerprint
    assert all(record.run_fingerprint == plan.fingerprint for record in result.records)


# --------------------------------------------------------------------------
# 実効値の配線 (E17-E19)
# --------------------------------------------------------------------------


def test_model_override_changes_both_the_request_and_the_vram_estimate(
    tmp_path: Path,
) -> None:
    """★ E17 / D-19 guard: モデルを変えると 4 つすべてが変わる。

    (a) リクエストの ``model`` (b) マニフェストの ``profile.models[0]``
    (c) ``vram.weights_gib`` のキー (d) ``vram_estimate_gib``。
    片方だけが変わる実装は「20B に投げて 14B の見積りを記録する」。
    """
    transport = RecordingTransport()

    result = execute(build_plan(tmp_path), transport, tmp_path)

    assert set(sent_models(transport)) == {"qwen3:8b-q4_K_M", "gpt-oss:20b"}

    first, second = result.manifests
    assert first.profile.models[0].model_id == "qwen3-8b"
    assert second.profile.models[0].model_id == "gpt-oss-20b"
    assert set(first.vram.weights_gib) == {"qwen3-8b"}
    assert set(second.vram.weights_gib) == {"gpt-oss-20b"}

    estimates = {record.model_id: record.vram_estimate_gib for record in result.records}
    assert estimates["qwen3-8b"] != estimates["gpt-oss-20b"]
    assert estimates["qwen3-8b"] == pytest.approx(first.vram.total_gib)
    assert estimates["gpt-oss-20b"] == pytest.approx(second.vram.total_gib)


def test_context_tokens_reach_the_request_the_report_and_the_fingerprint(
    tmp_path: Path,
) -> None:
    """★ E18 guard: 実効コンテキスト長が 3 か所すべてに届く。

    (a) リクエストの ``options.num_ctx`` (b) ``report.md`` の
    「設定コンテキスト長」列 (c) ``run_fingerprint``。表に出ないと、比較を
    読む人は「どちらのモデルが長いコンテキストで測られたか」を判別できない。
    """
    suite = load_suite(write_suite(tmp_path))
    varied = replace_first_case(
        suite, dataclasses.replace(suite.models[0], context_tokens=8192)
    )
    transport = RecordingTransport()

    plan = build_plan(tmp_path, suite=varied)
    result = execute(plan, transport, tmp_path)

    first_body = request_bodies(transport)[0]
    options = first_body["options"]
    assert isinstance(options, dict)
    assert options["num_ctx"] == 8192
    assert measured(result.records, 0)[0].context_tokens == 8192

    report = render_report(result)
    header_line = next(
        line for line in report.splitlines() if line.startswith("| model_id |")
    )
    header = [cell.strip() for cell in header_line.strip("|").split("|")]
    column = header.index(CONTEXT_LENGTH_COLUMN)
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in report.splitlines()
        if line.startswith("| qwen3-8b |") or line.startswith("| gpt-oss-20b |")
    ]
    assert [row[column] for row in rows] == ["8192", "16384"]

    assert plan.fingerprint != build_plan(tmp_path).fingerprint


def test_suite_generation_overrides_reach_the_request_and_the_manifest(
    tmp_path: Path,
) -> None:
    """★ E19 guard: 生成パラメータ 4 種がリクエストとマニフェストに届く。"""
    suite = load_suite(write_suite(tmp_path))
    varied = replace_first_case(
        suite,
        dataclasses.replace(
            suite.models[0],
            temperature=0.1,
            top_p=0.5,
            max_output_tokens=256,
            seed=42,
        ),
    )
    transport = RecordingTransport()

    plan = build_plan(tmp_path, suite=varied)
    result = execute(plan, transport, tmp_path)

    options = request_bodies(transport)[0]["options"]
    assert isinstance(options, dict)
    assert options["temperature"] == pytest.approx(0.1)
    assert options["top_p"] == pytest.approx(0.5)
    assert options["num_predict"] == 256
    assert options["seed"] == 42

    generation = result.manifests[0].generation
    assert generation.temperature == pytest.approx(0.1)
    assert generation.top_p == pytest.approx(0.5)
    assert generation.max_output_tokens == 256
    assert generation.seed == 42

    record = measured(result.records, 0)[0]
    assert record.temperature == pytest.approx(0.1)
    assert record.top_p == pytest.approx(0.5)
    assert record.max_output_tokens == 256
    assert record.seed == 42
    assert plan.fingerprint != build_plan(tmp_path).fingerprint


# --------------------------------------------------------------------------
# 実行順序とウォームアップ (D-22)
# --------------------------------------------------------------------------


def test_execution_is_model_major_and_warmups_come_first(tmp_path: Path) -> None:
    result = execute(build_plan(tmp_path), RecordingTransport(), tmp_path)

    assert [
        (record.case_index, record.phase, record.prompt_id) for record in result.records
    ] == [
        (0, "warmup", "summarize"),
        (0, "measure", "summarize"),
        (0, "measure", "translate"),
        (1, "warmup", "summarize"),
        (1, "measure", "summarize"),
        (1, "measure", "translate"),
    ]
    assert [record.sequence_index for record in result.records] == list(
        range(len(result.records))
    )


def test_warmup_runs_are_recorded_but_excluded_from_aggregates(
    tmp_path: Path,
) -> None:
    """★ E20 / D-22 guard: 回数で HTTP と warmup レコードだけが変わる。

    ``measure`` の件数と集計値は ``warmup_runs`` に影響されない。ウォーム
    アップを集計に混ぜる実装では、中央値か n が動いてここが落ちる。
    """
    medians: list[float | None] = []
    for warmup_runs in (0, 1, 2):
        suite = load_suite(write_suite(tmp_path))
        varied = dataclasses.replace(
            suite, suite=dataclasses.replace(suite.suite, warmup_runs=warmup_runs)
        )
        transport = RecordingTransport()

        result = execute(
            build_plan(tmp_path, suite=varied),
            transport,
            tmp_path,
            name=f"results-{warmup_runs}",
        )

        warmups = [r for r in result.records if r.phase == "warmup"]
        measures = [r for r in result.records if r.phase == "measure"]
        assert len(warmups) == MODEL_COUNT * warmup_runs, warmup_runs
        assert transport.call_count == MODEL_COUNT * (warmup_runs + PROMPT_COUNT)
        assert len(measures) == MODEL_COUNT * PROMPT_COUNT
        assert all(record.error is None for record in result.records)

        summary = result.summary.models[0]
        assert summary.attempted == PROMPT_COUNT
        assert (
            summary.generation_tokens_per_second.coverage
            == f"{PROMPT_COUNT}/{PROMPT_COUNT}"
        )
        medians.append(summary.generation_tokens_per_second.median)

    assert medians[0] == medians[1] == medians[2]
    assert medians[0] == pytest.approx(7.0), "eval_count / eval_duration 由来の値"


# --------------------------------------------------------------------------
# 速度指標の欠測 (D-21)
# --------------------------------------------------------------------------


def test_responses_without_timings_are_labelled_wallclock(
    tmp_path: Path, tmp_config: ConfigWriter
) -> None:
    """★ D-21 guard: timings を返さない経路では eval_* が null になる。"""
    config_path = tmp_config({'kind = "ollama"': 'kind = "openai_compatible"'})
    transport = RecordingTransport()

    result = execute(build_plan(tmp_path, config_path=config_path), transport, tmp_path)

    record = measured(result.records, 0)[0]
    assert record.eval_tokens_per_second is None
    assert record.prompt_tokens_per_second is None
    assert record.wallclock_tokens_per_second is not None
    assert record.generation_tokens_per_second == record.wallclock_tokens_per_second
    assert record.generation_tokens_per_second_source == "wallclock"
    assert result.summary.models[0].generation_tokens_per_second_source == "wallclock"


def test_zero_completion_tokens_never_become_zero_speeds(tmp_path: Path) -> None:
    """★ D-21 guard: 代表値が null になり、0.0 がレコードに現れない。"""

    def handle(request: httpx.Request) -> httpx.Response:
        payload = dict(NATIVE_SUCCESS_PAYLOAD)
        payload["eval_count"] = 0
        payload["eval_duration"] = 0
        return httpx.Response(200, json=payload)

    result = execute(build_plan(tmp_path), RecordingTransport(handle), tmp_path)

    record = measured(result.records, 0)[0]
    assert record.completion_tokens == 0
    assert record.eval_tokens_per_second is None
    assert record.wallclock_tokens_per_second is None
    assert record.generation_tokens_per_second is None
    assert record.generation_tokens_per_second_source is None

    summary = result.summary.models[0]
    assert summary.generation_tokens_per_second.median is None
    assert summary.generation_tokens_per_second.coverage == f"0/{PROMPT_COUNT}"


# --------------------------------------------------------------------------
# VRAM 実測 (D-23)
# --------------------------------------------------------------------------


def test_probe_readings_land_on_every_record_of_the_model(tmp_path: Path) -> None:
    probe = FakeProbe(
        [
            GpuMemory(name="gpu", used_mib=600, total_mib=16384),
            GpuMemory(name="gpu", used_mib=5720, total_mib=16384),
            GpuMemory(name="gpu", used_mib=13500, total_mib=16384),
        ]
    )

    result = execute(build_plan(tmp_path), RecordingTransport(), tmp_path, probe=probe)

    assert probe.call_count == MODEL_COUNT + 1, "アイドル基準は実行全体で 1 回だけ"
    assert result.vram_idle_mib == 600
    first = measured(result.records, 0)[0]
    second = measured(result.records, 1)[0]
    assert first.vram_used_mib == 5720
    assert first.vram_increment_gib == pytest.approx((5720 - 600) / 1024)
    assert second.vram_used_mib == 13500
    assert second.vram_increment_gib == pytest.approx((13500 - 600) / 1024)


def test_idle_baseline_is_measured_once_and_shared_across_all_models(
    tmp_path: Path,
) -> None:
    """★ 不具合再現の固定 (2026-08-23 実機実行で発見)。

    アイドル基準をモデルごとに測り直すと、2 モデル目以降の
    ``vram_increment_gib`` は「前のモデルがロード済みの状態からの差」に
    なり、実測値が意味を成さなくなる (実機では 20b で 1.37、8b で -5.24 と
    いう誤った値が記録された)。プローブが呼ばれるたびに異なる値を返す
    フェイクを注入し、全モデルのレコードが同じ ``idle`` (先頭の読み取り値)
    を基準にしていることを検証する。
    """
    probe = FakeProbe(
        [
            GpuMemory(name="gpu", used_mib=844, total_mib=16384),
            GpuMemory(name="gpu", used_mib=12246, total_mib=16384),
            GpuMemory(name="gpu", used_mib=13650, total_mib=16384),
        ]
    )

    result = execute(build_plan(tmp_path), RecordingTransport(), tmp_path, probe=probe)

    assert probe.call_count == MODEL_COUNT + 1, "アイドル基準は実行全体で 1 回だけ"
    assert result.vram_idle_mib == 844
    first = measured(result.records, 0)[0]
    second = measured(result.records, 1)[0]
    # 両モデルとも同じ idle (844) を基準に計算されている。
    assert first.vram_increment_gib == pytest.approx((12246 - 844) / 1024)
    assert second.vram_increment_gib == pytest.approx((13650 - 844) / 1024)
    # モデルごとに idle を測り直す不具合が再発すると、2 モデル目の基準が
    # 前のモデルの used (12246) にすり替わり、大幅に小さい誤った増分に
    # なる。これが起きていないことを固定する。
    assert second.vram_increment_gib != pytest.approx((13650 - 12246) / 1024)


def test_records_stay_null_when_no_gpu_probe_is_available(tmp_path: Path) -> None:
    """CI (nvidia-smi 無し) でも実行は完走し、実測列は null のまま。"""
    result = execute(build_plan(tmp_path), RecordingTransport(), tmp_path)

    assert all(record.vram_used_mib is None for record in result.records)
    assert all(record.vram_increment_gib is None for record in result.records)
    assert result.summary.models[0].vram_estimate_gib > 0.0


# --------------------------------------------------------------------------
# 失敗の扱い
# --------------------------------------------------------------------------


THREE_MODEL_SUITE = SUITE_TEXT.replace(
    '[[prompts]]\nid = "summarize"',
    '[[models]]\nmodel_id = "qwen3-14b"\nprofile = "rag_default"\n\n'
    '[[prompts]]\nid = "summarize"',
)


def test_one_failing_model_does_not_stop_the_others(tmp_path: Path) -> None:
    """1 モデル目が ModelNotFoundError でも 2・3 モデル目は完走する。"""

    def handle(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "qwen3:8b-q4_K_M":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(200, json=NATIVE_SUCCESS_PAYLOAD)

    suite_path = tmp_path / "suite.toml"
    suite_path.write_text(THREE_MODEL_SUITE, encoding="utf-8")
    plan = plan_run(
        load_suite(suite_path), suite_path, load_config(DEFAULT_CONFIG), DEFAULT_CONFIG
    )

    result = execute(plan, RecordingTransport(handle), tmp_path)

    failed = measured(result.records, 0)
    assert [record.error.type for record in failed if record.error is not None] == [
        "ModelNotFoundError"
    ] * PROMPT_COUNT
    assert all(record.response_text is None for record in failed)
    for case_index in (1, 2):
        survivors = measured(result.records, case_index)
        assert all(record.error is None for record in survivors)
        assert all(record.response_text for record in survivors)

    assert result.summary.models[0].generation_tokens_per_second.coverage == "0/0"
    assert result.summary.models[0].attempted == PROMPT_COUNT
    assert result.summary.models[1].generation_tokens_per_second.n == PROMPT_COUNT


OVER_BUDGET_SUITE = SUITE_TEXT.replace(
    '[[models]]\nmodel_id = "qwen3-8b"\nprofile = "lightweight"',
    '[[models]]\nmodel_id = "qwen3-14b"\nprofile = "rag_default"\n'
    "context_tokens = 32768",
)


def test_an_over_budget_model_is_skipped_without_any_http(tmp_path: Path) -> None:
    """★ D-04 guard (ハーネス経路): 予算超過は HTTP を 1 バイトも出さない。"""
    suite_path = tmp_path / "suite.toml"
    suite_path.write_text(OVER_BUDGET_SUITE, encoding="utf-8")
    plan = plan_run(
        load_suite(suite_path), suite_path, load_config(DEFAULT_CONFIG), DEFAULT_CONFIG
    )
    transport = RecordingTransport()

    assert plan.cases[0].starts_without_budget_abort is False
    assert plan.request_count == 1 * (1 + PROMPT_COUNT), "予算内のモデルの分だけ"

    result = execute(plan, transport, tmp_path)

    skipped = measured(result.records, 0)
    assert [record.error.type for record in skipped if record.error is not None] == [
        "VramBudgetExceededError"
    ] * PROMPT_COUNT
    assert not [r for r in result.records if r.case_index == 0 and r.phase == "warmup"]
    assert sent_models(transport) == ["gpt-oss:20b"] * (1 + PROMPT_COUNT)
    assert result.summary.models[0].succeeded == 0
    assert result.summary.models[1].succeeded == PROMPT_COUNT


# --------------------------------------------------------------------------
# 出力 (マニフェスト)
# --------------------------------------------------------------------------


def test_each_model_writes_one_manifest_under_the_results_directory(
    tmp_path: Path,
) -> None:
    result = execute(build_plan(tmp_path), RecordingTransport(), tmp_path)

    manifests_dir = tmp_path / "results" / "manifests"
    written = sorted(manifests_dir.glob("*.json"))
    assert len(written) == MODEL_COUNT
    assert sorted(result.manifest_paths) == written
    assert result.started_at_utc.endswith("Z")
    assert all(record.run_id == result.run_id for record in result.records)
