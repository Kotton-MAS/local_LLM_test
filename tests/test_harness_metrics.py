"""レコードの速度指標と集計 (harness/records.py) のテスト。

このファイルの中心は **欠測を 0 で埋めないこと** (D-21) と
**ウォームアップを捨てずに集計からだけ外すこと** (D-22) である。

``llmkit.ChatResult.tokens_per_second`` は計測不能を 0.0 で表す仕様なので、
それをそのまま記録すると「実測 0 トークン/秒」と「未計測」が区別できなくなり、
中央値が静かにゼロ方向へ歪む。ここではその 0.0 が実際に出ることを確かめた
うえで、レコードには ``None`` が載ることを固定する。

HTTP は 1 バイトも発行しない (:class:`~llmkit.ChatResult` を直接組み立てる)。
"""

from __future__ import annotations

import dataclasses

import pytest

from harness.gpu import GpuMemory
from harness.records import (
    MIXED_SPEED_SOURCE,
    RECORD_KEYS,
    RECORD_SCHEMA_VERSION,
    RecordError,
    ResponseFacts,
    RunPhase,
    RunRecord,
    SpeedMetrics,
    SpeedSource,
    VramReading,
    summarize_records,
    wallclock_tokens_per_second,
)
from llmkit import ChatResult, ChatTimings, TokenUsage


def chat_result(
    *,
    completion_tokens: int = 40,
    latency_s: float = 2.0,
    timings: ChatTimings | None = None,
) -> ChatResult:
    """速度指標の掃引に使う :class:`ChatResult`。"""
    return ChatResult(
        text="テスト応答",
        model="qwen3:8b-q4_K_M",
        finish_reason="stop",
        usage=TokenUsage(
            prompt_tokens=11,
            completion_tokens=completion_tokens,
            total_tokens=11 + completion_tokens,
        ),
        latency_s=latency_s,
        timings=timings,
    )


NATIVE_TIMINGS = ChatTimings(
    prompt_eval_count=11,
    prompt_eval_seconds=0.5,
    eval_count=40,
    eval_seconds=1.0,
)

#: 計測値が 1 つも返らなかったネイティブ応答 (プロンプトキャッシュヒット等)。
EMPTY_TIMINGS = ChatTimings(
    prompt_eval_count=None,
    prompt_eval_seconds=None,
    eval_count=None,
    eval_seconds=None,
)


def record(
    *,
    case_index: int = 0,
    model_id: str = "qwen3-8b",
    phase: RunPhase = "measure",
    prompt_id: str = "summarize",
    sequence_index: int = 0,
    generation: float | None = None,
    source: SpeedSource | None = None,
    prompt_speed: float | None = None,
    wallclock: float | None = None,
    error: RecordError | None = None,
    vram_used_mib: int | None = 5000,
    vram_increment_gib: float | None = 4.2,
) -> RunRecord:
    """集計テスト用の最小レコード。速度以外の値は固定する。"""
    return RunRecord(
        schema_version=RECORD_SCHEMA_VERSION,
        run_id="runid0000000",
        run_fingerprint="f" * 64,
        sequence_index=sequence_index,
        case_index=case_index,
        model_id=model_id,
        served_name=f"{model_id}:served",
        quantization="Q4_K",
        serving_runtime="ollama",
        profile_name="lightweight",
        context_tokens=16384,
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=512,
        seed=0,
        phase=phase,
        prompt_id=prompt_id,
        prompt_sha256="a" * 64,
        response_text=None if error is not None else "テスト応答",
        finish_reason=None if error is not None else "stop",
        prompt_tokens=None if error is not None else 11,
        completion_tokens=None if error is not None else 40,
        total_tokens=None if error is not None else 51,
        latency_s=None if error is not None else 2.0,
        eval_tokens_per_second=generation if source == "eval" else None,
        prompt_tokens_per_second=prompt_speed,
        wallclock_tokens_per_second=wallclock,
        generation_tokens_per_second=generation,
        generation_tokens_per_second_source=source,
        vram_estimate_gib=7.26,
        vram_budget_gib=14.0,
        vram_used_mib=vram_used_mib,
        vram_total_mib=16384,
        vram_increment_gib=vram_increment_gib,
        error=error,
    )


# --------------------------------------------------------------------------
# 速度指標 (D-21)
# --------------------------------------------------------------------------


def test_eval_timings_are_preferred_and_labelled() -> None:
    """計測値があれば eval を代表値にし、出典ラベルを添える。"""
    metrics = SpeedMetrics.from_chat_result(
        chat_result(timings=NATIVE_TIMINGS, latency_s=2.0)
    )

    assert metrics.eval_tokens_per_second == pytest.approx(40.0)
    assert metrics.prompt_tokens_per_second == pytest.approx(22.0)
    assert metrics.wallclock_tokens_per_second == pytest.approx(20.0)
    assert metrics.generation_tokens_per_second == pytest.approx(40.0)
    assert metrics.generation_tokens_per_second_source == "eval"


def test_missing_timings_fall_back_to_the_wallclock_with_its_own_label() -> None:
    """OpenAI 互換経路のように timings が無い応答は壁時計 + ラベル。"""
    metrics = SpeedMetrics.from_chat_result(chat_result(timings=None, latency_s=2.0))

    assert metrics.eval_tokens_per_second is None
    assert metrics.prompt_tokens_per_second is None
    assert metrics.wallclock_tokens_per_second == pytest.approx(20.0)
    assert metrics.generation_tokens_per_second == pytest.approx(20.0)
    assert metrics.generation_tokens_per_second_source == "wallclock"


def test_zero_completion_tokens_are_recorded_as_null_not_zero() -> None:
    """★ D-21 guard: ``tokens_per_second`` の 0.0 をそのまま記録しない。

    llmkit 側は互換のため計測不能を 0.0 で返す。その 0.0 が実在することを
    確かめたうえで、レコードには ``None`` が載ることを固定する。
    """
    result = chat_result(completion_tokens=0, latency_s=2.0, timings=EMPTY_TIMINGS)

    assert result.tokens_per_second == 0.0, "前提: llmkit は欠測を 0.0 で表す"

    metrics = SpeedMetrics.from_chat_result(result)

    assert wallclock_tokens_per_second(result) is None
    assert metrics.wallclock_tokens_per_second is None
    assert metrics.eval_tokens_per_second is None
    assert metrics.generation_tokens_per_second is None
    assert metrics.generation_tokens_per_second_source is None


def test_zero_latency_is_recorded_as_null_not_zero() -> None:
    """レイテンシが計測不能 (0 秒) でも 0.0 を書かない。"""
    result = chat_result(latency_s=0.0, timings=None)

    assert result.tokens_per_second == 0.0

    metrics = SpeedMetrics.from_chat_result(result)

    assert metrics.wallclock_tokens_per_second is None
    assert metrics.generation_tokens_per_second is None
    assert metrics.generation_tokens_per_second_source is None


def test_incomplete_timings_do_not_produce_an_eval_rate() -> None:
    """eval_count はあるが eval_duration が無い応答は壁時計へ落ちる。"""
    timings = dataclasses.replace(NATIVE_TIMINGS, eval_seconds=None)

    metrics = SpeedMetrics.from_chat_result(chat_result(timings=timings))

    assert metrics.eval_tokens_per_second is None
    assert metrics.generation_tokens_per_second_source == "wallclock"


def test_failed_attempts_have_no_speed_values_at_all() -> None:
    metrics = SpeedMetrics.missing()
    facts = ResponseFacts.missing()

    assert dataclasses.astuple(metrics) == (None,) * 6
    assert dataclasses.astuple(facts) == (None,) * 5


# --------------------------------------------------------------------------
# VRAM 実測 (D-23)
# --------------------------------------------------------------------------


def test_vram_increment_is_the_difference_from_the_idle_reading() -> None:
    reading = VramReading.from_probe(
        idle=GpuMemory(name="RTX 5070 Ti", used_mib=600, total_mib=16384),
        current=GpuMemory(name="RTX 5070 Ti", used_mib=8792, total_mib=16384),
    )

    assert reading.used_mib == 8792
    assert reading.total_mib == 16384
    assert reading.increment_gib == pytest.approx((8792 - 600) / 1024)


def test_vram_is_null_when_the_probe_cannot_read_anything() -> None:
    """プローブが使えない環境 (CI) でも 0 を書かない。"""
    reading = VramReading.from_probe(idle=None, current=None)

    assert (reading.used_mib, reading.total_mib, reading.increment_gib) == (
        None,
        None,
        None,
    )


def test_vram_increment_is_null_when_only_the_idle_reading_is_missing() -> None:
    reading = VramReading.from_probe(
        idle=None, current=GpuMemory(name="gpu", used_mib=8792, total_mib=16384)
    )

    assert reading.used_mib == 8792
    assert reading.increment_gib is None


# --------------------------------------------------------------------------
# レコードの形
# --------------------------------------------------------------------------


def test_record_dict_keys_match_the_declared_schema() -> None:
    """``records.jsonl`` の 1 行のキー集合は RECORD_KEYS と完全一致する。"""
    payload = record().to_dict()

    assert tuple(payload) == RECORD_KEYS
    assert len(RECORD_KEYS) == len(set(RECORD_KEYS))


def test_error_records_serialize_the_type_and_message() -> None:
    payload = record(
        error=RecordError(type="ModelNotFoundError", message="無い")
    ).to_dict()

    assert payload["error"] == {"type": "ModelNotFoundError", "message": "無い"}
    assert payload["response_text"] is None


def test_successful_records_have_no_error_key_content() -> None:
    assert record().to_dict()["error"] is None


# --------------------------------------------------------------------------
# 集計 (D-21 / D-22)
# --------------------------------------------------------------------------


def test_missing_timings_are_recorded_as_null_and_excluded_from_aggregates() -> None:
    """★ E21 guard: 欠測は分母にも分子にも入れず、n/N で件数を併記する。

    3 件のうち 1 件が欠測のとき、中央値は残り 2 件だけで計算される。0 を
    混ぜると 20.0 (= median(0, 30, 40)) ではなく 30.0 が返り、n も 3 に
    水増しされる。
    """
    records = [
        record(sequence_index=0, generation=30.0, source="eval", wallclock=25.0),
        record(sequence_index=1, generation=40.0, source="eval", wallclock=35.0),
        record(sequence_index=2, generation=None, source=None, wallclock=None),
    ]

    summary = summarize_records(records)
    model = summary.models[0]

    assert model.generation_tokens_per_second.median == pytest.approx(35.0)
    assert model.generation_tokens_per_second.n == 2
    assert model.generation_tokens_per_second.total == 3
    assert model.generation_tokens_per_second.coverage == "2/3"
    assert model.wallclock_tokens_per_second.median == pytest.approx(30.0)
    assert model.attempted == 3
    assert model.succeeded == 3


def test_a_metric_with_no_values_at_all_stays_null() -> None:
    """全件欠測でも 0.0 にならない (レポートは ``—`` を出せる)。"""
    summary = summarize_records([record(generation=None, source=None)])

    assert summary.models[0].generation_tokens_per_second.median is None
    assert summary.models[0].generation_tokens_per_second.n == 0
    assert summary.models[0].generation_tokens_per_second_source is None


def test_aggregate_uses_the_median_not_the_mean() -> None:
    """コールド実行の外れ値に引きずられないこと (§3 ソフト制約)。"""
    values = [5.0, 30.0, 31.0, 32.0, 33.0]
    records = [
        record(sequence_index=index, generation=value, source="eval")
        for index, value in enumerate(values)
    ]

    median = summarize_records(records).models[0].generation_tokens_per_second.median

    assert median == pytest.approx(31.0)
    assert median != pytest.approx(sum(values) / len(values))


def test_warmup_records_are_kept_but_excluded_from_aggregates() -> None:
    """★ D-22 guard (レコード層): warmup は残すが集計に入れない。"""
    records = [
        record(sequence_index=0, phase="warmup", generation=1.0, source="eval"),
        record(sequence_index=1, phase="measure", generation=30.0, source="eval"),
        record(sequence_index=2, phase="measure", generation=40.0, source="eval"),
    ]

    summary = summarize_records(records)

    assert summary.warmup_record_count == 1
    assert summary.measure_record_count == 2
    assert summary.models[0].generation_tokens_per_second.median == pytest.approx(35.0)
    assert summary.models[0].attempted == 2


def test_failed_records_count_in_the_denominator_but_not_in_the_median() -> None:
    """失敗は N に数え n から外す (「速いから少ない」と読めなくする)。"""
    records = [
        record(sequence_index=0, generation=30.0, source="eval"),
        record(
            sequence_index=1,
            error=RecordError(type="UpstreamError", message="失敗"),
            generation=None,
        ),
    ]

    summary = summarize_records(records)
    model = summary.models[0]

    assert model.generation_tokens_per_second.median == pytest.approx(30.0)
    assert model.attempted == 2
    assert model.succeeded == 1
    assert summary.error_record_count == 1


def test_each_model_case_becomes_one_row_in_declaration_order() -> None:
    records = [
        record(case_index=1, model_id="gpt-oss-20b", generation=20.0, source="eval"),
        record(case_index=0, model_id="qwen3-8b", generation=50.0, source="eval"),
    ]

    summary = summarize_records(records)

    assert [model.model_id for model in summary.models] == ["qwen3-8b", "gpt-oss-20b"]
    assert [model.case_index for model in summary.models] == [0, 1]


def test_mixed_speed_sources_are_not_hidden_behind_one_label() -> None:
    """分母の違う値が混ざったことを表の上で隠さない。"""
    records = [
        record(sequence_index=0, generation=30.0, source="eval"),
        record(sequence_index=1, generation=20.0, source="wallclock", wallclock=20.0),
    ]

    summary = summarize_records(records)

    assert summary.models[0].generation_tokens_per_second_source == MIXED_SPEED_SOURCE


def test_vram_measurements_survive_into_the_summary() -> None:
    summary = summarize_records([record(vram_used_mib=8792, vram_increment_gib=8.0)])

    assert summary.models[0].vram_used_mib == 8792
    assert summary.models[0].vram_increment_gib == pytest.approx(8.0)


def test_vram_measurements_stay_null_when_the_probe_is_unavailable() -> None:
    summary = summarize_records([record(vram_used_mib=None, vram_increment_gib=None)])

    assert summary.models[0].vram_used_mib is None
    assert summary.models[0].vram_increment_gib is None
