"""比較結果の書き出し (``report.md`` / ``records.jsonl`` / ``run.json``)。

出力先は ``results/<suite_id>/<YYYYmmddTHHMMSSZ>-<fingerprint[:12]>/`` で、
同じディレクトリの ``manifests/`` に :mod:`llmkit` の実行マニフェストが
モデルごとに 1 本入る (書き出すのは :func:`harness.runner.run_suite`)。

★ このモジュールの中核は 2 つある。

1. **欠測セルは ``—`` であり、0 を書かない** (D-21)。集計は
   ``phase == "measure"`` かつ ``error is None`` のレコードだけを対象にし、
   「測定できた件数 / 試行件数」を ``n/N`` で必ず併記する。0 を書くと
   「実測 0 トークン/秒」と「未計測」が表の上で区別できなくなる。
2. **``run.json`` だけでフィンガープリントを再計算できる** (D-20)。ハッシュ値
   だけを残すと「何が変わったから変わったのか」が追えないため、
   :func:`harness.runner.fingerprint_inputs` が返す入力そのものを
   ``reproduction`` に載せる。レポートが自分自身を検証できる状態にする。

応答本文は Markdown 側に**先頭 400 文字だけ**を折りたたみで置き、全文は
``records.jsonl`` に持つ (§3 ソフト制約)。Markdown の構造を応答内容で壊さない
よう、本文は引用ブロック (``> ``) にして書き出す。コードフェンスで囲むと、
応答自身がフェンスを含んでいたときに文書が崩れる。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from harness.records import (
    RECORD_SCHEMA_VERSION,
    MetricSummary,
    ModelSummary,
    RunRecord,
)
from harness.runner import CasePlan, RunPlan, RunResult, fingerprint_inputs
from harness.suite import PromptSpec
from llmkit import ConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "COMPARISON_TABLE_HEADER",
    "DIRECTORY_TIMESTAMP_FORMAT",
    "MISSING_CELL",
    "RECORDS_FILENAME",
    "REPORT_FILENAME",
    "RESPONSE_PREVIEW_CHARS",
    "RUN_JSON_FILENAME",
    "ReportPaths",
    "build_run_json",
    "render_records_jsonl",
    "render_report",
    "run_output_dir",
    "write_run_outputs",
]

REPORT_FILENAME = "report.md"
RECORDS_FILENAME = "records.jsonl"
RUN_JSON_FILENAME = "run.json"

#: 欠測セル。**数値 0 を書かない** (D-21)。
MISSING_CELL = "—"

#: Markdown に載せる応答本文の文字数。全文は ``records.jsonl`` にある。
RESPONSE_PREVIEW_CHARS = 400

#: 出力ディレクトリ名の時刻部分 (ISO8601 はコロンを含み扱いにくいため詰める)。
DIRECTORY_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: モデル比較表の列。**列名は受け入れ条件2 の機械検証に直結するため変えない**。
COMPARISON_TABLE_HEADER: tuple[str, ...] = (
    "model_id",
    "served_name",
    "quantization",
    "設定コンテキスト長",
    "生成速度 t/s (中央値)",
    "速度出典",
    "プロンプト処理速度 t/s (中央値)",
    "壁時計速度 t/s (中央値)",
    "VRAM 見積り GiB",
    "VRAM 実測増分 GiB",
    "測定数 n/N",
)


@dataclass(frozen=True, slots=True)
class ReportPaths:
    """書き出した成果物の場所。"""

    directory: Path
    report: Path
    records: Path
    run_json: Path


def run_output_dir(
    results_root: Path, *, suite_id: str, moment: datetime, fingerprint: str
) -> Path:
    """``<results_root>/<suite_id>/<時刻>-<fingerprint[:12]>`` を組み立てる。

    ``suite_id`` は :class:`~harness.suite.SuiteMeta` が ``[a-z0-9_-]+`` に
    制限しており、パス区切りや ``..`` は入口で拒否されている。

    Args:
        results_root: 比較結果を集める根ディレクトリ (既定 ``results/``)。
        suite_id: スイート ID。
        moment: 実行開始時刻。``run_suite`` に渡す ``clock`` と同じ値にする
            (ディレクトリ名と ``started_at_utc`` を食い違わせないため)。
        fingerprint: :func:`harness.runner.run_fingerprint` の値。
    """
    stamp = moment.astimezone(UTC).strftime(DIRECTORY_TIMESTAMP_FORMAT)
    return results_root / suite_id / f"{stamp}-{fingerprint[:12]}"


# --------------------------------------------------------------------------
# セルの整形 (欠測は — 、0 を書かない)
# --------------------------------------------------------------------------


def _format_speed(summary: MetricSummary) -> str:
    """速度の中央値。1 件も測れていなければ ``—`` (0.0 ではない)。"""
    if summary.median is None:
        return MISSING_CELL
    return f"{summary.median:.1f}"


def _format_gib(value: float | None) -> str:
    if value is None:
        return MISSING_CELL
    return f"{value:.2f}"


def _format_text(value: str | None) -> str:
    if value is None:
        return MISSING_CELL
    return value


def _escape_cell(value: str) -> str:
    """表のセルに入れる文字列から Markdown の表構造を壊す文字を除く。"""
    return value.replace("|", r"\|").replace("\n", " ")


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(_escape_cell(cell) for cell in header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines.extend(
        "| " + " | ".join(_escape_cell(cell) for cell in row) + " |" for row in rows
    )
    return lines


def _comparison_row(model: ModelSummary) -> tuple[str, ...]:
    """比較表 1 行。``n`` は測定できた件数、``N`` は試行件数 (D-21)。"""
    return (
        model.model_id,
        model.served_name,
        model.quantization,
        str(model.context_tokens),
        _format_speed(model.generation_tokens_per_second),
        _format_text(model.generation_tokens_per_second_source),
        _format_speed(model.prompt_tokens_per_second),
        _format_speed(model.wallclock_tokens_per_second),
        _format_gib(model.vram_estimate_gib),
        _format_gib(model.vram_increment_gib),
        f"{model.generation_tokens_per_second.n}/{model.attempted}",
    )


# --------------------------------------------------------------------------
# report.md
# --------------------------------------------------------------------------


def _reproduction_rows(result: RunResult) -> list[tuple[str, str]]:
    plan = result.plan
    return [
        ("run_fingerprint", result.fingerprint),
        ("run_id", result.run_id),
        ("実行開始 (UTC)", result.started_at_utc),
        ("スイート", str(plan.suite_path)),
        (
            "suite_sha256",
            f"{plan.suite_sha256} "
            "(ベースファイルのハッシュ。実効入力は reproduction を参照)",
        ),
        ("絞り込み (--models/--limit)", plan.filters.describe()),
        ("設定", str(plan.config_path)),
        ("config_sha256", plan.config_sha256),
        ("ウォームアップ回数", str(plan.warmup_runs)),
        ("プロンプト数", str(len(plan.suite.prompts))),
        ("モデル数", str(len(plan.cases))),
        ("レコードスキーマ版", RECORD_SCHEMA_VERSION),
    ]


def _quote_block(text: str) -> list[str]:
    """本文を引用ブロックにする (Markdown の構造を応答内容で壊さない)。"""
    lines = text.splitlines() or [""]
    return [f"> {line}" if line else ">" for line in lines]


def _preview(text: str) -> tuple[str, bool]:
    if len(text) <= RESPONSE_PREVIEW_CHARS:
        return text, False
    return text[:RESPONSE_PREVIEW_CHARS], True


def _record_facts(record: RunRecord) -> str:
    """応答 1 件の速度・トークン数の 1 行要約。欠測は ``—``。"""
    speed = (
        MISSING_CELL
        if record.generation_tokens_per_second is None
        else f"{record.generation_tokens_per_second:.1f} t/s"
    )
    source = _format_text(record.generation_tokens_per_second_source)
    completion = (
        MISSING_CELL
        if record.completion_tokens is None
        else str(record.completion_tokens)
    )
    return (
        f"- 生成速度: {speed} (出典: {source}) / "
        f"完了トークン: {completion} / "
        f"finish_reason: {_format_text(record.finish_reason)}"
    )


def _response_block(record: RunRecord | None) -> list[str]:
    """1 モデル分の応答。失敗は「空の応答」ではなく失敗として書く。"""
    if record is None:
        return ["- 記録がありません。"]
    if record.error is not None:
        return [
            f"- 失敗: {record.error.type}",
            "",
            *_quote_block(record.error.message),
        ]
    text = record.response_text
    if text is None:
        return ["- 応答本文がありません。"]
    preview, truncated = _preview(text)
    summary = (
        f"応答 (先頭 {RESPONSE_PREVIEW_CHARS} 文字 / 全文は "
        f"{RECORDS_FILENAME} の sequence_index={record.sequence_index})"
        if truncated
        else f"応答 (全文 / {RECORDS_FILENAME} の "
        f"sequence_index={record.sequence_index})"
    )
    return [
        _record_facts(record),
        "",
        "<details>",
        f"<summary>{summary}</summary>",
        "",
        *_quote_block(preview),
        "",
        "</details>",
    ]


def _prompt_section(
    result: RunResult,
    prompt: PromptSpec,
    measured: dict[tuple[int, str], RunRecord],
) -> list[str]:
    lines = [f"### {prompt.id}", ""]
    if prompt.tags:
        lines.extend([f"タグ: {', '.join(prompt.tags)}", ""])
    if prompt.system:
        lines.extend(["system:", "", *_quote_block(prompt.system), ""])
    lines.extend(["プロンプト:", "", *_quote_block(prompt.text), ""])
    for case in result.plan.cases:
        lines.extend([f"#### {case.model_id}", ""])
        lines.extend(_response_block(measured.get((case.case_index, prompt.id))))
        lines.append("")
    return lines


def render_report(result: RunResult) -> str:
    """``report.md`` の本文を組み立てる。

    構成は「再現条件ヘッダ → モデル比較表 → プロンプトごとの応答」。
    比較表の列は :data:`COMPARISON_TABLE_HEADER` で固定する。
    """
    plan = result.plan
    lines: list[str] = [
        f"# モデル比較レポート: {plan.suite_id}",
        "",
    ]
    if plan.suite.suite.description:
        lines.extend([plan.suite.suite.description, ""])
    lines.extend(["## 再現条件", ""])
    lines.extend(
        _markdown_table(
            ("項目", "値"),
            [(name, value) for name, value in _reproduction_rows(result)],
        )
    )
    lines.extend(
        [
            "",
            f"実行マニフェスト: `{plan.suite_id}` の各モデル 1 本ずつ "
            f"(同ディレクトリの `manifests/`)。",
            "",
            "## モデル比較",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            COMPARISON_TABLE_HEADER,
            [_comparison_row(model) for model in result.summary.models],
        )
    )
    lines.extend(
        [
            "",
            f"欠測は `{MISSING_CELL}`。集計は `phase=measure` かつ成功した"
            "レコードのみで、`測定数 n/N` は「測定できた件数 / 試行件数」。",
            "",
            "## 応答",
            "",
        ]
    )
    measured = {
        (record.case_index, record.prompt_id): record
        for record in result.records
        if record.phase == "measure"
    }
    for prompt in plan.suite.prompts:
        lines.extend(_prompt_section(result, prompt, measured))
    return "\n".join(lines).rstrip("\n") + "\n"


# --------------------------------------------------------------------------
# records.jsonl
# --------------------------------------------------------------------------


def render_records_jsonl(records: Sequence[RunRecord]) -> str:
    """1 行 1 レコードの JSONL (``sequence_index`` 昇順、warmup を含む)。

    各行のキーの並びは :data:`harness.records.RECORD_KEYS` と一致する
    (``sort_keys`` を使わず :meth:`RunRecord.to_dict` の宣言順のまま出す)。
    """
    ordered = sorted(records, key=lambda record: record.sequence_index)
    return "".join(
        json.dumps(record.to_dict(), ensure_ascii=False) + "\n" for record in ordered
    )


# --------------------------------------------------------------------------
# run.json
# --------------------------------------------------------------------------


def _reproduction(plan: RunPlan) -> dict[str, object]:
    """フィンガープリントの入力そのもの (計画時と同じ値)。

    ``run.json`` からこの辞書だけを取り出して
    :func:`harness.runner.fingerprint_digest` にかけると、記録された
    ``run_fingerprint`` と一致する。
    """
    return fingerprint_inputs(
        plan.suite,
        [case.manifest for case in plan.cases],
        suite_sha256=plan.suite_sha256,
        config_sha256=plan.config_sha256,
    )


def _metric_json(summary: MetricSummary) -> dict[str, object]:
    return {"median": summary.median, "n": summary.n, "total": summary.total}


def _model_json(model: ModelSummary, case: CasePlan) -> dict[str, object]:
    return {
        "case_index": model.case_index,
        "model_id": model.model_id,
        "served_name": model.served_name,
        "quantization": model.quantization,
        "profile_name": case.profile_name,
        "context_tokens": model.context_tokens,
        "generation_tokens_per_second": _metric_json(
            model.generation_tokens_per_second
        ),
        "generation_tokens_per_second_source": (
            model.generation_tokens_per_second_source
        ),
        "prompt_tokens_per_second": _metric_json(model.prompt_tokens_per_second),
        "wallclock_tokens_per_second": _metric_json(model.wallclock_tokens_per_second),
        "vram_estimate_gib": model.vram_estimate_gib,
        "vram_budget_gib": case.estimate.budget_gib,
        "vram_within_budget": case.starts_without_budget_abort,
        "vram_used_mib": model.vram_used_mib,
        "vram_increment_gib": model.vram_increment_gib,
        "attempted": model.attempted,
        "succeeded": model.succeeded,
    }


def _relative_to(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_output_dir(result: RunResult, output_dir: Path | None) -> Path:
    """出力先の出所を 1 つにする (F-8-001)。

    ``result.results_dir`` (``run_suite`` に渡した値) が唯一の出典。省略時は
    それをそのまま使う。明示的に渡された値が食い違う場合は、絶対パスの
    fallback (:func:`_relative_to` の ``ValueError`` 経路) が黙って
    ``run.json`` に混入するより先に ``ConfigError`` にする。
    """
    if output_dir is None or output_dir == result.results_dir:
        return result.results_dir
    msg = (
        "output_dir が run_suite() に渡した results_dir と一致しません "
        f"(output_dir={output_dir}, results_dir={result.results_dir})"
    )
    raise ConfigError(
        msg,
        remediation=(
            "write_run_outputs() / build_run_json() の output_dir を省略するか、"
            "run_suite() に渡した results_dir と同じ Path を渡してください"
        ),
    )


def build_run_json(
    result: RunResult, output_dir: Path | None = None
) -> dict[str, object]:
    """``run.json`` の中身 (再現条件の構造化版 + 集計値)。

    秘密は書かない。``runtime`` はマニフェスト由来で ``api_key_env``
    (環境変数「名」) しか持たない (D-05)。

    Args:
        result: :func:`harness.runner.run_suite` の結果。
        output_dir: 省略時は ``result.results_dir`` を使う。明示的に渡す値は
            それと一致していなければならない (:func:`_resolve_output_dir`)。

    Raises:
        ConfigError: ``output_dir`` が ``result.results_dir`` と食い違う場合。
    """
    target = _resolve_output_dir(result, output_dir)
    plan = result.plan
    summary = result.summary
    cases = {case.case_index: case for case in plan.cases}
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "run_id": result.run_id,
        "started_at_utc": result.started_at_utc,
        "run_fingerprint": result.fingerprint,
        # 実行開始直後に 1 回だけ測ったアイドル基準 (全モデル共通)。
        # プローブが使えない環境では None (D-23)。
        "vram_idle_mib": result.vram_idle_mib,
        "reproduction": _reproduction(plan),
        "suite": {
            "id": plan.suite_id,
            "path": str(plan.suite_path),
            # ベースファイルのハッシュ。--models/--limit の絞り込みは
            # 反映されない。実効入力は reproduction を参照 (F-8-002)。
            "sha256": plan.suite_sha256,
            "description": plan.suite.suite.description,
            "warmup_runs": plan.warmup_runs,
            "prompt_ids": list(plan.suite.prompt_ids),
            "filters": plan.filters.to_json(),
        },
        "config": {"path": str(plan.config_path), "sha256": plan.config_sha256},
        "models": [
            _model_json(model, cases[model.case_index]) for model in summary.models
        ],
        "counts": {
            "records": len(result.records),
            "warmup": summary.warmup_record_count,
            "measure": summary.measure_record_count,
            "error": summary.error_record_count,
        },
        "manifests": [_relative_to(path, target) for path in result.manifest_paths],
    }


# --------------------------------------------------------------------------
# 書き出し
# --------------------------------------------------------------------------


def write_run_outputs(result: RunResult, output_dir: Path | None = None) -> ReportPaths:
    """``report.md`` / ``records.jsonl`` / ``run.json`` を書き出す。

    ``output_dir`` は省略時 ``result.results_dir`` (:func:`harness.runner.run_suite`
    に渡した ``results_dir``) を使う。明示的に渡す場合はそれと一致していなければ
    ``ConfigError`` になる (:func:`_resolve_output_dir`, F-8-001)。値の出所を
    1 つにすることで、``_relative_to`` の絶対パス fallback (= ホストのユーザー名を
    含むパスが ``run.json`` の ``manifests[]`` に書かれる経路) を到達不能にする。

    Raises:
        ConfigError: 出力先に書けない場合、または ``output_dir`` が食い違う場合。
    """
    target = _resolve_output_dir(result, output_dir)
    paths = ReportPaths(
        directory=target,
        report=target / REPORT_FILENAME,
        records=target / RECORDS_FILENAME,
        run_json=target / RUN_JSON_FILENAME,
    )
    payload = json.dumps(
        build_run_json(result, target),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    try:
        target.mkdir(parents=True, exist_ok=True)
        paths.report.write_text(render_report(result), encoding="utf-8")
        paths.records.write_text(render_records_jsonl(result.records), encoding="utf-8")
        paths.run_json.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        msg = f"比較結果を書き出せません: {target}"
        raise ConfigError(
            msg,
            remediation=f"出力先ディレクトリの権限を確認してください ({exc.strerror})",
        ) from exc
    logger.info("比較結果を書き出しました: %s", target)
    return paths
