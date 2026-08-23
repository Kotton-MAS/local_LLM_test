"""比較ハーネスのコマンドラインインタフェース (stdlib ``argparse``)。

サブコマンドは 1 つだけ::

    python -m harness.cli run --suite suites/ja_basic.toml \
        --config configs/default.toml [--dry-run] [--models a,b] [--limit N]

- ``--dry-run`` は計画だけを出す。**HTTP を 1 バイトも発行せず exit 0**。
  ``run_fingerprint`` は計画段階で確定するため、実行前に「この条件で回す」を
  提示できる (D-20)。
- 人間向けの出力は ``sys.stdout`` / ``sys.stderr`` への書き込みで行う
  (ruff T20 が ``print()`` を機械的に禁止している)。
- ``--models`` / ``--limit`` の絞り込みは**入力そのものを変える**ため、
  ``run_fingerprint`` はフル実行と一致しない。一致させてしまうと「8 問中 1 問
  だけ回した結果」と「8 問回した結果」が同じ再現条件を名乗る。
- 絞り込みの結果が 0 件になる場合は :class:`llmkit.ConfigError` に翻訳する。
  そのまま :class:`~harness.suite.ComparisonSuite` を組み立てると、利用者に
  pydantic の内部エラー (llmkit の例外階層の外) がそのまま出る。

失敗時は :class:`llmkit.LlmkitError` の「対処方法つきメッセージ」を stderr に
出して **終了コード 1**。ケース単位の失敗は記録して次のモデルへ進み、
**全モデルが 1 件も測定できなかったときだけ** exit 1 にする。
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import httpx

from harness.gpu import NvidiaSmiProbe, VramProbe
from harness.records import ModelSummary
from harness.report import MISSING_CELL, ReportPaths, run_output_dir, write_run_outputs
from harness.runner import Clock, RunPlan, SuiteFilters, plan_run, run_suite
from harness.suite import ComparisonSuite, load_suite
from llmkit import ConfigError, LlmkitError, load_config

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_RESULTS_ROOT",
    "DEFAULT_SUITE_PATH",
    "EXIT_ERROR",
    "EXIT_OK",
    "build_parser",
    "limit_prompts",
    "main",
    "select_models",
]

DEFAULT_SUITE_PATH = Path("suites/ja_basic.toml")
DEFAULT_CONFIG_PATH = Path("configs/default.toml")
DEFAULT_RESULTS_ROOT = Path("results")

EXIT_OK = 0
EXIT_ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    """``run`` サブコマンドだけを持つパーサを組み立てる。"""
    parser = argparse.ArgumentParser(
        prog="harness",
        description="プロンプト集とモデル一覧の比較を実行して results/ に記録する",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="比較スイートを実行する")
    run.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE_PATH,
        help=f"比較スイート TOML (既定: {DEFAULT_SUITE_PATH})",
    )
    run.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"ベース設定 TOML (既定: {DEFAULT_CONFIG_PATH})",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="実行計画と run_fingerprint だけを出す (HTTP を発行しない)",
    )
    run.add_argument(
        "--models",
        default=None,
        help="対象モデルを model_id のカンマ区切りで絞り込む (既定: 全件)",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="先頭 N 問だけを使う (既定: 全問)",
    )
    return parser


# --------------------------------------------------------------------------
# 入力の絞り込み (結果が 0 件なら ConfigError に翻訳する)
# --------------------------------------------------------------------------


def select_models(suite: ComparisonSuite, spec: str) -> ComparisonSuite:
    """``--models`` の指定でスイートの ``[[models]]`` を絞り込む。

    並びはスイートの宣言順を保つ (比較表の行順が指定順で入れ替わらない)。
    未知の ``model_id`` は黙って捨てず失敗させる。捨てると「打ち間違えた
    モデルの結果が表から抜けたまま完走する」ため。

    Raises:
        ConfigError: 未知の ``model_id`` を含む場合、または結果が 0 件の場合。
    """
    requested = [name.strip() for name in spec.split(",") if name.strip()]
    known = [case.model_id for case in suite.models]
    unknown = [name for name in requested if name not in known]
    selected = tuple(case for case in suite.models if case.model_id in set(requested))
    if unknown or not selected:
        detail = (
            f"未知の model_id: {', '.join(unknown)}"
            if unknown
            else "対象が 1 件も選ばれませんでした"
        )
        msg = (
            f"--models '{spec}' からスイートの対象モデルを決められません "
            f"({detail})。スイートの model_id: {', '.join(known)}"
        )
        raise ConfigError(
            msg,
            remediation=(
                "--models にスイートの model_id をカンマ区切りで指定してください"
            ),
        )
    return dataclasses.replace(suite, models=selected)


def limit_prompts(suite: ComparisonSuite, limit: int) -> ComparisonSuite:
    """``--limit`` で先頭 N 問だけに絞り込む。

    Raises:
        ConfigError: ``limit`` が 1 未満の場合。
    """
    if limit < 1:
        msg = f"--limit は 1 以上の整数です (指定値: {limit})"
        raise ConfigError(
            msg, remediation="--limit を省略するか 1 以上の値を指定してください"
        )
    return dataclasses.replace(suite, prompts=suite.prompts[:limit])


def _load_filtered_suite(
    suite_path: Path, *, models: str | None, limit: int | None
) -> ComparisonSuite:
    suite = load_suite(suite_path)
    if models is not None:
        suite = select_models(suite, models)
    if limit is not None:
        suite = limit_prompts(suite, limit)
    return suite


# --------------------------------------------------------------------------
# 表示
# --------------------------------------------------------------------------


def _write(stream: TextIO, message: str) -> None:
    stream.write(f"{message}\n")


def _report_plan(plan: RunPlan, stdout: TextIO) -> None:
    """実行計画の要約。``--dry-run`` はこれと出力先の予定を出して終わる。"""
    _write(stdout, f"スイート        : {plan.suite_path} (id={plan.suite_id})")
    _write(stdout, f"suite_sha256    : {plan.suite_sha256}")
    _write(stdout, f"設定            : {plan.config_path}")
    _write(stdout, f"config_sha256   : {plan.config_sha256}")
    _write(stdout, f"run_fingerprint : {plan.fingerprint}")
    _write(
        stdout,
        f"プロンプト      : {len(plan.suite.prompts)} 問 / "
        f"ウォームアップ {plan.warmup_runs} 回",
    )
    _write(stdout, f"予定リクエスト数: {plan.request_count}")
    _write(stdout, "モデル:")
    for case in plan.cases:
        verdict = (
            "予算内" if case.starts_without_budget_abort else "予算超過 (スキップ)"
        )
        _write(
            stdout,
            f"  [{case.case_index}] {case.model_id} "
            f"profile={case.profile_name} served={case.spec.served_name} "
            f"ctx={case.config.generation.context_tokens} "
            f"VRAM 見積り {case.estimate.total_gib:.2f} / "
            f"予算 {case.estimate.budget_gib:.2f} GiB -> {verdict}",
        )


def _format_model_line(model: ModelSummary) -> str:
    median = model.generation_tokens_per_second.median
    speed = MISSING_CELL if median is None else f"{median:.1f} t/s"
    source = model.generation_tokens_per_second_source or MISSING_CELL
    return (
        f"  [{model.case_index}] {model.model_id} 生成速度 {speed} "
        f"(出典: {source}) 測定数 "
        f"{model.generation_tokens_per_second.n}/{model.attempted}"
    )


def _report_result(
    paths: ReportPaths, models: Sequence[ModelSummary], stdout: TextIO
) -> None:
    _write(stdout, f"出力先          : {paths.directory}")
    _write(stdout, f"レポート        : {paths.report}")
    _write(stdout, f"レコード        : {paths.records}")
    _write(stdout, f"再現条件        : {paths.run_json}")
    _write(stdout, "結果:")
    for model in models:
        _write(stdout, _format_model_line(model))


# --------------------------------------------------------------------------
# 実行
# --------------------------------------------------------------------------


def _get_path(namespace: argparse.Namespace, name: str) -> Path:
    value: object = getattr(namespace, name)
    return value if isinstance(value, Path) else Path(str(value))


def _get_optional_str(namespace: argparse.Namespace, name: str) -> str | None:
    value: object = getattr(namespace, name)
    return None if value is None else str(value)


def _get_optional_int(namespace: argparse.Namespace, name: str) -> int | None:
    value: object = getattr(namespace, name)
    return value if isinstance(value, int) else None


def _get_flag(namespace: argparse.Namespace, name: str) -> bool:
    return bool(getattr(namespace, name))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _run(
    args: argparse.Namespace,
    *,
    http_client: httpx.Client | None,
    probe: VramProbe,
    results_root: Path,
    clock: Clock,
    stdout: TextIO,
) -> int:
    suite_path = _get_path(args, "suite")
    config_path = _get_path(args, "config")
    models_filter = _get_optional_str(args, "models")
    limit_filter = _get_optional_int(args, "limit")
    suite = _load_filtered_suite(suite_path, models=models_filter, limit=limit_filter)
    filters = SuiteFilters(models=models_filter, limit=limit_filter)
    plan = plan_run(
        suite, suite_path, load_config(config_path), config_path, filters=filters
    )

    if _get_flag(args, "dry_run"):
        _write(stdout, "モード          : dry-run (HTTP を発行しません)")
        _report_plan(plan, stdout)
        _write(
            stdout,
            f"出力先 (予定)   : {results_root / plan.suite_id}/"
            f"<開始時刻>-{plan.fingerprint[:12]}/",
        )
        return EXIT_OK

    _report_plan(plan, stdout)
    moment = clock()
    output_dir = run_output_dir(
        results_root,
        suite_id=plan.suite_id,
        moment=moment,
        fingerprint=plan.fingerprint,
    )
    result = run_suite(
        plan,
        probe=probe,
        http_client=http_client,
        results_dir=output_dir,
        clock=lambda: moment,
    )
    # output_dir は run_suite(results_dir=...) に渡した値そのもの。
    # write_run_outputs は result.results_dir を出所として使うため、ここでは
    # 渡す必要が無い (渡す場合は一致していなければ ConfigError, F-8-001)。
    paths = write_run_outputs(result)
    _report_result(paths, result.summary.models, stdout)

    if any(model.succeeded > 0 for model in result.summary.models):
        return EXIT_OK
    _write(stdout, "全モデルで 1 件も測定できませんでした。")
    return EXIT_ERROR


def main(
    argv: Sequence[str] | None = None,
    *,
    http_client: httpx.Client | None = None,
    probe: VramProbe | None = None,
    results_root: Path | None = None,
    clock: Clock | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """CLI 本体。成功で 0、:class:`LlmkitError` を捕捉したら 1 を返す。

    ``http_client`` / ``probe`` / ``results_root`` / ``clock`` は
    ``llmkit.cli`` と同じ注入点で、テストが実 HTTP・実プロセス・``results/``
    への書き込みを一切起こさずに全経路を回せるようにするためにある。
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(argv)

    try:
        return _run(
            args,
            http_client=http_client,
            probe=probe if probe is not None else NvidiaSmiProbe(),
            results_root=results_root
            if results_root is not None
            else DEFAULT_RESULTS_ROOT,
            clock=clock if clock is not None else _utc_now,
            stdout=out,
        )
    except LlmkitError as exc:
        _write(err, f"エラー: {exc}")
        logger.debug("CLI が %s で終了します", type(exc).__name__, exc_info=exc)
        return EXIT_ERROR


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    sys.exit(main())
