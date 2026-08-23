"""コマンドラインインタフェース (stdlib ``argparse``)。

サブコマンドは 2 つだけ::

    python -m llmkit.cli doctor --config configs/default.toml
    python -m llmkit.cli chat "こんにちは" --config configs/default.toml

- 人間向けの出力は ``print()`` ではなく ``sys.stdout`` / ``sys.stderr`` への
  書き込みで行う (ruff T20 が ``print()`` を機械的に禁止している)。
- 失敗時は :class:`llmkit.errors.LlmkitError` の「対処方法つきメッセージ」を
  そのまま stderr に出し、**終了コード 1** を返す。
- ``logging.basicConfig`` は ``__main__`` ガード内でのみ呼ぶ (既存の main.py と同じ)。
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import httpx

from llmkit.bootstrap import BootstrapResult, bootstrap
from llmkit.client import ChatMessage, ChatResult
from llmkit.errors import LlmkitError

logger = logging.getLogger(__name__)

__all__ = ["build_parser", "main"]

DEFAULT_CONFIG_PATH = Path("configs/default.toml")
_PROBE_PROMPT = "ping"

EXIT_OK = 0
EXIT_ERROR = 1


def build_parser() -> argparse.ArgumentParser:
    """``doctor`` / ``chat`` の 2 サブコマンドを持つパーサを組み立てる。"""
    parser = argparse.ArgumentParser(
        prog="llmkit",
        description="設定駆動のローカル LLM 推論クライアント (Phase 1)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="設定・VRAM 見積り・ランタイム疎通をまとめて診断する"
    )
    _add_common_arguments(doctor)

    chat = subparsers.add_parser("chat", help="一問一答で 1 回だけ推論する")
    chat.add_argument("prompt", help="ユーザー発話")
    _add_common_arguments(chat)

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"TOML 設定ファイル (既定: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="使用する VRAM プロファイル (既定: vram.active_profile)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="実行マニフェストの出力先 (既定: outputs/runs)",
    )


def _get_path(namespace: argparse.Namespace, name: str) -> Path:
    value: object = getattr(namespace, name)
    return value if isinstance(value, Path) else Path(str(value))


def _get_optional_path(namespace: argparse.Namespace, name: str) -> Path | None:
    value: object = getattr(namespace, name)
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(str(value))


def _get_str(namespace: argparse.Namespace, name: str) -> str:
    value: object = getattr(namespace, name)
    return str(value)


def _get_optional_str(namespace: argparse.Namespace, name: str) -> str | None:
    value: object = getattr(namespace, name)
    return None if value is None else str(value)


def _write(stream: TextIO, message: str) -> None:
    stream.write(f"{message}\n")


def _report_startup(result: BootstrapResult, stdout: TextIO) -> None:
    """起動シーケンスの結果を人間向けに要約する。"""
    _write(stdout, f"設定       : {result.manifest.config_path}")
    _write(stdout, f"設定ハッシュ: {result.manifest.config_sha256}")
    _write(stdout, f"プロファイル: {result.profile.name}")
    _write(stdout, f"VRAM       : {result.estimate.summary()}")
    _write(
        stdout,
        f"API 経路   : {result.manifest.runtime.api_style} "
        f"(runtime.kind={result.manifest.runtime.kind})",
    )
    _write(stdout, f"接続先     : {result.endpoint_url}")
    _write(stdout, f"モデル実名 : {result.served_name}")
    if result.manifest_path is not None:
        _write(stdout, f"マニフェスト: {result.manifest_path}")


def _speed_summary(chat_result: ChatResult) -> str:
    """速度指標の表示文字列 (F-4-004 / F-5-001)。

    まず ``measured_tokens_per_second`` (欠測を ``None`` で表す) で計測不能
    かどうかを判定する。``eval_count`` 欠測などで ``None`` のときは、
    ``ChatResult.tokens_per_second`` の 0.0 をそのまま出すと「実測 0 t/s」に
    見えてしまうため、数値の代わりに計測不能である旨を出す (これが唯一の
    リポジトリ内消費者であり、この判定を通さずに ``tokens_per_second`` を
    直接表示しない)。

    計測できたときは従来どおり、``tokens_per_second`` (latency_s ベースの
    壁時計値) を主表示にし、ネイティブ経路で ``timings`` が得られれば
    ``eval_tokens_per_second`` (Ollama 内部の純粋な生成時間のみが分母) も
    併記して、両者の差から再ロードの有無を読み手が判別できるようにする。
    """
    if chat_result.measured_tokens_per_second is None:
        return "速度計測不能 (eval_count 欠測)"
    summary = f"{chat_result.tokens_per_second:.1f} t/s (latency_s ベース)"
    timings = chat_result.timings
    if timings is not None and timings.eval_tokens_per_second is not None:
        summary += f" / eval {timings.eval_tokens_per_second:.1f} t/s"
    return summary


def _run_doctor(
    args: argparse.Namespace, http_client: httpx.Client | None, stdout: TextIO
) -> int:
    result = bootstrap(
        _get_path(args, "config"),
        profile_name=_get_optional_str(args, "profile"),
        http_client=http_client,
        output_dir=_get_optional_path(args, "output_dir"),
    )
    _report_startup(result, stdout)

    _write(stdout, "疎通確認   : 推論ランタイムへ 1 回リクエストします...")
    chat_result = result.client.chat([ChatMessage(role="user", content=_PROBE_PROMPT)])
    _write(
        stdout,
        f"疎通確認 OK: model={chat_result.model} "
        f"finish_reason={chat_result.finish_reason} "
        f"{_speed_summary(chat_result)}",
    )
    return EXIT_OK


def _run_chat(
    args: argparse.Namespace, http_client: httpx.Client | None, stdout: TextIO
) -> int:
    result = bootstrap(
        _get_path(args, "config"),
        profile_name=_get_optional_str(args, "profile"),
        http_client=http_client,
        output_dir=_get_optional_path(args, "output_dir"),
    )
    chat_result = result.client.chat(
        [ChatMessage(role="user", content=_get_str(args, "prompt"))]
    )
    _write(stdout, chat_result.text)
    _write(
        stdout,
        f"--- {chat_result.usage.completion_tokens} tokens / "
        f"{chat_result.latency_s:.2f} s = "
        f"{_speed_summary(chat_result)} "
        f"(finish_reason={chat_result.finish_reason})",
    )
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    http_client: httpx.Client | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """CLI 本体。成功で 0、:class:`LlmkitError` を捕捉したら 1 を返す。"""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(argv)
    command = _get_str(args, "command")

    try:
        if command == "doctor":
            return _run_doctor(args, http_client, out)
        return _run_chat(args, http_client, out)
    except LlmkitError as exc:
        _write(err, f"エラー: {exc}")
        logger.debug("CLI が %s で終了します", type(exc).__name__, exc_info=exc)
        return EXIT_ERROR


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    sys.exit(main())
