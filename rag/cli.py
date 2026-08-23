"""RAG 索引のコマンドラインインタフェース (stdlib ``argparse``)。

サブコマンドは 2 つだけ::

    python -m rag.cli index  --settings vaults/sample.toml \
        [--config configs/default.toml] [--dry-run] [--rebuild]
    python -m rag.cli status --settings vaults/sample.toml \
        [--config configs/default.toml]

### ``--settings`` が vault を与える唯一の入口

``--vault <dir>`` は**作らない**。ディレクトリだけを渡されても ``vault_id`` と
索引の出力先が決まらず、決めるには設定ファイルとコマンドラインの 2 か所に
出典ができる (D-27 の趣旨)。索引の再現条件は設定ファイルの内容だけで閉じる。

### 画面に出すのは件数と fingerprint だけ

相対パスも出さない。実 vault を索引した結果をそのまま報告に貼れる状態を保つ
ためで、パスを 1 つでも出すと「この出力は貼ってよいか」を毎回人間が判断する
ことになる (判断させれば必ず漏れる)。索引ディレクトリの位置は設定ファイルの
``index.dir`` にあり、画面に出すのはその**直下のファイル名**だけにする
(``rag/indexer.py`` の例外メッセージが ``path.name`` だけを載せるのと同じ扱い)。

### ``--dry-run`` は HTTP を 1 バイトも出さない

計画 (:func:`rag.plan_index`) は vault の読み取りと sha256 だけで完結し、
埋め込みクライアントを 1 度も組み立てない。「出さないように気をつける」のでは
なく、**出す道具をその経路に持ち込まない**ことで保証する
(``harness/cli.py`` の ``--dry-run`` と同じ形)。

### ``--rebuild`` はマニフェストを読まない

全再構築の入口は :func:`rag.build_index` の ``manifest=None`` 1 つだけである
(§9 T6 決定1)。フラグを別に足すと「全再構築か」の判定が
:func:`rag.plan_index` (fingerprint 不一致) と実行本体 (フラグ) の 2 か所に
分かれる。``--rebuild`` は ``load_manifest`` を**呼ばない**ことで、既存の 1 つの
入口をそのまま使う。

人間向けの出力は ``print()`` ではなく ``sys.stdout`` / ``sys.stderr`` への
書き込みで行う (ruff T20 が ``print()`` を機械的に禁止している)。失敗時は
:class:`llmkit.LlmkitError` の「対処方法つきメッセージ」を stderr に出して
**終了コード 1**。1 ノート単位の失敗は記録して次のノートへ進み、**1 ノートも
索引できなかったときだけ** exit 1 にする (``harness/cli.py`` と同じ方針)。
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from llmkit import (
    AppConfig,
    EmbeddingClient,
    LlmkitError,
    create_embedding_client,
    load_config,
)
from rag.indexer import (
    CHUNKS_FILENAME,
    MANIFEST_FILENAME,
    IndexManifest,
    IndexPlan,
    IndexResult,
    build_index,
    chunks_path,
    load_manifest,
    plan_index,
)
from rag.settings import RagSettings, load_settings
from rag.store import JsonlVectorStore, VectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "EXIT_ERROR",
    "EXIT_OK",
    "build_parser",
    "main",
]

DEFAULT_CONFIG_PATH = Path("configs/default.toml")

EXIT_OK = 0
EXIT_ERROR = 1

#: 「未構築」の表示。``status`` はこれを出して exit 0 で終わる (索引がまだ
#: 無いことは異常ではなく、``index`` を回せば解消する状態であるため)。
_NOT_BUILT = "未構築"

#: 欠測の表示 (``harness/report.py`` の ``MISSING_CELL`` と同じ字)。埋め込みを
#: 1 度も行わず既存のマニフェストも無い実行では次元が分からないので、**0 で
#: 埋めずに**この字を出す (D-07)。0 を出すと「0 次元のベクトルを作った」と
#: 読める嘘になる。
_MISSING = "—"


def build_parser() -> argparse.ArgumentParser:
    """``index`` / ``status`` の 2 サブコマンドを持つパーサを組み立てる。"""
    parser = argparse.ArgumentParser(
        prog="rag",
        description="設定 TOML で指定した vault を埋め込んで索引に永続化する",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="索引を作成・更新する")
    _add_common_arguments(index)
    index.add_argument(
        "--dry-run",
        action="store_true",
        help="計画と index_fingerprint だけを出す (HTTP を発行しない)",
    )
    index.add_argument(
        "--rebuild",
        action="store_true",
        help="既存のマニフェストを使わず全ノートを埋め込み直す",
    )

    status = subparsers.add_parser("status", help="索引の状態を表示する")
    _add_common_arguments(status)

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """2 サブコマンドに共通の引数 (vault の出典と推論設定)。"""
    parser.add_argument(
        "--settings",
        type=Path,
        required=True,
        help="索引設定 TOML (vault の場所・索引の出力先・分割条件の唯一の出典)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"推論設定 TOML (既定: {DEFAULT_CONFIG_PATH})",
    )


# --------------------------------------------------------------------------
# 表示 (件数と fingerprint だけ。パスは 1 つも出さない)
# --------------------------------------------------------------------------


def _write(stream: TextIO, message: str) -> None:
    stream.write(f"{message}\n")


def _yes_no(value: bool) -> str:
    if value:
        return "はい"
    return "いいえ"


def _report_header(settings: RagSettings, plan: IndexPlan, stdout: TextIO) -> None:
    """どの vault を、いまの設定でどう見ているか。``index`` の 2 経路と共有する。"""
    _write(stdout, f"vault            : {settings.vault_id}")
    _write(stdout, f"index_fingerprint: {plan.fingerprint}")


def _report_plan(plan: IndexPlan, stdout: TextIO) -> None:
    """計画の内訳。``--dry-run`` と ``status`` が共有する。"""
    _write(stdout, f"全再構築         : {_yes_no(plan.full_rebuild)}")
    _write(stdout, f"対象ノート       : {plan.total_notes} 件")
    _write(stdout, f"  新規           : {len(plan.new)} 件")
    _write(stdout, f"  変更           : {len(plan.changed)} 件")
    _write(stdout, f"  変更なし       : {len(plan.unchanged)} 件")
    _write(stdout, f"  削除           : {len(plan.deleted)} 件")
    _write(stdout, f"再処理           : {len(plan.pending)} 件")


def _report_destination(stdout: TextIO) -> None:
    """出力先の**ファイル名**だけを出す (ディレクトリは画面に出さない)。"""
    _write(
        stdout,
        f"出力先 (予定)    : index.dir 直下の {MANIFEST_FILENAME} / {CHUNKS_FILENAME}",
    )


def _report_result(settings: RagSettings, result: IndexResult, stdout: TextIO) -> None:
    """実行結果の要約。:class:`rag.IndexResult` は件数しか持たない。"""
    dimensions = _MISSING if result.dimensions is None else f"{result.dimensions}"
    _write(stdout, f"vault            : {settings.vault_id}")
    _write(stdout, f"index_fingerprint: {result.fingerprint}")
    _write(stdout, f"索引したノート   : {result.indexed_notes} 件")
    _write(stdout, f"再処理しない     : {result.skipped_notes} 件")
    _write(stdout, f"失敗したノート   : {result.failed_notes} 件")
    _write(stdout, f"削除したノート   : {result.deleted_notes} 件")
    _write(stdout, f"埋め込みチャンク : {result.embedded_chunks} 件")
    _write(stdout, f"リクエスト       : {result.request_count} 回")
    _write(stdout, f"次元             : {dimensions}")
    _write(stdout, f"所要             : {result.elapsed_s:.3f} 秒")


def _report_manifest(
    manifest: IndexManifest | None, plan: IndexPlan, stdout: TextIO
) -> None:
    """既存の索引の状態。まだ無ければ「未構築」とだけ出す。

    「まだ索引が無い」ことは異常ではなく ``index`` を回せば解消する状態なので、
    ここで失敗にはしない (:func:`_status` は exit 0 を返す)。
    """
    if manifest is None:
        _write(stdout, f"索引             : {_NOT_BUILT}")
        return
    matches = manifest.index_fingerprint == plan.fingerprint
    _write(stdout, "索引             : 構築済み")
    _write(stdout, f"fingerprint 一致 : {_yes_no(matches)}")
    _write(stdout, f"索引済みノート   : {manifest.totals.notes} 件")
    _write(stdout, f"索引済みチャンク : {manifest.totals.chunks} 件")
    _write(stdout, f"次元             : {manifest.embedding.dimensions}")


# --------------------------------------------------------------------------
# 実行
# --------------------------------------------------------------------------


def _get_path(namespace: argparse.Namespace, name: str) -> Path:
    value: object = getattr(namespace, name)
    return value if isinstance(value, Path) else Path(str(value))


def _get_flag(namespace: argparse.Namespace, name: str) -> bool:
    return bool(getattr(namespace, name))


def _get_command(namespace: argparse.Namespace) -> str:
    return str(namespace.command)


def _index_mode(rebuild: bool) -> str:
    """``index`` の実行モード表示 (``--rebuild`` を明示する)。"""
    if rebuild:
        return "索引 (全再構築)"
    return "索引 (差分更新)"


def _open_store(settings: RagSettings) -> JsonlVectorStore:
    """既存の索引を読む (まだ無ければ空のストアになる。**書き込みはしない**)。"""
    return JsonlVectorStore(chunks_path(settings))


def _resolve_client(
    config: AppConfig, embedding_client: EmbeddingClient | None
) -> EmbeddingClient:
    """埋め込みクライアント。注入が無ければ ``llmkit`` に組み立てさせる。

    ``rag`` は推論ランタイムに直接触れないため (D-25)、HTTP クライアントを
    受け取る口も持たない。テストは ``llmkit.create_embedding_client`` に
    ``httpx.MockTransport`` を注入したものをそのまま渡す (§9 T7 決定2)。
    """
    if embedding_client is not None:
        return embedding_client
    return create_embedding_client(config)


def _index(
    args: argparse.Namespace,
    *,
    embedding_client: EmbeddingClient | None,
    stdout: TextIO,
) -> int:
    settings = load_settings(_get_path(args, "settings"))
    config = load_config(_get_path(args, "config"))
    store = _open_store(settings)
    rebuild = _get_flag(args, "rebuild")
    # --rebuild は「既存のマニフェストを読まない」で表現する。全再構築の入口を
    # build_index(manifest=None) の 1 つに保つため (§9 T6 決定1)。
    manifest = None if rebuild else load_manifest(settings)

    if _get_flag(args, "dry_run"):
        return _dry_run(settings, config, store=store, manifest=manifest, stdout=stdout)

    _write(stdout, f"モード           : {_index_mode(rebuild)}")
    result = build_index(
        settings,
        config,
        embedding_client=_resolve_client(config, embedding_client),
        store=store,
        manifest=manifest,
    )
    _report_result(settings, result, stdout)
    if result.indexed_notes > 0 or result.failed_notes == 0:
        return EXIT_OK
    _write(stdout, "1 件のノートも索引できませんでした。")
    return EXIT_ERROR


def _dry_run(
    settings: RagSettings,
    config: AppConfig,
    *,
    store: VectorStore,
    manifest: IndexManifest | None,
    stdout: TextIO,
) -> int:
    """計画だけを出して終わる。**HTTP 0 回・書き込み 0 バイト・exit 0**。"""
    plan = plan_index(settings, config, store=store, manifest=manifest)
    _write(stdout, "モード           : dry-run (HTTP を発行しません)")
    _report_header(settings, plan, stdout)
    _report_plan(plan, stdout)
    _report_destination(stdout)
    return EXIT_OK


def _status(args: argparse.Namespace, *, stdout: TextIO) -> int:
    """索引の状態を出す。まだ無くても異常ではないので exit 0 を返す。"""
    settings = load_settings(_get_path(args, "settings"))
    config = load_config(_get_path(args, "config"))
    store = _open_store(settings)
    manifest = load_manifest(settings)
    plan = plan_index(settings, config, store=store, manifest=manifest)
    _report_header(settings, plan, stdout)
    _report_manifest(manifest, plan, stdout)
    _report_plan(plan, stdout)
    return EXIT_OK


def _run(
    args: argparse.Namespace,
    *,
    embedding_client: EmbeddingClient | None,
    stdout: TextIO,
) -> int:
    if _get_command(args) == "status":
        return _status(args, stdout=stdout)
    return _index(args, embedding_client=embedding_client, stdout=stdout)


def main(
    argv: Sequence[str] | None = None,
    *,
    embedding_client: EmbeddingClient | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """CLI 本体。成功で 0、:class:`LlmkitError` を捕捉したら 1 を返す。

    ``embedding_client`` / ``stdout`` / ``stderr`` は ``harness/cli.py`` と同じ
    注入点で、テストが実 HTTP も実 stdout への書き込みも起こさずに全経路を
    回せるようにするためにある。``harness/cli.py`` が持つ ``http_client`` を
    ここには置かない (§9 T7 決定2: ``rag`` は ``httpx`` を import しない)。
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(argv)

    try:
        return _run(args, embedding_client=embedding_client, stdout=out)
    except LlmkitError as exc:
        _write(err, f"エラー: {exc}")
        logger.debug("CLI が %s で終了します", type(exc).__name__, exc_info=exc)
        return EXIT_ERROR


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    sys.exit(main())
