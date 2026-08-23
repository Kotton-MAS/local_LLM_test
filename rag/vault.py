"""vault (Obsidian のノート置き場) に触れる **唯一の** モジュール。

このモジュールが背負う制約は 2 つある。

1. **書き込み API を 1 つも持たない** (§3 論点5)。``open`` の書き込みモード・
   ``write_text`` / ``write_bytes`` / ``mkdir`` / ``touch`` / ``unlink`` /
   ``rename`` / ``replace`` / ``chmod`` / ``shutil.*`` / ``os.remove`` は
   このファイルに 1 つも現れない。``tests/test_rag_vault.py`` が AST で検査する
   (D-30 の静的側)。索引処理は利用者の一次資料を読むだけであり、書き換える理由が
   1 つも無い。「バグで壊した」は不可逆な事故になる。
2. **ログ・例外に絶対パスとノート本文を出さない** (CLAUDE.md のログ出力ルール)。
   実 vault のパスは利用者名を含み、本文は個人情報そのものになり得る。外へ出す
   識別子は vault ルートからの相対パス (POSIX 表記) と件数だけにする。

選択の規則は「ホワイトリスト → ブラックリスト」の順で、
``include_globs`` に一致したものだけを候補にし、``exclude_globs`` で落とす。
候補に上がらないものは最初から読まない。
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import re
import stat
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath

from llmkit import ConfigError
from rag.settings import RagSettings

logger = logging.getLogger(__name__)

__all__ = [
    "VaultFile",
    "iter_vault_files",
    "read_note_bytes",
    "read_note_text",
]


@dataclasses.dataclass(frozen=True)
class VaultFile:
    """索引対象として選ばれた vault 内のファイル 1 件。

    Attributes:
        relpath: vault ルートからの相対パス (POSIX 表記)。外へ出す唯一の識別子。
        mtime_ns: 最終更新時刻 (ナノ秒)。差分更新の高速経路に使う (3b / D-29)。
        size: バイト数。同上。
    """

    relpath: str
    mtime_ns: int
    size: int


@functools.lru_cache(maxsize=256)
def _compile_glob(pattern: str) -> re.Pattern[str]:
    """glob パターンを相対パス (POSIX 表記) 用の正規表現に変換する。

    ``pathlib.Path.match`` は ``**`` を再帰として扱わず、``PurePath.full_match``
    は Python 3.13 以降にしか無い。``fnmatch`` は ``*`` が ``/`` を跨ぐ。
    include と exclude で意味がずれると「除外したつもりのものが索引される」ので、
    両方が同じこの実装を通る。

    規則:

    - ``*`` は ``/`` を跨がない 0 文字以上、``?`` は ``/`` 以外の 1 文字。
    - ``**`` は 0 個以上のディレクトリ階層。
    - 末尾の ``/**`` はそのディレクトリ自身にも一致する (走査の枝刈りに使う)。
    """
    segments = pattern.split("/")
    parts: list[str] = []
    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        if segment == "**":
            if is_last:
                # 末尾の ** は「以下すべて」。直前に付けた '/' を戻して
                # ディレクトリ自身にも一致させる。
                if parts and parts[-1] == "/":
                    parts.pop()
                    parts.append("(?:/.*)?")
                else:
                    parts.append(".*")
            else:
                parts.append("(?:[^/]+/)*")
            continue
        parts.append(_translate_segment(segment))
        if not is_last:
            parts.append("/")
    return re.compile(r"\A" + "".join(parts) + r"\Z")


def _translate_segment(segment: str) -> str:
    """glob の 1 セグメントを正規表現に直す (``*`` は ``/`` を跨がない)。"""
    translated: list[str] = []
    for character in segment:
        if character == "*":
            translated.append("[^/]*")
        elif character == "?":
            translated.append("[^/]")
        else:
            translated.append(re.escape(character))
    return "".join(translated)


def _matches_any(relpath: str, patterns: Iterable[str]) -> bool:
    return any(
        _compile_glob(pattern).match(relpath) is not None for pattern in patterns
    )


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _scan_vault(settings: RagSettings) -> list[VaultFile]:
    """vault を 1 回走査して、選ばれたファイルを相対パス順に返す。

    シンボリックリンクは **辿らない**。リンク先が vault の外にあれば索引が
    vault の外の内容を取り込むことになり、リンクが循環していれば走査が終わらない。
    """
    root = settings.vault_dir
    exclude_globs = settings.exclude_globs
    selected: list[VaultFile] = []
    # 末尾 /** (または **) のパターンは _is_prunable の等価性保証により、
    # 一致すれば配下のファイルにも確実に一致する。枝刈りされたディレクトリの
    # 中身は歩かないため確認しようが無く、確認する必要も無い。監視対象は
    # 「一致しても枝刈りされない = 実際にファイル単位まで見て初めて分かる」
    # 非 /** パターンだけに絞る。
    watched_patterns = tuple(
        pattern for pattern in exclude_globs if not _is_prunable(pattern)
    )
    file_hits: dict[str, int] = dict.fromkeys(watched_patterns, 0)
    directory_hits: dict[str, bool] = dict.fromkeys(watched_patterns, False)
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in subdirectories:
            dir_relpath = _relpath_of(current / name, root)
            for pattern in watched_patterns:
                if not directory_hits[pattern] and _matches_any(
                    dir_relpath, (pattern,)
                ):
                    directory_hits[pattern] = True
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if not _is_excluded_directory(current / name, root, exclude_globs)
        )
        for filename in sorted(filenames):
            candidate = current / filename
            relpath = _relpath_of(candidate, root)
            if watched_patterns and _matches_any(relpath, settings.include_globs):
                for pattern in watched_patterns:
                    if _matches_any(relpath, (pattern,)):
                        file_hits[pattern] += 1
            vault_file = _accept_file(candidate, root, settings)
            if vault_file is not None:
                selected.append(vault_file)
    selected.sort(key=lambda entry: entry.relpath)
    _warn_about_exclude_globs_matching_no_file(file_hits, directory_hits)
    return selected


def _warn_about_exclude_globs_matching_no_file(
    file_hits: dict[str, int], directory_hits: dict[str, bool]
) -> None:
    """ファイル単位で 1 件も除外しなかった除外パターンを警告する。

    ``exclude_globs`` は「一致すれば落とす」ブラックリストであり、真の判定は
    常にファイル単位 (:func:`_matches_any`) が行う (§9 T2 決定6)。この意味論
    自体は正しいが、``.gitignore`` の書き味に慣れた利用者が ``.trash`` の
    ように末尾 ``/**`` を付け忘れると、ディレクトリ自身には一致するのに配下の
    ファイルには 1 件も一致せず、**除外したつもりのファイルが黙って索引に
    残る** (round-10 レビュー)。ディレクトリにすら一致しなかったパターン
    (この vault に単に存在しないパス。既定の ``.git/**`` など) は設定ミスとは
    限らないため警告しない。末尾 ``/**`` のパターンは :func:`_is_prunable` の
    等価性保証により常に安全なので、そもそも監視対象に含めない (呼び出し元
    ``_scan_vault`` 参照)。

    相対パスもファイル名も出さない。パターン文字列と件数 0 だけを載せる
    (CLAUDE.md のログ出力ルール)。
    """
    for pattern, hits in file_hits.items():
        if hits > 0 or not directory_hits[pattern]:
            continue
        logger.warning(
            "除外パターンがディレクトリには一致しましたが、"
            "ファイル単位では 1 件も一致しませんでした (0 件): %s "
            "(サブツリー全体を除外するには '%s/**' のように書いてください)",
            pattern,
            pattern.rstrip("/"),
        )


def _relpath_of(path: Path, root: Path) -> str:
    return PurePosixPath(path.relative_to(root)).as_posix()


def _is_prunable(pattern: str) -> bool:
    """パターンがディレクトリ丸ごとの枝刈りに使って安全かを判定する。

    末尾が ``/**`` のパターン、または ``**`` そのものは、ディレクトリ自身に
    一致した時点で配下のすべてのファイルにも確実に一致する (``_compile_glob``
    の「末尾の ``/**`` はディレクトリ自身にも一致する」規則により、その正規表現
    は ``directory`` にも ``directory/任意の配下`` にも一致するため)。この形の
    パターンに限れば、枝刈りとファイル単位の除外判定 (:func:`_matches_any`) は
    結果が一致する。

    それ以外のパターン (例: ``notes/*``) はディレクトリ自身には一致しても
    配下のファイルには一致しないことがある。``notes/*`` は
    ``notes/2024`` という**ディレクトリ**には一致するが、``notes/*`` は
    ``/`` を跨がないため配下の ``notes/2024/keep.md`` には一致しない。
    このパターンを枝刈りに使うと、ファイル単位では除外対象でない
    ``notes/2024/keep.md`` までディレクトリごと消えてしまう (F-9-002)。
    """
    return pattern == "**" or pattern.endswith("/**")


def _is_excluded_directory(
    directory: Path, root: Path, exclude_globs: Iterable[str]
) -> bool:
    """走査に入る前にディレクトリごと落とせるかを判定する (枝刈り)。

    ``.git/**`` のような巨大なディレクトリを開かずに済ませるための最適化。
    枝刈りの対象は :func:`_is_prunable` が真を返すパターンに限る。それ以外の
    パターンによる除外はファイル単位の判定 (:func:`_accept_file` 内の
    ``_matches_any``) に委ね、ここでは落とさない (F-9-002: 限定しないと
    枝刈りとファイル単位の判定が等価でなくなり、除外していないファイルが
    黙って消える)。
    """
    if directory.is_symlink():
        return True
    prunable_globs = tuple(
        pattern for pattern in exclude_globs if _is_prunable(pattern)
    )
    return _matches_any(_relpath_of(directory, root), prunable_globs)


def _accept_file(
    candidate: Path, root: Path, settings: RagSettings
) -> VaultFile | None:
    """1 ファイルが索引対象かを判定し、対象なら :class:`VaultFile` を作る。"""
    if candidate.is_symlink():
        logger.debug("シンボリックリンクをスキップ: %s", _relpath_of(candidate, root))
        return None
    relpath = _relpath_of(candidate, root)
    if not _matches_any(relpath, settings.include_globs):
        return None
    if _matches_any(relpath, settings.exclude_globs):
        logger.debug("除外パターンに一致: %s", relpath)
        return None
    if not _is_within(candidate.resolve(), root):
        logger.debug("vault ルートの外を指すためスキップ: %s", relpath)
        return None
    try:
        status = candidate.stat()
    except OSError:
        logger.warning("状態を取得できないためスキップします: %s", relpath)
        return None
    if not stat.S_ISREG(status.st_mode):
        return None
    return VaultFile(relpath=relpath, mtime_ns=status.st_mtime_ns, size=status.st_size)


def iter_vault_files(settings: RagSettings) -> Iterator[VaultFile]:
    """索引対象のファイルを相対パスの昇順で列挙する。

    ``include_globs`` に一致したものだけを候補にし、``exclude_globs`` で落とす。
    シンボリックリンク、および ``resolve()`` した結果が vault ルートの配下に
    ならないものはスキップする。

    Args:
        settings: 解決済みの索引設定。

    Yields:
        選ばれたファイル 1 件ごとの :class:`VaultFile`。
    """
    selected = _scan_vault(settings)
    logger.info(
        "索引対象を選択しました: vault=%s files=%d", settings.vault_id, len(selected)
    )
    yield from selected


def _resolve_note_path(settings: RagSettings, relpath: str) -> Path:
    """相対パスを vault ルート配下の実ファイルに解決する。

    ``..`` や絶対パス、シンボリックリンク経由で vault の外に出る経路をここで塞ぐ。
    :func:`iter_vault_files` を経由しない呼び出しでも同じ保証が要る。
    """
    if (
        not relpath
        or PurePosixPath(relpath).is_absolute()
        or Path(relpath).is_absolute()
    ):
        msg = f"vault 内の相対パスではありません: {relpath}"
        raise ConfigError(
            msg, remediation="vault ルートからの相対パスを指定してください"
        )
    candidate = settings.vault_dir / PurePosixPath(relpath)
    if candidate.is_symlink() or not _is_within(
        candidate.resolve(), settings.vault_dir
    ):
        msg = f"vault ルートの外を指す参照です: {relpath}"
        raise ConfigError(
            msg, remediation="vault の内側にある実ファイルを指定してください"
        )
    return candidate


def read_note_bytes(settings: RagSettings, relpath: str) -> bytes:
    """ノートの生バイト列を読む。

    Args:
        settings: 解決済みの索引設定。
        relpath: vault ルートからの相対パス (POSIX 表記)。

    Raises:
        ConfigError: vault の外を指す参照か、読み取りに失敗した場合。
            メッセージには相対パスだけを載せ、本文と絶対パスは載せない。
    """
    path = _resolve_note_path(settings, relpath)
    try:
        return path.read_bytes()
    except OSError as exc:
        msg = f"ノートを読めません: {relpath}"
        raise ConfigError(
            msg,
            remediation=(
                f"読み取り権限とファイルの存在を確認してください ({exc.strerror})"
            ),
        ) from exc


def read_note_text(settings: RagSettings, relpath: str) -> str:
    """ノートを UTF-8 (``errors="strict"``) で復号して返す。

    復号を厳格にするのは、壊れた文字を ``�`` に置き換えると「読めた」ことに
    なってしまい、化けた本文がそのまま索引に入るため。

    Raises:
        ConfigError: vault の外を指す参照、読み取り失敗、UTF-8 として不正な場合。
    """
    raw = read_note_bytes(settings, relpath)
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        msg = f"ノートを UTF-8 として読めません: {relpath} (位置 {exc.start})"
        raise ConfigError(
            msg, remediation="ファイルを UTF-8 で保存し直してください"
        ) from exc
