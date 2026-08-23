"""読み取り専用ローダ (``rag/vault.py``) の検証。

**vault を 1 バイトも書き換えない**ことを 2 つの層で固定する (§3 論点5):

- 静的: ``test_no_vault_module_calls_a_write_api`` (D-30 guard / AST)
- 動的: ``test_indexing_leaves_every_vault_file_byte_identical`` (D-30 guard)

動的側は本サイクル時点ではまだ索引本体が無いため、**vault 全体をパースする
処理**の前後でスナップショットを比較する。3b で索引本体に差し替える。
スナップショットは ``(相対パス, size, st_mtime_ns, st_mode, sha256)`` の集合で、
エントリの増減も見る。``st_atime`` は読み取りで必ず変わるため比較しない。
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import stat
from pathlib import Path

import pytest
from conftest import write_rag_settings

import rag
from llmkit import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent
VAULT_MODULE = REPO_ROOT / "rag" / "vault.py"

#: vault を書き換え得る API。1 つでも現れたら D-30 が崩れる。
_WRITE_ATTRIBUTE_NAMES = frozenset(
    {
        "write_text",
        "write_bytes",
        "writelines",
        "write",
        "mkdir",
        "makedirs",
        "touch",
        "unlink",
        "remove",
        "rmdir",
        "rmtree",
        "rename",
        "replace",
        "chmod",
        "chown",
        "symlink_to",
        "hardlink_to",
        "truncate",
        "utime",
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "move",
    }
)

#: ``open()`` に許す読み取り専用モード。
_READ_ONLY_MODES = frozenset({"r", "rb", "rt", "br", "tr"})

#: 書き込み系ヘルパを持ち込むだけで検査をすり抜けられる import。
_WRITE_CAPABLE_MODULES = frozenset({"shutil", "tempfile"})

#: ``(size, st_mtime_ns, st_mode, sha256)``。``st_atime`` は含めない。
type Fingerprint = tuple[int, int, int, str]


def snapshot_tree(root: Path) -> dict[str, Fingerprint]:
    """vault 配下の全エントリの指紋を採る (``st_atime`` は含めない)。"""
    entries: dict[str, Fingerprint] = {}
    for path in sorted(root.rglob("*")):
        status = path.lstat()
        digest = ""
        if stat.S_ISREG(status.st_mode) and not path.is_symlink():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries[path.relative_to(root).as_posix()] = (
            status.st_size,
            status.st_mtime_ns,
            status.st_mode,
            digest,
        )
    return entries


def parse_whole_vault(settings: rag.RagSettings) -> list[rag.ParsedNote]:
    """索引処理が vault に対して行う読み取りを一通り行う。

    3b では ``build_index`` に差し替える。読み取り経路 (列挙 → バイト列 →
    テキスト → パース) をすべて通すことが目的。
    """
    parsed: list[rag.ParsedNote] = []
    for entry in rag.iter_vault_files(settings):
        raw = rag.read_note_bytes(settings, entry.relpath)
        text = rag.read_note_text(settings, entry.relpath)
        assert raw.decode("utf-8") == text
        parsed.append(rag.parse_note(entry.relpath, text))
    return parsed


# --------------------------------------------------------------------------
# D-30 guard (静的)
# --------------------------------------------------------------------------


def test_no_vault_module_calls_a_write_api() -> None:
    """D-30 guard (静的): ``rag/vault.py`` に書き込み API が 1 つも無い。

    AST で見る (文字列一致だと docstring に並べた API 名を違反と誤検出する)。
    """
    tree = ast.parse(VAULT_MODULE.read_text(encoding="utf-8"))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [
                f"import {alias.name}"
                for alias in node.names
                if alias.name.split(".")[0] in _WRITE_CAPABLE_MODULES
            ]
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.split(".")[0] in _WRITE_CAPABLE_MODULES
        ):
            offenders.append(f"from {node.module} import ...")
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _WRITE_ATTRIBUTE_NAMES
        ):
            offenders.append(f".{node.func.attr}()")
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = _open_mode(node)
            if mode is None or mode not in _READ_ONLY_MODES:
                offenders.append(f"open() モード={mode!r}")

    assert not offenders, f"rag/vault.py の書き込み経路: {offenders}"


def _open_mode(node: ast.Call) -> str | None:
    """``open()`` 呼び出しのモード引数 (リテラルでなければ ``None``)。"""
    mode_node: ast.expr | None = node.args[1] if len(node.args) > 1 else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    if mode_node is None:
        return "r"
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return mode_node.value
    return None


def test_the_write_api_guard_actually_detects_a_write() -> None:
    """上の検査が「何を見ても通る」ものになっていないことの確認 (変異検証の縮小版)。

    ``rag/vault.py`` に 1 行足した状態を模した木を同じ検査に通す。
    """
    mutated = ast.parse(
        VAULT_MODULE.read_text(encoding="utf-8") + '\nPath("x").write_text("y")\n'
    )
    offenders = [
        f".{node.func.attr}()"
        for node in ast.walk(mutated)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _WRITE_ATTRIBUTE_NAMES
    ]

    assert offenders == [".write_text()"]


# --------------------------------------------------------------------------
# D-30 guard (動的)
# --------------------------------------------------------------------------


def test_indexing_leaves_every_vault_file_byte_identical(
    sample_vault_copy: Path,
) -> None:
    """D-30 guard (動的): 読み取り処理の前後で vault が完全に一致する。"""
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))
    before = snapshot_tree(sample_vault_copy)

    parsed = parse_whole_vault(settings)

    after = snapshot_tree(sample_vault_copy)
    assert len(parsed) == 11
    assert before == after
    assert set(before) >= {
        "notes",
        "notes/project-alpha.md",
        ".obsidian/app.json",
        ".trash/deleted.md",
        "attachments/pixel.png",
    }
    # 索引の出力先は vault の外にあるため、vault の中に何も増えていない。
    assert not settings.index_dir.is_relative_to(settings.vault_dir)


def test_the_snapshot_detects_a_change(sample_vault_copy: Path) -> None:
    """上のスナップショット比較が変更を検出できることの確認。"""
    before = snapshot_tree(sample_vault_copy)
    (sample_vault_copy / "notes" / "no-heading.md").write_text("x", encoding="utf-8")
    changed = snapshot_tree(sample_vault_copy)
    (sample_vault_copy / "notes" / "added.md").write_text("y", encoding="utf-8")
    added = snapshot_tree(sample_vault_copy)

    assert before != changed, "内容の変更を検出できていません"
    assert set(changed) != set(added), "エントリの増加を検出できていません"


# --------------------------------------------------------------------------
# 選択規則 (L311 / E26)
# --------------------------------------------------------------------------


def test_iter_vault_files_selects_only_notes(sample_vault_copy: Path) -> None:
    """L311: ``.obsidian/`` / 添付 / ``.trash/`` / ``*.excalidraw.md`` を索引しない。"""
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))

    selected = [entry.relpath for entry in rag.iter_vault_files(settings)]

    assert len(selected) == 11
    assert all(relpath.startswith("notes/") for relpath in selected)
    assert selected == sorted(selected), "相対パスの昇順で返らない"
    assert not [path for path in selected if path.startswith(".obsidian/")]
    assert not [path for path in selected if path.startswith("attachments/")]
    assert not [path for path in selected if path.startswith(".trash/")]
    assert not [path for path in selected if path.endswith(".excalidraw.md")]
    assert "notes/日本語 ファイル名.md" in selected
    assert "notes/empty.md" in selected


def test_include_globs_are_a_whitelist(sample_vault_copy: Path) -> None:
    """ホワイトリストに一致しないものは除外パターンに関係なく拾わない。"""
    settings = rag.load_settings(
        write_rag_settings(sample_vault_copy.parent, include_globs=["**/*.canvas"])
    )

    selected = [entry.relpath for entry in rag.iter_vault_files(settings)]

    assert selected == ["attachments/board.canvas"]


def test_exclude_globs_change_the_selected_files(sample_vault_copy: Path) -> None:
    """E26: ``exclude_globs`` に 1 パターン足すと対象件数が減る。"""
    directory = sample_vault_copy.parent
    baseline = rag.load_settings(write_rag_settings(directory, name="base.toml"))
    tightened = rag.load_settings(
        write_rag_settings(
            directory,
            name="tight.toml",
            exclude_globs=[
                *rag.DEFAULT_EXCLUDE_GLOBS,
                "notes/frontmatter-*.md",
            ],
        )
    )

    before = {entry.relpath for entry in rag.iter_vault_files(baseline)}
    after = {entry.relpath for entry in rag.iter_vault_files(tightened)}

    assert len(before) == 11
    assert len(after) == 9
    assert before - after == {
        "notes/frontmatter-rich.md",
        "notes/frontmatter-broken.md",
    }


@pytest.mark.parametrize(
    ("exclude_globs", "expected_selected"),
    [
        (["notes/*"], {"notes/2024/keep.md", "top.md"}),
        (["notes"], {"notes/draft.md", "notes/2024/keep.md", "top.md"}),
        (["**/2024"], {"notes/draft.md", "notes/2024/keep.md", "top.md"}),
    ],
    ids=["notes-star", "notes-exact", "double-star-2024"],
)
def test_directory_pruning_never_drops_files_that_file_level_matching_keeps(
    tmp_path: Path, exclude_globs: list[str], expected_selected: set[str]
) -> None:
    """F-9-002 回帰: 末尾が ``/**`` でない ``exclude_globs`` は枝刈りに使わない。

    ``notes/*`` はディレクトリ ``notes/2024`` 自身には一致するが、``*`` は
    ``/`` を跨がないため配下の ``notes/2024/keep.md`` には一致しない。枝刈りを
    ファイル単位の判定と区別せずに使うと、除外対象でないファイルがディレクトリ
    ごと黙って消える。掃引には末尾 ``/**`` でないパターン (``notes/*`` /
    ``notes`` / ``**/2024``) を含める。
    """
    vault_dir = tmp_path / "vault"
    (vault_dir / "notes" / "2024").mkdir(parents=True)
    (vault_dir / "notes" / "draft.md").write_text("draft", encoding="utf-8")
    (vault_dir / "notes" / "2024" / "keep.md").write_text("keep", encoding="utf-8")
    (vault_dir / "top.md").write_text("top", encoding="utf-8")

    settings = rag.load_settings(
        write_rag_settings(tmp_path, vault_dir="vault", exclude_globs=exclude_globs)
    )

    selected = {entry.relpath for entry in rag.iter_vault_files(settings)}

    assert selected == expected_selected


@pytest.mark.parametrize(
    ("exclude_globs", "expect_warning"),
    [([".trash"], True), ([".trash/**"], False)],
    ids=["missing-slash-star-warns", "slash-star-suffix-is-silent"],
)
def test_exclude_globs_that_exclude_no_file_are_warned_about(
    sample_vault_copy: Path,
    caplog: pytest.LogCaptureFixture,
    exclude_globs: list[str],
    expect_warning: bool,
) -> None:
    """round-10 レビュー: 除外し損ねを利用者に伝える (F-9-002 の裏返し)。

    ``.trash`` (末尾 ``/**`` 無し) はディレクトリ ``.trash`` 自身には一致する
    のに、ファイル単位の判定 (真の除外条件) では ``.trash/deleted.md`` に
    一致せず、除外したつもりのファイルが黙って索引に残る。``.trash/**`` へ
    書き直すと同じディレクトリが正しく除外され、警告は出ない。
    """
    settings = rag.load_settings(
        write_rag_settings(sample_vault_copy.parent, exclude_globs=exclude_globs)
    )

    with caplog.at_level(logging.WARNING, logger="rag.vault"):
        selected = {entry.relpath for entry in rag.iter_vault_files(settings)}

    warnings = [record.getMessage() for record in caplog.records]
    matching = [message for message in warnings if ".trash" in message]

    if expect_warning:
        assert matching, warnings
        assert any(".trash" in message for message in matching)
        assert any("/**" in message for message in matching)  # dir/** の対処
        # 相対パスもファイル本文も出さない (パターン文字列と件数だけ)。
        assert all("deleted.md" not in message for message in warnings)
        assert all(str(sample_vault_copy) not in message for message in warnings)
        # 意味論自体は変えない: .trash/deleted.md は依然として除外されない。
        assert ".trash/deleted.md" in selected
    else:
        assert not matching, matching
        assert ".trash/deleted.md" not in selected


def test_the_sample_vault_defaults_never_trigger_the_exclude_warning(
    sample_vault_copy: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """既定の ``exclude_globs`` (すべて末尾 ``/**``) では警告が出ない。"""
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))

    with caplog.at_level(logging.WARNING, logger="rag.vault"):
        list(rag.iter_vault_files(settings))

    assert caplog.records == []


def test_vault_file_carries_the_change_detection_inputs(
    sample_vault_copy: Path,
) -> None:
    """``mtime_ns`` / ``size`` が実ファイルの値と一致する (3b の高速経路の入力)。"""
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))

    entries = {entry.relpath: entry for entry in rag.iter_vault_files(settings)}

    target = sample_vault_copy / "notes" / "project-alpha.md"
    status = target.stat()
    assert entries["notes/project-alpha.md"].size == status.st_size
    assert entries["notes/project-alpha.md"].mtime_ns == status.st_mtime_ns
    assert entries["notes/empty.md"].size == 0


# --------------------------------------------------------------------------
# シンボリックリンク・権限・vault の外
# --------------------------------------------------------------------------


def test_symlinks_are_never_followed(sample_vault_copy: Path, tmp_path: Path) -> None:
    """vault の外を指すリンクは対象にならない (中身も取り込まない)。"""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.md").write_text("# 外部のノート\n", encoding="utf-8")
    (sample_vault_copy / "notes" / "linked.md").symlink_to(outside_dir / "secret.md")
    (sample_vault_copy / "linked-dir").symlink_to(outside_dir, target_is_directory=True)
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))

    selected = [entry.relpath for entry in rag.iter_vault_files(settings)]

    assert "notes/linked.md" not in selected
    assert "linked-dir/secret.md" not in selected
    assert len(selected) == 11


def test_reading_through_a_symlink_is_refused(
    sample_vault_copy: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("# 外部\n", encoding="utf-8")
    (sample_vault_copy / "notes" / "linked.md").symlink_to(outside)
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))

    with pytest.raises(ConfigError) as excinfo:
        rag.read_note_text(settings, "notes/linked.md")

    assert "vault ルートの外" in str(excinfo.value)


@pytest.mark.parametrize(
    "relpath", ["../outside.md", "notes/../../outside.md", "/etc/hostname", ""]
)
def test_paths_outside_the_vault_are_refused(
    sample_vault_copy: Path, relpath: str
) -> None:
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))

    with pytest.raises(ConfigError):
        rag.read_note_bytes(settings, relpath)


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root は権限を無視するため 0o555 の検査が恒真になる"
)
def test_a_read_only_vault_can_still_be_indexed(sample_vault_copy: Path) -> None:
    """書き込み権限を落とした vault でも読み取りが完走する。"""
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))
    _chmod_tree(sample_vault_copy, dir_mode=0o555, file_mode=0o444)
    try:
        parsed = parse_whole_vault(settings)
        assert len(parsed) == 11
        with pytest.raises(OSError):
            (sample_vault_copy / "notes" / "blocked.md").write_text("x")
    finally:
        _chmod_tree(sample_vault_copy, dir_mode=0o755, file_mode=0o644)


def _chmod_tree(root: Path, *, dir_mode: int, file_mode: int) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(dir_mode if path.is_dir() else file_mode)
    root.chmod(dir_mode)


def test_a_file_that_disappears_between_listing_and_stat_is_skipped(
    sample_vault_copy: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F-9-028: ``stat()`` の ``OSError`` は列挙全体を止めず、その 1 件だけ

    スキップして WARNING を残す。ファイル列挙後に対象が消える・権限で弾かれる
    等、実運用で起き得る経路 (``rag/vault.py`` の未検査分岐)。
    """
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))
    target = sample_vault_copy / "notes" / "project-alpha.md"
    real_stat = Path.stat

    def flaky_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        # is_symlink() は内部で lstat() (follow_symlinks=False) を呼ぶため、
        # そちらは通常どおり動かし、_accept_file が直接呼ぶ stat() (既定の
        # follow_symlinks=True) だけを失敗させる。
        if self == target and follow_symlinks:
            raise OSError("stat 失敗 (テスト用)")
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    with caplog.at_level(logging.WARNING, logger="rag.vault"):
        selected = [entry.relpath for entry in rag.iter_vault_files(settings)]

    assert "notes/project-alpha.md" not in selected
    assert len(selected) == 10
    assert any(
        "notes/project-alpha.md" in record.getMessage() for record in caplog.records
    )


def test_missing_note_raises_a_config_error_naming_only_the_relative_path(
    sample_vault_copy: Path,
) -> None:
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))

    with pytest.raises(ConfigError) as excinfo:
        rag.read_note_bytes(settings, "notes/absent.md")

    message = str(excinfo.value)
    assert "notes/absent.md" in message
    assert str(sample_vault_copy) not in message


def test_non_utf8_notes_are_refused_without_leaking_content(
    sample_vault_copy: Path,
) -> None:
    """壊れた文字を置換して「読めた」ことにしない。本文も例外に載せない。"""
    (sample_vault_copy / "notes" / "cp932.md").write_bytes(
        "# 見出し\n秘密の本文\n".encode("cp932")
    )
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))

    with pytest.raises(ConfigError) as excinfo:
        rag.read_note_text(settings, "notes/cp932.md")

    message = str(excinfo.value)
    assert "notes/cp932.md" in message
    assert "UTF-8" in message
    assert "秘密" not in message


# --------------------------------------------------------------------------
# ログ (CLAUDE.md のログ出力ルール)
# --------------------------------------------------------------------------


def test_logs_never_contain_absolute_paths_or_note_text(
    sample_vault_copy: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """ログに出るのは相対パスと件数だけ。"""
    settings = rag.load_settings(write_rag_settings(sample_vault_copy.parent))
    (sample_vault_copy / "notes" / "linked.md").symlink_to(
        sample_vault_copy / "notes" / "no-heading.md"
    )

    with caplog.at_level(logging.DEBUG, logger="rag.vault"):
        parsed = parse_whole_vault(settings)

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert log_text, "ログが 1 行も出ていません"
    assert str(sample_vault_copy) not in log_text
    assert "/home/" not in log_text
    for note in parsed:
        for line in note.body.splitlines():
            if len(line.strip()) > 8:
                assert line.strip() not in log_text
    assert "files=11" in log_text
