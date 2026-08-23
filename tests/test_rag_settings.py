"""索引設定 (``rag/settings.py``) の検証。

固定する性質:

- 相対パスは**設定ファイルからの相対**として解決する (起動場所に依存しない)。
- **vault の場所の出典は ``vault.dir`` ただ 1 つ** (E30 guard)。
- ``index.dir`` が ``vault.dir`` 配下に解決される設定は拒否する (§3 論点5 の設定層)。
- 例外は ``llmkit.ConfigError`` で、メッセージには**設定に書かれたままの文字列**と
  対処が入る (解決後の絶対パスを載せない)。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from conftest import SAMPLE_VAULT_CONFIG, SAMPLE_VAULT_DIR, write_rag_settings

import rag
from llmkit import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_settings_reads_every_declared_field(tmp_path: Path) -> None:
    """宣言したキーがすべて設定ファイルから届く (既定値で埋まっていない)。"""
    shutil.copytree(SAMPLE_VAULT_DIR, tmp_path / "vault")
    settings_path = tmp_path / "rag.toml"
    settings_path.write_text(
        "\n".join(
            [
                "[vault]",
                'id = "custom-id_1"',
                'dir = "vault"',
                'include_globs = ["notes/*.md"]',
                'exclude_globs = ["notes/empty.md"]',
                "",
                "[index]",
                'dir = "out/index"',
                "",
                "[chunk]",
                "max_tokens = 80",
                "cjk_chars_per_token = 2.0",
                "ascii_chars_per_token = 3.5",
                'heading_separator = " / "',
                "",
                "[embed]",
                "batch_size = 4",
                "",
            ]
        ),
        encoding="utf-8",
    )

    settings = rag.load_settings(settings_path)

    assert settings.vault_id == "custom-id_1"
    assert settings.vault_dir == (tmp_path / "vault").resolve()
    assert settings.index_dir == (tmp_path / "out" / "index").resolve()
    assert settings.include_globs == ("notes/*.md",)
    assert settings.exclude_globs == ("notes/empty.md",)
    assert settings.chunk.max_tokens == 80
    assert settings.chunk.cjk_chars_per_token == 2.0
    assert settings.chunk.ascii_chars_per_token == 3.5
    assert settings.chunk.heading_separator == " / "
    assert settings.embed.batch_size == 4
    assert settings.source_path == settings_path


def test_defaults_cover_every_optional_section(sample_vault_copy: Path) -> None:
    """``[vault]`` だけで読め、既定値が仕様書 §4 T2 の表と一致する。"""
    settings_path = sample_vault_copy.parent / "minimal.toml"
    settings_path.write_text(
        '[vault]\nid = "sample"\ndir = "vault"\n', encoding="utf-8"
    )

    settings = rag.load_settings(settings_path)

    assert settings.include_globs == rag.DEFAULT_INCLUDE_GLOBS == ("**/*.md",)
    assert settings.exclude_globs == rag.DEFAULT_EXCLUDE_GLOBS
    assert settings.exclude_globs == (
        ".obsidian/**",
        ".trash/**",
        ".git/**",
        "**/*.excalidraw.md",
    )
    assert settings.chunk.max_tokens == 240
    assert settings.chunk.cjk_chars_per_token == 1.0
    assert settings.chunk.ascii_chars_per_token == 4.0
    assert settings.chunk.heading_separator == " > "
    assert settings.embed.batch_size == 16
    # index.dir 既定は data/index/<vault.id> (設定ファイルからの相対)。
    assert (
        settings.index_dir == (sample_vault_copy.parent / "data/index/sample").resolve()
    )


def test_vault_dir_is_the_only_source_of_the_vault_location(tmp_path: Path) -> None:
    """E30: ``vault.dir`` を変えると索引されるノート集合が変わる。

    vault の場所がコード・テストのどこにもハードコードされていないことの検証。
    2 つの vault は同じプロセス・同じ設定ファイル名で読み、違うのは
    ``vault.dir`` の 1 行だけにする。
    """
    first = tmp_path / "vault-a"
    shutil.copytree(SAMPLE_VAULT_DIR, first)
    second = tmp_path / "vault-b" / "inner"
    second.mkdir(parents=True)
    (second / "only-here.md").write_text("# 別の vault\n", encoding="utf-8")

    settings_a = rag.load_settings(
        write_rag_settings(tmp_path, vault_dir="vault-a", name="a.toml")
    )
    settings_b = rag.load_settings(
        write_rag_settings(tmp_path, vault_dir="vault-b/inner", name="b.toml")
    )

    selected_a = {entry.relpath for entry in rag.iter_vault_files(settings_a)}
    selected_b = {entry.relpath for entry in rag.iter_vault_files(settings_b)}

    assert selected_b == {"only-here.md"}
    assert selected_a != selected_b
    assert len(selected_a) == 11
    assert settings_a.vault_dir != settings_b.vault_dir


def test_relative_paths_resolve_against_the_settings_file(
    sample_vault_copy: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """カレントディレクトリを変えても同じ vault を指す。"""
    settings_path = write_rag_settings(sample_vault_copy.parent)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    settings = rag.load_settings(settings_path)

    assert settings.vault_dir == sample_vault_copy.resolve()
    assert settings.index_dir == (sample_vault_copy.parent / "index").resolve()


def test_index_dir_inside_the_vault_is_rejected(sample_vault_copy: Path) -> None:
    """§3 論点5 (設定層): 索引の出力先を vault の中に置けない。"""
    settings_path = write_rag_settings(
        sample_vault_copy.parent, index_dir="vault/.index"
    )

    with pytest.raises(ConfigError) as excinfo:
        rag.load_settings(settings_path)

    assert "index.dir" in str(excinfo.value)
    assert "対処" in str(excinfo.value)


def test_index_dir_that_reaches_the_vault_through_dotdot_is_rejected(
    sample_vault_copy: Path,
) -> None:
    """``..`` を挟んで vault の中を指す書き方も resolve() 後に落ちる。"""
    settings_path = write_rag_settings(
        sample_vault_copy.parent, index_dir="index/../vault/notes"
    )

    with pytest.raises(ConfigError):
        rag.load_settings(settings_path)


def test_index_dir_equal_to_the_vault_is_rejected(sample_vault_copy: Path) -> None:
    settings_path = write_rag_settings(sample_vault_copy.parent, index_dir="vault")

    with pytest.raises(ConfigError):
        rag.load_settings(settings_path)


def test_missing_vault_dir_is_a_config_error_with_remediation(tmp_path: Path) -> None:
    settings_path = write_rag_settings(tmp_path, vault_dir="does-not-exist")

    with pytest.raises(ConfigError) as excinfo:
        rag.load_settings(settings_path)

    message = str(excinfo.value)
    assert "vault.dir" in message
    assert "does-not-exist" in message
    assert "対処" in message
    # 解決後の絶対パスを載せない (実 vault のパスを出さないため)。
    assert str(tmp_path) not in message


def test_vault_dir_pointing_at_a_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "not-a-dir").write_text("", encoding="utf-8")
    settings_path = write_rag_settings(tmp_path, vault_dir="not-a-dir")

    with pytest.raises(ConfigError) as excinfo:
        rag.load_settings(settings_path)

    assert "ディレクトリではありません" in str(excinfo.value)


@pytest.mark.parametrize("vault_id", ["../escape", "Sample", "a/b", ""])
def test_invalid_vault_id_is_rejected(sample_vault_copy: Path, vault_id: str) -> None:
    """``vault.id`` は索引ディレクトリ名になるため構文レベルで縛る。"""
    settings_path = write_rag_settings(sample_vault_copy.parent, vault_id=vault_id)

    with pytest.raises(ConfigError) as excinfo:
        rag.load_settings(settings_path)

    assert "vault" in str(excinfo.value)


def test_unknown_keys_are_rejected(sample_vault_copy: Path) -> None:
    """打ち間違えたキーが黙って無視されない (extra='forbid')。"""
    settings_path = sample_vault_copy.parent / "typo.toml"
    settings_path.write_text(
        '[vault]\nid = "sample"\ndir = "vault"\nexcludes = ["x"]\n', encoding="utf-8"
    )

    with pytest.raises(ConfigError) as excinfo:
        rag.load_settings(settings_path)

    assert "excludes" in str(excinfo.value)


def test_missing_settings_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        rag.load_settings(tmp_path / "absent.toml")

    assert "対処" in str(excinfo.value)


def test_broken_toml_is_a_config_error(tmp_path: Path) -> None:
    settings_path = tmp_path / "broken.toml"
    settings_path.write_text("[vault\nid = 'sample'\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        rag.load_settings(settings_path)

    assert "TOML" in str(excinfo.value)


def test_empty_include_globs_is_rejected(sample_vault_copy: Path) -> None:
    """ホワイトリストが空なら索引対象が 0 件になる。設定ミスとして落とす。"""
    settings_path = write_rag_settings(sample_vault_copy.parent, include_globs=())

    with pytest.raises(ConfigError) as excinfo:
        rag.load_settings(settings_path)

    assert "include_globs" in str(excinfo.value)


def test_the_shipped_sample_settings_load_and_point_outside_the_vault() -> None:
    """コミット済みの ``vaults/sample.toml`` がそのまま読める。"""
    settings = rag.load_settings(SAMPLE_VAULT_CONFIG)

    assert settings.vault_id == "sample"
    assert settings.vault_dir == SAMPLE_VAULT_DIR.resolve()
    assert settings.index_dir == (REPO_ROOT / "data" / "index" / "sample").resolve()
    assert not settings.index_dir.is_relative_to(settings.vault_dir)


def test_the_shipped_sample_settings_contain_no_absolute_paths() -> None:
    """公開リポジトリに実 vault の絶対パスを持ち込まない。"""
    text = SAMPLE_VAULT_CONFIG.read_text(encoding="utf-8")

    assert "/home/" not in text
    assert "~" not in text
    assert os.sep + os.sep not in text
