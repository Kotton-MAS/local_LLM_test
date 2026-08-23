"""RAG 索引の CLI (``rag/cli.py``) のテスト。

このファイルが固定する性質は 4 つある。

1. **``--dry-run`` は HTTP を 1 バイトも出さず、索引ディレクトリを作らない。**
   埋め込みは時間を食うので、「この条件で回す」を実行前に確かめられることが
   唯一の防御になる (``harness/cli.py`` の ``--dry-run`` と同じ扱い)。
2. **``--settings`` が vault を与える唯一の入口** (**E36**)。索引されるノートの
   集合は設定ファイルの内容だけで決まる (要件書 L309 の直接検証)。
3. **``index.dir`` が索引の唯一の出力先** (**E35**)。出力先を変えると別の索引が
   でき、元の索引は 1 バイトも変わらない。
4. **画面に出るのは件数と fingerprint だけ。** 絶対パスもノート本文も ``[[``
   も 0 件で、実 vault を索引した出力をそのまま報告に貼れる。

埋め込みは ``httpx.MockTransport`` + 決定論的フェイク (``tests/conftest.py``)
で行う。実 HTTP は 1 バイトも出さないが、``llmkit`` の例外翻訳表は本番と同じ
経路を通る。書き出し先は必ず ``tmp_path`` で、リポジトリには 1 バイトも書かない。
"""

from __future__ import annotations

import ast
import io
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from conftest import (
    DEFAULT_CONFIG,
    SAMPLE_VAULT_DIR,
    RecordingTransport,
    fake_embedding_transport,
    write_rag_settings,
)

import llmkit
import rag
import rag.cli
from rag.cli import EXIT_ERROR, EXIT_OK, main

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 合成 vault の索引対象ノート数 / チャンク総数 (``tests/test_rag_indexer.py``
#: と同じ前提)。CLI の表示がこの数字と一致することを確かめる。
SAMPLE_NOTE_COUNT = 11
SAMPLE_CHUNK_COUNT = 24

#: 既定 ``batch_size=16`` で 24 チャンクを送るときの要求回数 (= ⌈24/16⌉)。
SAMPLE_REQUEST_COUNT = 2

#: E36 で使う別 vault のノート数。合成 vault (11) と**違う**数にする。
#: どちらのノートも本文を持つので、全件失敗の検証にも使える (合成 vault は
#: 0 チャンクのノートを 2 件含み、それらは要求を出さずに確定するため
#: 「1 件も索引できなかった」状態を作れない、§9 T6 決定14)。
OTHER_NOTE_COUNT = 2


# --------------------------------------------------------------------------
# 小道具
# --------------------------------------------------------------------------


class Invocation:
    """CLI を 1 回呼んだ結果 (終了コード・出力・発行した HTTP)。"""

    def __init__(
        self, code: int, stdout: str, stderr: str, transport: RecordingTransport
    ) -> None:
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.transport = transport

    @property
    def http_calls(self) -> int:
        return self.transport.call_count

    @property
    def fields(self) -> dict[str, str]:
        """``ラベル : 値`` 形式の出力を辞書にする (件数の検証に使う)。

        ラベルは全角の桁そろえで空白が入るため、両端を strip して比べる。
        """
        table: dict[str, str] = {}
        for line in self.stdout.splitlines():
            if ":" not in line:
                continue
            label, _, value = line.partition(":")
            table[label.strip()] = value.strip()
        return table

    def count(self, label: str) -> int:
        """``<n> 件`` / ``<n> 回`` の数値部分。"""
        return int(self.fields[label].split(" ")[0])


def invoke(
    *arguments: str,
    transport: RecordingTransport | None = None,
    config_path: Path = DEFAULT_CONFIG,
    inject_client: bool = True,
) -> Invocation:
    """``python -m rag.cli ...`` と同じ経路を実 HTTP なしで呼ぶ。

    注入点は :class:`llmkit.EmbeddingClient` であってトランスポートではない
    (§9 T7 決定2)。``inject_client=False`` は「注入が無くても HTTP を出さない」
    経路 (``--dry-run`` / ``status``) の検証に使う。埋め込みクライアントを
    組み立てさせないので、その経路が本当にランタイムを要らないことを構造として
    示せる。
    """
    recorder = transport if transport is not None else fake_embedding_transport()
    stdout = io.StringIO()
    stderr = io.StringIO()
    with recorder.client() as http_client:
        client = (
            llmkit.create_embedding_client(
                llmkit.load_config(config_path), http_client=http_client
            )
            if inject_client
            else None
        )
        code = main(
            [*arguments, "--config", str(config_path)],
            embedding_client=client,
            stdout=stdout,
            stderr=stderr,
        )
    return Invocation(code, stdout.getvalue(), stderr.getvalue(), recorder)


def sample_settings_file(
    tmp_path: Path, *, vault_name: str = "vault", **overrides: str
) -> Path:
    """合成 vault の**複製**を指す設定ファイルを ``tmp_path`` に書き出す。

    複製を使うのは、テストが vault を書き換える場合に備えるため。コミット済みの
    合成 vault 自体は 1 バイトも変更しない。
    """
    vault = tmp_path / vault_name
    if not vault.exists():
        shutil.copytree(SAMPLE_VAULT_DIR, vault)
    return write_rag_settings(tmp_path, vault_dir=vault_name, **overrides)


def other_vault_settings_file(tmp_path: Path) -> Path:
    """合成 vault とは**別の** vault を指す設定ファイル (E36)。"""
    notes = tmp_path / "other-vault" / "notes"
    notes.mkdir(parents=True)
    (notes / "alpha.md").write_text("# Alpha\n\nアルファの本文。\n", encoding="utf-8")
    (notes / "beta.md").write_text("# Beta\n\nベータの本文。\n", encoding="utf-8")
    return write_rag_settings(
        tmp_path,
        vault_dir="other-vault",
        index_dir="other-index",
        vault_id="other",
        name="other.toml",
    )


def settings_of(path: Path) -> rag.RagSettings:
    return rag.load_settings(path)


def artifact_bytes(settings: rag.RagSettings) -> dict[str, bytes]:
    """索引成果物の中身 (バイト一致の比較に使う)。"""
    return {
        path.name: path.read_bytes()
        for path in (rag.manifest_path(settings), rag.chunks_path(settings))
    }


def note_bodies() -> tuple[str, ...]:
    """合成 vault のノート本文の行 (出力に 1 行も現れてはいけない)。"""
    lines: list[str] = []
    for note in sorted(SAMPLE_VAULT_DIR.rglob("*.md")):
        lines += [
            line.strip()
            for line in note.read_text(encoding="utf-8").splitlines()
            if len(line.strip()) >= 10
        ]
    return tuple(lines)


def absolute_path_tokens(text: str) -> list[str]:
    """出力に現れた絶対パスらしきトークン。

    ``/`` で始まる 2 文字以上のトークンを絶対パスとみなす。出力に載ってよい
    ``manifest.json / chunks.jsonl`` の区切りは前後が空白なので一致しない。
    """
    return [token for token in text.split() if re.fullmatch(r"/\S+", token)]


def leaks(text: str) -> list[str]:
    """報告にそのまま貼れない要素 (絶対パス / 本文 / wiki リンク) を集める。"""
    found = absolute_path_tokens(text)
    found += [body for body in note_bodies() if body in text]
    if "[[" in text:
        found.append("[[")
    return found


def upstream_failure(_texts: tuple[str, ...]) -> httpx.Response:
    """埋め込み要求を必ず 5xx にする (``UpstreamError`` へ翻訳される)。"""
    return httpx.Response(500, json={"error": {"message": "boom"}})


def connection_refused(_texts: tuple[str, ...]) -> httpx.Response:
    """接続断 (``RuntimeUnavailableError`` へ翻訳される)。"""
    msg = "connection refused"
    raise httpx.ConnectError(msg)


# --------------------------------------------------------------------------
# --dry-run (HTTP 0 回・書き込み 0 バイト・exit 0)
# --------------------------------------------------------------------------


def test_dry_run_issues_no_http_and_creates_no_index_directory(
    tmp_path: Path,
) -> None:
    """★ 受け入れ基準: ランタイム無しで exit 0、HTTP 0 回、出力先も作らない。"""
    settings_path = sample_settings_file(tmp_path)

    invocation = invoke("index", "--settings", str(settings_path), "--dry-run")

    assert invocation.code == EXIT_OK
    assert invocation.http_calls == 0
    assert not settings_of(settings_path).index_dir.exists()


def test_dry_run_needs_no_embedding_client_at_all(tmp_path: Path) -> None:
    """計画の経路は埋め込みクライアントを 1 度も組み立てない (構造的な保証)。"""
    settings_path = sample_settings_file(tmp_path)

    invocation = invoke(
        "index", "--settings", str(settings_path), "--dry-run", inject_client=False
    )

    assert invocation.code == EXIT_OK
    assert invocation.http_calls == 0


def test_dry_run_prints_the_plan_and_the_fingerprint(tmp_path: Path) -> None:
    settings_path = sample_settings_file(tmp_path)

    invocation = invoke("index", "--settings", str(settings_path), "--dry-run")

    assert "dry-run" in invocation.fields["モード"]
    assert len(invocation.fields["index_fingerprint"]) == 64
    assert invocation.count("対象ノート") == SAMPLE_NOTE_COUNT
    assert invocation.count("新規") == SAMPLE_NOTE_COUNT
    assert invocation.count("再処理") == SAMPLE_NOTE_COUNT


def test_dry_run_names_the_output_files_without_naming_a_directory(
    tmp_path: Path,
) -> None:
    """出力先は**ファイル名**だけを出す (ディレクトリを画面に出さない)。"""
    settings_path = sample_settings_file(tmp_path, index_dir="zz-output-dir")

    invocation = invoke("index", "--settings", str(settings_path), "--dry-run")

    destination = invocation.fields["出力先 (予定)"]
    assert rag.MANIFEST_FILENAME in destination
    assert rag.CHUNKS_FILENAME in destination
    assert "zz-output-dir" not in invocation.stdout


def test_dry_run_after_an_index_reports_nothing_to_reprocess(tmp_path: Path) -> None:
    """索引済みの vault に対する計画は「再処理 0 件」になる。"""
    settings_path = sample_settings_file(tmp_path)
    invoke("index", "--settings", str(settings_path))

    invocation = invoke("index", "--settings", str(settings_path), "--dry-run")

    assert invocation.http_calls == 0
    assert invocation.count("再処理") == 0
    assert invocation.count("変更なし") == SAMPLE_NOTE_COUNT


# --------------------------------------------------------------------------
# index (本実行)
# --------------------------------------------------------------------------


def test_an_index_writes_both_artifacts_and_reports_only_counts(
    tmp_path: Path,
) -> None:
    settings_path = sample_settings_file(tmp_path)

    invocation = invoke("index", "--settings", str(settings_path))

    settings = settings_of(settings_path)
    assert invocation.code == EXIT_OK
    assert rag.manifest_path(settings).is_file()
    assert rag.chunks_path(settings).is_file()
    assert invocation.count("索引したノート") == SAMPLE_NOTE_COUNT
    assert invocation.count("埋め込みチャンク") == SAMPLE_CHUNK_COUNT
    assert invocation.count("リクエスト") == SAMPLE_REQUEST_COUNT
    assert invocation.http_calls == SAMPLE_REQUEST_COUNT


def test_a_second_index_run_reprocesses_nothing(tmp_path: Path) -> None:
    """要件書 L310: 変更のないノートを 1 件も再処理しない。"""
    settings_path = sample_settings_file(tmp_path)
    invoke("index", "--settings", str(settings_path))
    before = artifact_bytes(settings_of(settings_path))

    invocation = invoke("index", "--settings", str(settings_path))

    assert invocation.code == EXIT_OK
    assert invocation.http_calls == 0
    assert invocation.count("索引したノート") == 0
    assert invocation.count("埋め込みチャンク") == 0
    assert invocation.count("再処理しない") == SAMPLE_NOTE_COUNT
    assert artifact_bytes(settings_of(settings_path)) == before


# --------------------------------------------------------------------------
# --rebuild (逃げ道: fingerprint が一致していても全再構築)
# --------------------------------------------------------------------------


def test_rebuild_reindexes_everything_even_when_the_fingerprint_matches(
    tmp_path: Path,
) -> None:
    """★ 受け入れ基準: ``--rebuild`` は「変更なし」を無視して全件やり直す。"""
    settings_path = sample_settings_file(tmp_path)
    first = invoke("index", "--settings", str(settings_path))
    baseline = artifact_bytes(settings_of(settings_path))

    invocation = invoke("index", "--settings", str(settings_path), "--rebuild")

    assert invocation.code == EXIT_OK
    assert invocation.fields["index_fingerprint"] == first.fields["index_fingerprint"]
    assert invocation.count("索引したノート") == SAMPLE_NOTE_COUNT
    assert invocation.count("埋め込みチャンク") == SAMPLE_CHUNK_COUNT
    assert invocation.count("リクエスト") == SAMPLE_REQUEST_COUNT
    assert invocation.http_calls == SAMPLE_REQUEST_COUNT
    assert invocation.count("再処理しない") == 0
    # 同じ入力を同じ規則で埋め直したので、成果物はバイト一致に戻る。
    assert artifact_bytes(settings_of(settings_path)) == baseline


def test_rebuild_reports_the_full_rebuild_mode(tmp_path: Path) -> None:
    settings_path = sample_settings_file(tmp_path)

    invocation = invoke("index", "--settings", str(settings_path), "--rebuild")

    assert "全再構築" in invocation.fields["モード"]


def test_rebuild_with_dry_run_plans_a_full_rebuild_without_any_http(
    tmp_path: Path,
) -> None:
    settings_path = sample_settings_file(tmp_path)
    invoke("index", "--settings", str(settings_path))

    invocation = invoke(
        "index", "--settings", str(settings_path), "--rebuild", "--dry-run"
    )

    assert invocation.http_calls == 0
    assert invocation.fields["全再構築"] == "はい"
    assert invocation.count("再処理") == SAMPLE_NOTE_COUNT


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def test_status_on_an_unbuilt_index_says_so_and_exits_zero(tmp_path: Path) -> None:
    """★ 受け入れ基準: 未構築は異常ではない (``index`` を回せば解消する)。"""
    settings_path = sample_settings_file(tmp_path)

    invocation = invoke("status", "--settings", str(settings_path))

    assert invocation.code == EXIT_OK
    assert invocation.http_calls == 0
    assert invocation.fields["索引"] == "未構築"
    assert invocation.count("対象ノート") == SAMPLE_NOTE_COUNT
    assert not settings_of(settings_path).index_dir.exists()


def test_status_after_an_index_reports_a_matching_fingerprint_and_no_work(
    tmp_path: Path,
) -> None:
    """★ 受け入れ基準: ``index`` → ``status`` で「一致 / 再処理 0 件」。"""
    settings_path = sample_settings_file(tmp_path)
    indexed = invoke("index", "--settings", str(settings_path))
    manifest = rag.load_manifest(settings_of(settings_path))
    assert manifest is not None

    invocation = invoke("status", "--settings", str(settings_path), inject_client=False)

    assert invocation.code == EXIT_OK
    assert invocation.http_calls == 0
    assert invocation.fields["索引"] == "構築済み"
    assert invocation.fields["fingerprint 一致"] == "はい"
    assert invocation.fields["index_fingerprint"] == indexed.fields["index_fingerprint"]
    assert invocation.count("索引済みノート") == SAMPLE_NOTE_COUNT
    assert invocation.count("索引済みチャンク") == SAMPLE_CHUNK_COUNT
    assert invocation.count("再処理") == 0
    assert invocation.fields["次元"] == str(manifest.embedding.dimensions)


def test_status_reports_a_mismatch_when_the_chunk_settings_change(
    tmp_path: Path,
) -> None:
    """前提が変われば「fingerprint 一致: いいえ」と出る (それでも exit 0)。"""
    settings_path = sample_settings_file(tmp_path)
    invoke("index", "--settings", str(settings_path))
    settings_path.write_text(
        settings_path.read_text(encoding="utf-8") + "\n[chunk]\nmax_tokens = 120\n",
        encoding="utf-8",
    )

    invocation = invoke("status", "--settings", str(settings_path))

    assert invocation.code == EXIT_OK
    assert invocation.fields["fingerprint 一致"] == "いいえ"
    assert invocation.count("再処理") == SAMPLE_NOTE_COUNT


def test_status_never_writes_anything(tmp_path: Path) -> None:
    """``status`` は読むだけ。成果物のバイト列を 1 ビットも動かさない。"""
    settings_path = sample_settings_file(tmp_path)
    invoke("index", "--settings", str(settings_path))
    before = artifact_bytes(settings_of(settings_path))

    invoke("status", "--settings", str(settings_path))

    assert artifact_bytes(settings_of(settings_path)) == before


# --------------------------------------------------------------------------
# E35: index.dir が索引の唯一の出力先
# --------------------------------------------------------------------------


def test_index_dir_is_the_only_output_location(tmp_path: Path) -> None:
    """**E35**: 出力先を変えると別の索引ができ、元の索引は無傷のまま。

    出力先が設定の 1 か所で決まらないと、「別の索引を作ったつもりが既存の索引を
    上書きしていた」事故が静かに起きる。索引は再生成できるが、再生成には埋め込みの
    時間が丸ごとかかる。
    """
    first_path = sample_settings_file(tmp_path)
    second_path = write_rag_settings(tmp_path, index_dir="index-b", name="b.toml")
    invoke("index", "--settings", str(first_path))
    first = settings_of(first_path)
    second = settings_of(second_path)
    baseline = artifact_bytes(first)

    invocation = invoke("index", "--settings", str(second_path))

    assert invocation.code == EXIT_OK
    assert first.index_dir != second.index_dir
    assert rag.manifest_path(second).is_file(), "新しい出力先に索引ができていない"
    assert invocation.count("索引したノート") == SAMPLE_NOTE_COUNT, (
        "別の出力先なのに既存の索引を再利用してしまっている"
    )
    assert artifact_bytes(first) == baseline, "元の索引が書き換えられた"


# --------------------------------------------------------------------------
# E36: --settings が vault を与える唯一の入口 (要件書 L309)
# --------------------------------------------------------------------------


def test_the_indexed_note_set_follows_the_given_config(tmp_path: Path) -> None:
    """**E36**: 索引されるノートの集合は渡した設定だけで決まる。

    ``--vault <dir>`` を作らなかったことの裏返しでもある。vault の出典が設定
    ファイルとコマンドラインの 2 か所にあると、``vault_id`` (= 索引の既定の
    置き場所) と実際に読んだ vault が食い違う組み合わせが作れてしまう。
    """
    sample_path = sample_settings_file(tmp_path)
    other_path = other_vault_settings_file(tmp_path)

    sample_run = invoke("index", "--settings", str(sample_path))
    other_run = invoke("index", "--settings", str(other_path))

    sample_manifest = rag.load_manifest(settings_of(sample_path))
    other_manifest = rag.load_manifest(settings_of(other_path))
    assert sample_manifest is not None
    assert other_manifest is not None
    assert sample_run.count("索引したノート") == SAMPLE_NOTE_COUNT
    assert other_run.count("索引したノート") == OTHER_NOTE_COUNT
    sample_notes = {note.relpath for note in sample_manifest.notes}
    other_notes = {note.relpath for note in other_manifest.notes}
    assert sample_notes.isdisjoint(other_notes), "別 vault なのに同じノート集合"
    assert sample_run.fields["vault"] != other_run.fields["vault"]


def test_the_cli_has_no_vault_option(tmp_path: Path) -> None:
    """``--vault`` を受け付けない (vault の出典を 2 か所にしない、D-27 の趣旨)。"""
    settings_path = sample_settings_file(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        rag.cli.build_parser().parse_args(
            ["index", "--settings", str(settings_path), "--vault", str(tmp_path)]
        )

    assert excinfo.value.code == 2


def test_settings_is_required() -> None:
    """``--settings`` を省略すると argparse が使い方を出して落ちる。

    既定値を持たせない: 省略した実行が「たまたま出荷済みの合成 vault を索引
    する」ことになると、実 vault を索引したつもりの実行が黙って別の索引を
    更新する (§9 T7 決定3)。
    """
    with pytest.raises(SystemExit) as excinfo:
        rag.cli.build_parser().parse_args(["index"])

    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "command",
    ["index", "status"],
    ids=["index", "status"],
)
def test_both_subcommands_take_the_settings_and_the_config(command: str) -> None:
    """2 サブコマンドは同じ 2 つの入力 (索引設定 / 推論設定) を取る。"""
    namespace = rag.cli.build_parser().parse_args(
        [command, "--settings", "s.toml", "--config", "c.toml"]
    )

    assert namespace.command == command
    assert namespace.settings == Path("s.toml")
    assert namespace.config == Path("c.toml")


def test_an_unknown_subcommand_is_rejected() -> None:
    """``index`` / ``status`` 以外は受け付けない (検索は次サイクル)。"""
    with pytest.raises(SystemExit) as excinfo:
        rag.cli.build_parser().parse_args(["search", "--settings", "s.toml"])

    assert excinfo.value.code == 2


# --------------------------------------------------------------------------
# 失敗 (exit 1 + 対処つきメッセージ)
# --------------------------------------------------------------------------


def _missing_settings(tmp_path: Path) -> Path:
    return tmp_path / "does-not-exist.toml"


def _index_dir_inside_the_vault(tmp_path: Path) -> Path:
    sample_settings_file(tmp_path)
    return write_rag_settings(
        tmp_path, index_dir="vault/index", name="inside-vault.toml"
    )


def _index_dir_that_git_would_track(tmp_path: Path) -> Path:
    """リポジトリ内の追跡され得る場所。**設定を読んだ時点で落ちるので作られない**。"""
    sample_settings_file(tmp_path)
    tracked = (REPO_ROOT / "rag" / "generated-by-a-test").resolve()
    return write_rag_settings(tmp_path, index_dir=str(tracked), name="tracked.toml")


@pytest.mark.parametrize(
    ("make_settings", "expected"),
    [
        (_missing_settings, "読めません"),
        (_index_dir_inside_the_vault, "vault.dir の配下"),
        (_index_dir_that_git_would_track, "追跡"),
    ],
    ids=["missing-file", "index-inside-the-vault", "index-would-be-tracked"],
)
def test_a_config_error_exits_one_with_a_remediation(
    tmp_path: Path,
    make_settings: Callable[[Path], Path],
    expected: str,
) -> None:
    """★ 受け入れ基準: ``ConfigError`` は exit 1 + 対処つきメッセージ。

    どれも「実行してみて初めて壊れる」たぐいの設定であり、索引を書き始める前に
    落ちなければならない (書き始めてから落ちると、vault の中や追跡され得る場所に
    ノート本文が残る)。
    """
    settings_path = make_settings(tmp_path)

    invocation = invoke("index", "--settings", str(settings_path))

    assert invocation.code == EXIT_ERROR
    assert invocation.stdout == ""
    assert invocation.stderr.startswith("エラー: ")
    assert expected in invocation.stderr
    assert "してください" in invocation.stderr, "対処が書かれていない"


def test_a_config_error_leaves_the_vault_untouched(tmp_path: Path) -> None:
    """設定が落ちる実行は vault にも索引にも 1 バイトも書かない。"""
    settings_path = _index_dir_inside_the_vault(tmp_path)
    vault = tmp_path / "vault"
    before = sorted(path.relative_to(vault) for path in vault.rglob("*"))

    invocation = invoke("index", "--settings", str(settings_path))

    assert invocation.code == EXIT_ERROR
    assert sorted(path.relative_to(vault) for path in vault.rglob("*")) == before


def test_a_run_where_every_note_fails_exits_one(tmp_path: Path) -> None:
    """★ 受け入れ基準: 全ノート失敗は exit 1。"""
    settings_path = other_vault_settings_file(tmp_path)
    transport = fake_embedding_transport(intercept=upstream_failure)

    invocation = invoke("index", "--settings", str(settings_path), transport=transport)

    assert invocation.code == EXIT_ERROR
    assert invocation.count("索引したノート") == 0
    assert invocation.count("失敗したノート") == OTHER_NOTE_COUNT
    assert "索引できませんでした" in invocation.stdout
    # 1 度も埋め込めていないので次元は分からない。0 で埋めない (D-07)。
    assert invocation.fields["次元"] == "—"


def test_a_partially_failing_run_still_exits_zero(tmp_path: Path) -> None:
    """1 ノートの失敗で索引全体を落とさない (``harness/cli.py`` と同じ方針)。"""
    settings_path = other_vault_settings_file(tmp_path)
    transport = fake_embedding_transport(
        intercept=lambda texts: (
            upstream_failure(texts)
            if any("アルファ" in text for text in texts)
            else None
        )
    )

    invocation = invoke("index", "--settings", str(settings_path), transport=transport)

    assert invocation.code == EXIT_OK
    assert invocation.count("索引したノート") == 1
    assert invocation.count("失敗したノート") == 1


def test_a_runtime_failure_is_reported_as_an_error_and_exits_one(
    tmp_path: Path,
) -> None:
    """ランタイム障害は :class:`llmkit.LlmkitError` として捕捉され exit 1。"""
    settings_path = sample_settings_file(tmp_path)
    transport = fake_embedding_transport(intercept=connection_refused)

    invocation = invoke("index", "--settings", str(settings_path), transport=transport)

    assert invocation.code == EXIT_ERROR
    assert invocation.stderr.startswith("エラー: ")
    assert not leaks(invocation.stderr), "障害メッセージにパスか本文が漏れている"


# --------------------------------------------------------------------------
# 出力に貼ってはいけないものが 1 つも出ない
# --------------------------------------------------------------------------


def successful_invocations(settings_path: Path) -> tuple[Invocation, ...]:
    """受け入れ検証で実際に回す 5 手順 (dry-run → index → status → 再実行 →
    ``--rebuild``)。"""
    return (
        invoke("index", "--settings", str(settings_path), "--dry-run"),
        invoke("index", "--settings", str(settings_path)),
        invoke("status", "--settings", str(settings_path)),
        invoke("index", "--settings", str(settings_path)),
        invoke("index", "--settings", str(settings_path), "--rebuild"),
    )


def test_the_output_never_carries_a_path_a_body_or_a_wiki_link(
    tmp_path: Path,
) -> None:
    """★ 受け入れ基準: stdout / stderr に絶対パス・本文・``[[`` が 0 件。

    実 vault を索引した出力をそのまま報告に貼れる状態を保つための機械検査。
    「気をつけて貼る」に頼ると、貼るたびに人間が判断することになる。
    """
    settings_path = sample_settings_file(tmp_path)
    offenders: list[str] = []
    for invocation in successful_invocations(settings_path):
        offenders += leaks(invocation.stdout)
        offenders += leaks(invocation.stderr)

    assert not offenders, f"報告に貼れない要素が出力に現れた: {offenders}"


def test_the_leak_check_actually_detects_something(tmp_path: Path) -> None:
    """上の検査が「何も見ていない」で通っていないことの確認。"""
    bodies = note_bodies()

    assert bodies, "合成 vault から本文の行を 1 つも拾えていない"
    assert leaks(f"出力先: {tmp_path}")
    assert leaks(bodies[0])
    assert leaks("[[project-alpha]]")


def test_the_output_never_names_the_vault_or_the_index_directory(
    tmp_path: Path,
) -> None:
    """相対パスも出さない (vault と索引のディレクトリ名が 1 度も出ない)。"""
    settings_path = sample_settings_file(
        tmp_path, vault_name="zz-vault-dir", index_dir="zz-index-dir"
    )

    for invocation in successful_invocations(settings_path):
        combined = invocation.stdout + invocation.stderr
        assert "zz-vault-dir" not in combined
        assert "zz-index-dir" not in combined


def test_the_cli_writes_only_to_the_injected_streams(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """出力は注入されたストリームだけに行き、実 stdout を汚さない。"""
    settings_path = sample_settings_file(tmp_path)

    invocation = invoke("index", "--settings", str(settings_path))

    captured = capsys.readouterr()
    assert invocation.stdout != ""
    assert invocation.stderr == ""
    assert captured.out == ""
    assert captured.err == ""


# --------------------------------------------------------------------------
# 層の境界 (cli は L4 の入口であって公開 API ではない)
# --------------------------------------------------------------------------


def test_the_cli_is_deliberately_not_part_of_the_public_api() -> None:
    """``rag.__all__`` に ``cli`` を含めない (``harness`` と同じ扱い)。"""
    assert set(rag.cli.__all__), "rag.cli に __all__ がありません"
    assert set(rag.cli.__all__).isdisjoint(rag.__all__)
    assert "cli" not in rag.__all__


def test_the_cli_never_imports_an_http_library() -> None:
    """D-25 の補強: CLI も ``httpx`` を知らない (§9 T7 決定2)。

    ``rag/`` から推論ランタイムへの直接経路が無いことは
    ``tests/test_rag_layout.py::test_rag_never_talks_to_the_runtime_directly``
    が全モジュールについて固定しているが、CLI は「テストのために HTTP
    クライアントを受け取る」誘惑が最も強い場所なので、ここでも名指しで
    確かめる。注入点は :class:`llmkit.EmbeddingClient` であり、トランスポートは
    ``llmkit`` の内側に閉じている。
    """
    tree = ast.parse((REPO_ROOT / "rag" / "cli.py").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]

    forbidden = {"httpx", "requests", "urllib", "http", "socket"}
    offenders = [name for name in imported if name.split(".")[0] in forbidden]
    assert not offenders, f"rag/cli.py が HTTP を直接扱っている: {offenders}"
