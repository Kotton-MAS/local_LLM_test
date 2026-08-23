"""要件書 Phase 3 受け入れ条件の機械検証。

``docs/localllmrequirements.md`` L309-L314 / L319 の 7 条件と、このモジュールの
7 つのテスト関数を **1 対 1** に対応させる。各 docstring の先頭に対応行を書き、
各テストは自分が対応する行が受け入れ条件のままであること (``- [`` で始まる)
を最初に確かめる。条件が増えたらテストも増やす。

7 条件はすべて **1 つの経路** (設定 TOML → :func:`rag.build_index` → 索引成果物)
を実際に回して測る。3a で満たした L311-L314 / L319 は個別モジュールのテスト
(``test_rag_vault.py`` / ``test_rag_parser.py`` / ``test_rag_chunker.py``) に
散っていたが、そこで見ているのは中間表現 (``ParsedNote`` / ``Chunk``) であり、
「索引に何が入ったか」ではない。受け入れ条件が問うているのは索引の中身なので、
ここでは **``chunks.jsonl`` を読み直したレコード**に対して主張する。

**L315-L318 を含めない理由**: この 4 条件 (クエリへの回答と参照位置 / 評価質問
セット 20 問以上の精度測定 / 既知の失敗パターン 4 種 / 根拠が無いときに「情報が
ない」と返す) は検索・リランキング・回答生成の実装を要求しており、本サイクル
(3b = 索引の構築と差分更新) のスコープ外である (仕様書
``docs/plans/2026-08-23-phase3b-indexing.md`` §3 ハード制約)。未実装の条件に
テストを対応させると、何も検証しないテストが緑になるか、恒常的に赤になるかの
どちらかにしかならない。**次サイクルで実装するときに、テストとチェックボックスを
同時に増やす。**

実 HTTP は 1 バイトも発行しない (D-02)。埋め込みは ``httpx.MockTransport`` +
決定論的フェイク (``tests/conftest.py`` の :func:`fake_embedding_vector`) で、
書き出し先は必ず ``tmp_path`` (リポジトリの ``data/`` へは 1 バイトも書かない)。
"""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

from conftest import (
    DEFAULT_CONFIG,
    FAKE_EMBEDDING_DIMENSIONS,
    RecordingTransport,
    fake_embedding_transport,
    write_rag_settings,
)

import llmkit
import rag

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "docs" / "localllmrequirements.md"

#: 要件書の行番号 -> 対応するテスト関数名。
ACCEPTANCE_MAP = {
    309: "test_l309_a_vault_path_produces_an_index",
    310: "test_l310_an_unchanged_rerun_reprocesses_no_note",
    311: "test_l311_obsidian_and_attachments_never_enter_the_index",
    312: "test_l312_wikilinks_are_resolved_in_every_indexed_body",
    313: "test_l313_frontmatter_is_metadata_and_never_body",
    314: "test_l314_indexed_chunks_keep_their_heading_hierarchy",
    319: "test_l319_indexing_leaves_every_vault_file_untouched",
}

#: 合成 vault の索引対象ノート数 (``tests/test_rag_vault.py`` と同じ前提)。
SAMPLE_NOTE_COUNT = 11

#: 合成 vault のチャンク総数 (``tests/test_rag_chunker.py`` と同じ前提)。
SAMPLE_CHUNK_COUNT = 24

#: 索引対象外であることを検査する側の「実在する」入力 (検査の空回り防止)。
EXCLUDED_VAULT_ENTRIES = (
    ".obsidian/app.json",
    ".trash/deleted.md",
    "attachments/pixel.png",
    "attachments/board.canvas",
    "notes/diagram.excalidraw.md",
)


def requirement_line(number: int) -> str:
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    return lines[number - 1]


def assert_is_acceptance_line(number: int) -> None:
    """参照している行が受け入れ条件のままであること。

    ``- [x]`` (完了) でも落ちないよう、チェック状態ではなく**チェックボックス
    そのもの**を見る (Phase 3 の完了と同時にこのテストが落ちるのは無意味)。
    """
    assert requirement_line(number).startswith("- ["), number


# --------------------------------------------------------------------------
# 実行ヘルパ (実 HTTP なし・書き出し先は tmp_path)
# --------------------------------------------------------------------------


def sample_settings(vault_copy: Path) -> rag.RagSettings:
    """合成 vault の複製を **vault パスとして与える**設定を書いて読み込む。

    vault の場所を与える入口は設定 TOML の ``[vault] dir`` だけ (仕様書 §5 T7
    Q1 = a) なので、「vault パスを指定する」(L309) はこの 1 行を書くことに
    等しい。パスは設定ファイルからの相対で解決されるため、実 vault の絶対
    パスを 1 文字も書かずに掃引できる。
    """
    return rag.load_settings(write_rag_settings(vault_copy.parent))


def index_once(
    settings: rag.RagSettings, *, transport: RecordingTransport | None = None
) -> rag.IndexResult:
    """CLI (``python -m rag.cli index``) が行う手順そのままで 1 回索引する。

    マニフェストもストアも**毎回ディスクから読み直す**。プロセスを跨いだ
    再実行と同じ条件にしないと、メモリに残った状態のおかげで「再処理して
    いない」が成立してしまう。
    """
    recorder = transport if transport is not None else fake_embedding_transport()
    config = llmkit.load_config(DEFAULT_CONFIG)
    with recorder.client() as http_client:
        return rag.build_index(
            settings,
            config,
            embedding_client=llmkit.create_embedding_client(
                config, http_client=http_client
            ),
            store=rag.JsonlVectorStore(rag.chunks_path(settings)),
            manifest=rag.load_manifest(settings),
        )


def stored_records(settings: rag.RagSettings) -> tuple[rag.ChunkRecord, ...]:
    """``chunks.jsonl`` を**読み直した**レコード (索引に実際に入ったもの)。"""
    return tuple(rag.JsonlVectorStore(rag.chunks_path(settings)).iter_records())


def records_of(settings: rag.RagSettings, relpath: str) -> tuple[rag.ChunkRecord, ...]:
    return tuple(
        record for record in stored_records(settings) if record.relpath == relpath
    )


def manifest_of(settings: rag.RagSettings) -> rag.IndexManifest:
    manifest = rag.load_manifest(settings)
    assert manifest is not None, "manifest.json が書かれていません"
    return manifest


def index_digests(settings: rag.RagSettings) -> dict[str, str]:
    """索引ディレクトリの ``ファイル名 -> sha256``。未作成なら空の辞書。"""
    if not settings.index_dir.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(settings.index_dir.iterdir())
        if path.is_file()
    }


#: ``(size, st_mtime_ns, st_mode, sha256)``。``st_atime`` は含めない。
type Fingerprint = tuple[int, int, int, str]


def snapshot_tree(root: Path) -> dict[str, Fingerprint]:
    """vault 配下の全エントリの指紋を採る (``tests/test_rag_vault.py`` と同型)。"""
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


# --------------------------------------------------------------------------
# 受け入れ条件
# --------------------------------------------------------------------------


def test_l309_a_vault_path_produces_an_index(sample_vault_copy: Path) -> None:
    """要件書 L309: vault パスを指定するとインデックスが構築される。

    入力は設定 TOML の ``[vault] dir`` 1 行だけで、出力は ``index.dir`` 配下の
    2 ファイル (``chunks.jsonl`` / ``manifest.json``)。ファイルが在るだけでは
    足りない (空でも「生成された」と言えてしまう) ため、11 ノート 24 チャンク
    分のベクトルが実際に入っていること・マニフェストの集計が中身と一致する
    ことまで見る。
    """
    assert_is_acceptance_line(309)

    settings = sample_settings(sample_vault_copy)
    transport = fake_embedding_transport()

    result = index_once(settings, transport=transport)

    assert result.indexed_notes == SAMPLE_NOTE_COUNT
    assert result.embedded_chunks == SAMPLE_CHUNK_COUNT
    assert result.failed_notes == 0
    assert result.dimensions == FAKE_EMBEDDING_DIMENSIONS
    assert transport.call_count == result.request_count

    assert set(index_digests(settings)) == {"chunks.jsonl", "manifest.json"}
    records = stored_records(settings)
    assert len(records) == SAMPLE_CHUNK_COUNT
    assert {len(record.vector) for record in records} == {FAKE_EMBEDDING_DIMENSIONS}

    manifest = manifest_of(settings)
    assert manifest.totals.notes == SAMPLE_NOTE_COUNT
    assert manifest.totals.chunks == SAMPLE_CHUNK_COUNT
    assert manifest.index_fingerprint == rag.index_fingerprint(
        settings, llmkit.resolve_embedding_spec(llmkit.load_config(DEFAULT_CONFIG))
    )
    # 索引の出力先は vault の外にある (D-30 設定層)。
    assert not settings.index_dir.is_relative_to(settings.vault_dir)


def test_l310_an_unchanged_rerun_reprocesses_no_note(sample_vault_copy: Path) -> None:
    """要件書 L310: 再実行時、変更のないノートは再処理されない (差分更新)。

    「再処理しない」を件数だけで測ると、埋め込みを飛ばしつつ索引を作り直す
    実装 (中身が変わる) を見逃す。逆にバイト一致だけを測ると、毎回全件を
    埋め直しても同じ結果になるので気づけない。**要求 0 回**と**成果物の
    バイト一致**を同時に主張する。
    """
    assert_is_acceptance_line(310)

    settings = sample_settings(sample_vault_copy)
    first = index_once(settings)
    before = index_digests(settings)

    transport = fake_embedding_transport()
    second = index_once(settings, transport=transport)

    assert first.embedded_chunks == SAMPLE_CHUNK_COUNT
    assert second.embedded_chunks == 0
    assert second.indexed_notes == 0
    assert second.skipped_notes == SAMPLE_NOTE_COUNT
    assert second.request_count == 0
    assert transport.call_count == 0, "変更が無いのに埋め込みを要求している"
    assert index_digests(settings) == before


def test_l311_obsidian_and_attachments_never_enter_the_index(
    sample_vault_copy: Path,
) -> None:
    """要件書 L311: ``.obsidian/`` と添付ファイルがインデックスに含まれない。

    ``iter_vault_files`` の選択規則ではなく**索引成果物**を見る。列挙が正しく
    ても、索引側が別経路でファイルを拾えば条件は破れる。除外対象が vault に
    実在することも併せて主張し、「そもそも無いから通った」を防ぐ。
    """
    assert_is_acceptance_line(311)

    settings = sample_settings(sample_vault_copy)
    index_once(settings)

    for relpath in EXCLUDED_VAULT_ENTRIES:
        assert (sample_vault_copy / relpath).exists(), relpath

    indexed = {record.relpath for record in stored_records(settings)}
    indexed |= {note.relpath for note in manifest_of(settings).notes}

    assert len(indexed) == SAMPLE_NOTE_COUNT
    assert all(relpath.startswith("notes/") for relpath in indexed), indexed
    assert not indexed & set(EXCLUDED_VAULT_ENTRIES)
    assert not [path for path in indexed if path.endswith(".excalidraw.md")]
    assert not [path for path in indexed if not path.endswith(".md")]


def test_l312_wikilinks_are_resolved_in_every_indexed_body(
    sample_vault_copy: Path,
) -> None:
    """要件書 L312: ``[[note|alias]]`` が ``alias`` に解決され ``[[`` が残らない。

    索引に入った全レコードの ``body`` を走査する。加えて、リンク先が
    ``links`` としてメタデータ側に残っていること (捨てていないこと) と、
    別名が本文に現れることを 1 件で具体的に確かめる。
    """
    assert_is_acceptance_line(312)

    settings = sample_settings(sample_vault_copy)
    index_once(settings)

    records = stored_records(settings)
    offenders = [
        record.chunk_id
        for record in records
        if "[[" in record.body or "]]" in record.body
    ]
    assert not offenders, f"wikilink 記法が索引に残っています: {offenders}"

    review = records_of(settings, "notes/weekly-review.md")
    assert review, "検査対象のノートが索引に入っていません"
    body = "\n".join(record.body for record in review)
    assert "Alpha の設計メモ" in body, "別名に解決されていません"
    assert "制約の一覧" in body
    assert "project-alpha" in review[0].links


def test_l313_frontmatter_is_metadata_and_never_body(sample_vault_copy: Path) -> None:
    """要件書 L313: frontmatter が本文に混入せず、メタデータとして取得できる。

    「混入しない」(索引レコードの ``body``) と「取得できる」(``tags`` および
    :func:`rag.parse_note` の ``frontmatter``) の両方を見る。片方だけだと、
    frontmatter を丸ごと捨てる実装でも「混入していない」で通ってしまう。
    """
    assert_is_acceptance_line(313)

    settings = sample_settings(sample_vault_copy)
    index_once(settings)

    relpath = "notes/frontmatter-rich.md"
    records = records_of(settings, relpath)
    assert records, "検査対象のノートが索引に入っていません"

    for record in records:
        for leaked in ("title:", "tags:", "aliases:", "created:", "draft:", "---"):
            assert leaked not in record.body, (leaked, record.chunk_id)
        assert record.tags == ("設計", "テンプレート")

    note = rag.parse_note(relpath, rag.read_note_text(settings, relpath))
    assert note.title == "設計テンプレート"
    assert note.frontmatter["aliases"] == ("テンプレ", "design-template")
    assert note.frontmatter["description"] == "frontmatter の各記法をまとめた見本"


def test_l314_indexed_chunks_keep_their_heading_hierarchy(
    sample_vault_copy: Path,
) -> None:
    """要件書 L314: チャンクが見出し階層のコンテキストを保持している。

    保持しているだけでなく、**埋め込みに渡した文字列に効いている**ことまで
    見る。``embed_text`` は永続化しない (D-41) ので、レコードから
    :func:`rag.render_embed_text` で再構成した文字列の先頭に見出し経路が
    現れることを確かめる。
    """
    assert_is_acceptance_line(314)

    settings = sample_settings(sample_vault_copy)
    index_once(settings)

    records = records_of(settings, "notes/project-alpha.md")
    assert records, "検査対象のノートが索引に入っていません"

    paths = {record.heading_path for record in records}
    assert ("プロジェクトAlpha", "設計", "データモデル") in paths, paths
    assert all(path[0] == "プロジェクトAlpha" for path in paths)

    separator = settings.chunk.heading_separator
    for record in records:
        rendered = rag.render_embed_text(record.heading_path, record.body, separator)
        prefix = separator.join(record.heading_path)
        assert rendered.startswith(prefix)
        assert rendered.endswith(record.body)
        assert prefix not in record.body, "見出し経路が本文にも二重に入っている"


def test_l319_indexing_leaves_every_vault_file_untouched(
    sample_vault_copy: Path,
) -> None:
    """要件書 L319: vault 内のファイルが実行前後で一切変更されていない。

    比べるのは相対パス・サイズ・``st_mtime_ns``・``st_mode``・sha256 で、
    エントリの増減も見る (索引が vault の隣に一時ファイルを作る実装を
    検出するため)。索引が**実際に成果物を書いた**ことも同時に主張し、
    「何もしなかったから一致した」を防ぐ。
    """
    assert_is_acceptance_line(319)

    settings = sample_settings(sample_vault_copy)
    before = snapshot_tree(sample_vault_copy)

    result = index_once(settings)

    after = snapshot_tree(sample_vault_copy)
    assert result.indexed_notes == SAMPLE_NOTE_COUNT
    assert result.embedded_chunks == SAMPLE_CHUNK_COUNT
    assert set(index_digests(settings)) == {"chunks.jsonl", "manifest.json"}
    assert after == before
