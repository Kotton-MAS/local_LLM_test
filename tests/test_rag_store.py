"""ベクトルストア (``rag/store.py``) の検証。

- ``test_every_store_implementation_passes_the_same_contract`` … **D-39 guard**。
  2 実装 (``JsonlVectorStore`` / ``InMemoryVectorStore``) を ``parametrize`` して
  同じ契約を回す。差分更新 (T6) は :class:`rag.VectorStore` しか知らないため、
  実装ごとに振る舞いが違うと「テストでは通るが実行すると壊れる」状態が作れる。
- ``test_the_persisted_record_never_stores_the_body_twice`` … **D-41 guard**。
  JSONL に ``embed_text`` / ``model`` / ``dimensions`` を書かず、埋め込み文字列は
  :func:`rag.render_embed_text` で ``heading_path`` + ``body`` から再構成できる
  ことを合成 vault の全チャンクで確かめる。
- ``test_committing_the_same_content_twice_is_byte_identical`` … 行順を
  ``(relpath, ordinal)`` 昇順に固定していることの証拠。**投入順を変えた 2 つの
  ストアを比較する**ので、ソートを外すと落ちる (D-37 のバイト一致の前提)。
- ``test_a_failed_commit_leaves_the_previous_index_untouched`` … 原子性。

ベクトルはテキストの sha256 から決定論的に作る (``fake_vector``)。「同じテキスト
なら同じベクトル」が成り立たないと、バイト一致の検査が原理的に書けない。
"""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from conftest import write_rag_settings

import rag
from llmkit import ConfigError
from rag import ChunkRecord, InMemoryVectorStore, JsonlVectorStore, VectorStore

#: 索引ファイル名。T5 が ``rag/indexer.py`` に定数として持つ (store はパスを
#: 受け取るだけで、置き場所も名前も知らない)。
CHUNKS_FILENAME = "chunks.jsonl"

#: テスト用の低次元。次元そのものは検査対象ではないので短くする。
DIMENSIONS = 8

#: 合成 vault のチャンク総数 (``tests/test_rag_chunker.py`` と同じ前提)。
SAMPLE_CHUNK_COUNT = 24

StoreFactory = Callable[[Path], VectorStore]


# --------------------------------------------------------------------------
# 小道具
# --------------------------------------------------------------------------


def fake_vector(text: str, dimensions: int = DIMENSIONS) -> tuple[float, ...]:
    """テキストから決定論的にベクトルを作る (同じテキストなら常に同じ)。"""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return tuple((digest[index] - 127.5) / 256.0 for index in range(dimensions))


def make_record(
    relpath: str,
    ordinal: int,
    *,
    body: str = "本文",
    dimensions: int = DIMENSIONS,
) -> ChunkRecord:
    """検査用の 1 レコード。``vector`` は本文から決まる。"""
    return ChunkRecord(
        chunk_id=f"{relpath}#{ordinal:04d}",
        relpath=relpath,
        ordinal=ordinal,
        part_index=0,
        heading_path=("タイトル", "見出し"),
        body=body,
        estimated_tokens=len(body),
        tags=("タグ",),
        links=(),
        vector=fake_vector(f"{relpath}#{ordinal}:{body}", dimensions),
    )


def record_from_chunk(chunk: rag.Chunk) -> ChunkRecord:
    """:class:`rag.Chunk` から永続化レコードを作る (T6 が行う変換と同じ形)。

    ``embed_text`` は**渡さない**。埋め込みへ送る文字列としては使うが、索引には
    残さない (D-41)。
    """
    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        relpath=chunk.relpath,
        ordinal=chunk.ordinal,
        part_index=chunk.part_index,
        heading_path=chunk.heading_path,
        body=chunk.body,
        estimated_tokens=chunk.estimated_tokens,
        tags=chunk.tags,
        links=chunk.links,
        vector=fake_vector(chunk.embed_text),
    )


def sample_chunks(vault_copy: Path) -> tuple[rag.RagSettings, tuple[rag.Chunk, ...]]:
    """合成 vault の全チャンク (``tests/test_rag_chunker.py`` と同じ組み立て)。"""
    settings = rag.load_settings(write_rag_settings(vault_copy.parent))
    chunks: list[rag.Chunk] = []
    for entry in rag.iter_vault_files(settings):
        parsed = rag.parse_note(
            entry.relpath, rag.read_note_text(settings, entry.relpath)
        )
        chunks.extend(rag.chunk_note(parsed, settings.chunk))
    return settings, tuple(chunks)


def fill(store: VectorStore, records: Sequence[ChunkRecord]) -> None:
    """レコード列をノート単位にまとめて投入する。"""
    grouped: dict[str, list[ChunkRecord]] = {}
    for record in records:
        if record.relpath not in grouped:
            grouped[record.relpath] = []
        grouped[record.relpath].append(record)
    for relpath, note_records in grouped.items():
        store.replace_note(relpath, note_records)


def make_jsonl_store(directory: Path) -> VectorStore:
    return JsonlVectorStore(directory / "index" / CHUNKS_FILENAME)


def make_memory_store(directory: Path) -> VectorStore:
    return InMemoryVectorStore()


STORE_FACTORIES: tuple[StoreFactory, ...] = (make_jsonl_store, make_memory_store)
STORE_IDS = ("jsonl", "memory")


# --------------------------------------------------------------------------
# D-39 guard: 2 実装が同じ契約を満たす
# --------------------------------------------------------------------------


@pytest.mark.parametrize("make_store", STORE_FACTORIES, ids=STORE_IDS)
def test_every_store_implementation_passes_the_same_contract(
    make_store: StoreFactory, tmp_path: Path
) -> None:
    """D-39 guard: :class:`rag.VectorStore` の契約を 2 実装で同一に固定する。

    差分更新 (T6) は Protocol しか知らないため、実装ごとに振る舞いが分かれると
    「片方の実装でしかテストしていない経路」が生まれる。とくに ``replace_note``
    が追記に化けると、1 文字編集しただけのノートの古いチャンクが索引に残り続け、
    例外もテスト失敗も出ないまま検索結果だけが二重になる。
    """
    store = make_store(tmp_path)

    # (1) 空のストアは 0 件。
    assert list(store.iter_records()) == []
    assert store.note_chunk_counts() == {}

    # (2) ノート単位で投入できる。
    store.replace_note("a.md", [make_record("a.md", 0), make_record("a.md", 1)])
    store.replace_note("b.md", [make_record("b.md", 0)])
    assert store.note_chunk_counts() == {"a.md": 2, "b.md": 1}

    # (3) 2 回目の replace_note は**置換**であって追記ではない。
    store.replace_note("a.md", [make_record("a.md", 0, body="書き直した本文")])
    assert store.note_chunk_counts() == {"a.md": 1, "b.md": 1}
    assert [
        record.body for record in store.iter_records() if record.relpath == "a.md"
    ] == ["書き直した本文"]

    # (4) note_chunk_counts() は iter_records() の集計と一致する。
    assert store.note_chunk_counts() == dict(
        Counter(record.relpath for record in store.iter_records())
    )

    # (5) 行順は (relpath, ordinal) 昇順で決まる (投入順に依存しない)。
    store.replace_note("a.md", [make_record("a.md", 1), make_record("a.md", 0)])
    assert [record.ordinal for record in store.iter_records()][:2] == [0, 1]

    # (6) 存在しないノートの delete_note は例外にならない。
    store.delete_note("missing.md")

    # (7) delete_note の後は 0 件。
    store.delete_note("a.md")
    store.delete_note("b.md")
    assert list(store.iter_records()) == []
    assert store.note_chunk_counts() == {}

    # (8) commit はどちらの実装でも呼べる (永続化の有無を呼び出し側が知らない)。
    store.commit()


@pytest.mark.parametrize("make_store", STORE_FACTORIES, ids=STORE_IDS)
def test_replacing_a_note_with_no_records_removes_it(
    make_store: StoreFactory, tmp_path: Path
) -> None:
    """空のチャンク列での置換は削除と同じ (0 件のノートを表に残さない)。

    残すと ``note_chunk_counts()`` に 0 件のノートが現れ、``iter_records()`` の
    集計 (0 件のノートは現れない) と食い違う。契約 (4) が壊れる。
    """
    store = make_store(tmp_path)
    store.replace_note("a.md", [make_record("a.md", 0)])

    store.replace_note("a.md", [])

    assert store.note_chunk_counts() == {}
    assert list(store.iter_records()) == []


@pytest.mark.parametrize("make_store", STORE_FACTORIES, ids=STORE_IDS)
def test_records_from_another_note_are_rejected(
    make_store: StoreFactory, tmp_path: Path
) -> None:
    """``relpath`` の食い違いは黙って受け入れない。

    受け入れると、そのノートを次に差し替えたときに他ノートのチャンクが道連れで
    消える (削除の単位が relpath なので、表の鍵と中身が食い違う)。
    """
    store = make_store(tmp_path)

    with pytest.raises(ConfigError) as excinfo:
        store.replace_note("a.md", [make_record("b.md", 0)])

    assert "a.md" in str(excinfo.value)


def test_both_implementations_satisfy_the_protocol(tmp_path: Path) -> None:
    """静的に :class:`rag.VectorStore` を満たすこと (mypy が検査する)。"""
    stores: tuple[VectorStore, ...] = (
        JsonlVectorStore(tmp_path / CHUNKS_FILENAME),
        InMemoryVectorStore(),
    )

    assert [type(store).__name__ for store in stores] == [
        "JsonlVectorStore",
        "InMemoryVectorStore",
    ]


# --------------------------------------------------------------------------
# JsonlVectorStore 固有: 往復・バイト一致・原子性
# --------------------------------------------------------------------------


def test_a_missing_index_file_starts_an_empty_store(tmp_path: Path) -> None:
    """索引がまだ無い状態は「空」であって失敗ではない (初回実行)。"""
    path = tmp_path / "index" / CHUNKS_FILENAME
    store = JsonlVectorStore(path)

    assert list(store.iter_records()) == []
    assert not path.exists()

    store.commit()

    assert path.is_file()
    assert path.read_text(encoding="utf-8") == ""


def test_a_jsonl_store_round_trips_every_record_exactly(
    sample_vault_copy: Path, tmp_path: Path
) -> None:
    """書いて読み直したレコードが**完全一致**する (float を含む)。

    合成 vault の全チャンクで回す。1 件でも値が変質すると、次回の実行で
    「変更なし」と判定されたノートのベクトルだけが静かに別物になる。
    """
    _, chunks = sample_chunks(sample_vault_copy)
    records = [record_from_chunk(chunk) for chunk in chunks]
    path = tmp_path / "index" / CHUNKS_FILENAME
    store = JsonlVectorStore(path)
    fill(store, records)
    store.commit()
    expected = list(store.iter_records())

    reloaded = list(JsonlVectorStore(path).iter_records())

    assert len(expected) == SAMPLE_CHUNK_COUNT
    assert reloaded == expected
    assert [record.vector for record in reloaded] == [
        record.vector for record in expected
    ]


def test_committing_the_same_content_twice_is_byte_identical(tmp_path: Path) -> None:
    """同じ内容なら ``chunks.jsonl`` は常に同じバイト列になる。

    **投入順とノート内の並び順を変えた 2 つのストアを比べる**。行順を
    ``(relpath, ordinal)`` 昇順に固定していないと、再実行のたびに成果物の
    バイト列が変わり、L310 の「バイト一致」を再現性の証拠に使えなくなる。
    """
    records = [
        make_record("notes/b.md", 0),
        make_record("notes/b.md", 1),
        make_record("a.md", 0),
        make_record("a.md", 1),
        make_record("a.md", 2),
    ]
    first_path = tmp_path / "first" / CHUNKS_FILENAME
    first = JsonlVectorStore(first_path)
    fill(first, records)
    first.commit()
    second_path = tmp_path / "second" / CHUNKS_FILENAME
    second = JsonlVectorStore(second_path)
    fill(second, list(reversed(records)))
    second.commit()

    assert first_path.read_bytes() == second_path.read_bytes()

    # 読み直して何も変えずに commit しても同じバイト列になる。
    before = first_path.read_bytes()
    JsonlVectorStore(first_path).commit()

    assert first_path.read_bytes() == before
    assert before.decode("utf-8").count("\n") == len(records)


def test_a_failed_commit_leaves_the_previous_index_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``commit()`` は原子的。失敗しても既存の索引は 1 バイトも変わらない。

    一時ファイルへ全量を書いてから :func:`os.replace` で差し替えるので、
    差し替えに失敗した時点の索引は「前回の完全な内容」である。失敗を
    :class:`ConfigError` に翻訳するのは ``harness/report.py`` と同じ形。
    """
    path = tmp_path / "index" / CHUNKS_FILENAME
    store = JsonlVectorStore(path)
    store.replace_note("a.md", [make_record("a.md", 0)])
    store.commit()
    before = path.read_bytes()
    store.replace_note("b.md", [make_record("b.md", 0)])
    real_replace = os.replace

    def guarded_replace(
        src: str | os.PathLike[str], dst: str | os.PathLike[str]
    ) -> None:
        if str(src).endswith(".tmp"):
            raise OSError(errno.EACCES, "Permission denied")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", guarded_replace)

    with pytest.raises(ConfigError) as excinfo:
        store.commit()

    assert path.read_bytes() == before
    assert sorted(item.name for item in path.parent.iterdir()) == [CHUNKS_FILENAME]
    assert "Permission denied" in excinfo.value.remediation


# --------------------------------------------------------------------------
# D-41 guard: 本文を 2 度書かない / モデル名と次元を持たない
# --------------------------------------------------------------------------


def test_the_persisted_record_never_stores_the_body_twice(
    sample_vault_copy: Path, tmp_path: Path
) -> None:
    """D-41 guard: JSONL に ``embed_text`` / ``model`` / ``dimensions`` が無い。

    埋め込み文字列を保存すると、``body`` だけ差し替わった (あるいは
    ``embed_text`` だけ差し替わった) レコードが索引に残せる状態になる。ここでは
    合成 vault の全チャンクについて、保存したキーが :class:`rag.ChunkRecord` の
    宣言と完全に一致し、本文が 1 行に 1 度しか現れず、
    :func:`rag.render_embed_text` による再構成が :func:`rag.chunk_note` の
    ``embed_text`` と一致することを確かめる。
    """
    settings, chunks = sample_chunks(sample_vault_copy)
    path = tmp_path / "index" / CHUNKS_FILENAME
    store = JsonlVectorStore(path)
    fill(store, [record_from_chunk(chunk) for chunk in chunks])
    store.commit()
    field_names = {field.name for field in dataclasses.fields(ChunkRecord)}
    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == SAMPLE_CHUNK_COUNT
    for line in lines:
        payload = json.loads(line)
        assert set(payload) == field_names, payload.keys()
        encoded_body = json.dumps(payload["body"], ensure_ascii=False)[1:-1]
        assert line.count(encoded_body) == 1, "本文が 1 行に 2 度現れています"

    by_chunk_id = {chunk.chunk_id: chunk for chunk in chunks}
    reconstructed = {
        record.chunk_id: rag.render_embed_text(
            record.heading_path, record.body, settings.chunk.heading_separator
        )
        for record in JsonlVectorStore(path).iter_records()
    }

    assert len(reconstructed) == SAMPLE_CHUNK_COUNT
    assert reconstructed == {
        chunk_id: chunk.embed_text for chunk_id, chunk in by_chunk_id.items()
    }


def test_an_unknown_key_in_the_index_is_rejected(tmp_path: Path) -> None:
    """未知のキーを持つ行は読み込み時に落ちる (D-41 を読み取り方向から守る)。

    ``embed_text`` を書き足した索引を「読めてしまう」と、書き出し側だけを直しても
    古い索引が黙って生き残る。
    """
    path = tmp_path / CHUNKS_FILENAME
    payload = json.loads(
        json.dumps(
            {
                "chunk_id": "a.md#0000",
                "relpath": "a.md",
                "ordinal": 0,
                "part_index": 0,
                "heading_path": ["タイトル"],
                "body": "本文",
                "estimated_tokens": 2,
                "tags": [],
                "links": [],
                "vector": [0.5],
                "embed_text": "タイトル\n\n本文",
            }
        )
    )
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        JsonlVectorStore(path)

    assert "embed_text" in str(excinfo.value)


# --------------------------------------------------------------------------
# 読み込みの厳格さ (D-07 / D-08)
# --------------------------------------------------------------------------


def test_a_broken_line_never_leaks_the_note_body(tmp_path: Path) -> None:
    """壊れた行は :class:`ConfigError`。例外に本文も絶対パスも載せない。

    索引の中身はノート本文そのものであり、例外メッセージに載せると
    CLAUDE.md のログ出力ルールに反する経路が索引側にできる。
    """
    secret = "極秘のノート本文"
    path = tmp_path / "index" / CHUNKS_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"body": secret, "ordinal": "ゼロ"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as excinfo:
        JsonlVectorStore(path)

    message = f"{excinfo.value} {excinfo.value.remediation}"
    assert secret not in message
    assert str(tmp_path) not in message
    assert f"{CHUNKS_FILENAME}:1" in message


def test_a_line_that_is_not_json_is_a_config_error(tmp_path: Path) -> None:
    """JSON として壊れている行も同じ経路で :class:`ConfigError` になる。"""
    path = tmp_path / CHUNKS_FILENAME
    path.write_text('{"chunk_id": "a.md#0000"\n', encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        JsonlVectorStore(path)

    assert f"{CHUNKS_FILENAME}:1" in str(excinfo.value)


def test_mixed_vector_dimensions_are_rejected(tmp_path: Path) -> None:
    """次元の違うベクトルが同居した索引は読み込み時に落ちる。

    次元はレコードに書かない (D-41) ので、ファイル単位の整合はここでしか
    見られない。同居を許すとコサイン類似度が無意味になっても何も落ちない
    (§2.2 の「静かに壊れる」と同型)。
    """
    path = tmp_path / CHUNKS_FILENAME
    store = JsonlVectorStore(path)
    store.replace_note("a.md", [make_record("a.md", 0)])
    store.replace_note("b.md", [make_record("b.md", 0, dimensions=DIMENSIONS - 1)])
    store.commit()

    with pytest.raises(ConfigError) as excinfo:
        JsonlVectorStore(path)

    assert "次元" in str(excinfo.value)
