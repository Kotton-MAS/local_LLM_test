"""索引の再現条件・マニフェスト・差分の計画 (``rag/indexer.py``) の検証。

- ``test_the_fingerprint_covers_every_input_that_changes_a_vector`` …
  **D-35 guard**。「入れる」と「入れない」を **1 本のテストで両方向から**固定
  する。片方だけを固定すると、入力を足す変異 (``source_path`` を入れる) も
  落とす変異 (``chunk`` を外す) もどちらか一方が検出できない。
- ``test_changing_chunk_settings_invalidates_the_whole_index`` … **E31**。
- ``test_the_token_algorithm_is_part_of_the_fingerprint`` … **E32**。近似
  トークナイザの CJK 範囲表を差し替えると fingerprint が動くこと (手書きの
  ``VERSION`` 定数では上げ忘れが起きる、D-19 と同型)。
- ``test_changing_the_embedding_model_invalidates_the_index`` … **E34**。
  §2.2 の「次元も意味空間も違うベクトルが同居する」欠陥を塞ぐ中核。

差分更新の**実行**側 (T6 / :func:`rag.build_index`) は同じファイルの後半で
検証する:

- ``test_touching_a_note_without_changing_bytes_reindexes_nothing`` …
  **D-36 guard**。mtime を差分判定にもマニフェストにも使わない。
- ``test_an_unchanged_rerun_leaves_the_index_byte_identical`` … **D-37 guard**。
  要件書 L310 そのもの。確定の順序 (``store.commit()`` → ``write_manifest()``)
  は ``test_the_manifest_is_written_only_after_the_store_is_committed`` が固定する。
- ``test_a_failed_note_is_never_recorded_as_indexed`` … **D-38 guard**。部分的に
  成功したチャンクを載せると、次回は ``sha256`` が一致して「変更なし」と判定
  され、**欠けたまま永久に治らない**。

埋め込みは ``httpx.MockTransport`` + 決定論的フェイク (``tests/conftest.py`` の
:func:`fake_embedding_vector`) で行う。実 HTTP は 1 バイトも出さないが、
``llmkit`` の例外翻訳表は本番と同じ経路を通る。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import httpx
import pytest
from conftest import (
    DEFAULT_CONFIG,
    FAKE_EMBEDDING_DIMENSIONS,
    FAKE_EMBEDDING_MODEL,
    EmbeddingIntercept,
    RecordingTransport,
    fake_embedding_transport,
    fake_embedding_vector,
    write_config_variant,
    write_rag_settings,
)

import harness
import llmkit
import rag
from llmkit import (
    AppConfig,
    ConfigError,
    ContextLengthError,
    ModelSpec,
    OutOfMemoryError,
    RuntimeUnavailableError,
    UpstreamError,
)
from rag import chunker
from rag.indexer import fingerprint_digest, fingerprint_inputs

#: 合成 vault の索引対象ノート数 (``tests/test_rag_vault.py`` と同じ前提)。
SAMPLE_NOTE_COUNT = 11

#: 合成 vault のチャンク総数 (``tests/test_rag_chunker.py`` と同じ前提)。
SAMPLE_CHUNK_COUNT = 24

#: テスト用の低次元。次元そのものは検査対象ではないので短くする。
DIMENSIONS = 8

#: 索引を書いたランタイムが名乗ったモデル名 (T6 が実際の応答から埋める)。
REPORTED_MODEL = "test-embedding"


# --------------------------------------------------------------------------
# 小道具
# --------------------------------------------------------------------------


def fake_vector(text: str) -> tuple[float, ...]:
    """テキストから決定論的にベクトルを作る (``tests/test_rag_store.py`` と同型)。"""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return tuple((digest[index] - 127.5) / 256.0 for index in range(DIMENSIONS))


def make_settings(tmp_path: Path) -> rag.RagSettings:
    """fingerprint の掃引用の設定。**ファイルには 1 バイトも触れない**。

    :func:`rag.load_settings` を通さずに組み立てるのは、``vault_dir`` /
    ``index_dir`` / ``source_path`` を実在しないパスへ振るため (fingerprint は
    パスを読まないので、掃引にディレクトリを用意する必要が無いこと自体が
    D-35 の主張の一部である)。
    """
    return rag.RagSettings(
        vault_id="sample",
        vault_dir=tmp_path / "vault",
        index_dir=tmp_path / "index",
        include_globs=rag.DEFAULT_INCLUDE_GLOBS,
        exclude_globs=rag.DEFAULT_EXCLUDE_GLOBS,
        chunk=rag.ChunkSettings(),
        embed=rag.EmbedSettings(),
        source_path=tmp_path / "rag.toml",
    )


def local_config() -> AppConfig:
    """``configs/default.toml`` (``is_local = true``。api_key を要求しない)。"""
    return llmkit.load_config(DEFAULT_CONFIG)


def sample_settings(vault_copy: Path) -> rag.RagSettings:
    """合成 vault の複製を指す設定 (索引先は ``tmp_path/index``)。"""
    return rag.load_settings(write_rag_settings(vault_copy.parent))


def note_digest(settings: rag.RagSettings, relpath: str) -> str:
    """ノートの生バイト列の sha256 (差分判定の唯一の材料、D-36)。"""
    return hashlib.sha256(rag.read_note_bytes(settings, relpath)).hexdigest()


def indexed_state(
    settings: rag.RagSettings, config: AppConfig
) -> tuple[rag.IndexManifest, rag.InMemoryVectorStore]:
    """「いま索引済み」の状態 (マニフェスト + ストア) を合成 vault から作る。

    チャンク数は実際に :func:`rag.chunk_note` を通した値なので、空ノート
    (0 チャンク) も現実どおり 0 件として記録される。T4 決定4 の申し送り
    (0 チャンクのノートはストアの表に現れない) が踏まれる経路である。
    """
    spec = llmkit.resolve_embedding_spec(config)
    notes: list[rag.ManifestNote] = []
    records: list[rag.ChunkRecord] = []
    for entry in rag.iter_vault_files(settings):
        parsed = rag.parse_note(
            entry.relpath, rag.read_note_text(settings, entry.relpath)
        )
        chunks = rag.chunk_note(parsed, settings.chunk)
        records += [
            rag.ChunkRecord(
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
            for chunk in chunks
        ]
        notes.append(
            rag.ManifestNote(
                relpath=entry.relpath,
                sha256=note_digest(settings, entry.relpath),
                chunk_count=len(chunks),
            )
        )
    manifest = rag.IndexManifest(
        schema_version=rag.INDEX_SCHEMA_VERSION,
        index_fingerprint=rag.index_fingerprint(settings, spec),
        fingerprint_inputs=fingerprint_inputs(settings, spec),
        embedding=rag.ManifestEmbedding(
            reported_model=REPORTED_MODEL, dimensions=DIMENSIONS
        ),
        notes=tuple(notes),
        totals=rag.ManifestTotals(
            notes=len(notes), chunks=sum(note.chunk_count for note in notes)
        ),
    )
    return manifest, rag.InMemoryVectorStore(records)


def replace_chunk(
    settings: rag.RagSettings, chunk: rag.ChunkSettings
) -> rag.RagSettings:
    return dataclasses.replace(settings, chunk=chunk)


# --------------------------------------------------------------------------
# D-35 guard: fingerprint に入れるもの / 入れないもの
# --------------------------------------------------------------------------

#: fingerprint を **1 ビットも動かしてはいけない**掃引 (§4 論点1 の「入れない」)。
#: どれもベクトルの中身に影響しない値であり、入れると偽の全再構築を生む。
Mutation = Callable[
    [rag.RagSettings, AppConfig, Path], tuple[rag.RagSettings, AppConfig]
]

_IRRELEVANT_MUTATIONS: tuple[tuple[str, Mutation], ...] = (
    (
        "source_path",
        lambda settings, config, tmp_path: (
            dataclasses.replace(settings, source_path=tmp_path / "moved" / "rag.toml"),
            config,
        ),
    ),
    (
        "vault_dir",
        lambda settings, config, tmp_path: (
            dataclasses.replace(settings, vault_dir=tmp_path / "another-vault"),
            config,
        ),
    ),
    (
        "index_dir",
        lambda settings, config, tmp_path: (
            dataclasses.replace(settings, index_dir=tmp_path / "another-index"),
            config,
        ),
    ),
    (
        "runtime.base_url",
        lambda settings, config, tmp_path: (
            settings,
            llmkit.load_config(
                write_config_variant(
                    tmp_path,
                    {"http://localhost:11434/v1": "http://127.0.0.1:18888/v1"},
                    name="other-base-url.toml",
                )
            ),
        ),
    ),
    (
        "embed.batch_size",
        lambda settings, config, tmp_path: (
            dataclasses.replace(
                settings,
                embed=rag.EmbedSettings(batch_size=settings.embed.batch_size * 4),
            ),
            config,
        ),
    ),
    (
        "include_globs",
        lambda settings, config, tmp_path: (
            dataclasses.replace(
                settings, include_globs=(*settings.include_globs, "**/*.markdown")
            ),
            config,
        ),
    ),
    (
        "exclude_globs",
        lambda settings, config, tmp_path: (
            dataclasses.replace(
                settings, exclude_globs=(*settings.exclude_globs, "archive/**")
            ),
            config,
        ),
    ),
)


def vector_changing_variants(
    settings: rag.RagSettings, spec: ModelSpec
) -> Iterator[tuple[str, rag.RagSettings, ModelSpec]]:
    """fingerprint が**必ず動かなければならない**掃引 (§4 論点1 の「入れる」)。

    CJK 範囲表だけは値の差し替えではなくモジュール属性の差し替えになるため、
    テスト本体で ``monkeypatch`` を使って別に確かめる。
    """
    chunk = settings.chunk
    yield (
        "chunk.max_tokens",
        replace_chunk(settings, dataclasses.replace(chunk, max_tokens=200)),
        spec,
    )
    yield (
        "chunk.cjk_chars_per_token",
        replace_chunk(settings, dataclasses.replace(chunk, cjk_chars_per_token=1.5)),
        spec,
    )
    yield (
        "chunk.ascii_chars_per_token",
        replace_chunk(settings, dataclasses.replace(chunk, ascii_chars_per_token=3.0)),
        spec,
    )
    yield (
        "chunk.heading_separator",
        replace_chunk(settings, dataclasses.replace(chunk, heading_separator=" / ")),
        spec,
    )
    yield (
        "embedding.model_id",
        settings,
        dataclasses.replace(spec, model_id="another-embedding"),
    )
    yield (
        "embedding.served_name",
        settings,
        dataclasses.replace(spec, served_name="another/served-name"),
    )
    yield ("vault_id", dataclasses.replace(settings, vault_id="other"), spec)


@pytest.mark.parametrize(
    ("label", "mutate"),
    _IRRELEVANT_MUTATIONS,
    ids=[label for label, _ in _IRRELEVANT_MUTATIONS],
)
def test_the_fingerprint_covers_every_input_that_changes_a_vector(
    label: str,
    mutate: Mutation,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-35 guard: 入力の**過不足**を 1 本で固定する。

    ``index_fingerprint`` は「前提が変わったら再処理する」ための値なので、
    2 方向の誤りがどちらも致命的になる。

    - 入れ忘れ → ``max_tokens`` を変えた再実行が「変更なし」と判定され、別々の
      前提で切られたチャンクが同じ索引に同居する (§2.2、例外もテスト失敗も
      出ない)。
    - 入れすぎ → 設定ファイルを移動しただけで全再構築が走り、実 vault の絶対
      パス (利用者名を含む) が索引成果物と画面に出る経路ができる。

    したがってこのテストは、掃引した 1 項目で**変わらない**ことと、ベクトルを
    変える 7 項目 + CJK 範囲表で**変わる**ことを同時に主張する。
    """
    settings = make_settings(tmp_path)
    config = local_config()
    spec = llmkit.resolve_embedding_spec(config)
    baseline = rag.index_fingerprint(settings, spec)

    mutated_settings, mutated_config = mutate(settings, config, tmp_path)
    assert (mutated_settings, mutated_config) != (settings, config), (
        f"掃引が値を変えていません: {label}"
    )
    mutated = rag.index_fingerprint(
        mutated_settings, llmkit.resolve_embedding_spec(mutated_config)
    )

    assert mutated == baseline, (
        f"ベクトルに影響しない値が fingerprint を変えた: {label}"
    )

    variants = vector_changing_variants(settings, spec)
    for name, changed_settings, changed_spec in variants:
        assert rag.index_fingerprint(changed_settings, changed_spec) != baseline, (
            f"ベクトルを変える入力が fingerprint に入っていない: {name}"
        )
    with monkeypatch.context() as patched:
        patched.setattr(
            chunker, "_CJK_RANGES", (*chunker._CJK_RANGES, (0x1F300, 0x1F5FF))
        )
        assert rag.index_fingerprint(settings, spec) != baseline, (
            "近似トークナイザの CJK 範囲表が fingerprint に入っていない"
        )


def test_the_digest_uses_the_same_canonical_json_rules_as_the_harness(
    tmp_path: Path,
) -> None:
    """``harness.fingerprint_digest`` と同じ dict に同じ値を返す。

    層構造上 ``rag`` は ``harness`` を import できないため、正規化 JSON の
    3 行だけは重複させている (§3 ソフト制約)。重複を許す条件が「両者が一致する
    ことをテストが固定していること」なので、ここで固定する。
    """
    settings = make_settings(tmp_path)
    spec = llmkit.resolve_embedding_spec(local_config())
    inputs = fingerprint_inputs(settings, spec)

    assert fingerprint_digest is not harness.fingerprint_digest
    assert fingerprint_digest(inputs) == harness.fingerprint_digest(inputs)

    # 非 ASCII・入れ子・浮動小数・真偽値・None を含む dict でも一致する
    # (ensure_ascii / sort_keys の指定が片方だけ違うと、ここで差が出る)。
    payload: dict[str, object] = {
        "日本語": ["あ", 1, 2.5, True, None],
        "b": {"z": 1, "a": {"深い": "値"}},
    }
    assert fingerprint_digest(payload) == harness.fingerprint_digest(payload)


# --------------------------------------------------------------------------
# E31 / E32 / E34: 前提が変わったら索引全体が無効になる
# --------------------------------------------------------------------------

ChunkMutation = Callable[[rag.ChunkSettings], rag.ChunkSettings]

_CHUNK_MUTATIONS: tuple[tuple[str, ChunkMutation], ...] = (
    ("max_tokens", lambda chunk: dataclasses.replace(chunk, max_tokens=200)),
    (
        "cjk_chars_per_token",
        lambda chunk: dataclasses.replace(chunk, cjk_chars_per_token=1.5),
    ),
    (
        "ascii_chars_per_token",
        lambda chunk: dataclasses.replace(chunk, ascii_chars_per_token=3.0),
    ),
    (
        "heading_separator",
        lambda chunk: dataclasses.replace(chunk, heading_separator=" / "),
    ),
)


@pytest.mark.parametrize(
    ("label", "mutate"),
    _CHUNK_MUTATIONS,
    ids=[label for label, _ in _CHUNK_MUTATIONS],
)
def test_changing_chunk_settings_invalidates_the_whole_index(
    label: str, mutate: ChunkMutation, sample_vault_copy: Path
) -> None:
    """E31: ``[chunk]`` の 4 キーはどれも索引全体の前提である。

    バイト列が 1 ビットも変わっていないノートまで ``changed`` に入ることが
    要点。``sha256`` だけで判定すると、240 で切ったチャンクと 200 で切った
    チャンクが同じ索引に同居したまま誰も気づかない (§2.2)。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    manifest, store = indexed_state(settings, config)

    before = rag.plan_index(settings, config, store=store, manifest=manifest)
    assert before.pending == (), label
    assert len(before.unchanged) == SAMPLE_NOTE_COUNT

    changed_settings = replace_chunk(settings, mutate(settings.chunk))
    after = rag.plan_index(changed_settings, config, store=store, manifest=manifest)

    assert after.fingerprint != before.fingerprint, label
    assert after.full_rebuild is True
    assert after.unchanged == ()
    assert len(after.changed) == SAMPLE_NOTE_COUNT
    assert after.new == ()


def test_the_token_algorithm_is_part_of_the_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E32: CJK 範囲表そのものが fingerprint に入る。

    ``chunk_algorithm_sha256`` を手書きの ``VERSION = 1`` 定数にすると、表を
    変えたのに定数を上げ忘れた実行が「前提は同じ」と名乗れる (D-19 と同型)。
    表から導出していればそれが起こらないことをここで固定する。

    逆に、**並べ替えただけでは変わらない**ことも固定する。
    :func:`rag.chunker._is_cjk` は ``any`` で判定するので並び順は挙動に影響
    せず、意味の変わらない編集で全再構築が走ってはならない。
    """
    settings = make_settings(tmp_path)
    spec = llmkit.resolve_embedding_spec(local_config())
    original = chunker._CJK_RANGES
    before_inputs = fingerprint_inputs(settings, spec)
    before = rag.index_fingerprint(settings, spec)

    monkeypatch.setattr(chunker, "_CJK_RANGES", (*original, (0x1F300, 0x1F5FF)))
    after_inputs = fingerprint_inputs(settings, spec)

    assert (
        after_inputs["chunk_algorithm_sha256"]
        != before_inputs["chunk_algorithm_sha256"]
    )
    assert rag.index_fingerprint(settings, spec) != before

    monkeypatch.setattr(chunker, "_CJK_RANGES", tuple(reversed(original)))

    assert rag.index_fingerprint(settings, spec) == before


def test_changing_the_embedding_model_invalidates_the_index(
    sample_vault_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E34: ``profiles[active].embedding`` を変えると索引全体が無効になる。

    §2.2 の中核。モデルを差し替えると次元も意味空間も違うベクトルが生まれる
    ので、既存のチャンクを 1 件でも再利用してはならない。

    2 つの設定は ``embedding`` の 1 行だけが違う (``is_local = false`` は
    どちらにも入れてある: カタログには埋め込みモデルが 1 つしか無く、外部 API
    経路でないと「別の埋め込みモデル」を作れないため)。
    """
    monkeypatch.setenv("LLMKIT_API_KEY", "sk-test-do-not-leak-0123456789")
    external = {"is_local = true": "is_local = false"}
    config = llmkit.load_config(
        write_config_variant(tmp_path, external, name="embedding-a.toml")
    )
    other = llmkit.load_config(
        write_config_variant(
            tmp_path,
            {**external, 'embedding = "ruri-v3-310m"': 'embedding = "other-embedder"'},
            name="embedding-b.toml",
        )
    )
    settings = sample_settings(sample_vault_copy)
    manifest, store = indexed_state(settings, config)

    before = rag.plan_index(settings, config, store=store, manifest=manifest)
    plan = rag.plan_index(settings, other, store=store, manifest=manifest)

    assert before.pending == ()

    assert plan.fingerprint != manifest.index_fingerprint
    assert plan.full_rebuild is True
    assert plan.unchanged == ()
    assert len(plan.changed) == SAMPLE_NOTE_COUNT


# --------------------------------------------------------------------------
# マニフェスト
# --------------------------------------------------------------------------


def test_writing_the_same_manifest_twice_is_byte_identical(
    sample_vault_copy: Path,
) -> None:
    """L310 の前提: 同じ内容の manifest は常に同じバイト列になる。

    時刻・run_id・実行ごとに変わる値を 1 つでも持たせると、無変更の再実行で
    成果物がバイト一致するという受け入れ条件が原理的に満たせなくなる。
    """
    settings = sample_settings(sample_vault_copy)
    manifest, _ = indexed_state(settings, local_config())

    path = rag.write_manifest(settings, manifest)
    first = path.read_bytes()
    second_path = rag.write_manifest(settings, manifest)

    assert second_path == path
    assert second_path.read_bytes() == first


def test_a_manifest_round_trips_through_the_index_directory(
    sample_vault_copy: Path,
) -> None:
    """書いて読み直すと完全に同じ :class:`rag.IndexManifest` になる。"""
    settings = sample_settings(sample_vault_copy)
    manifest, _ = indexed_state(settings, local_config())

    rag.write_manifest(settings, manifest)
    loaded = rag.load_manifest(settings)

    assert loaded == manifest
    assert loaded is not None
    assert loaded.totals.notes == SAMPLE_NOTE_COUNT
    assert loaded.totals.chunks == SAMPLE_CHUNK_COUNT


def test_the_manifest_records_no_time_no_run_id_and_no_absolute_path(
    sample_vault_copy: Path,
) -> None:
    """マニフェストに載ってよい 6 キーを固定する (D-37 の前提)。

    絶対パスを載せると、実 vault のパス (利用者名を含む) が ``status`` 表示と
    ログに出る経路ができる。時刻・run_id を載せるとバイト一致が恒偽になる
    (D-20 が ``run_id`` / ``started_at_utc`` を除外したのと同じ理由)。
    """
    settings = sample_settings(sample_vault_copy)
    manifest, _ = indexed_state(settings, local_config())
    path = rag.write_manifest(settings, manifest)

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert set(payload) == {
        "schema_version",
        "index_fingerprint",
        "fingerprint_inputs",
        "embedding",
        "notes",
        "totals",
    }
    for forbidden in ("run_id", "started_at", "mtime", "timestamp", "elapsed"):
        assert forbidden not in text
    assert str(settings.vault_dir) not in text
    assert str(settings.index_dir) not in text
    assert str(settings.source_path) not in text
    assert not [note for note in payload["notes"] if note["relpath"].startswith("/")]


def test_load_manifest_returns_none_before_the_first_index(
    sample_vault_copy: Path,
) -> None:
    """索引がまだ無い状態は例外ではなく ``None`` (欠測を 0 で埋めない、D-07)。"""
    settings = sample_settings(sample_vault_copy)

    assert rag.load_manifest(settings) is None
    assert not settings.index_dir.exists()


def test_a_broken_manifest_never_leaks_the_index_directory_path(
    sample_vault_copy: Path,
) -> None:
    """壊れた manifest は :class:`ConfigError`。メッセージはファイル名だけ。"""
    settings = sample_settings(sample_vault_copy)
    settings.index_dir.mkdir(parents=True)
    rag.manifest_path(settings).write_text("{ これは JSON ではない", encoding="utf-8")

    with pytest.raises(ConfigError) as error:
        rag.load_manifest(settings)

    message = str(error.value)
    assert rag.MANIFEST_FILENAME in message
    assert str(settings.index_dir) not in message


def test_a_manifest_whose_digest_disagrees_with_its_inputs_is_rejected(
    sample_vault_copy: Path,
) -> None:
    """記録した fingerprint と、その入力から再計算した値が食い違ったら止める。

    両方を載せる以上、食い違いは書き換えか破損の証拠であり、どちらが本当かを
    決められない。そのまま差分判定に使うと「前提が変わったのに変わっていない
    と名乗る索引」を信じることになる。
    """
    settings = sample_settings(sample_vault_copy)
    manifest, _ = indexed_state(settings, local_config())
    rag.write_manifest(settings, manifest)
    path = rag.manifest_path(settings)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["index_fingerprint"] = "0" * 64
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="index_fingerprint"):
        rag.load_manifest(settings)


def test_an_unknown_schema_version_is_never_silently_overwritten(
    sample_vault_copy: Path,
) -> None:
    """知らない形式の索引は黙って使わず、作り直しを促して止まる。"""
    settings = sample_settings(sample_vault_copy)
    manifest, _ = indexed_state(settings, local_config())
    rag.write_manifest(settings, manifest)
    path = rag.manifest_path(settings)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = rag.INDEX_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="--rebuild"):
        rag.load_manifest(settings)


def test_totals_that_disagree_with_the_notes_cannot_be_written(
    sample_vault_copy: Path,
) -> None:
    """``totals`` は ``notes[]`` から導出した値としか書けない (D-19 の趣旨)。

    読み込み側では検査しない。手で壊れた manifest は拒否ではなく**再処理**で
    直すのが :func:`rag.plan_index` の役割であり、そこを塞ぐと自己修復の経路が
    到達不能になる。
    """
    settings = sample_settings(sample_vault_copy)
    manifest, _ = indexed_state(settings, local_config())
    inconsistent = dataclasses.replace(
        manifest,
        totals=rag.ManifestTotals(
            notes=manifest.totals.notes, chunks=manifest.totals.chunks + 1
        ),
    )

    with pytest.raises(ConfigError, match="totals"):
        rag.write_manifest(settings, inconsistent)

    assert not rag.manifest_path(settings).exists()


# --------------------------------------------------------------------------
# 計画 (HTTP 0 回・書き込み 0 バイト)
# --------------------------------------------------------------------------


def test_planning_never_touches_the_runtime_or_the_index_directory(
    sample_vault_copy: Path,
) -> None:
    """合成 vault の 11 ノートを HTTP 0 回・書き込み 0 バイトで分類する。

    埋め込みクライアントは実際に生成しておく (「クライアントが無いから 0 回」
    ではなく「あっても呼ばない」ことの証拠にする)。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    transport = RecordingTransport()
    with transport.client() as http_client:
        llmkit.create_embedding_client(config, http_client=http_client)
        plan = rag.plan_index(
            settings, config, store=rag.InMemoryVectorStore(), manifest=None
        )

    assert transport.call_count == 0
    assert not settings.index_dir.exists()
    assert plan.total_notes == SAMPLE_NOTE_COUNT
    assert len(plan.new) == SAMPLE_NOTE_COUNT
    assert plan.changed == ()
    assert plan.unchanged == ()
    assert plan.deleted == ()
    assert plan.full_rebuild is True
    assert plan.pending == plan.new
    assert plan.fingerprint == rag.index_fingerprint(
        settings, llmkit.resolve_embedding_spec(config)
    )


def test_an_unchanged_vault_plans_no_work_at_all(sample_vault_copy: Path) -> None:
    """L310 の計画側: 何も変えずに計画し直すと再処理は 0 件。

    空ノート (0 チャンク) がストアの表に現れないこと (T4 決定4) を理由に
    毎回 ``changed`` へ落ちないことも、ここで同時に固定される。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    manifest, store = indexed_state(settings, config)

    plan = rag.plan_index(settings, config, store=store, manifest=manifest)

    assert plan.full_rebuild is False
    assert len(plan.unchanged) == SAMPLE_NOTE_COUNT
    assert plan.pending == ()
    assert plan.deleted == ()
    assert manifest.totals.chunks == SAMPLE_CHUNK_COUNT
    assert len(store.note_chunk_counts()) < SAMPLE_NOTE_COUNT, (
        "0 チャンクのノートがストアに現れない前提が崩れています (T4 決定4)"
    )


def test_only_the_edited_note_is_planned_for_reprocessing(
    sample_vault_copy: Path,
) -> None:
    """1 ノートを編集したら、そのノートだけが ``changed`` になる。"""
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    manifest, store = indexed_state(settings, config)
    target = "notes/project-alpha.md"
    edited = sample_vault_copy / target
    edited.write_text(
        edited.read_text(encoding="utf-8") + "\n追記した 1 行。\n", encoding="utf-8"
    )

    plan = rag.plan_index(settings, config, store=store, manifest=manifest)

    assert plan.changed == (target,)
    assert len(plan.unchanged) == SAMPLE_NOTE_COUNT - 1
    assert plan.new == ()
    assert plan.deleted == ()
    assert plan.full_rebuild is False


def test_a_note_removed_from_the_vault_is_planned_for_deletion(
    sample_vault_copy: Path,
) -> None:
    """vault から消えたノートは ``deleted`` に入り、他は再処理されない。"""
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    manifest, store = indexed_state(settings, config)
    target = "notes/weekly-review.md"
    (sample_vault_copy / target).unlink()

    plan = rag.plan_index(settings, config, store=store, manifest=manifest)

    assert plan.deleted == (target,)
    assert plan.pending == ()
    assert len(plan.unchanged) == SAMPLE_NOTE_COUNT - 1
    assert target not in plan.note_digests


def test_a_note_added_to_the_vault_is_the_only_new_one(
    sample_vault_copy: Path,
) -> None:
    """新しいノートだけが ``new`` に入る。"""
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    manifest, store = indexed_state(settings, config)
    (sample_vault_copy / "notes" / "added.md").write_text(
        "# 追加\n\n新しく足した本文。\n", encoding="utf-8"
    )

    plan = rag.plan_index(settings, config, store=store, manifest=manifest)

    assert plan.new == ("notes/added.md",)
    assert plan.changed == ()
    assert len(plan.unchanged) == SAMPLE_NOTE_COUNT
    assert plan.total_notes == SAMPLE_NOTE_COUNT + 1


def test_a_manifest_chunk_count_that_disagrees_with_the_store_is_repaired(
    sample_vault_copy: Path,
) -> None:
    """自己修復: マニフェストとストアの食い違いは ``changed`` として直す。

    埋め込みの途中で落ちた実行は「マニフェストには載っているのにチャンクが
    欠けている」状態を残し得る。ノートのバイト列は変わっていないので、
    ``sha256`` だけで判定すると**永久に直らない** (§2.2 と同型の静かな破壊)。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    manifest, store = indexed_state(settings, config)
    rag.write_manifest(settings, manifest)
    path = rag.manifest_path(settings)
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_entry = next(note for note in payload["notes"] if note["chunk_count"] > 0)
    target_entry["chunk_count"] -= 1
    payload["totals"]["chunks"] -= 1
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded = rag.load_manifest(settings)

    plan = rag.plan_index(settings, config, store=store, manifest=loaded)

    assert plan.changed == (target_entry["relpath"],)
    assert len(plan.unchanged) == SAMPLE_NOTE_COUNT - 1
    assert plan.full_rebuild is False


def test_chunks_missing_from_the_store_are_reindexed(sample_vault_copy: Path) -> None:
    """ストア側が欠けている場合も同じ経路で直る (突き合わせは双方向)。"""
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    manifest, store = indexed_state(settings, config)
    target = next(note.relpath for note in manifest.notes if note.chunk_count > 0)
    store.delete_note(target)

    plan = rag.plan_index(settings, config, store=store, manifest=manifest)

    assert plan.changed == (target,)
    assert len(plan.unchanged) == SAMPLE_NOTE_COUNT - 1


def test_a_fingerprint_mismatch_marks_every_present_note_as_changed(
    sample_vault_copy: Path,
) -> None:
    """前提が変わった実行では ``new`` と ``changed`` を区別しない。

    どちらも全量再埋め込みになるため区別に情報が無く、「``unchanged`` が空で
    ある」ことだけが T6 の依存する不変条件になる。マニフェストに無いノートも
    ``changed`` 側に入る。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    manifest, store = indexed_state(settings, config)
    (sample_vault_copy / "notes" / "added.md").write_text(
        "# 追加\n\n新しく足した本文。\n", encoding="utf-8"
    )
    stale = dataclasses.replace(
        manifest, index_fingerprint="0" * 64, notes=manifest.notes[:-1]
    )

    plan = rag.plan_index(settings, config, store=store, manifest=stale)

    assert plan.full_rebuild is True
    assert plan.unchanged == ()
    assert plan.new == ()
    assert len(plan.changed) == SAMPLE_NOTE_COUNT + 1
    assert "notes/added.md" in plan.changed


def test_the_plan_carries_the_digest_of_every_note_it_saw(
    sample_vault_copy: Path,
) -> None:
    """``note_digests`` は次のマニフェストの材料。読み直しを 2 回しない。"""
    settings = sample_settings(sample_vault_copy)
    config = local_config()

    plan = rag.plan_index(
        settings, config, store=rag.InMemoryVectorStore(), manifest=None
    )

    assert set(plan.note_digests) == set(plan.new)
    for relpath, digest in plan.note_digests.items():
        assert digest == note_digest(settings, relpath)
    assert not isinstance(plan.note_digests, dict), (
        "計画が可変な dict をそのまま渡しています (呼び出し側が書き換えられる)"
    )


def test_planning_reads_the_vault_only_through_the_vault_module(
    sample_vault_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-30: 計画中の vault 読み取りは ``rag.vault.read_note_bytes`` だけを通る。

    ``rag/indexer.py`` は ``vault_dir`` を 1 度も参照しない (静的検査は
    ``tests/test_rag_layout.py``)。ここでは実行時にも、読み取りがその 1 関数
    しか呼ばれないことを確かめる。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    seen: list[str] = []
    original = rag.read_note_bytes

    def record(target: rag.RagSettings, relpath: str) -> bytes:
        seen.append(relpath)
        return original(target, relpath)

    monkeypatch.setattr(rag.indexer, "read_note_bytes", record)
    plan = rag.plan_index(
        settings, config, store=rag.InMemoryVectorStore(), manifest=None
    )

    assert sorted(seen) == sorted(plan.new)
    assert len(seen) == SAMPLE_NOTE_COUNT


def test_the_plan_never_sends_a_request_even_with_a_live_client(
    sample_vault_copy: Path,
) -> None:
    """``--dry-run`` の土台: 応答を返す口が開いていてもリクエストは 0 件。"""
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    manifest, store = indexed_state(settings, config)
    requests: list[httpx.Request] = []
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json={"data": [], "model": "x"}),
        requests=requests,
    )
    with transport.client() as http_client:
        llmkit.create_embedding_client(config, http_client=http_client)
        rag.plan_index(settings, config, store=store, manifest=manifest)

    assert requests == []


# --------------------------------------------------------------------------
# 索引の実行 (T6: build_index) — 小道具
# --------------------------------------------------------------------------


def index_once(
    settings: rag.RagSettings,
    config: AppConfig,
    *,
    transport: RecordingTransport | None = None,
    store: rag.VectorStore | None = None,
) -> rag.IndexResult:
    """CLI (T7) が行う手順そのままで 1 回索引する。

    マニフェストもストアも**毎回ディスクから読み直す**。プロセスを跨いだ
    再実行と同じ条件にしないと、メモリに残った状態のおかげで通ってしまう
    (差分更新の検証としては意味が無くなる)。
    """
    recorder = transport if transport is not None else fake_embedding_transport()
    target = store if store is not None else rag.JsonlVectorStore(chunks_path(settings))
    with recorder.client() as http_client:
        return rag.build_index(
            settings,
            config,
            embedding_client=llmkit.create_embedding_client(
                config, http_client=http_client
            ),
            store=target,
            manifest=rag.load_manifest(settings),
        )


def chunks_path(settings: rag.RagSettings) -> Path:
    return rag.chunks_path(settings)


def index_digests(settings: rag.RagSettings) -> dict[str, str]:
    """索引ディレクトリの ``ファイル名 -> sha256``。未作成なら空の辞書。

    「1 バイトも書かずに中断した」の検査に使う (ファイルの増減も見る)。
    """
    if not settings.index_dir.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(settings.index_dir.iterdir())
        if path.is_file()
    }


def stored_records(settings: rag.RagSettings) -> dict[str, tuple[rag.ChunkRecord, ...]]:
    """``chunks.jsonl`` を読み直して ``relpath -> レコード`` にする。"""
    grouped: dict[str, list[rag.ChunkRecord]] = {}
    for record in rag.JsonlVectorStore(chunks_path(settings)).iter_records():
        if record.relpath not in grouped:
            grouped[record.relpath] = []
        grouped[record.relpath].append(record)
    return {relpath: tuple(records) for relpath, records in grouped.items()}


def stored_order(settings: rag.RagSettings) -> list[tuple[str, int]]:
    """``chunks.jsonl`` の**ファイル上の行順**を ``(relpath, ordinal)`` で返す。

    :meth:`rag.JsonlVectorStore.iter_records` は並べ替えてから返すので、
    行順そのものを見るには生の行を読むしかない。行順が固定されていないと、
    同じ内容の索引がバイト一致しない (D-37)。
    """
    text = chunks_path(settings).read_text(encoding="utf-8")
    return [
        (json.loads(line)["relpath"], json.loads(line)["ordinal"])
        for line in text.splitlines()
        if line.strip()
    ]


def manifest_notes(settings: rag.RagSettings) -> dict[str, rag.ManifestNote]:
    manifest = rag.load_manifest(settings)
    assert manifest is not None, "マニフェストが書かれていません"
    return {note.relpath: note for note in manifest.notes}


def note_chunks(settings: rag.RagSettings, relpath: str) -> tuple[rag.Chunk, ...]:
    """1 ノートを実際に分割した結果 (期待値を手で書かないため)。"""
    parsed = rag.parse_note(relpath, rag.read_note_text(settings, relpath))
    return rag.chunk_note(parsed, settings.chunk)


def embed_texts_of(settings: rag.RagSettings, relpath: str) -> frozenset[str]:
    """そのノートが埋め込みへ送るテキスト (障害注入の対象を指定するのに使う)。"""
    return frozenset(chunk.embed_text for chunk in note_chunks(settings, relpath))


def every_note_body(settings: rag.RagSettings) -> tuple[str, ...]:
    """合成 vault の全チャンクの本文 (ログ・例外に現れてはいけない文字列)。"""
    return tuple(
        chunk.body
        for entry in rag.iter_vault_files(settings)
        for chunk in note_chunks(settings, entry.relpath)
    )


def failing_response(exception_kind: str) -> httpx.Response:
    """``llmkit`` の翻訳表が指定の例外に変換する応答を組み立てる。

    例外クラスを直接送出せず**応答**を作るのは、``rag`` から見た障害が常に
    ``llmkit`` の翻訳を通った後の形であることを、テストの側でも守るため。
    """
    if exception_kind == "context_length":
        return httpx.Response(400, json={"error": "maximum context length exceeded"})
    if exception_kind == "out_of_memory":
        return httpx.Response(500, json={"error": "CUDA error: out of memory"})
    return httpx.Response(500, json={"error": "internal server error"})


def intercept_texts(
    targets: frozenset[str], response: httpx.Response
) -> EmbeddingIntercept:
    """``targets`` のどれかを含む要求だけを ``response`` に差し替える。"""

    def intercept(texts: tuple[str, ...]) -> httpx.Response | None:
        if targets.isdisjoint(texts):
            return None
        return response

    return intercept


class DelegatingStore:
    """``commit()`` だけを差し替えられるストア (書き出し順序の証拠を取る道具)。

    :class:`rag.VectorStore` を満たす薄い委譲。``rag.JsonlVectorStore`` を継承
    しないのは、Protocol を受け取る側が実装の継承関係に依存していないことを
    同時に示すため。
    """

    def __init__(self, inner: rag.VectorStore, *, commit_error: Exception) -> None:
        self._inner = inner
        self._commit_error = commit_error
        self.commit_calls = 0

    def note_chunk_counts(self) -> Mapping[str, int]:
        return self._inner.note_chunk_counts()

    def replace_note(self, relpath: str, records: Sequence[rag.ChunkRecord]) -> None:
        self._inner.replace_note(relpath, records)

    def delete_note(self, relpath: str) -> None:
        self._inner.delete_note(relpath)

    def iter_records(self) -> Iterator[rag.ChunkRecord]:
        return self._inner.iter_records()

    def commit(self) -> None:
        self.commit_calls += 1
        raise self._commit_error


# --------------------------------------------------------------------------
# L309: 初回の索引
# --------------------------------------------------------------------------


def test_the_first_index_embeds_every_chunk_and_writes_both_artifacts(
    sample_vault_copy: Path,
) -> None:
    """L309: 11 ノート / 24 チャンクが埋め込まれ、成果物が 2 つ書かれる。"""
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    transport = fake_embedding_transport()

    result = index_once(settings, config, transport=transport)

    assert result.indexed_notes == SAMPLE_NOTE_COUNT
    assert result.embedded_chunks == SAMPLE_CHUNK_COUNT
    assert result.failed_notes == 0
    assert result.skipped_notes == 0
    assert result.deleted_notes == 0
    assert result.dimensions == FAKE_EMBEDDING_DIMENSIONS
    # 既定 batch_size=16 なので 24 チャンク = 2 要求 (ノート数 11 ではない
    # = バッチがノート境界を跨いでいる)。
    assert result.request_count == 2
    assert transport.call_count == result.request_count
    assert set(index_digests(settings)) == {"manifest.json", "chunks.jsonl"}
    assert sum(len(records) for records in stored_records(settings).values()) == (
        SAMPLE_CHUNK_COUNT
    )
    assert stored_order(settings) == sorted(stored_order(settings))


def test_every_stored_vector_matches_the_text_that_was_embedded(
    sample_vault_copy: Path,
) -> None:
    """レコードのベクトルが、そのレコードから再構成できる文字列の埋め込みである。

    ``embed_text`` は永続化しない (D-41) ので、``heading_path`` + ``body`` から
    :func:`rag.render_embed_text` で再構成する。ここが一致しないと、検索時に
    問い合わせを埋め込む空間と索引の空間がずれる (例外は 1 つも出ない)。
    """
    settings = sample_settings(sample_vault_copy)
    index_once(settings, local_config())

    for records in stored_records(settings).values():
        for record in records:
            text = rag.render_embed_text(
                record.heading_path, record.body, settings.chunk.heading_separator
            )
            assert record.vector == fake_embedding_vector(text)


def test_a_768_dimensional_runtime_round_trips_through_the_index(
    sample_vault_copy: Path,
) -> None:
    """実機と同じ 768 次元でも索引が成立する (既定の低次元は検査の都合)。"""
    settings = sample_settings(sample_vault_copy)
    config = local_config()

    result = index_once(
        settings, config, transport=fake_embedding_transport(dimensions=768)
    )

    assert result.dimensions == 768
    manifest = rag.load_manifest(settings)
    assert manifest is not None
    assert manifest.embedding.dimensions == 768
    assert manifest.embedding.reported_model == FAKE_EMBEDDING_MODEL
    vectors = {
        len(record.vector)
        for records in stored_records(settings).values()
        for record in records
    }
    assert vectors == {768}


def test_a_note_without_chunks_is_recorded_so_it_is_never_reprocessed(
    sample_vault_copy: Path,
) -> None:
    """0 チャンクのノートも**マニフェストに載る**。

    載せないと毎回「新規」として計画され、要求は 0 回のまま再処理され続ける
    (静かに残る欠陥なので、件数の一致では気づけない)。
    """
    settings = sample_settings(sample_vault_copy)
    index_once(settings, local_config())

    notes = manifest_notes(settings)

    assert notes["notes/empty.md"].chunk_count == 0
    assert notes["notes/whitespace-only.md"].chunk_count == 0
    assert "notes/empty.md" not in stored_records(settings)


# --------------------------------------------------------------------------
# L310 / D-36 / D-37: 再実行で 1 件も再処理しない
# --------------------------------------------------------------------------


def test_an_unchanged_rerun_leaves_the_index_byte_identical(
    sample_vault_copy: Path,
) -> None:
    """D-37 guard (L310): 無変更の再実行は再埋め込み 0 / 要求 0 / バイト一致。

    「再処理しない」は**再埋め込みをしない**ことであってファイルを書かない
    ことではないので、``chunks.jsonl`` は書き直されたうえでバイト一致する
    (行順を ``(relpath, ordinal)`` に固定しているため)。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    index_once(settings, config)
    before = index_digests(settings)

    transport = fake_embedding_transport()
    result = index_once(settings, config, transport=transport)

    assert result.embedded_chunks == 0
    assert result.indexed_notes == 0
    assert result.request_count == 0
    assert transport.call_count == 0
    assert result.skipped_notes == SAMPLE_NOTE_COUNT
    assert index_digests(settings) == before


def test_touching_a_note_without_changing_bytes_reindexes_nothing(
    sample_vault_copy: Path,
) -> None:
    """D-36 guard: 差分判定はバイト列の sha256 だけ。mtime を見ない。

    Obsidian の同期・バックアップ・``git checkout`` は内容を変えずに mtime を
    動かす。mtime を判定に使うと、そのたびに全ノートの再埋め込みが走る
    (実 vault で 34 ノート分の無駄な往復になる)。逆に mtime を
    マニフェストに書くと、内容が同じでも成果物がバイト一致しなくなり
    L310 が原理的に満たせない。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    index_once(settings, config)
    before = index_digests(settings)

    for entry in rag.iter_vault_files(settings):
        path = sample_vault_copy / entry.relpath
        os.utime(path, ns=(entry.mtime_ns + 10**9, entry.mtime_ns + 10**9))
    transport = fake_embedding_transport()
    result = index_once(settings, config, transport=transport)

    assert [entry.mtime_ns for entry in rag.iter_vault_files(settings)] != [], (
        "掃引の前提として vault にノートが存在すること"
    )
    assert result.embedded_chunks == 0
    assert result.indexed_notes == 0
    assert transport.call_count == 0
    assert index_digests(settings) == before
    assert "mtime" not in rag.manifest_path(settings).read_text(encoding="utf-8")


def test_batch_size_changes_request_count_but_not_the_index(
    sample_vault_copy: Path,
) -> None:
    """E28: ``embed.batch_size`` は要求回数だけを変え、索引はバイト一致。

    まとめ方を変えても個々のベクトルは変わらないため、``batch_size`` を
    ``index_fingerprint`` に入れてはいけない (入れると 4→16 の変更だけで
    全再構築が走る)。
    """
    config = local_config()
    small = dataclasses.replace(
        sample_settings(sample_vault_copy),
        embed=rag.EmbedSettings(batch_size=4),
        index_dir=sample_vault_copy.parent / "index-small",
    )
    large = dataclasses.replace(
        small,
        embed=rag.EmbedSettings(batch_size=16),
        index_dir=sample_vault_copy.parent / "index-large",
    )

    small_transport = fake_embedding_transport()
    large_transport = fake_embedding_transport()
    small_result = index_once(small, config, transport=small_transport)
    large_result = index_once(large, config, transport=large_transport)

    assert small_result.request_count == 6  # ceil(24 / 4)
    assert large_result.request_count == 2  # ceil(24 / 16)
    assert small_transport.call_count == 6
    assert large_transport.call_count == 2
    assert small_result.embedded_chunks == large_result.embedded_chunks
    assert small_result.fingerprint == large_result.fingerprint
    assert index_digests(small) == index_digests(large)


def test_only_the_edited_note_is_reembedded(sample_vault_copy: Path) -> None:
    """E29: 1 ノートを編集したら、そのノートのチャンクだけが再埋め込みされる。

    他ノートのベクトルは**バイト一致**する。ここが崩れると、差分更新が
    「速いだけで毎回別の索引を作る」ものになる。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    index_once(settings, config)
    before = stored_records(settings)
    target = "notes/weekly-review.md"

    path = sample_vault_copy / target
    path.write_text(
        path.read_text(encoding="utf-8") + "\n追記した 1 行。\n", encoding="utf-8"
    )
    expected = len(note_chunks(settings, target))
    transport = fake_embedding_transport()
    result = index_once(settings, config, transport=transport)

    assert result.indexed_notes == 1
    assert result.embedded_chunks == expected
    assert result.skipped_notes == SAMPLE_NOTE_COUNT - 1
    assert transport.call_count == 1  # ceil(3 / 16)
    after = stored_records(settings)
    assert after[target] != before[target]
    assert {
        relpath: records for relpath, records in after.items() if relpath != target
    } == {relpath: records for relpath, records in before.items() if relpath != target}
    assert manifest_notes(settings)[target].sha256 == note_digest(settings, target)


def test_excluding_a_note_deletes_its_chunks_without_reembedding_the_rest(
    sample_vault_copy: Path,
) -> None:
    """E33: ``exclude_globs`` を 1 つ足しても fingerprint は動かない。

    glob は「索引対象の集合」を変えるだけで、残るノートから出るベクトルには
    1 ビットも影響しない。fingerprint に入れると 1 パターン足しただけで
    全ノートの再埋め込みが走る。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    first = index_once(settings, config)
    before = stored_records(settings)
    target = "notes/project-alpha.md"

    narrowed = dataclasses.replace(
        settings, exclude_globs=(*settings.exclude_globs, target)
    )
    transport = fake_embedding_transport()
    result = index_once(narrowed, config, transport=transport)

    assert result.fingerprint == first.fingerprint
    assert result.embedded_chunks == 0
    assert result.indexed_notes == 0
    assert result.deleted_notes == 1
    assert transport.call_count == 0
    after = stored_records(narrowed)
    assert target not in after
    assert target not in manifest_notes(narrowed)
    assert after == {
        relpath: records for relpath, records in before.items() if relpath != target
    }


def test_changing_the_chunk_settings_reembeds_every_chunk(
    sample_vault_copy: Path,
) -> None:
    """E31 (実行側): ``max_tokens`` を変えると全チャンクが作り直される。

    §2.2 が「防ぐべき唯一最大の欠陥」と呼ぶ状態 (240 で切ったチャンクと 200 で
    切ったチャンクの同居) を実際に潰していることの確認。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    first = index_once(settings, config)

    retuned = dataclasses.replace(
        settings, chunk=dataclasses.replace(settings.chunk, max_tokens=120)
    )
    transport = fake_embedding_transport()
    result = index_once(retuned, config, transport=transport)

    assert result.fingerprint != first.fingerprint
    assert result.indexed_notes == SAMPLE_NOTE_COUNT
    assert result.skipped_notes == 0
    expected = sum(
        len(note_chunks(retuned, entry.relpath))
        for entry in rag.iter_vault_files(retuned)
    )
    assert result.embedded_chunks == expected > SAMPLE_CHUNK_COUNT
    dimensions = {
        len(record.vector)
        for records in stored_records(retuned).values()
        for record in records
    }
    assert dimensions == {FAKE_EMBEDDING_DIMENSIONS}
    assert sum(len(records) for records in stored_records(retuned).values()) == expected


def test_a_note_deleted_from_the_vault_loses_its_chunks(
    sample_vault_copy: Path,
) -> None:
    """vault から消えたノートは索引からも消える (再埋め込みは 0 件)。"""
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    index_once(settings, config)
    target = "notes/code-fence.md"

    (sample_vault_copy / target).unlink()
    transport = fake_embedding_transport()
    result = index_once(settings, config, transport=transport)

    assert result.deleted_notes == 1
    assert result.embedded_chunks == 0
    assert transport.call_count == 0
    assert target not in stored_records(settings)
    assert target not in manifest_notes(settings)


# --------------------------------------------------------------------------
# D-38: 失敗の扱い (ノート単位のトランザクション / 例外種別による分岐)
# --------------------------------------------------------------------------

#: バッチをノート単位に切って再送しても直らない障害 = そのノート固有の問題。
_NOTE_SCOPED_CASES = (
    ("upstream", UpstreamError),
    ("context_length", ContextLengthError),
)

#: 次のノートでも必ず再発する障害。即座に中断する。
_FATAL_CASES = (
    ("unavailable", RuntimeUnavailableError),
    ("out_of_memory", OutOfMemoryError),
)


@pytest.mark.parametrize(
    ("label", "expected"),
    _NOTE_SCOPED_CASES,
    ids=[label for label, _ in _NOTE_SCOPED_CASES],
)
def test_a_failed_note_is_never_recorded_as_indexed(
    label: str,
    expected: type[Exception],
    sample_vault_copy: Path,
) -> None:
    """D-38 guard: 部分的に成功したチャンクを索引にもマニフェストにも残さない。

    半分だけ載せると、次回は ``sha256`` が一致するので「変更なし」と判定され、
    **欠けたまま永久に治らない** (§2.2 と同型の静かな破壊)。したがって
    「失敗したノートが 1 件も載らない」ことと「次回そのノートだけが再試行
    される」ことを 1 本で固定する。

    ``batch_size=4`` なので、失敗するノート (9 チャンク) は複数のバッチに
    跨がり、そのうち 1 つは別のノートと同居する。それでも巻き添えは 0 件。
    """
    settings = dataclasses.replace(
        sample_settings(sample_vault_copy), embed=rag.EmbedSettings(batch_size=4)
    )
    config = local_config()
    target = "notes/project-alpha.md"
    chunks = note_chunks(settings, target)
    assert len(chunks) == 9, "掃引の前提 (複数のバッチに跨がるノート)"
    # **最後のチャンクだけ**を失敗させる。先行する 7 チャンクは埋め込みに
    # 成功しており、それを索引へ書けば「9 チャンクのうち 7 件だけが載った
    # ノート」ができる。次回は sha256 が一致するので二度と直らない。
    transport = fake_embedding_transport(
        intercept=intercept_texts(
            frozenset({chunks[-1].embed_text}), failing_response(label)
        )
    )

    result = index_once(settings, config, transport=transport)

    assert issubclass(expected, llmkit.LlmkitError)
    assert transport.call_count > 4, "先行チャンクの埋め込みが成功していない"
    assert result.failed_notes == 1
    assert result.indexed_notes == SAMPLE_NOTE_COUNT - 1
    assert result.embedded_chunks == SAMPLE_CHUNK_COUNT - 9
    stored = stored_records(settings)
    assert target not in stored
    assert target not in manifest_notes(settings)
    assert len(manifest_notes(settings)) == SAMPLE_NOTE_COUNT - 1
    assert sum(len(records) for records in stored.values()) == SAMPLE_CHUNK_COUNT - 9

    # 次の実行は、そのノートだけを再試行する (他の 10 ノートは触らない)。
    before = stored_records(settings)
    healthy = fake_embedding_transport()
    retry = index_once(settings, config, transport=healthy)

    assert retry.indexed_notes == 1
    assert retry.embedded_chunks == 9
    assert retry.failed_notes == 0
    assert healthy.call_count == 3  # ceil(9 / 4)
    after = stored_records(settings)
    assert len(after[target]) == 9
    assert {key: value for key, value in after.items() if key != target} == before
    # 後から書き足したノートも行順に割り込む (書き込み順が漏れない、D-37)。
    assert stored_order(settings) == sorted(stored_order(settings))


def test_a_failed_note_never_blocks_the_notes_that_share_its_batch(
    sample_vault_copy: Path,
) -> None:
    """バッチはノート境界を跨ぐが、失敗はノート境界を跨がない。

    ``batch_size=4`` の 3 番目のバッチは ``no-heading`` (1 チャンク) と
    ``project-alpha`` (9 チャンクの先頭 3 件) の同居になる。バッチ単位で
    失敗を記録すると、健全な ``no-heading`` が索引から静かに落ちる。実装は
    失敗したバッチを**ノート単位に切り直して送り直す**ので、落ちるのは
    原因のノートだけになる。

    切り分けの要求 (2 件) が増える一方で、失敗が確定したノートの残り
    チャンク (6 件 = 2 バッチ分) は送られない。
    """
    settings = dataclasses.replace(
        sample_settings(sample_vault_copy), embed=rag.EmbedSettings(batch_size=4)
    )
    config = local_config()
    target = "notes/project-alpha.md"
    transport = fake_embedding_transport(
        intercept=intercept_texts(
            embed_texts_of(settings, target), failing_response("upstream")
        )
    )

    result = index_once(settings, config, transport=transport)

    stored = stored_records(settings)
    assert "notes/no-heading.md" in stored
    assert target not in stored
    # 通常 6 バッチ。3 番目が失敗して 2 件に切り直され、その後は失敗した
    # ノートを飛ばして詰め直すので 7 要求で終わる (24/4 + 2 - 1)。
    assert transport.call_count == 7
    assert result.request_count == transport.call_count


@pytest.mark.parametrize(
    ("label", "expected"),
    _FATAL_CASES,
    ids=[label for label, _ in _FATAL_CASES],
)
def test_a_fatal_runtime_failure_stops_the_index_immediately(
    label: str,
    expected: type[llmkit.LlmkitError],
    sample_vault_copy: Path,
) -> None:
    """ランタイム側の障害は即座に中断する (34 回同じエラーを出さない)。

    それまでに**完成した**ノートは確定済みで、未処理のノートはマニフェストに
    載らないので次回そのまま再試行される。
    """
    settings = dataclasses.replace(
        sample_settings(sample_vault_copy), embed=rag.EmbedSettings(batch_size=4)
    )
    config = local_config()
    sent = 0

    def intercept(texts: tuple[str, ...]) -> httpx.Response | None:
        nonlocal sent
        sent += 1
        if sent < 2:
            return None
        if label == "unavailable":
            raise httpx.ConnectError("ランタイムに接続できません")
        return failing_response(label)

    transport = fake_embedding_transport(intercept=intercept)

    with pytest.raises(expected) as caught:
        index_once(settings, config, transport=transport)

    assert transport.call_count == 2
    assert transport.call_count < SAMPLE_NOTE_COUNT, "ノートごとに再発させている"
    # 1 要求目で完成した 2 ノート (と 0 チャンクの 2 ノート) は確定済み。
    assert set(stored_records(settings)) == {
        "notes/code-fence.md",
        "notes/frontmatter-broken.md",
    }
    assert set(manifest_notes(settings)) == {
        "notes/code-fence.md",
        "notes/empty.md",
        "notes/frontmatter-broken.md",
        "notes/whitespace-only.md",
    }
    assert str(sample_vault_copy) not in str(caught.value)


def test_the_manifest_is_written_only_after_the_store_is_committed(
    sample_vault_copy: Path,
) -> None:
    """D-37 の順序: ``store.commit()`` → :func:`rag.write_manifest`。

    逆順にすると、マニフェストだけ進んだ状態でクラッシュしたとき、次回の実行が
    ``sha256`` の一致を見て「変更なし」と判定し、**チャンクが欠けたまま永久に
    固定される**。この順序ならクラッシュ時に古いのはマニフェストの側になり、
    次回はそのノートを再処理するだけで済む (安全側に倒れる)。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    index_once(settings, config)
    before = index_digests(settings)
    target = "notes/tags-and-links.md"
    path = sample_vault_copy / target
    path.write_text(
        path.read_text(encoding="utf-8") + "\n更新した 1 行。\n", encoding="utf-8"
    )

    broken = DelegatingStore(
        rag.JsonlVectorStore(chunks_path(settings)),
        commit_error=ConfigError(
            "索引を書き出せません: chunks.jsonl",
            remediation="index.dir の権限と空き容量を確認してください",
        ),
    )
    with pytest.raises(ConfigError):
        index_once(settings, config, store=broken)

    assert broken.commit_calls == 1
    assert index_digests(settings) == before, (
        "ストアの確定に失敗したのにマニフェストが進んでいる"
    )
    assert manifest_notes(settings)[target].sha256 != note_digest(settings, target)


@pytest.mark.parametrize(
    ("label", "model", "dimensions"),
    (
        ("model", "another-embedding", FAKE_EMBEDDING_DIMENSIONS),
        ("dimensions", FAKE_EMBEDDING_MODEL, 16),
    ),
    ids=["model", "dimensions"],
)
def test_a_runtime_that_reports_a_different_embedding_never_writes(
    label: str,
    model: str,
    dimensions: int,
    sample_vault_copy: Path,
) -> None:
    """D-35 の第 2 条項: 申告の食い違いは **1 バイトも書く前に**中断する。

    設定に書いたモデル ID は宣言であり、応答の ``model`` と次元が実際である。
    宣言が同じまま実際が変わると ``index_fingerprint`` は動かないので、
    差分更新は「前提は同じ」と信じて既存のベクトルを残す。次元も意味空間も
    違うベクトルが同居してもコサイン類似度は例外を出さない (§2.2 と同型)。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    index_once(settings, config)
    before = index_digests(settings)
    target = "notes/no-heading.md"
    path = sample_vault_copy / target
    path.write_text(
        path.read_text(encoding="utf-8") + "\n差し替え後の 1 行。\n", encoding="utf-8"
    )

    transport = fake_embedding_transport(model=model, dimensions=dimensions)
    with pytest.raises(ConfigError) as caught:
        index_once(settings, config, transport=transport)

    assert index_digests(settings) == before, f"{label} の食い違いで索引が書かれた"
    assert "--rebuild" in caught.value.remediation
    assert str(sample_vault_copy) not in str(caught.value)
    assert str(settings.index_dir) not in str(caught.value)


# --------------------------------------------------------------------------
# ログ・出力に本文と絶対パスを出さない
# --------------------------------------------------------------------------


def test_indexing_never_logs_a_note_body_or_an_absolute_path(
    sample_vault_copy: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """成功経路と失敗経路の両方で、本文も絶対パスも 1 文字も出さない。

    索引は本文を全量ディスクへ書く最初の処理であり、``index_dir`` は
    gitignore されるがログと画面はされない。``DEBUG`` まで全部拾って検査する。
    """
    settings = dataclasses.replace(
        sample_settings(sample_vault_copy), embed=rag.EmbedSettings(batch_size=4)
    )
    config = local_config()
    bodies = every_note_body(settings)
    target = "notes/project-alpha.md"

    with caplog.at_level(logging.DEBUG):
        index_once(settings, config)
        path = sample_vault_copy / target
        path.write_text(
            path.read_text(encoding="utf-8") + "\n追記。\n", encoding="utf-8"
        )
        failing = fake_embedding_transport(
            intercept=intercept_texts(
                embed_texts_of(settings, target), failing_response("upstream")
            )
        )
        index_once(settings, config, transport=failing)

    captured = capsys.readouterr()
    combined = caplog.text + captured.out + captured.err

    assert "索引が完了しました" in combined, "検査対象のログが 1 件も出ていない"
    assert "ノートを索引できませんでした" in combined, "失敗経路を通っていない"
    assert captured.out == ""
    assert captured.err == ""
    leaked = [body for body in bodies if body in combined]
    assert leaked == [], f"ノート本文がログに出ている ({len(leaked)} 件)"
    assert str(sample_vault_copy) not in combined
    assert str(settings.index_dir) not in combined
    assert str(settings.source_path) not in combined


def test_a_note_that_cannot_be_chunked_is_skipped_not_fatal(
    sample_vault_copy: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """チャンク分割に失敗したノートを飛ばして索引を続けること。

    ``_prepare`` は ``parse_note`` / ``chunk_note`` の ``ConfigError`` を
    捕まえて、そのノートだけを失敗に計上し次へ進む (T6 決定4)。この経路は
    HTTP を 1 度も出さないため、埋め込み失敗を模した既存テスト
    (``test_a_failed_note_is_never_recorded_as_indexed``) では踏まれない。

    実際に踏まれるのは「見出し経路だけで上限を使い切る」ような設定を
    利用者が書いたとき (chunker が ``ConfigError`` を送出する)。壊れた
    ノートが 1 件あるだけで vault 全体の索引が不可能になるのを防ぐ、という
    保証がここにかかっている。

    埋め込みの成否とは独立した経路なので、失敗ノートがマニフェストに
    載らないこと・次回そのノートだけが再試行されることまで固定する。
    """
    settings = sample_settings(sample_vault_copy)
    config = local_config()
    broken = "notes/project-alpha.md"
    real_chunk_note = rag.chunk_note

    def refuse_one_note(parsed: rag.ParsedNote, chunk: object) -> object:
        if parsed.relpath == broken:
            msg = "見出し経路だけで上限を使い切りました"
            raise ConfigError(msg, remediation="chunk.max_tokens を上げてください")
        return real_chunk_note(parsed, chunk)  # type: ignore[arg-type]

    monkeypatch.setattr(rag.indexer, "chunk_note", refuse_one_note)

    result = index_once(settings, config)

    assert result.failed_notes == 1, result
    assert result.indexed_notes == 10, "他のノートは索引されるべき"
    assert result.embedded_chunks > 0, "壊れた 1 件で全体が止まってはいけない"

    manifest = rag.load_manifest(settings)
    assert manifest is not None
    recorded = {note.relpath for note in manifest.notes}
    assert broken not in recorded, (
        f"チャンク分割に失敗したノートがマニフェストに載っている: {broken}"
    )

    # 次回はそのノートだけが再試行される (sha256 は一致しているが未記録のため)。
    monkeypatch.undo()
    again = index_once(settings, config)
    assert again.indexed_notes == 1, again
    assert again.skipped_notes == 10, again
