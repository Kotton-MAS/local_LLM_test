"""索引の再現条件 (fingerprint)・マニフェスト・差分の計画 (D-35)。

このモジュールの中核は :func:`index_fingerprint` である。差分更新は「前提が
変わっていないノートを再処理しない」仕組みであり、**前提そのものが変わったこと
を検出できないと索引が静かに壊れる**。``max_tokens`` を 240→200 にして再実行
すると、ファイルのバイト列は 1 ビットも変わらないので全ノートが「変更なし」と
判定され、240 で切ったチャンクと 200 で切ったチャンクが同じ索引に同居する。
埋め込みモデルを差し替えた場合はさらに悪く、次元も意味空間も違うベクトルが
同居してコサイン類似度が無意味になる。どちらも例外を出さず、テストも落ちず、
検索精度だけが理由不明に劣化する (D-19 と同型)。

### fingerprint に入れるもの / 入れないもの (§4 論点1)

入れる (どれか 1 つでも変わると、同じノートから**別のベクトル**が出る):

* ``schema_version`` — 索引成果物の形式
* ``vault_id``
* ``chunk`` の 4 キー (``max_tokens`` / ``cjk_chars_per_token`` /
  ``ascii_chars_per_token`` / ``heading_separator``)
* ``chunk_algorithm_sha256`` — 近似トークナイザの CJK 範囲表そのものから導出。
  手書きの ``VERSION = 1`` 定数にしない (**手で上げ忘れる**のは D-19 と同型)
* ``embedding`` の ``model_id`` / ``served_name``

**入れない**: ``source_path`` / vault のルート / 索引の出力先 /
``runtime.base_url`` / ``embed.batch_size`` / ``include_globs`` /
``exclude_globs`` / mtime / 時刻 / run_id。(このモジュールは vault の場所を
表す設定項目を、名前としても 1 度も書かない: ``tests/test_rag_layout.py`` の
``test_only_the_vault_and_settings_modules_know_where_the_vault_is`` が
``rag/vault.py`` と ``rag/settings.py`` 以外での参照を一律に落とす。)

いずれも**生成されるベクトルに 1 ビットも影響しない**。入れると偽の全再構築を
生むうえ、パス類は実 vault の絶対パス (利用者名を含む) を索引成果物と画面に
書き出す唯一の経路になる。glob を変えたときの「索引対象の集合の変化」は
:func:`plan_index` の集合差分 (追加 → 新規、消失 → 削除) が正しく処理する。

### マニフェストは時刻も run_id も絶対パスも持たない

無変更の再実行で成果物が**バイト一致**することが受け入れ条件 (要件書 L310)
なので、実行のたびに変わる値を 1 つでも持たせるとその条件が原理的に満たせなく
なる (D-20 が ``run_id`` / ``started_at_utc`` を除外したのと同じ理由)。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as pydantic_dataclass

from llmkit import (
    AppConfig,
    ConfigError,
    ContextLengthError,
    EmbeddingBatch,
    EmbeddingClient,
    LlmkitError,
    ModelNotFoundError,
    ModelSpec,
    OutOfMemoryError,
    RuntimeUnavailableError,
    UpstreamError,
    resolve_embedding_spec,
)
from rag import chunker
from rag.chunker import chunk_note
from rag.parser import parse_note
from rag.settings import RagSettings
from rag.store import ChunkRecord, VectorStore
from rag.vault import iter_vault_files, read_note_bytes, read_note_text

logger = logging.getLogger(__name__)

__all__ = [
    "CHUNKS_FILENAME",
    "INDEX_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "IndexManifest",
    "IndexPlan",
    "IndexResult",
    "ManifestEmbedding",
    "ManifestNote",
    "ManifestTotals",
    "build_index",
    "chunks_path",
    "index_fingerprint",
    "load_manifest",
    "manifest_path",
    "plan_index",
    "write_manifest",
]
# ``fingerprint_inputs`` / ``fingerprint_digest`` は公開関数だが __all__ に
# 入れない (§9 T5 決定)。``harness`` が同じ名前を公開しており、
# ``tests/test_rag_layout.py::test_rag_and_harness_never_import_each_other`` が
# ``rag.__all__`` と ``harness.__all__`` の互いに素であることを要求するため。
# 名前をそろえること自体は意図的で、両者が同じ dict に同じ digest を返すことを
# ``tests/test_rag_indexer.py`` が固定する。

#: 索引成果物 (manifest + chunks.jsonl) の形式バージョン。fingerprint に入る
#: ので、形式を変えれば既存の索引は全再構築になる。
INDEX_SCHEMA_VERSION = 1

#: 索引ディレクトリの中のファイル名。出典はここ 1 か所だけ (D-27 の趣旨)。
MANIFEST_FILENAME = "manifest.json"
CHUNKS_FILENAME = "chunks.jsonl"

_FORBID_EXTRA = ConfigDict(extra="forbid")

NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
#: 小文字 16 進 64 桁。``pattern`` は Rust 正規表現なので ``\A`` / ``\Z`` は
#: 使えず、``^`` / ``$`` で全体一致を表す。
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


# --------------------------------------------------------------------------
# fingerprint (§4 論点1 / D-35)
# --------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _chunk_algorithm_sha256() -> str:
    """近似トークナイザの CJK 範囲表そのものから導出したハッシュ。

    手書きの ``VERSION = 1`` 定数にしないのは、**表を変えたのに定数を上げ忘れ
    た**実行が「前提は同じ」と名乗れてしまうため (D-19 と同型)。表から導出すれば
    忘れようがない。

    ``rag.chunker`` の定数を実行時に読むのは、``from ... import`` で束ねると
    束ねた時点の値に固定され、表の差し替えが伝わらなくなるため。範囲は昇順に
    並べ替えてから畳む: :func:`rag.chunker._is_cjk` は ``any`` で判定するので
    表の**並び順は挙動に影響しない**。並び順まで拾うと、意味の変わらない
    並べ替えが全再構築を引き起こす。
    """
    ranges = sorted((start, end) for start, end in chunker._CJK_RANGES)
    return _sha256_text(json.dumps([list(pair) for pair in ranges]))


def fingerprint_inputs(settings: RagSettings, spec: ModelSpec) -> dict[str, object]:
    """フィンガープリントの入力を素の辞書で返す (マニフェストに載せる形)。

    ハッシュ値だけを記録すると「何が変わったから変わったのか」が追えない。
    入力そのものを残し、:func:`fingerprint_digest` で再計算できるようにする
    (``harness/runner.py`` と同じ方針)。

    Args:
        settings: 解決済みの索引設定。**パス類は 1 つも読まない**。
        spec: 埋め込みモデルの解決結果 (:func:`llmkit.resolve_embedding_spec`)。

    Returns:
        正規化 JSON にできる素の辞書。
    """
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "vault_id": settings.vault_id,
        "chunk": {
            "max_tokens": settings.chunk.max_tokens,
            "cjk_chars_per_token": settings.chunk.cjk_chars_per_token,
            "ascii_chars_per_token": settings.chunk.ascii_chars_per_token,
            "heading_separator": settings.chunk.heading_separator,
        },
        "chunk_algorithm_sha256": _chunk_algorithm_sha256(),
        "embedding": {
            "model_id": spec.model_id,
            "served_name": spec.served_name,
        },
    }


def fingerprint_digest(inputs: Mapping[str, object]) -> str:
    """正規化 JSON (``sort_keys=True``, ``ensure_ascii=False``) の sha256。

    ``harness.runner.fingerprint_digest`` と**同じ規則**。層構造上 ``rag`` は
    ``harness`` を import できず、この 3 行を L2 (``llmkit``) に上げるのも
    「推論ランタイムの抽象」という責務から外れるため、重複を受け入れる。
    D-32 が禁じた「独自の近似式の複製」とは性質が違う (こちらは stdlib 呼び出し
    1 行で、両者が同じ dict に同じ値を返すことをテストが固定する)。
    """
    canonical = json.dumps(dict(inputs), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def index_fingerprint(settings: RagSettings, spec: ModelSpec) -> str:
    """索引の再現条件を 1 つの文字列にまとめる (D-35)。

    この値が一致する限り、同じノート (同じバイト列) からは同じベクトルが出る。
    一致しない場合は差分更新を行わず**全再構築**する。
    """
    return fingerprint_digest(fingerprint_inputs(settings, spec))


# --------------------------------------------------------------------------
# マニフェスト
# --------------------------------------------------------------------------


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class ManifestNote:
    """索引済みノート 1 件の記録。

    Attributes:
        relpath: vault ルートからの相対パス (POSIX 表記)。**絶対パスは持たない**。
        sha256: ノートの生バイト列のハッシュ。差分判定はこれだけで決まる
            (mtime を判定にもマニフェストにも使わない、D-36)。
        chunk_count: そのノートが持つチャンク数。ストアの実データと突き合わせて
            「manifest には載っているのにチャンクが欠けている」状態を検出する。
    """

    relpath: NonEmptyStr
    sha256: Sha256Hex
    chunk_count: NonNegativeInt


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class ManifestEmbedding:
    """ランタイムが**実際に名乗った**埋め込みモデルと次元 (D-35 の第 2 条項)。

    設定に書いたモデル ID (``fingerprint_inputs.embedding``) は宣言であり、
    ここに入るのは応答由来の実際である。両者は別物なので別枠で持ち、次回実行の
    申告と食い違ったら書き出す前に中断する (実装は T6)。
    """

    reported_model: NonEmptyStr
    dimensions: PositiveInt


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class ManifestTotals:
    """``notes[]`` の要約。``status`` 表示のための便宜であり、真実は ``notes[]``。

    導出できる値を持つ以上、両者が食い違った manifest は書けてはならない。
    :func:`write_manifest` が書き出す前に一致を確かめる (読み込み側では
    確かめない: 手で壊れた manifest は拒否ではなく**再処理**で直すのが
    :func:`plan_index` の役割であるため)。
    """

    notes: NonNegativeInt
    chunks: NonNegativeInt


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class IndexManifest:
    """索引 1 つ分のマニフェスト。**時刻・run_id・絶対パスを 1 つも持たない**。

    書き出す型と読み込む型を 1 つにする (``rag/store.py`` の
    :class:`~rag.ChunkRecord` と同じ方針)。``extra="forbid"`` なので、未知の
    キーを持つ manifest は読み込み時に落ちる。

    Attributes:
        schema_version: :data:`INDEX_SCHEMA_VERSION`。
        index_fingerprint: :func:`index_fingerprint` の値。
        fingerprint_inputs: その入力そのもの (何が変わったのかを追えるように)。
        embedding: ランタイムが名乗ったモデル名と次元。
        notes: 索引済みノート (``relpath`` 昇順)。
        totals: ``notes[]`` の要約。
    """

    schema_version: int
    index_fingerprint: Sha256Hex
    fingerprint_inputs: dict[str, object]
    embedding: ManifestEmbedding
    notes: tuple[ManifestNote, ...]
    totals: ManifestTotals


_MANIFEST_ADAPTER: TypeAdapter[IndexManifest] = TypeAdapter(IndexManifest)

_REBUILD_REMEDIATION = "索引を作り直してください (--rebuild)"


def manifest_path(settings: RagSettings) -> Path:
    """``manifest.json`` のパス。索引ディレクトリの中だけを指す。"""
    return settings.index_dir / MANIFEST_FILENAME


def chunks_path(settings: RagSettings) -> Path:
    """``chunks.jsonl`` のパス (:class:`rag.JsonlVectorStore` に渡す)。"""
    return settings.index_dir / CHUNKS_FILENAME


def _manifest_payload(manifest: IndexManifest) -> dict[str, object]:
    """マニフェストの JSON 表現。キーを明示して書く (``rag/store.py`` と同じ)。"""
    return {
        "schema_version": manifest.schema_version,
        "index_fingerprint": manifest.index_fingerprint,
        "fingerprint_inputs": dict(manifest.fingerprint_inputs),
        "embedding": {
            "reported_model": manifest.embedding.reported_model,
            "dimensions": manifest.embedding.dimensions,
        },
        "notes": [
            {
                "relpath": note.relpath,
                "sha256": note.sha256,
                "chunk_count": note.chunk_count,
            }
            for note in manifest.notes
        ],
        "totals": {
            "notes": manifest.totals.notes,
            "chunks": manifest.totals.chunks,
        },
    }


def _require_consistent_totals(manifest: IndexManifest) -> None:
    """``totals`` が ``notes[]`` から導出した値と一致することを確かめる。

    書き出す側でしか見ない。ここを通さずに書かれた manifest は 1 つも無い
    ので、「集計だけがずれた索引」は生成経路そのものが存在しなくなる。
    """
    expected_notes = len(manifest.notes)
    expected_chunks = sum(note.chunk_count for note in manifest.notes)
    if (manifest.totals.notes, manifest.totals.chunks) == (
        expected_notes,
        expected_chunks,
    ):
        return
    msg = (
        f"マニフェストの totals が notes[] と一致しません "
        f"(totals={manifest.totals.notes}/{manifest.totals.chunks}, "
        f"notes[]={expected_notes}/{expected_chunks})"
    )
    raise ConfigError(
        msg,
        remediation="totals は notes[] から導出してください (集計を手で書かない)",
    )


def write_manifest(settings: RagSettings, manifest: IndexManifest) -> Path:
    """マニフェストを索引ディレクトリへ書き出す。

    同じ内容なら**常に同じバイト列**になる (``sort_keys=True`` + ``indent=2``)。
    書き出しの形は ``harness/report.py`` に合わせ、``OSError`` は
    :class:`ConfigError` に翻訳する (メッセージにファイル名以外のパスを載せない)。

    Args:
        settings: 解決済みの索引設定 (``index_dir`` だけを使う)。
        manifest: 書き出す内容。

    Returns:
        書き出したファイルのパス。

    Raises:
        ConfigError: ``totals`` が ``notes[]`` と食い違う場合、または書き出しに
            失敗した場合。
    """
    _require_consistent_totals(manifest)
    payload = json.dumps(
        _manifest_payload(manifest), ensure_ascii=False, indent=2, sort_keys=True
    )
    path = manifest_path(settings)
    try:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        msg = f"マニフェストを書き出せません: {path.name}"
        raise ConfigError(
            msg,
            remediation=(
                f"index.dir の権限と空き容量を確認してください ({exc.strerror})"
            ),
        ) from exc
    logger.debug(
        "マニフェストを書き出しました: file=%s notes=%d chunks=%d",
        path.name,
        manifest.totals.notes,
        manifest.totals.chunks,
    )
    return path


def load_manifest(settings: RagSettings) -> IndexManifest | None:
    """マニフェストを読む。まだ索引が無ければ ``None``。

    例外メッセージには**ファイル名だけ**を載せ、解決済みの絶対パスも中身も
    載せない (``rag/store.py`` と同じ扱い)。

    Args:
        settings: 解決済みの索引設定 (``index_dir`` だけを使う)。

    Returns:
        読み込んだ :class:`IndexManifest`。ファイルが無ければ ``None``。

    Raises:
        ConfigError: 読み取りに失敗した / スキーマが不正 / 形式バージョンが
            違う / 記録された fingerprint がその入力から再計算した値と
            食い違う場合。
    """
    path = manifest_path(settings)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"マニフェストを読み込めません: {path.name}"
        raise ConfigError(
            msg,
            remediation=f"index.dir の権限を確認するか、{_REBUILD_REMEDIATION}",
        ) from exc
    try:
        manifest = _MANIFEST_ADAPTER.validate_json(text)
    except ValidationError as exc:
        msg = f"マニフェストを読めません: {path.name} ({_format_validation_error(exc)})"
        raise ConfigError(msg, remediation=_REBUILD_REMEDIATION) from exc
    _require_known_schema_version(path, manifest)
    _require_matching_digest(path, manifest)
    return manifest


def _format_validation_error(exc: ValidationError) -> str:
    """``キー名: 理由`` の一覧。**入力値そのものは載せない**。"""
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return " / ".join(parts)


def _require_known_schema_version(path: Path, manifest: IndexManifest) -> None:
    """知らない形式バージョンの索引を黙って上書きしない。"""
    if manifest.schema_version == INDEX_SCHEMA_VERSION:
        return
    msg = (
        f"マニフェストの形式バージョンが違います: {path.name} "
        f"(索引={manifest.schema_version}, 実装={INDEX_SCHEMA_VERSION})"
    )
    raise ConfigError(msg, remediation=_REBUILD_REMEDIATION)


def _require_matching_digest(path: Path, manifest: IndexManifest) -> None:
    """記録された fingerprint が、記録された入力から再計算した値と一致するか。

    2 つを両方載せる以上、食い違った manifest は「どちらが本当か」を決められ
    ない。食い違いは書き換え・破損の証拠なので、そのまま差分判定に使わない。
    """
    recomputed = fingerprint_digest(manifest.fingerprint_inputs)
    if recomputed == manifest.index_fingerprint:
        return
    msg = (
        f"マニフェストの index_fingerprint が fingerprint_inputs と"
        f"一致しません: {path.name}"
    )
    raise ConfigError(msg, remediation=_REBUILD_REMEDIATION)


# --------------------------------------------------------------------------
# 差分の計画 (HTTP 0 回・書き込み 0 バイト)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IndexPlan:
    """次の索引実行で何をするかの計画。``--dry-run`` はこれを出すだけで足りる。

    Attributes:
        fingerprint: 今回の設定から計算した :func:`index_fingerprint`。
        full_rebuild: 既存の索引を再利用できないこと (マニフェストが無い、
            または fingerprint が一致しない)。
        unchanged: 再処理しないノート (``relpath`` 昇順)。
        changed: 内容が変わった / 索引と食い違うため作り直すノート。
        new: マニフェストに無いノート。
        deleted: マニフェストにあるが vault から消えたノート。
        note_digests: いま vault にあるノートの ``relpath -> sha256``。
            次のマニフェストを組み立てるのに使う (読み直しを 2 回しない)。
    """

    fingerprint: str
    full_rebuild: bool
    unchanged: tuple[str, ...]
    changed: tuple[str, ...]
    new: tuple[str, ...]
    deleted: tuple[str, ...]
    note_digests: Mapping[str, str]

    @property
    def pending(self) -> tuple[str, ...]:
        """再処理するノート (``new`` + ``changed``) を ``relpath`` 昇順で返す。"""
        return tuple(sorted(self.new + self.changed))

    @property
    def total_notes(self) -> int:
        """いま vault にあるノート数。"""
        return len(self.note_digests)


def _note_digests(settings: RagSettings) -> dict[str, str]:
    """索引対象のノートを ``relpath -> sha256`` にする。

    vault を読むのは :func:`rag.vault.read_note_bytes` だけを通す (D-30)。
    このモジュールは ``open`` も ``read_bytes`` も呼ばず、vault のルートを
    指す設定項目も 1 度も参照しない (参照できなければ vault 配下のパスを
    組み立てられないので、``read_text`` の許可があっても vault には届かない)。
    """
    return {
        entry.relpath: _sha256_bytes(read_note_bytes(settings, entry.relpath))
        for entry in iter_vault_files(settings)
    }


def plan_index(
    settings: RagSettings,
    config: AppConfig,
    *,
    store: VectorStore,
    manifest: IndexManifest | None,
) -> IndexPlan:
    """差分更新の計画を立てる。**HTTP を 1 バイトも出さず、1 バイトも書かない**。

    分類の規則:

    1. マニフェストが無ければ全件 ``new`` (再利用できる索引が存在しない)。
    2. fingerprint が一致しなければ全件 ``changed``。前提そのものが変わって
       いるので、バイト列が同じノートも作り直す (§2.2 の欠陥を塞ぐ核心)。
       ``new`` と ``changed`` の区別はこの場合何も意味しない (どちらも全量
       再埋め込みになる) ため、1 つの区分にまとめる。
    3. それ以外は ``sha256`` の一致で判定する (**mtime は見ない**、D-36)。
       加えて、マニフェストの ``chunk_count`` とストアの実データが食い違う
       ノートも ``changed`` に入れる (自己修復)。埋め込みの途中で落ちた実行が
       残した「マニフェストには載っているのにチャンクが欠けている」状態は、
       これが無いと sha256 が一致する限り**永久に直らない**。

    Args:
        settings: 解決済みの索引設定。
        config: 読み込み済み設定 (埋め込みモデルの出典、D-27)。
        store: 既存の索引 (チャンク数の突き合わせにだけ使う)。
        manifest: 既存のマニフェスト (:func:`load_manifest` の戻り値)。

    Returns:
        :class:`IndexPlan`。

    Raises:
        ConfigError: 埋め込みモデルを解決できない場合、または vault を読めない
            場合。
    """
    fingerprint = index_fingerprint(settings, resolve_embedding_spec(config))
    digests = _note_digests(settings)
    present = tuple(sorted(digests))
    if manifest is None:
        plan = IndexPlan(
            fingerprint=fingerprint,
            full_rebuild=True,
            unchanged=(),
            changed=(),
            new=present,
            deleted=(),
            note_digests=types.MappingProxyType(digests),
        )
    else:
        known = {note.relpath: note for note in manifest.notes}
        deleted = tuple(sorted(set(known) - set(digests)))
        if manifest.index_fingerprint != fingerprint:
            plan = IndexPlan(
                fingerprint=fingerprint,
                full_rebuild=True,
                unchanged=(),
                changed=present,
                new=(),
                deleted=deleted,
                note_digests=types.MappingProxyType(digests),
            )
        else:
            unchanged, changed, new = _classify(digests, known, store)
            plan = IndexPlan(
                fingerprint=fingerprint,
                full_rebuild=False,
                unchanged=unchanged,
                changed=changed,
                new=new,
                deleted=deleted,
                note_digests=types.MappingProxyType(digests),
            )
    logger.info(
        "索引を計画しました: vault=%s notes=%d new=%d changed=%d "
        "unchanged=%d deleted=%d rebuild=%s",
        settings.vault_id,
        plan.total_notes,
        len(plan.new),
        len(plan.changed),
        len(plan.unchanged),
        len(plan.deleted),
        plan.full_rebuild,
    )
    return plan


def _classify(
    digests: Mapping[str, str],
    known: Mapping[str, ManifestNote],
    store: VectorStore,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """``(unchanged, changed, new)`` を ``relpath`` 昇順で返す。

    ``Mapping.get`` は使わない (D-25 guard が ``.get()`` を一律に落とすため、
    ``in`` + 添字で書く)。
    """
    counts = store.note_chunk_counts()
    unchanged: list[str] = []
    changed: list[str] = []
    new: list[str] = []
    for relpath in sorted(digests):
        if relpath not in known:
            new.append(relpath)
            continue
        note = known[relpath]
        stored_chunks = _stored_chunk_count(counts, relpath)
        if note.sha256 == digests[relpath] and note.chunk_count == stored_chunks:
            unchanged.append(relpath)
        else:
            changed.append(relpath)
    return tuple(unchanged), tuple(changed), tuple(new)


def _stored_chunk_count(counts: Mapping[str, int], relpath: str) -> int:
    """ストアが持つチャンク数。表に無いノート (0 チャンク) は 0。

    ``Mapping.get`` を使えない (D-25 guard が ``.get()`` を一律に落とす) ため
    ``in`` + 添字で書く。三項演算子にすると ruff SIM401 が ``.get`` を勧め、
    ``if``/``else`` ブロックにすると SIM108 が三項演算子を勧めるので、
    早期 return の関数に切り出して両方と衝突しない形にする。
    """
    if relpath in counts:
        return counts[relpath]
    return 0


# --------------------------------------------------------------------------
# 索引の実行 (差分更新の本体、D-37 / D-38)
# --------------------------------------------------------------------------

#: 次のノートでも必ず再発する障害 (§4 論点5)。ここまでに完了したノートを
#: 確定してから送出し直す。34 回同じエラーを出すのは利用者への嫌がらせで、
#: 部分索引を作る時間も無駄になる。
_FATAL_ERRORS = (RuntimeUnavailableError, OutOfMemoryError, ModelNotFoundError)

#: そのノート固有の障害。記録して次のノートへ進む (1 件で索引全体を止めない)。
_NOTE_SCOPED_ERRORS = (ContextLengthError, UpstreamError)

#: 埋め込み待ちのチャンク 1 件 (``(ノート番号, チャンク)``)。
_QueueItem = tuple[int, chunker.Chunk]


@dataclasses.dataclass(frozen=True)
class IndexResult:
    """1 回の索引実行の結果。**件数だけを持ち、相対パスも本文も持たない**。

    CLI (T7) はこの値をそのまま画面に出す。実 vault の実行結果を報告に貼れる
    状態を保つため、ここに識別子を持たせない (どのノートが失敗したかは
    ``DEBUG`` ログとマニフェストの差分から分かる)。

    Attributes:
        indexed_notes: 埋め込み直してストアへ書いたノート数。
        skipped_notes: 前提もバイト列も変わらず**再処理しなかった**ノート数。
        failed_notes: そのノート固有の障害で飛ばしたノート数
            (マニフェストには新しい内容として載らない、D-38)。
        deleted_notes: vault から消えたためチャンクを取り除いたノート数。
        embedded_chunks: 索引に**書き込まれた**チャンク数。埋め込みに成功しても
            ノートが完成しなかった (= 破棄した) チャンクは数えない。
        request_count: 埋め込みランタイムへの要求回数 (失敗した要求も含む)。
        elapsed_s: 実行の所要時間 (秒)。
        fingerprint: この実行の :func:`index_fingerprint`。
        dimensions: ランタイムが返したベクトルの次元。埋め込みを 1 度も行わず、
            既存のマニフェストも無い場合は ``None`` (**0 で埋めない**、D-07)。
    """

    indexed_notes: int
    skipped_notes: int
    failed_notes: int
    deleted_notes: int
    embedded_chunks: int
    request_count: int
    elapsed_s: float
    fingerprint: str
    dimensions: int | None


@dataclasses.dataclass
class _NoteWork:
    """1 ノート分の作業状態 (チャンクと、そこまでに得たベクトル)。

    ``vectors`` が ``chunks`` と同じ長さになるまでストアには 1 件も書かない。
    これが「ノート単位のトランザクション」(D-38) の実体で、部分的に成功した
    チャンクはこのオブジェクトごと捨てられる。
    """

    relpath: str
    chunks: tuple[chunker.Chunk, ...]
    vectors: list[tuple[float, ...]] = dataclasses.field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """全チャンクのベクトルが揃ったか (揃って初めてストアへ書ける)。"""
        return len(self.vectors) == len(self.chunks)


def _note_groups(batch: Sequence[_QueueItem]) -> list[tuple[int, list[_QueueItem]]]:
    """バッチをノート単位の連続した区画に切る (順序を保つ)。

    待ち行列はノート順に並んでいるので、隣接する同じノート番号をまとめれば
    区画は一意に決まる。失敗したバッチをノート単位で送り直すのに使う。
    """
    groups: list[tuple[int, list[_QueueItem]]] = []
    for item in batch:
        index = item[0]
        if groups and groups[-1][0] == index:
            groups[-1][1].append(item)
        else:
            groups.append((index, [item]))
    return groups


class _IndexBuilder:
    """再処理対象のノートを埋め込んでストアへ書く実行主体。

    バッチ (``embed.batch_size`` 件) は**ノート境界を跨ぐ**。跨がせないと
    小さなノートが並ぶ vault で 1 件ずつの要求になり、要求回数が
    チャンク数に張り付く。一方で確定の単位はノートであり、
    :meth:`_store_note` はそのノートの全チャンクが揃ってからしか呼ばない。

    バッチが跨いだせいで「巻き添えで失敗するノート」が出ないよう、ノート固有の
    障害 (:data:`_NOTE_SCOPED_ERRORS`) で落ちたバッチは**ノート単位に切り直して
    送り直す** (:meth:`_isolate`)。これをしないと、長すぎるチャンクを 1 つ含む
    ノートと同じバッチに入っただけで健全なノートが索引から漏れる。
    """

    def __init__(
        self,
        settings: RagSettings,
        *,
        embedding_client: EmbeddingClient,
        store: VectorStore,
        baseline: ManifestEmbedding | None,
    ) -> None:
        self._settings = settings
        self._client = embedding_client
        self._store = store
        self._embedding = baseline
        self._failed: set[str] = set()
        self.indexed_counts: dict[str, int] = {}
        self.embedded_chunks = 0
        self.request_count = 0

    @property
    def embedding(self) -> ManifestEmbedding | None:
        """ランタイムが名乗ったモデルと次元 (1 度も要求しなければ既存の値)。"""
        return self._embedding

    @property
    def failed_notes(self) -> int:
        return len(self._failed)

    def run(self, pending: Sequence[str]) -> LlmkitError | None:
        """``pending`` を順に処理する。

        Returns:
            中断させた致命的な障害。最後まで進めたなら ``None``。
        """
        works: list[_NoteWork] = []
        for relpath in pending:
            work = self._prepare(relpath)
            if work is not None:
                works.append(work)
        for work in works:
            if not work.chunks:
                # 空ノート (0 チャンク) は要求を出さずに確定する。ここで
                # 確定しないとマニフェストに載らず、毎回「新規」として
                # 数え直される (要求は 0 回のままなので静かに残り続ける)。
                self._store_note(work)
        return self._embed_queue(works)

    def _prepare(self, relpath: str) -> _NoteWork | None:
        """1 ノートを読み、パースし、チャンクへ分割する (HTTP は出さない)。"""
        try:
            parsed = parse_note(relpath, read_note_text(self._settings, relpath))
            chunks = chunk_note(parsed, self._settings.chunk)
        except ConfigError as exc:
            self._fail(relpath, exc)
            return None
        return _NoteWork(relpath=relpath, chunks=chunks)

    def _embed_queue(self, works: Sequence[_NoteWork]) -> LlmkitError | None:
        """全ノートのチャンクを 1 本の待ち行列にして ``batch_size`` 件ずつ送る。"""
        queue: list[_QueueItem] = [
            (index, chunk) for index, work in enumerate(works) for chunk in work.chunks
        ]
        batch_size = self._settings.embed.batch_size
        position = 0
        while position < len(queue):
            batch: list[_QueueItem] = []
            while position < len(queue) and len(batch) < batch_size:
                item = queue[position]
                position += 1
                # 既に失敗が確定したノートの残りチャンクは送らない
                # (どうせ破棄するので、要求を出すだけ無駄になる)。
                if works[item[0]].relpath not in self._failed:
                    batch.append(item)
            if not batch:
                continue
            fatal = self._embed_batch(works, batch)
            if fatal is not None:
                return fatal
        return None

    def _embed_batch(
        self, works: Sequence[_NoteWork], batch: Sequence[_QueueItem]
    ) -> LlmkitError | None:
        try:
            outcome = self._embed(batch)
        except _FATAL_ERRORS as exc:
            # ノートを失敗として記録しない。これはノートの問題ではなく
            # ランタイムの問題であり、未処理のノートは既存の記録を保ったまま
            # 次回そのまま再試行される。
            return exc
        except _NOTE_SCOPED_ERRORS as exc:
            return self._isolate(works, batch, exc)
        self._assign(works, batch, outcome)
        return None

    def _isolate(
        self,
        works: Sequence[_NoteWork],
        batch: Sequence[_QueueItem],
        exc: LlmkitError,
    ) -> LlmkitError | None:
        """失敗したバッチをノート単位に切り直して、巻き添えを切り分ける。"""
        groups = _note_groups(batch)
        if len(groups) == 1:
            self._fail(works[groups[0][0]].relpath, exc)
            return None
        for index, items in groups:
            try:
                outcome = self._embed(items)
            except _FATAL_ERRORS as fatal:
                return fatal
            except _NOTE_SCOPED_ERRORS as note_error:
                self._fail(works[index].relpath, note_error)
            else:
                self._assign(works, items, outcome)
        return None

    def _embed(self, batch: Sequence[_QueueItem]) -> EmbeddingBatch:
        """1 要求分の埋め込み。**送った回数はここでだけ数える**。"""
        self.request_count += 1
        outcome = self._client.embed([chunk.embed_text for _, chunk in batch])
        self._require_consistent_embedding(outcome)
        return outcome

    def _require_consistent_embedding(self, outcome: EmbeddingBatch) -> None:
        """ランタイムの申告が既存の索引と食い違ったら書く前に止める (D-35)。

        設定に書いたモデル ID は宣言であり、応答の ``model`` / 次元が実際で
        ある。両者は別物なので、同じ ``index_fingerprint`` を名乗る索引に
        別のモデルのベクトルが積まれる経路がここにだけ残る。混ざっても例外は
        出ず、コサイン類似度が静かに無意味になる (§2.2 と同型)。

        1 度目の応答を基準にするので、実行の途中でランタイムがモデルを
        差し替えた場合も同じ検査で落ちる。
        """
        observed = ManifestEmbedding(
            reported_model=outcome.model, dimensions=outcome.dimensions
        )
        if self._embedding is None:
            self._embedding = observed
            return
        if self._embedding == observed:
            return
        msg = (
            f"ランタイムが名乗った埋め込みモデルが既存の索引と一致しません "
            f"(索引={self._embedding.reported_model}/"
            f"{self._embedding.dimensions}次元, "
            f"応答={observed.reported_model}/{observed.dimensions}次元)"
        )
        raise ConfigError(
            msg,
            remediation=(
                "次元も意味空間も違うベクトルが同居します。"
                f"モデルを戻すか、{_REBUILD_REMEDIATION}"
            ),
        )

    def _assign(
        self,
        works: Sequence[_NoteWork],
        items: Sequence[_QueueItem],
        outcome: EmbeddingBatch,
    ) -> None:
        """応答のベクトルを送った順にノートへ配り、揃ったノートを確定する。"""
        for (index, _chunk), vector in zip(items, outcome.vectors, strict=True):
            works[index].vectors.append(vector)
        for index, _items in _note_groups(items):
            work = works[index]
            if work.is_complete:
                self._store_note(work)

    def _store_note(self, work: _NoteWork) -> None:
        """1 ノート分のチャンクをまとめてストアへ置き換える (追記ではない)。"""
        records = tuple(
            ChunkRecord(
                chunk_id=chunk.chunk_id,
                relpath=chunk.relpath,
                ordinal=chunk.ordinal,
                part_index=chunk.part_index,
                heading_path=chunk.heading_path,
                body=chunk.body,
                estimated_tokens=chunk.estimated_tokens,
                tags=chunk.tags,
                links=chunk.links,
                vector=vector,
            )
            for chunk, vector in zip(work.chunks, work.vectors, strict=True)
        )
        self._store.replace_note(work.relpath, records)
        self.indexed_counts[work.relpath] = len(records)
        self.embedded_chunks += len(records)
        logger.debug(
            "ノートを索引しました: relpath=%s chunks=%d", work.relpath, len(records)
        )

    def _fail(self, relpath: str, exc: LlmkitError) -> None:
        """1 ノートの失敗を記録する。**本文も絶対パスも載せない**。"""
        self._failed.add(relpath)
        logger.warning(
            "ノートを索引できませんでした。記録して次へ進みます: relpath=%s error=%s",
            relpath,
            type(exc).__name__,
        )


def _discard_every_chunk(store: VectorStore) -> None:
    """全再構築のために既存のチャンクをすべて捨てる。

    前提 (``index_fingerprint``) が変わった実行では、既存のチャンクは別の
    規則で切られた別のモデルのベクトルかもしれない。残したまま上書きすると、
    今回の対象にならなかったノート (読めなくなった等) のチャンクだけが古い
    規則のまま索引に居座る。
    """
    for relpath in tuple(sorted(store.note_chunk_counts())):
        store.delete_note(relpath)


def _next_manifest(
    settings: RagSettings,
    plan: IndexPlan,
    previous: IndexManifest | None,
    builder: _IndexBuilder,
    spec: ModelSpec,
) -> IndexManifest | None:
    """次に書き出すマニフェスト。埋め込みが 1 度も成立しなければ ``None``。

    土台は「前回のマニフェストから、vault から消えたノートを除いたもの」で、
    そこへ**今回書き終えたノートだけ**を上書きする。この形にすると、

    * 再処理しなかったノートは前回の記録がそのまま残る、
    * 失敗したノートは**古い記録のまま**残る (ストアにも古いチャンクが残って
      いるので索引と一致し、次回は ``sha256`` の違いで必ず再試行される)、
    * 新規で失敗したノートは 1 度も載らない (D-38)、

    の 3 つが同時に成り立つ。全再構築ではストアを空にしているので土台も空に
    する (古い記録だけが残ると、チャンクの無いノートを索引済みと名乗る)。
    """
    embedding = builder.embedding
    if embedding is None:
        return None
    entries: dict[str, ManifestNote] = {}
    if previous is not None and not plan.full_rebuild:
        deleted = set(plan.deleted)
        entries = {
            note.relpath: note for note in previous.notes if note.relpath not in deleted
        }
    for relpath, chunk_count in builder.indexed_counts.items():
        entries[relpath] = ManifestNote(
            relpath=relpath,
            sha256=plan.note_digests[relpath],
            chunk_count=chunk_count,
        )
    notes = tuple(entries[relpath] for relpath in sorted(entries))
    return IndexManifest(
        schema_version=INDEX_SCHEMA_VERSION,
        index_fingerprint=plan.fingerprint,
        fingerprint_inputs=fingerprint_inputs(settings, spec),
        embedding=embedding,
        notes=notes,
        totals=ManifestTotals(
            notes=len(notes), chunks=sum(note.chunk_count for note in notes)
        ),
    )


def _finalize(
    settings: RagSettings,
    plan: IndexPlan,
    previous: IndexManifest | None,
    builder: _IndexBuilder,
    spec: ModelSpec,
    *,
    store: VectorStore,
) -> None:
    """確定は ``store.commit()`` → :func:`write_manifest` の順に固定する (D-37)。

    逆順にすると、マニフェストだけ進んだ状態でクラッシュしたとき、次回の実行が
    ``sha256`` の一致を見て「変更なし」と判定し、**チャンクが欠けたまま永久に
    固定される**。この順序ならクラッシュ時に古いのはマニフェストの側になり、
    次回はそのノートを再処理するだけで済む (安全側に倒れる)。
    """
    store.commit()
    manifest = _next_manifest(settings, plan, previous, builder, spec)
    if manifest is None:
        logger.info("埋め込みが 1 件も成立しなかったため、マニフェストを更新しません")
        return
    write_manifest(settings, manifest)


def build_index(
    settings: RagSettings,
    config: AppConfig,
    *,
    embedding_client: EmbeddingClient,
    store: VectorStore,
    manifest: IndexManifest | None,
) -> IndexResult:
    """差分更新を実行して索引を更新する (要件書 L309 / L310)。

    :func:`plan_index` が ``changed`` + ``new`` と判定したノートだけを読み直し、
    パース → チャンク → 埋め込み → :meth:`~rag.VectorStore.replace_note` の順に
    処理する。埋め込みのバッチは**ノート境界を跨ぐ**が、ストアへ書くのは
    そのノートの全チャンクが揃ってからで、部分的に成功したチャンクは破棄する
    (D-38)。半分だけ載せると次回は ``sha256`` が一致するので「変更なし」と
    判定され、**欠けたまま永久に治らない**。

    ``manifest=None`` を渡すと再利用できる索引が無い扱いになり、全再構築に
    なる (T7 の ``--rebuild`` はこの経路を使う)。

    Args:
        settings: 解決済みの索引設定。
        config: 読み込み済み設定 (埋め込みモデルの出典、D-27)。
        embedding_client: :class:`llmkit.EmbeddingClient`。HTTP はこの中だけ
            (``rag`` はランタイムに直接触れない、D-25)。
        store: 索引の書き込み先。
        manifest: 既存のマニフェスト (:func:`load_manifest` の戻り値)。

    Returns:
        :class:`IndexResult` (件数のみ。相対パスも本文も含まない)。

    Raises:
        ConfigError: ランタイムが名乗ったモデル名・次元が既存の索引と食い違う
            場合 (この場合は索引ディレクトリに **1 バイトも書かない**)、
            または索引を書き出せない場合。
        RuntimeUnavailableError: ランタイムに接続できない場合。
        OutOfMemoryError: ランタイム側で VRAM が枯渇した場合。
        ModelNotFoundError: ランタイム側にモデルが無い場合。
            上記 3 つは**次のノートでも必ず再発する**ため即座に中断するが、
            そこまでに完了したノートは確定してから送出し直す。
    """
    started = time.perf_counter()
    spec = resolve_embedding_spec(config)
    plan = plan_index(settings, config, store=store, manifest=manifest)
    if plan.full_rebuild:
        _discard_every_chunk(store)
    for relpath in plan.deleted:
        store.delete_note(relpath)
    logger.info(
        "索引を開始します: vault=%s pending=%d unchanged=%d deleted=%d rebuild=%s",
        settings.vault_id,
        len(plan.pending),
        len(plan.unchanged),
        len(plan.deleted),
        plan.full_rebuild,
    )
    builder = _IndexBuilder(
        settings,
        embedding_client=embedding_client,
        store=store,
        baseline=None if plan.full_rebuild or manifest is None else manifest.embedding,
    )
    fatal = builder.run(plan.pending)
    _finalize(settings, plan, manifest, builder, spec, store=store)
    if fatal is not None:
        logger.warning(
            "ランタイムの障害により索引を中断しました: error=%s indexed=%d",
            type(fatal).__name__,
            len(builder.indexed_counts),
        )
        raise fatal
    embedding = builder.embedding
    result = IndexResult(
        indexed_notes=len(builder.indexed_counts),
        skipped_notes=len(plan.unchanged),
        failed_notes=builder.failed_notes,
        deleted_notes=len(plan.deleted),
        embedded_chunks=builder.embedded_chunks,
        request_count=builder.request_count,
        elapsed_s=time.perf_counter() - started,
        fingerprint=plan.fingerprint,
        dimensions=None if embedding is None else embedding.dimensions,
    )
    logger.info(
        "索引が完了しました: vault=%s indexed=%d skipped=%d failed=%d "
        "deleted=%d chunks=%d requests=%d elapsed_s=%.3f",
        settings.vault_id,
        result.indexed_notes,
        result.skipped_notes,
        result.failed_notes,
        result.deleted_notes,
        result.embedded_chunks,
        result.request_count,
        result.elapsed_s,
    )
    return result
