"""ベクトルストアの抽象と 2 実装 (D-39 / D-41)。

索引の永続化をここに閉じる。差分更新 (3b T6) の粒度が**ノート単位**なので、
このモジュールが公開する更新 API も ``replace_note`` / ``delete_note`` の
ノート単位だけであり、チャンク単位の追加・削除は持たない。ノートの一部だけが
差し替わった中間状態を作れる API を用意すると、埋め込みの途中で失敗した実行が
「半分だけ新しいチャンク」を索引に残せてしまう (§4 論点5)。

### 検索メソッドを今は定義しない (D-39)

``search(vector, k)`` を今決めると、タグ絞り込み・スコア閾値・MMR を足す段階で
Protocol そのものを変えることになる。代わりに ``iter_records()`` (全走査) だけを
置く。合成 vault 規模 (数百チャンク) の総当たりコサイン類似度は純 Python でも
数十 ms であり、次サイクルの検索は store の**外**に書ける。

### 永続レコードに ``embed_text`` を入れない (D-41)

:class:`rag.chunker.Chunk` は ``body`` と ``embed_text`` を両方持つが、JSONL に
両方書くと「片方だけ差し替わったレコード」が索引に残せる状態になり、本文も
2 重に保存される。埋め込み文字列は :func:`rag.chunker.render_embed_text` で
``heading_path`` + ``body`` から**必ず再構成する** (組み立ての実装は 1 つだけ)。
同じ理由で**モデル名と次元もレコードに持たせない**。索引 1 つに対して 1 つしか
無い値をレコード数だけ複製すると、D-19 と同型の「片方だけ変わった」状態を
作れてしまうため、manifest (T5) に 1 か所だけ置く。

### 単一ファイル + 原子的な差し替え

``chunks.jsonl`` は 1 ファイルにまとめ、``commit()`` で一時ファイルへ全量を
書いてから :func:`os.replace` で差し替える。ノート単位にファイルを分けると
「一部のノートだけ新しい」部分書き込み状態が観測でき、原子性の保証が難しい。
「再処理しない」は**再埋め込みをしない**ことであってファイルを書かないことでは
ないため、全量の書き直しは受け入れる。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Protocol

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as pydantic_dataclass

from llmkit import ConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkRecord",
    "InMemoryVectorStore",
    "JsonlVectorStore",
    "VectorStore",
]

_FORBID_EXTRA = ConfigDict(extra="forbid")

NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonEmptyVector = Annotated[tuple[float, ...], Field(min_length=1)]


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class ChunkRecord:
    """索引に永続化するチャンク 1 件。

    ``pydantic`` の frozen dataclass にするのは、**書き出す型と読み込む型を
    1 つにする**ため (D-08。``BaseModel`` は継承しない)。別々に定義すると、
    書き出し側にフィールドを足しても読み込み側が黙って無視する状態が作れる。
    ``extra="forbid"`` なので、未知のキーを持つ行は読み込み時に落ちる
    (D-41 を読み取り方向からも守る)。

    Attributes:
        chunk_id: ``<relpath>#<ordinal を 4 桁 0 詰め>`` (:class:`rag.Chunk` と同じ)。
        relpath: vault ルートからの相対パス (POSIX 表記)。差分更新の単位。
        ordinal: ノート内の通し番号 (0 始まり)。行順の第 2 キー。
        part_index: 同じセクションを分割したときの通し番号 (0 始まり)。
        heading_path: ``(ノートタイトル, H1, H2, …)``。
        body: 本文だけ。**見出し経路の接頭辞を含まない**。
        estimated_tokens: 近似トークン数 (D-32)。
        tags: ノート単位のタグ。
        links: ノート単位のリンク先。
        vector: 埋め込みベクトル。**次元はここに書かない** (D-41)。長さの
            食い違いは読み込み時に :class:`ConfigError` として検出する。
    """

    chunk_id: NonEmptyStr
    relpath: NonEmptyStr
    ordinal: NonNegativeInt
    part_index: NonNegativeInt
    heading_path: tuple[str, ...]
    body: NonEmptyStr
    estimated_tokens: NonNegativeInt
    tags: tuple[str, ...]
    links: tuple[str, ...]
    vector: NonEmptyVector


_RECORD_ADAPTER: TypeAdapter[ChunkRecord] = TypeAdapter(ChunkRecord)


class VectorStore(Protocol):
    """索引の読み書き口。実装は :class:`JsonlVectorStore` と
    :class:`InMemoryVectorStore` の 2 つで、**同じ適合テストを通る**。

    検索メソッドは意図的に持たない (D-39)。
    """

    def note_chunk_counts(self) -> Mapping[str, int]:
        """``relpath -> チャンク数``。整合検査と計画に使う。"""
        ...

    def replace_note(self, relpath: str, records: Sequence[ChunkRecord]) -> None:
        """``relpath`` のチャンクを ``records`` で**置き換える** (追記しない)。"""
        ...

    def delete_note(self, relpath: str) -> None:
        """``relpath`` のチャンクをすべて取り除く (存在しなくても例外にしない)。"""
        ...

    def iter_records(self) -> Iterator[ChunkRecord]:
        """全レコードを ``(relpath, ordinal)`` 昇順で返す。"""
        ...

    def commit(self) -> None:
        """永続化を確定する (原子的)。永続化しない実装では何もしない。"""
        ...


def _ordered(notes: Mapping[str, tuple[ChunkRecord, ...]]) -> list[ChunkRecord]:
    """``(relpath, ordinal)`` 昇順に並べる。

    行順を出力の直前に 1 か所で決めることで、``replace_note`` の呼び出し順や
    dict の挿入順が成果物のバイト列に漏れない (D-37 のバイト一致の前提)。
    """
    return sorted(
        (record for records in notes.values() for record in records),
        key=lambda record: (record.relpath, record.ordinal),
    )


class _RecordTable:
    """``relpath -> チャンク`` の表。2 実装が共有する唯一の状態。

    契約 (適合テスト) の実体はここにあり、:class:`JsonlVectorStore` と
    :class:`InMemoryVectorStore` の違いは ``commit()`` だけになる。表の操作を
    2 か所に書くと、片方だけが「置換」で片方が「追記」という乖離が入り得る。
    """

    def __init__(self, records: Iterable[ChunkRecord] = ()) -> None:
        grouped: dict[str, list[ChunkRecord]] = {}
        for record in records:
            if record.relpath not in grouped:
                grouped[record.relpath] = []
            grouped[record.relpath].append(record)
        self._notes: dict[str, tuple[ChunkRecord, ...]] = {
            relpath: tuple(note_records) for relpath, note_records in grouped.items()
        }

    def note_chunk_counts(self) -> Mapping[str, int]:
        return {relpath: len(records) for relpath, records in self._notes.items()}

    def replace_note(self, relpath: str, records: Sequence[ChunkRecord]) -> None:
        mismatched = sorted(
            {record.relpath for record in records if record.relpath != relpath}
        )
        if mismatched:
            msg = (
                f"別のノートのチャンクを {relpath} に書き込もうとしました "
                f"(食い違い {len(mismatched)} 件)"
            )
            raise ConfigError(
                msg,
                remediation=(
                    "store.replace_note には同じ relpath のチャンクだけを渡してください"
                ),
            )
        if not records:
            self.delete_note(relpath)
            return
        self._notes[relpath] = tuple(records)

    def delete_note(self, relpath: str) -> None:
        if relpath in self._notes:
            del self._notes[relpath]

    def iter_records(self) -> Iterator[ChunkRecord]:
        return iter(_ordered(self._notes))

    def render(self) -> str:
        """JSONL 全文。同じ内容なら**常に同じバイト列**になる。"""
        return "".join(
            json.dumps(_payload(record), ensure_ascii=False, sort_keys=True) + "\n"
            for record in _ordered(self._notes)
        )


def _payload(record: ChunkRecord) -> dict[str, object]:
    """1 レコードの JSON 表現。

    ``dataclasses.asdict`` ではなくキーを明示するのは、``embed_text`` /
    ``model`` / ``dimensions`` を**足さない**という決定 (D-41) を読める形で
    残すため。フィールドの追加漏れ・余計なキーの追加は、書き出したキーの集合と
    :class:`ChunkRecord` の宣言を突き合わせるテストが検出する。
    """
    return {
        "chunk_id": record.chunk_id,
        "relpath": record.relpath,
        "ordinal": record.ordinal,
        "part_index": record.part_index,
        "heading_path": list(record.heading_path),
        "body": record.body,
        "estimated_tokens": record.estimated_tokens,
        "tags": list(record.tags),
        "links": list(record.links),
        "vector": list(record.vector),
    }


class InMemoryVectorStore:
    """永続化しないストア。テストと ``--dry-run`` の計画に使う。

    ``commit()`` は何もしない。「書かない」ことを明示的な実装にしておくと、
    永続化の有無を呼び出し側が分岐せずに済む (:class:`VectorStore` を
    受け取る関数はどちらの実装でも同じコードで動く)。
    """

    def __init__(self, records: Iterable[ChunkRecord] = ()) -> None:
        self._table = _RecordTable(records)

    def note_chunk_counts(self) -> Mapping[str, int]:
        return self._table.note_chunk_counts()

    def replace_note(self, relpath: str, records: Sequence[ChunkRecord]) -> None:
        self._table.replace_note(relpath, records)

    def delete_note(self, relpath: str) -> None:
        self._table.delete_note(relpath)

    def iter_records(self) -> Iterator[ChunkRecord]:
        return self._table.iter_records()

    def commit(self) -> None:
        """何もしない (永続化しない実装であることを明示する)。"""


class JsonlVectorStore:
    """単一の JSONL ファイルに永続化するストア。

    生成時にファイルを全読み込みし、以降の更新はメモリ上で行い、``commit()``
    で全量を書き直す。読み込み対象は**索引ディレクトリの中のファイルだけ**で、
    vault 相対パスを受け取る API を 1 つも持たない (``rag/settings.py`` と同じ
    位置づけ、D-30)。
    """

    def __init__(self, path: Path) -> None:
        """``path`` を読み込む (存在しなければ空のストアとして始める)。

        Args:
            path: ``chunks.jsonl`` のパス。索引ディレクトリの中を指す。

        Raises:
            ConfigError: 行が壊れている / 未知のキーを持つ / ベクトルの次元が
                そろっていない / 読み取りに失敗した場合。
        """
        self._path = path
        self._table = _RecordTable(_load_records(path))

    @property
    def path(self) -> Path:
        """永続化先。呼び出し側が出力先を報告するために読む (書き換えない)。"""
        return self._path

    def note_chunk_counts(self) -> Mapping[str, int]:
        return self._table.note_chunk_counts()

    def replace_note(self, relpath: str, records: Sequence[ChunkRecord]) -> None:
        self._table.replace_note(relpath, records)

    def delete_note(self, relpath: str) -> None:
        self._table.delete_note(relpath)

    def iter_records(self) -> Iterator[ChunkRecord]:
        return self._table.iter_records()

    def commit(self) -> None:
        """一時ファイルへ全量を書いてから :func:`os.replace` で差し替える。

        途中で失敗しても既存の索引は 1 バイトも変わらない。書き出しの形は
        ``harness/report.py`` に合わせ、``OSError`` は :class:`ConfigError` に
        翻訳する (例外メッセージにファイル名以外のパスを載せない)。

        Raises:
            ConfigError: 出力先に書けない場合。
        """
        payload = self._table.render()
        temporary = self._path.with_name(self._path.name + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self._path)
        except OSError as exc:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
            msg = f"索引を書き出せません: {self._path.name}"
            raise ConfigError(
                msg,
                remediation=(
                    f"index.dir の権限と空き容量を確認してください ({exc.strerror})"
                ),
            ) from exc
        logger.debug(
            "索引を書き出しました: file=%s records=%d",
            self._path.name,
            payload.count("\n"),
        )


def _load_records(path: Path) -> list[ChunkRecord]:
    """JSONL を厳格に読む。壊れた行・次元不一致は :class:`ConfigError`。

    例外メッセージには**ファイル名と行番号だけ**を載せ、行の中身 (ノート本文)
    も解決済みの絶対パスも載せない (CLAUDE.md のログ出力ルール、
    ``rag/settings.py`` と同じ扱い)。
    """
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"索引を読み込めません: {path.name}"
        raise ConfigError(
            msg,
            remediation=(
                "index.dir の権限を確認するか、索引を作り直してください (--rebuild)"
            ),
        ) from exc
    records: list[ChunkRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        records.append(_parse_line(path, line_number, line))
    _require_one_dimension(path, records)
    return records


def _parse_line(path: Path, line_number: int, line: str) -> ChunkRecord:
    try:
        return _RECORD_ADAPTER.validate_json(line)
    except ValidationError as exc:
        msg = (
            f"索引の行を読めません: {path.name}:{line_number} "
            f"({_format_validation_error(exc)})"
        )
        raise ConfigError(
            msg,
            remediation="索引を作り直してください (--rebuild)",
        ) from exc


def _format_validation_error(exc: ValidationError) -> str:
    """``キー名: 理由`` の一覧。**入力値そのものは載せない** (本文が漏れる)。

    JSON として壊れている行も ``pydantic`` は :class:`ValidationError` にする
    (``json_invalid``) ため、経路は 1 本で足りる。
    """
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return " / ".join(parts)


def _require_one_dimension(path: Path, records: Sequence[ChunkRecord]) -> None:
    """全レコードのベクトルが同じ次元であることを確かめる。

    次元はレコードに書かない (D-41) ので、ファイル単位の整合はここでしか
    見られない。次元の違うベクトルが同居したまま読み込めてしまうと、
    コサイン類似度が無意味になっても例外もテスト失敗も出ない (§2.2 と同型)。
    """
    dimensions = {len(record.vector) for record in records}
    if len(dimensions) <= 1:
        return
    msg = (
        f"索引の中でベクトルの次元がそろっていません: {path.name} "
        f"({len(dimensions)} 種類 / レコード {len(records)} 件)"
    )
    raise ConfigError(
        msg,
        remediation=(
            "埋め込みモデルを変えた索引が混ざっています。索引を作り直して"
            "ください (--rebuild)"
        ),
    )
