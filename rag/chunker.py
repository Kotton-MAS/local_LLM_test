"""見出しベースのチャンク分割と、外部依存の無いトークン数の**近似** (D-32)。

このモジュールは**純粋**である。ファイルにもネットワークにも触れず、入力は
:class:`rag.parser.ParsedNote` と :class:`rag.settings.ChunkSettings` だけで、
同じ入力からは常に同じチャンクが出る。3b の差分更新は「前提が変わっていない
ノートを再処理しない」仕組みなので、ここが決定論的でないと索引の中に別々の
前提で作られたチャンクが静かに混ざる。

### トークン数は「近似」である (D-32)

:func:`estimate_tokens` は文字種別 (CJK かそれ以外か) を数えて係数で割るだけの
近似器であり、埋め込みモデルの実トークナイザとは一致しない。``tiktoken`` は
新規依存であるうえ BPE が cl100k で ruri-v3 とも Qwen とも別物なので、依存を
増やしても正確さは得られない。ランタイムに数えさせる案は、チャンク境界の決定に
ネットワーク往復が要り D-02 (テストは実 HTTP を出さない) と両立しない。
要件書 L185 が求めているのは「上限 240 トークン**目安**(設定可能)」であり、
正確なトークン数ではない。

### 分割規則 (仕様書 §3 論点4 の 1〜5)

1. 見出しの検出は :func:`rag.parser.iter_headings` に委ね、フェンス判定を
   ここで再実装しない (規則が 2 か所に分かれると、片方だけ直った状態が生まれる)。
2. 見出し単位でセクションに切る。``heading_path`` は ``(ノートタイトル, H1, …)``
   で、**最初の見出しより前の本文にも文脈が付く**。
3. 上限を超えるセクションは **空行 (段落) 境界 → 行境界 → 文字境界**の順に
   分割する。**どの経路でも本文を 1 文字も落とさない**。
4. ``embed_text = heading_separator.join(heading_path) + "\\n\\n" + body``。
   **上限は ``embed_text`` に対して適用する**。``body`` だけを見て切ると、
   見出し経路の分だけ上限を静かに超えたテキストを埋め込みへ送ることになる。
5. 空文字列・空白のみのチャンクは作らない (空ノートは 0 チャンク)。
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable, Sequence

from llmkit import ConfigError
from rag.parser import Heading, ParsedNote, iter_headings
from rag.settings import ChunkSettings

logger = logging.getLogger(__name__)

__all__ = [
    "Chunk",
    "chunk_note",
    "estimate_tokens",
    "render_embed_text",
]

#: 見出し経路と本文の区切り。``embed_text`` の中でだけ使う。
_PREFIX_SEPARATOR = "\n\n"

#: CJK として数える符号位置の範囲 (両端を含む)。この表そのものが近似の定義で
#: あり、変えると ``index_fingerprint`` の前提が変わる (3b / D-28)。
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),  # CJK の約物
    (0x3040, 0x309F),  # ひらがな
    (0x30A0, 0x30FF),  # カタカナ
    (0x3400, 0x4DBF),  # CJK 統合漢字 拡張 A
    (0x4E00, 0x9FFF),  # CJK 統合漢字
    (0xAC00, 0xD7A3),  # ハングル音節
    (0xF900, 0xFAFF),  # CJK 互換漢字
    (0xFF00, 0xFF9F),  # 全角英数・記号と半角カナ
    (0x20000, 0x2FA1F),  # CJK 統合漢字 拡張 B 以降
)


def _is_cjk(character: str) -> bool:
    code_point = ord(character)
    return any(start <= code_point <= end for start, end in _CJK_RANGES)


def _count_chars(text: str) -> tuple[int, int]:
    """``(CJK 文字数, 総文字数)`` を返す。差分更新のための出発点。"""
    cjk_characters = sum(1 for character in text if _is_cjk(character))
    return cjk_characters, len(text)


def _tokens_from_counts(
    cjk_count: int,
    total_count: int,
    *,
    cjk_chars_per_token: float,
    ascii_chars_per_token: float,
) -> int:
    """D-32 の近似式そのもの。``(CJK 文字数, 総文字数)`` からトークン数を出す。

    ``ceil(CJK 文字数 / cjk_chars_per_token + その他の文字数 /
    ascii_chars_per_token)``。

    この関数が近似式の**唯一の実装**である。:func:`estimate_tokens`
    (``_count_chars`` との合成) と、文字境界フォールバック
    (:func:`_pack_characters` 内の ``tokens_for``、差分更新のため文字列を
    毎回数え直さない) の両方がここを通る。式を 2 か所に複製すると、
    ``estimate_tokens`` を差し替えても文字境界フォールバックの分割数が
    追随しない乖離が生まれる (round-10 レビュー)。
    """
    other_count = total_count - cjk_count
    return math.ceil(
        cjk_count / cjk_chars_per_token + other_count / ascii_chars_per_token
    )


def estimate_tokens(
    text: str, *, cjk_chars_per_token: float, ascii_chars_per_token: float
) -> int:
    """トークン数の**近似値**を返す (実トークナイザではない)。

    ``_count_chars`` (文字を数える) と ``_tokens_from_counts`` (式を適用する)
    の合成。外部トークナイザも推論ランタイムも使わないため、GPU もネット
    ワークも無い環境で決定論的に評価できる (D-32)。

    Args:
        text: 数える対象。
        cjk_chars_per_token: CJK 1 トークンあたりの文字数 (既定 1.0)。
        ascii_chars_per_token: CJK 以外 1 トークンあたりの文字数 (既定 4.0)。

    Returns:
        近似トークン数 (0 以上)。空文字列は 0。

    Note:
        係数が正であることは :class:`rag.settings.ChunkSettings` が保証する。
    """
    if not text:
        return 0
    cjk_characters, total_characters = _count_chars(text)
    return _tokens_from_counts(
        cjk_characters,
        total_characters,
        cjk_chars_per_token=cjk_chars_per_token,
        ascii_chars_per_token=ascii_chars_per_token,
    )


@dataclasses.dataclass(frozen=True)
class Chunk:
    """埋め込みの単位 1 件。

    自分で組み立てて書き出すだけの型なので素の frozen dataclass にする (決定23)。

    Attributes:
        chunk_id: ``<relpath>#<ordinal を 4 桁 0 詰め>``。決定論的で可読、
            同じ入力の再実行で完全に一致する。
        relpath: vault ルートからの相対パス (POSIX 表記)。
        ordinal: ノート内の通し番号 (0 始まり)。
        heading_path: ``(ノートタイトル, H1, H2, …)``。連続する重複は畳む。
        part_index: 同じセクションを分割したときの通し番号 (0 始まり)。
        body: 本文だけ。**見出し経路の接頭辞を含まない**。
        embed_text: 埋め込みへ渡す文字列。見出し経路 + ``\\n\\n`` + ``body``。
        estimated_tokens: ``embed_text`` の近似トークン数 (上限判定に使った値)。
        tags: ノート単位のタグ (チャンクごとには分けない)。
        links: ノート単位のリンク先 (同上)。
    """

    chunk_id: str
    relpath: str
    ordinal: int
    heading_path: tuple[str, ...]
    part_index: int
    body: str
    embed_text: str
    estimated_tokens: int
    tags: tuple[str, ...]
    links: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _Section:
    """見出しで区切った本文の 1 区画 (分割前)。"""

    heading_path: tuple[str, ...]
    body: str


def _heading_path(title: str, stack: Sequence[Heading]) -> tuple[str, ...]:
    """``(ノートタイトル, H1, H2, …)`` を組み立てる。

    **直前の要素と同じ文字列は加えない。** Obsidian では H1 をノートのタイトルと
    同じ文字列にする書き方が一般的で、そのまま並べると
    ``プロジェクトAlpha > プロジェクトAlpha > 設計`` のように同じ語が 2 回出る。
    情報が増えないのに ``embed_text`` の接頭辞が伸び、その分だけ本文に使える
    上限が削られる。離れた位置の同名見出し (``設計 > 概要 > 設計``) は文脈が
    違うため畳まない。
    """
    path: list[str] = [title]
    for heading in stack:
        if heading.text and heading.text != path[-1]:
            path.append(heading.text)
    return tuple(path)


def _iter_sections(parsed: ParsedNote) -> list[_Section]:
    """本文を見出し単位のセクションに切る (見出し行そのものは本文に残さない)。

    最初の見出しより前の本文も 1 セクションとして扱う。捨てると、見出しを
    後から付け足したノートの冒頭が索引から静かに消える。
    """
    lines = parsed.body.splitlines(keepends=True)
    headings = {heading.line_index: heading for heading in iter_headings(parsed.body)}
    sections: list[_Section] = []
    stack: list[Heading] = []
    path = _heading_path(parsed.title, stack)
    buffer: list[str] = []
    for line_index, line in enumerate(lines):
        if line_index in headings:
            heading = headings[line_index]
            sections.append(_Section(heading_path=path, body="".join(buffer)))
            buffer.clear()
            while stack and stack[-1].level >= heading.level:
                stack.pop()
            stack.append(heading)
            path = _heading_path(parsed.title, stack)
            continue
        buffer.append(line)
    sections.append(_Section(heading_path=path, body="".join(buffer)))
    return sections


def _paragraph_units(text: str) -> list[str]:
    """空行を境に段落へ切る。連結すると元の文字列に**完全に**戻る。

    空行は直前の段落の末尾に付ける。切り出した断片を捨てないため、分割の経路が
    どれであっても本文は失われない。
    """
    units: list[str] = []
    current: list[str] = []
    has_content = False
    for line in text.splitlines(keepends=True):
        current.append(line)
        if line.strip():
            has_content = True
            continue
        if has_content:
            units.append("".join(current))
            current = []
            has_content = False
    if current:
        units.append("".join(current))
    return units


def _pack(
    units: Sequence[str],
    fits: Callable[[str], bool],
    finer: Callable[[str], list[str]],
) -> list[str]:
    """単位を順に詰める。単体で上限に収まらない単位は ``finer`` で細かく分ける。

    連結すると入力に戻る (単位を捨てない・並べ替えない)。
    """
    parts: list[str] = []
    buffer = ""
    for unit in units:
        candidate = buffer + unit
        if fits(candidate):
            buffer = candidate
            continue
        if buffer:
            parts.append(buffer)
            buffer = ""
        if fits(unit):
            buffer = unit
            continue
        parts.extend(finer(unit))
    if buffer:
        parts.append(buffer)
    return parts


def _pack_characters(
    unit: str, heading_path: Sequence[str], settings: ChunkSettings
) -> list[str]:
    """文字境界フォールバック (規則3)。1 文字ずつ詰める。

    ``_pack`` の汎用実装をそのまま文字単位に適用すると、1 文字追加するたびに
    ``embed_text`` 全体 (見出し経路の接頭辞 + それまでのバッファ) を
    ``estimate_tokens`` で先頭から再計算することになる。バッファは
    ``max_tokens`` 相当でリセットされるとはいえ、リセットのたびに 0 から
    その文字数までの再走査を繰り返すため定数係数が非常に大きい (F-9-014)。

    ここでは見出し経路由来の接頭辞の CJK 文字数・総文字数を 1 回だけ数え、
    バッファについても CJK 文字数・総文字数だけを保持して 1 文字ぶんを
    ``±1`` するだけにする。連結すると ``unit`` に戻る (文字を 1 つも捨てない)。
    単体で上限を超える文字でも捨てずにそのまま 1 チャンクにする
    (``_require_room_for_the_body`` が「見出し経路だけで上限を使い切る」
    場合を先に ``ConfigError`` にしており、本文を捨てるより上限を超える
    ほうが害が小さいため)。

    ``tokens_for`` は :func:`_tokens_from_counts` の部分適用であり、独自に式を
    持たない (D-32)。文字列を毎回数え直さないという性能特性 (F-9-014) を保った
    まま、近似式そのものは ``estimate_tokens`` と同じ 1 か所を通る。
    """
    prefix_cjk, prefix_total = _count_chars(
        settings.heading_separator.join(heading_path) + _PREFIX_SEPARATOR
    )

    def tokens_for(cjk_count: int, total_count: int) -> int:
        return _tokens_from_counts(
            cjk_count,
            total_count,
            cjk_chars_per_token=settings.cjk_chars_per_token,
            ascii_chars_per_token=settings.ascii_chars_per_token,
        )

    parts: list[str] = []
    buffer_chars: list[str] = []
    buffer_cjk = 0
    for character in unit:
        is_cjk = _is_cjk(character)
        candidate_cjk = buffer_cjk + (1 if is_cjk else 0)
        candidate_total = len(buffer_chars) + 1
        if (
            tokens_for(prefix_cjk + candidate_cjk, prefix_total + candidate_total)
            <= settings.max_tokens
        ):
            buffer_chars.append(character)
            buffer_cjk = candidate_cjk
            continue
        if buffer_chars:
            parts.append("".join(buffer_chars))
        buffer_chars = [character]
        buffer_cjk = 1 if is_cjk else 0
    if buffer_chars:
        parts.append("".join(buffer_chars))
    return parts


def render_embed_text(heading_path: Sequence[str], body: str, separator: str) -> str:
    """埋め込みへ渡す文字列を組み立てる (規則4)。**唯一の実装**。

    ``separator.join(heading_path) + "\n\n" + body``。:func:`chunk_note` の
    ``Chunk.embed_text`` も、索引レコード (:class:`rag.ChunkRecord`) からの
    再構成も、必ずこの関数を通る。永続化では ``embed_text`` を保存せず
    ``heading_path`` + ``body`` から作り直すため (D-41)、組み立てが 2 か所に
    分かれると「保存時と再構成時で別の文字列」という、例外もテスト失敗も
    出ない乖離がそのまま検索精度の劣化になる (D-32 の ``_tokens_from_counts``
    と同じ理由で 1 か所に閉じる)。

    Args:
        heading_path: ``(ノートタイトル, H1, H2, …)``。
        body: 本文 (見出し経路の接頭辞を含まない)。
        separator: 見出し経路の連結子 (``ChunkSettings.heading_separator``)。

    Returns:
        埋め込み対象の文字列。
    """
    return separator.join(heading_path) + _PREFIX_SEPARATOR + body


def _embed_text(heading_path: Sequence[str], body: str, settings: ChunkSettings) -> str:
    """``settings`` から区切り文字を取り出して :func:`render_embed_text` に渡す。"""
    return render_embed_text(heading_path, body, settings.heading_separator)


def _estimate(text: str, settings: ChunkSettings) -> int:
    return estimate_tokens(
        text,
        cjk_chars_per_token=settings.cjk_chars_per_token,
        ascii_chars_per_token=settings.ascii_chars_per_token,
    )


def _split_body(
    body: str, heading_path: Sequence[str], settings: ChunkSettings
) -> list[str]:
    """上限に収まるように本文を分割する (段落 → 行 → 文字の順、規則3)。"""
    fits = _fits_predicate(heading_path, settings)
    if fits(body):
        return [body]

    def by_characters(unit: str) -> list[str]:
        return _pack_characters(unit, heading_path, settings)

    def by_lines(unit: str) -> list[str]:
        return _pack(unit.splitlines(keepends=True), fits, by_characters)

    return _pack(_paragraph_units(body), fits, by_lines)


def _fits_predicate(
    heading_path: Sequence[str], settings: ChunkSettings
) -> Callable[[str], bool]:
    """上限判定を返す。**判定の対象は ``embed_text``** である (規則4)。

    ``body`` だけを見て切ると、見出し経路の分だけ上限を超えたテキストが
    埋め込みへ渡る。超過は例外にもテスト失敗にもならず、埋め込み側で黙って
    切り詰められて検索精度だけが落ちる。
    """

    def fits(candidate: str) -> bool:
        embed_text = _embed_text(heading_path, candidate, settings)
        return _estimate(embed_text, settings) <= settings.max_tokens

    return fits


def _require_room_for_the_body(
    relpath: str, heading_path: Sequence[str], settings: ChunkSettings
) -> None:
    """見出し経路だけで上限を使い切っていないことを確かめる。

    使い切っていると、本文を 1 文字も落とさない唯一の出力が「1 文字ずつの
    チャンクの山」になる。それは索引としては壊れており、静かに作るより設定の
    誤りとして落とすほうがよい。

    例外メッセージには**相対パスと件数・トークン数だけ**を載せる。見出し文字列は
    ノート本文の一部なのでログにも例外にも出さない (CLAUDE.md ログ出力ルール)。
    """
    prefix_tokens = _estimate(_embed_text(heading_path, "", settings), settings)
    if prefix_tokens < settings.max_tokens:
        return
    msg = (
        f"見出し経路だけで chunk.max_tokens を使い切ります: {relpath} "
        f"(見出し {len(heading_path)} 段 / 近似 {prefix_tokens} トークン "
        f">= 上限 {settings.max_tokens})"
    )
    raise ConfigError(
        msg,
        remediation=(
            "chunk.max_tokens を大きくするか、chunk.heading_separator を"
            "短くしてください"
        ),
    )


def chunk_note(parsed: ParsedNote, settings: ChunkSettings) -> tuple[Chunk, ...]:
    """パース済みノート 1 件をチャンクへ分割する。

    Args:
        parsed: :func:`rag.parser.parse_note` の結果。
        settings: ``[chunk]`` セクションの設定 (``RagSettings.chunk``)。

    Returns:
        本文の出現順に並んだ :class:`Chunk` の組。空ノート・空白のみのノートは
        空の組。

    Raises:
        ConfigError: 見出し経路だけで ``max_tokens`` を使い切る場合。
    """
    chunks: list[Chunk] = []
    ordinal = 0
    for section in _iter_sections(parsed):
        body = section.body.strip()
        if not body:
            continue
        _require_room_for_the_body(parsed.relpath, section.heading_path, settings)
        parts = [
            stripped
            for part in _split_body(body, section.heading_path, settings)
            if (stripped := part.strip())
        ]
        for part_index, part in enumerate(parts):
            embed_text = _embed_text(section.heading_path, part, settings)
            chunks.append(
                Chunk(
                    chunk_id=f"{parsed.relpath}#{ordinal:04d}",
                    relpath=parsed.relpath,
                    ordinal=ordinal,
                    heading_path=section.heading_path,
                    part_index=part_index,
                    body=part,
                    embed_text=embed_text,
                    estimated_tokens=_estimate(embed_text, settings),
                    tags=parsed.tags,
                    links=parsed.links,
                )
            )
            ordinal += 1
    logger.debug(
        "チャンクを作成しました: relpath=%s chunks=%d", parsed.relpath, len(chunks)
    )
    return tuple(chunks)
