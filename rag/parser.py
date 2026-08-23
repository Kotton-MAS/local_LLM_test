"""Obsidian Markdown のパース (frontmatter / wikilink / タグ / 見出し)。

このモジュールは**ファイルに触れない**。入力は相対パスと本文文字列であり、
vault の読み取りは :mod:`rag.vault` だけが行う (§3 論点5 の「構造」層)。

### frontmatter (D-31)

1 行目が厳密に ``---`` で、それ以降に閉じの ``---`` がある場合だけを frontmatter と
みなす。閉じが無ければ**全体を本文**として扱う。閉じの判定を緩めると、
``---`` を水平線として使ったノートの本文冒頭が黙って消える。

パースは YAML のサブセット (スカラ / インラインリスト / ブロックリスト /
クォート文字列) だけを解釈する。PyYAML を入れないのは新規依存を足さないため
(D-18 と同じ理由)。**解釈できない行があっても例外にせず**、``frontmatter_raw`` に
ブロックの全文を残す。解釈できなかった行を本文へ混ぜることは決してしない。
混ぜると ``tags: [社外秘]`` のようなメタ情報が検索対象の本文に化ける。

### wikilink と見出しの非対称 (D-34)

**wikilink はコードフェンスの内側でも解決する。見出し検出はフェンスの内側を
無視する。** 両者の目的が違うためで、この非対称は意図的である。

- 見出しはチャンクの境界を決める。フェンス内の ``# コメント`` で本文を切ると、
  コードブロックが途中で分断される。
- wikilink は表示テキストとメタデータの抽出であり、フェンス内に残すと
  ``[[`` ``]]`` という記法上の記号が埋め込みテキストに混入する。フェンス内の
  リンクだけ別扱いにすると、同じ記法が場所によって別の意味を持つことになる。

``tests/test_rag_parser.py::test_wikilinks_inside_code_fences_are_resolved_but_headings_are_not``
が両方向を固定している。
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

__all__ = [
    "FrontmatterValue",
    "Heading",
    "ParsedNote",
    "iter_headings",
    "parse_note",
]

#: frontmatter の値として持てる型。スカラは文字列のまま持つ (日付や真偽値を
#: 勝手に型変換すると、YAML の全機能を自前で持つ方向に引きずられる)。
type FrontmatterValue = str | tuple[str, ...]

#: frontmatter の開始・終了行 (厳密一致)。
_FRONTMATTER_FENCE = "---"

#: ``[[note#heading]]`` を本文に展開するときの区切り。
_LINK_HEADING_SEPARATOR = " > "

#: ``![[...]]`` / ``[[...]]``。入れ子は扱わない (Obsidian も許していない)。
_WIKILINK_PATTERN = re.compile(r"(!?)\[\[([^\[\]\n]+)\]\]")

#: 本文中のインラインタグ。直前が単語構成文字・``/``・``#`` の場合は取らない
#: (URL の ``#fragment`` と ``##`` 見出しを除くため)。``# `` 形式の見出しは
#: ``#`` の直後が空白なので、この文字クラスには一致しない。
_INLINE_TAG_PATTERN = re.compile(r"(?<![\w/#])#([\w一-龥ぁ-んァ-ヶー/-]+)")

#: 見出し行 (フェンスの外側でのみ適用する)。
_HEADING_PATTERN = re.compile(r"\A(#{1,6})\s+(.*)\Z")

#: コードフェンスの開始・終了 (``` または ~~~)。
_FENCE_PATTERN = re.compile(r"\A\s{0,3}(`{3,}|~{3,})(.*)\Z")

#: ノートとして扱う拡張子。これ以外の埋め込み (画像・PDF) は本文から取り除く。
_NOTE_SUFFIXES = frozenset({"", ".md"})


@dataclasses.dataclass(frozen=True)
class Heading:
    """本文中の見出し 1 つ (フェンスの外側にあるものだけ)。

    Attributes:
        level: ``#`` の個数 (1〜6)。
        text: 見出しの文字列 (``#`` と前後の空白を除いたもの)。
        line_index: 本文を ``splitlines()`` したときの 0 始まりの行番号。
    """

    level: int
    text: str
    line_index: int


@dataclasses.dataclass(frozen=True)
class ParsedNote:
    """パース済みのノート 1 件。

    Attributes:
        relpath: vault ルートからの相対パス (POSIX 表記)。
        title: frontmatter の ``title``。無ければファイル名の stem。
        frontmatter: 解釈できた frontmatter の項目。
        frontmatter_raw: frontmatter ブロックの全文 (無ければ空文字列)。
            解釈できなかった行もここには残る (D-31)。
        tags: frontmatter の ``tags`` と本文中のインラインタグ (重複を除く)。
        links: ``[[...]]`` の参照先 (重複を除く)。
        embeds: ``![[...]]`` の参照先 (重複を除く)。
        body: wikilink を展開した本文。frontmatter は含まない。
    """

    relpath: str
    title: str
    frontmatter: Mapping[str, FrontmatterValue]
    frontmatter_raw: str
    tags: tuple[str, ...]
    links: tuple[str, ...]
    embeds: tuple[str, ...]
    body: str


def _split_frontmatter(text: str) -> tuple[str, str]:
    """``(frontmatter_raw, body)`` に分ける。frontmatter が無ければ前者は空。

    1 行目が厳密に ``---`` で、2 行目以降に ``---`` の行がある場合だけを
    frontmatter とする (D-31)。
    """
    if not text.startswith(_FRONTMATTER_FENCE):
        return "", text
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != _FRONTMATTER_FENCE:
        return "", text
    start = len(lines[0])
    cursor = start
    for line in lines[1:]:
        if line.strip() == _FRONTMATTER_FENCE:
            # 本文は閉じ行の直後から末尾まで。改行を含めて 1 文字も加工しない。
            return text[start:cursor], text[cursor + len(line) :]
        cursor += len(line)
    # 閉じが無い: 全体を本文として扱う (捨てない)。
    return "", text


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _split_inline_list(value: str) -> tuple[str, ...]:
    inner = value[1:-1].strip()
    if not inner:
        return ()
    return tuple(
        _strip_quotes(item.strip()) for item in inner.split(",") if item.strip()
    )


def _parse_frontmatter(raw: str) -> Mapping[str, FrontmatterValue]:
    """YAML のサブセットとして frontmatter を読む。

    解釈できない行は**黙って飛ばす** (例外にしない)。全文は
    ``ParsedNote.frontmatter_raw`` に残るため、情報は失われない (D-31)。
    """
    parsed: dict[str, FrontmatterValue] = {}
    block_list_key: str | None = None
    block_list_items: list[str] = []

    def flush() -> None:
        nonlocal block_list_key
        if block_list_key is not None:
            parsed[block_list_key] = tuple(block_list_items)
            block_list_key = None
            block_list_items.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and block_list_key is not None:
            block_list_items.append(_strip_quotes(stripped[2:].strip()))
            continue
        key, separator, value = line.partition(":")
        if not separator or line[:1].isspace() or not key.strip():
            # ネストしたマッピングや YAML の未対応記法。飛ばす。
            flush()
            continue
        flush()
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            parsed[key] = _split_inline_list(value)
        elif value:
            parsed[key] = _strip_quotes(value)
        else:
            block_list_key = key
            block_list_items.clear()
    flush()
    return parsed


def _is_note_target(target: str) -> bool:
    """``[[...]]`` の参照先がノートかどうか (拡張子で判定する)。"""
    return PurePosixPath(target).suffix.lower() in _NOTE_SUFFIXES


def _display_text(target: str, heading: str, alias: str) -> str:
    if alias:
        return alias
    if target and heading:
        return f"{target}{_LINK_HEADING_SEPARATOR}{heading}"
    if heading:
        return heading
    return target


def _resolve_wikilinks(
    body: str, self_name: str
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """本文中の wikilink を表示テキストへ展開し、参照先を集める。

    フェンスの内側かどうかで挙動を変えない (D-34)。

    Returns:
        ``(展開後の本文, links, embeds)``。参照先は出現順で重複を除く。
    """
    links: list[str] = []
    embeds: list[str] = []
    # 重複判定は list の `in` (線形探索) ではなく dict の keys で行う。
    # ノート内のユニークな参照数を m とすると `in` の線形探索は m 回ずつ増える
    # ため、重複除去全体が O(m^2) になる (F-9-013、ホットパス: parse_note は
    # ノート 1 件ごとに毎回呼ばれる)。_dedupe() と同じ「dict で O(1) 判定」の
    # 方針をここにも合わせる。
    seen_links: dict[str, None] = {}
    seen_embeds: dict[str, None] = {}

    def replace(match: re.Match[str]) -> str:
        is_embed = match.group(1) == "!"
        inner = match.group(2)
        target_part, _, alias = inner.partition("|")
        target, _, heading = target_part.partition("#")
        target = target.strip()
        heading = heading.strip()
        alias = alias.strip()
        reference = target if target else self_name
        collected = embeds if is_embed else links
        seen = seen_embeds if is_embed else seen_links
        if reference not in seen:
            seen[reference] = None
            collected.append(reference)
        if is_embed and not _is_note_target(target):
            # 画像・PDF などの埋め込みは本文から取り除く (記法も名前も残さない)。
            return ""
        return _display_text(target, heading, alias)

    return _WIKILINK_PATTERN.sub(replace, body), tuple(links), tuple(embeds)


def _frontmatter_tags(frontmatter: Mapping[str, FrontmatterValue]) -> tuple[str, ...]:
    # Mapping.get を使わない: rag/ には HTTP クライアントの .get を含む
    # 属性呼び出しを 1 つも置かないという境界検査 (D-25 guard) に合わせている。
    if "tags" not in frontmatter:
        return ()
    value = frontmatter["tags"]
    if isinstance(value, str):
        return (value.lstrip("#"),) if value else ()
    return tuple(item.lstrip("#") for item in value if item)


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for value in values:
        if value:
            seen.setdefault(value, None)
    return tuple(seen)


def iter_headings(body: str) -> tuple[Heading, ...]:
    """本文から見出しを取り出す。**コードフェンスの内側は見出しにしない** (D-34)。

    Args:
        body: frontmatter を除いた本文。

    Returns:
        出現順の :class:`Heading` の組。
    """
    headings: list[Heading] = []
    fence: str | None = None
    for line_index, line in enumerate(body.splitlines()):
        fence_match = _FENCE_PATTERN.match(line)
        if fence_match is not None:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        heading_match = _HEADING_PATTERN.match(line)
        if heading_match is None:
            continue
        headings.append(
            Heading(
                level=len(heading_match.group(1)),
                text=heading_match.group(2).strip(),
                line_index=line_index,
            )
        )
    return tuple(headings)


def parse_note(relpath: str, text: str) -> ParsedNote:
    """ノート 1 件をパースする。例外は出さない (壊れた frontmatter も含む)。

    Args:
        relpath: vault ルートからの相対パス (POSIX 表記)。``title`` の既定値と
            ``[[#heading]]`` の参照先 (自ノート) の決定に使う。
        text: ノートの本文 (UTF-8 で復号済み)。

    Returns:
        :class:`ParsedNote`。
    """
    frontmatter_raw, raw_body = _split_frontmatter(text)
    frontmatter = _parse_frontmatter(frontmatter_raw)
    self_name = PurePosixPath(relpath).stem
    # Mapping.get を使わない (D-25 guard が .get() を落とす)。ruff の SIM401 も
    # 三項演算子の形にすると .get を勧めてくるため、if 文で書く。
    title = self_name
    if "title" in frontmatter:
        declared = frontmatter["title"]
        if isinstance(declared, str) and declared:
            title = declared
    body, links, embeds = _resolve_wikilinks(raw_body, self_name)
    inline_tags = tuple(match.group(1) for match in _INLINE_TAG_PATTERN.finditer(body))
    return ParsedNote(
        relpath=relpath,
        title=title,
        frontmatter=frontmatter,
        frontmatter_raw=frontmatter_raw,
        tags=_dedupe(_frontmatter_tags(frontmatter) + inline_tags),
        links=_dedupe(links),
        embeds=_dedupe(embeds),
        body=body,
    )
