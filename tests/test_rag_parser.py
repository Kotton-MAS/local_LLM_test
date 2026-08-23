"""Obsidian パーサ (``rag/parser.py``) の検証。

要件書 L312 (wikilink) / L313 (frontmatter) の直接検証と、2 つの guard を含む。

- ``test_unparsable_frontmatter_never_leaks_into_the_body`` … D-31
- ``test_wikilinks_inside_code_fences_are_resolved_but_headings_are_not`` … D-34

パーサはファイルに触れないため、記法の網羅はインライン文字列で、実データの
検証は合成 vault (``vaults/sample/``) で行う。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_rag_settings

import rag

# --------------------------------------------------------------------------
# 合成 vault 全体
# --------------------------------------------------------------------------


def parsed_sample_notes(directory: Path) -> dict[str, rag.ParsedNote]:
    settings = rag.load_settings(write_rag_settings(directory.parent))
    return {
        entry.relpath: rag.parse_note(
            entry.relpath, rag.read_note_text(settings, entry.relpath)
        )
        for entry in rag.iter_vault_files(settings)
    }


def test_every_sample_note_parses_without_raising(sample_vault_copy: Path) -> None:
    """12 ノート (うち索引対象 11) がすべて例外なくパースできる。"""
    notes = parsed_sample_notes(sample_vault_copy)

    assert len(notes) == 11
    assert all(isinstance(note.body, str) for note in notes.values())
    assert notes["notes/empty.md"].body == ""
    assert notes["notes/whitespace-only.md"].body.strip() == ""


def test_no_wikilink_syntax_survives_in_any_body(sample_vault_copy: Path) -> None:
    """L312: 全ノートの ``body`` に ``[[`` と ``]]`` が 1 つも残らない。"""
    notes = parsed_sample_notes(sample_vault_copy)

    offenders = [
        relpath
        for relpath, note in notes.items()
        if "[[" in note.body or "]]" in note.body
    ]

    assert not offenders, f"wikilink 記法が本文に残っています: {offenders}"
    # 検査が「そもそもリンクが無い」で通っていないことの確認。
    assert notes["notes/weekly-review.md"].links == ("project-alpha", "weekly-review")


# --------------------------------------------------------------------------
# wikilink (L312)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("markup", "display", "links", "embeds"),
    [
        ("[[note]]", "note", ("note",), ()),
        ("[[note|alias]]", "alias", ("note",), ()),
        ("[[note#heading]]", "note > heading", ("note",), ()),
        ("[[note#heading|alias]]", "alias", ("note",), ()),
        ("[[#heading]]", "heading", ("self",), ()),
        ("![[image.png]]", "", (), ("image.png",)),
        ("![[note]]", "note", (), ("note",)),
    ],
)
def test_wikilink_forms_resolve_to_the_documented_display_text(
    markup: str, display: str, links: tuple[str, ...], embeds: tuple[str, ...]
) -> None:
    """仕様書 §4 T2 の表をそのまま固定する。"""
    note = rag.parse_note("self.md", f"本文 {markup} 続き\n")

    assert note.body == f"本文 {display} 続き\n"
    assert note.links == links
    assert note.embeds == embeds


def test_the_sample_vault_carries_every_wikilink_form(sample_vault_copy: Path) -> None:
    """``weekly-review.md`` が 5 記法をすべて含み、すべて展開される。"""
    note = parsed_sample_notes(sample_vault_copy)["notes/weekly-review.md"]

    assert "[[" not in note.body
    assert "project-alpha の設計節を書いた" in note.body
    assert "Alpha の設計メモ" in note.body
    assert "project-alpha > データモデル" in note.body
    assert "制約の一覧" in note.body
    assert "今週やったこと の続き" in note.body
    assert note.links == ("project-alpha", "weekly-review")


def test_attachment_embeds_are_removed_from_the_body(sample_vault_copy: Path) -> None:
    """``![[pixel.png]]`` は本文から消え、``embeds`` にだけ残る。"""
    note = parsed_sample_notes(sample_vault_copy)["notes/tags-and-links.md"]

    assert note.embeds == ("pixel.png",)
    assert "pixel.png" not in note.body
    assert "![[" not in note.body


def test_repeated_links_are_deduplicated_in_order() -> None:
    note = rag.parse_note("self.md", "[[b]] [[a]] [[b|別名]] [[a#見出し]]\n")

    assert note.links == ("b", "a")
    assert note.body == "b a 別名 a > 見出し\n"


# --------------------------------------------------------------------------
# frontmatter (L313 / D-31)
# --------------------------------------------------------------------------


def test_frontmatter_keeps_metadata_out_of_the_body(sample_vault_copy: Path) -> None:
    """L313: ``frontmatter-rich.md`` のメタ情報が本文に漏れない。"""
    note = parsed_sample_notes(sample_vault_copy)["notes/frontmatter-rich.md"]

    assert note.title == "設計テンプレート"
    assert note.frontmatter["tags"] == ("設計", "テンプレート")
    assert note.frontmatter["aliases"] == ("テンプレ", "design-template")
    assert note.frontmatter["created"] == "2026-08-23"
    assert note.frontmatter["description"] == "frontmatter の各記法をまとめた見本"
    assert "title:" not in note.body
    assert "tags:" not in note.body
    assert "---" not in note.body
    assert note.body.lstrip().startswith("# 設計テンプレート")
    assert note.tags == ("設計", "テンプレート")


def test_unparsable_frontmatter_never_leaks_into_the_body(
    sample_vault_copy: Path,
) -> None:
    """D-31 guard: frontmatter と本文の境界が両方向に厳密であること。

    1. 閉じのある frontmatter に**解釈できない行**があっても例外にせず、
       その行は ``frontmatter_raw`` に残って**本文には現れない**。
    2. 閉じの無い frontmatter (``frontmatter-broken.md``) は frontmatter として
       扱わず、**ファイル全体を本文**にする。閉じの判定を緩めると、水平線や
       書きかけのノートの冒頭が黙って本文から消える。
    """
    weird = rag.parse_note(
        "weird.md",
        "---\n"
        "title: 変な frontmatter\n"
        "nested:\n"
        "  child: 値\n"
        "- 宙に浮いた項目\n"
        "?? 解釈できない行\n"
        "---\n"
        "\n"
        "# 本文の見出し\n",
    )

    assert weird.title == "変な frontmatter"
    assert "?? 解釈できない行" in weird.frontmatter_raw
    assert "child: 値" in weird.frontmatter_raw
    assert "?? 解釈できない行" not in weird.body
    assert "child: 値" not in weird.body
    assert "宙に浮いた項目" not in weird.body
    assert weird.body.strip() == "# 本文の見出し"

    broken = parsed_sample_notes(sample_vault_copy)["notes/frontmatter-broken.md"]

    assert broken.frontmatter_raw == ""
    assert broken.frontmatter == {}
    assert broken.title == "frontmatter-broken"
    assert broken.body.startswith("---\ntitle: 閉じ忘れ")
    assert "閉じ忘れのノート" in broken.body


def test_a_leading_horizontal_rule_is_not_frontmatter(sample_vault_copy: Path) -> None:
    """1 行目が三本線ちょうどでなければ frontmatter ではない (厳密一致)。"""
    note = parsed_sample_notes(sample_vault_copy)["notes/horizontal-rule.md"]

    assert note.frontmatter_raw == ""
    assert note.body.startswith("----")
    assert "\n---\n" in note.body


def test_frontmatter_scalars_lists_and_quotes() -> None:
    note = rag.parse_note(
        "x.md",
        "---\n"
        "title: 'クォート付き'\n"
        'inline: [a, "b", c]\n'
        "block:\n"
        "  - 1つ目\n"
        "  - 2つ目\n"
        "empty_list: []\n"
        "number: 42\n"
        "---\n"
        "本文\n",
    )

    assert note.frontmatter["title"] == "クォート付き"
    assert note.frontmatter["inline"] == ("a", "b", "c")
    assert note.frontmatter["block"] == ("1つ目", "2つ目")
    assert note.frontmatter["empty_list"] == ()
    assert note.frontmatter["number"] == "42"
    assert note.body == "本文\n"


def test_title_falls_back_to_the_file_stem(sample_vault_copy: Path) -> None:
    notes = parsed_sample_notes(sample_vault_copy)

    assert notes["notes/no-heading.md"].title == "no-heading"
    assert notes["notes/日本語 ファイル名.md"].title == "日本語 ファイル名"
    assert notes["notes/project-alpha.md"].title == "プロジェクトAlpha"


# --------------------------------------------------------------------------
# 見出しとコードフェンス (D-34)
# --------------------------------------------------------------------------


def test_wikilinks_inside_code_fences_are_resolved_but_headings_are_not(
    sample_vault_copy: Path,
) -> None:
    """D-34 guard: フェンス内で wikilink は解決し、見出しは検出しない。

    非対称は意図的であり、**両方向**を固定する。片方だけを検査すると、
    「フェンス内も一律に見出しにする」変更と「フェンス内は wikilink も
    触らない」変更のどちらかが無警告で通る。
    """
    note = parsed_sample_notes(sample_vault_copy)["notes/code-fence.md"]
    headings = [heading.text for heading in rag.iter_headings(note.body)]

    # 見出し: フェンスの外側だけ。
    assert headings == ["コードフェンス", "補足"]
    assert "# 見出しに見える行" in note.body, "フェンス内の行が本文から消えている"
    assert "# これも見出しではない" in note.body
    # wikilink: フェンスの内側でも解決される。
    assert "[[project-alpha|Alpha]]" not in note.body
    assert "Alpha のパスを表示する" in note.body
    assert note.links == ("project-alpha",)


def test_headings_track_both_fence_markers() -> None:
    body = (
        "# 見出し1\n"
        "```\n"
        "# フェンス内(バッククォート)\n"
        "```\n"
        "## 見出し2\n"
        "~~~text\n"
        "### フェンス内(チルダ)\n"
        "~~~\n"
        "###### 見出し6\n"
        "#見出しではない(空白なし)\n"
        "####### 7個は見出しではない\n"
    )

    headings = rag.iter_headings(body)

    assert [heading.text for heading in headings] == ["見出し1", "見出し2", "見出し6"]
    assert [heading.level for heading in headings] == [1, 2, 6]
    assert [heading.line_index for heading in headings] == [0, 4, 8]


def test_sample_note_heading_hierarchy(sample_vault_copy: Path) -> None:
    """``project-alpha.md`` が H1>H2>H3 の 3 階層を持つ (T3 の分割経路の入力)。"""
    note = parsed_sample_notes(sample_vault_copy)["notes/project-alpha.md"]
    headings = rag.iter_headings(note.body)

    levels = {heading.text: heading.level for heading in headings}
    assert levels["プロジェクトAlpha"] == 1
    assert levels["設計"] == 2
    assert levels["データモデル"] == 3
    assert note.title == "プロジェクトAlpha"


# --------------------------------------------------------------------------
# タグ
# --------------------------------------------------------------------------


def test_inline_tags_exclude_url_fragments_and_headings(
    sample_vault_copy: Path,
) -> None:
    """URL の ``#fragment`` と見出しの ``# `` をタグにしない。"""
    note = parsed_sample_notes(sample_vault_copy)["notes/tags-and-links.md"]

    assert note.tags == ("レシピ", "料理/和食")
    assert "installation" not in note.tags
    assert "タグとリンク" not in note.tags


def test_tags_are_not_removed_from_the_body(sample_vault_copy: Path) -> None:
    """タグは抽出するだけで本文からは消さない (本文の意味が変わるため)。"""
    note = parsed_sample_notes(sample_vault_copy)["notes/tags-and-links.md"]

    assert "#レシピ" in note.body
    assert "#料理/和食" in note.body


def test_frontmatter_tags_and_inline_tags_are_merged() -> None:
    note = rag.parse_note(
        "x.md",
        "---\ntags: [設計, '#共有']\n---\n本文 #設計 と #新しいタグ\n",
    )

    assert note.tags == ("設計", "共有", "新しいタグ")


def test_tags_never_come_from_a_link_target() -> None:
    """``[[note#heading]]`` の ``#`` を展開後にタグとして拾わない。"""
    note = rag.parse_note("self.md", "[[note#見出し]]\n")

    assert note.tags == ()
    assert note.body == "note > 見出し\n"
