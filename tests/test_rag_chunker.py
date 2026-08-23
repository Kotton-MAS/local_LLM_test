"""チャンク分割 (``rag/chunker.py``) の検証。

要件書 L314 (見出し階層がチャンクへ引き継がれる) の直接検証と、有効性観点の
2 軸 (E24 / E25) を含む。

- ``test_token_estimate_is_deterministic_and_configurable`` … D-32 guard / E25
- ``test_character_boundary_fallback_shares_the_token_formula`` … D-32 guard
  (文字境界フォールバックが独自の式を持たないこと)
- ``test_max_tokens_changes_the_chunk_boundaries`` … E24

**最重要の不変条件**は
``test_chunk_bodies_reconstruct_every_section_without_losing_a_character``
であり、合成 vault の全ノートに対して回す。分割の経路 (段落 → 行 → 文字) の
どれを通っても本文が 1 文字も消えないことを、チャンカの実装とは独立に
組み立てた期待値と突き合わせて確かめる。ここが唯一、**無音のテキスト欠落**を
検出できる防壁である。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from conftest import write_rag_settings

import rag
import rag.chunker as chunker_module
from llmkit import ConfigError

# --------------------------------------------------------------------------
# 合成 vault を読むための小道具 (tests/test_rag_parser.py と同じ組み立て)
# --------------------------------------------------------------------------


def sample_settings(directory: Path) -> rag.RagSettings:
    return rag.load_settings(write_rag_settings(directory.parent))


def parsed_sample_notes(directory: Path) -> dict[str, rag.ParsedNote]:
    settings = sample_settings(directory)
    return {
        entry.relpath: rag.parse_note(
            entry.relpath, rag.read_note_text(settings, entry.relpath)
        )
        for entry in rag.iter_vault_files(settings)
    }


def chunked_sample_vault(
    directory: Path, chunk_settings: rag.ChunkSettings | None = None
) -> dict[str, tuple[rag.Chunk, ...]]:
    settings = sample_settings(directory)
    effective = settings.chunk if chunk_settings is None else chunk_settings
    return {
        relpath: rag.chunk_note(note, effective)
        for relpath, note in parsed_sample_notes(directory).items()
    }


def squeeze(text: str) -> str:
    """空白 (改行を含む) をすべて落とした正規形。"""
    return "".join(text.split())


def body_without_heading_lines(note: rag.ParsedNote) -> str:
    """チャンカを使わずに「セクション本文の総和」を組み立てる。

    見出し行はチャンクの ``body`` ではなく ``heading_path`` に入るため、
    期待値からも取り除く。``rag.chunker`` の内部は一切使わない。
    """
    heading_lines = {heading.line_index for heading in rag.iter_headings(note.body)}
    return "\n".join(
        line
        for line_index, line in enumerate(note.body.splitlines())
        if line_index not in heading_lines
    )


def estimate(text: str, chunk: rag.ChunkSettings) -> int:
    return rag.estimate_tokens(
        text,
        cjk_chars_per_token=chunk.cjk_chars_per_token,
        ascii_chars_per_token=chunk.ascii_chars_per_token,
    )


def tokens(text: str, *, cjk: float = 1.0, other: float = 4.0) -> int:
    """既定係数での近似トークン数 (係数を掃引するテスト用の短縮形)。"""
    return rag.estimate_tokens(
        text, cjk_chars_per_token=cjk, ascii_chars_per_token=other
    )


# --------------------------------------------------------------------------
# 上限 (規則4) — 判定の対象は embed_text
# --------------------------------------------------------------------------


def test_every_chunk_of_the_sample_vault_stays_within_the_limit(
    sample_vault_copy: Path,
) -> None:
    """全チャンクで ``estimate_tokens(embed_text) <= max_tokens``。

    上限を ``body`` に対して適用する実装に変えると、見出し経路の分だけ超過した
    チャンクが出てここが落ちる (仕様書 §5 変異検証4)。
    """
    chunk_settings = rag.ChunkSettings()
    chunked = chunked_sample_vault(sample_vault_copy, chunk_settings)

    offenders = [
        (chunk.chunk_id, chunk.estimated_tokens)
        for chunks in chunked.values()
        for chunk in chunks
        if estimate(chunk.embed_text, chunk_settings) > chunk_settings.max_tokens
    ]

    assert not offenders, f"上限を超えたチャンク: {offenders}"
    # 検査が「そもそも分割していない」で通っていないことの確認。
    assert any(
        chunk.part_index > 0 for chunks in chunked.values() for chunk in chunks
    ), "上限超過で分割されたセクションが 1 つも無い"


def test_estimated_tokens_is_the_number_used_for_the_limit(
    sample_vault_copy: Path,
) -> None:
    """``Chunk.estimated_tokens`` は ``embed_text`` の近似値である。"""
    chunk_settings = rag.ChunkSettings()
    chunked = chunked_sample_vault(sample_vault_copy, chunk_settings)

    for chunks in chunked.values():
        for chunk in chunks:
            assert chunk.estimated_tokens == estimate(chunk.embed_text, chunk_settings)
            assert chunk.estimated_tokens >= estimate(chunk.body, chunk_settings)


# --------------------------------------------------------------------------
# ★ 最重要の不変条件: 本文が 1 文字も失われない (規則3)
# --------------------------------------------------------------------------


def test_chunk_bodies_reconstruct_every_section_without_losing_a_character(
    sample_vault_copy: Path,
) -> None:
    """1 ノートのチャンクの ``body`` を順に連結すると元の本文に一致する。

    比較は空白正規化のうえで行う (段落境界の空行や行末の改行はチャンクの端で
    落ちるため)。**無音のテキスト欠落を検出する唯一の防壁**なので、合成 vault の
    全ノートに対して回す。
    """
    notes = parsed_sample_notes(sample_vault_copy)
    chunk_settings = rag.ChunkSettings()

    mismatches: list[str] = []
    for relpath, note in notes.items():
        chunks = rag.chunk_note(note, chunk_settings)
        joined = squeeze("".join(chunk.body for chunk in chunks))
        if joined != squeeze(body_without_heading_lines(note)):
            mismatches.append(relpath)

    assert not mismatches, f"本文が復元できないノート: {mismatches}"


#: 合成 vault の最も深い見出し経路は近似 20 トークンあり、それを下回る上限は
#: 設定として成立しない (``ConfigError``。下の
#: ``test_a_heading_path_that_fills_the_whole_budget_is_a_config_error`` が固定)。
_LIMIT_SWEEP = (240, 120, 80, 40, 30)


@pytest.mark.parametrize("max_tokens", _LIMIT_SWEEP)
def test_no_character_is_lost_at_any_limit(
    sample_vault_copy: Path, max_tokens: int
) -> None:
    """上限を変えて分割経路 (段落 / 行 / 文字) を掃引しても本文は保たれる。"""
    notes = parsed_sample_notes(sample_vault_copy)
    chunk_settings = rag.ChunkSettings(max_tokens=max_tokens)

    for relpath, note in notes.items():
        chunks = rag.chunk_note(note, chunk_settings)
        joined = squeeze("".join(chunk.body for chunk in chunks))
        assert joined == squeeze(body_without_heading_lines(note)), relpath


def test_an_indivisible_unit_longer_than_the_limit_is_cut_at_character_boundaries() -> (
    None
):
    """空行も改行も無い 1 行が上限を超える場合は**文字境界**で切る (規則3)。

    ここで諦めて 1 チャンクにすると上限が破れ、切り捨てると本文が消える。
    どちらも例外を出さずに索引だけが壊れるため、テストで固定する。
    """
    long_line = "あ" * 600
    note = rag.parse_note("notes/long-line.md", f"# 長い行\n\n{long_line}\n")
    chunk_settings = rag.ChunkSettings(max_tokens=240)

    chunks = rag.chunk_note(note, chunk_settings)

    assert len(chunks) > 1, "上限を超える 1 行が分割されていない"
    assert all(
        estimate(chunk.embed_text, chunk_settings) <= chunk_settings.max_tokens
        for chunk in chunks
    )
    assert "".join(chunk.body for chunk in chunks) == long_line
    assert [chunk.part_index for chunk in chunks] == list(range(len(chunks)))


def test_paragraph_boundaries_are_preferred_over_line_boundaries() -> None:
    """段落ごとに収まるなら、段落の途中では切らない (規則3 の優先順位)。"""
    paragraphs = ["あ" * 100, "い" * 100, "う" * 100]
    note = rag.parse_note("notes/paras.md", "\n\n".join(paragraphs) + "\n")

    chunks = rag.chunk_note(note, rag.ChunkSettings(max_tokens=150))

    assert [chunk.body for chunk in chunks] == paragraphs


def test_line_boundaries_are_used_when_a_paragraph_does_not_fit() -> None:
    """段落単位で収まらないときは行境界で切る (文字境界へは落ちない)。"""
    lines = [f"{index}行目の内容です。" + "か" * 40 for index in range(6)]
    note = rag.parse_note("notes/lines.md", "\n".join(lines) + "\n")

    chunks = rag.chunk_note(note, rag.ChunkSettings(max_tokens=120))

    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.body.splitlines():
            assert line in lines, f"行の途中で切れています: {line!r}"


# --------------------------------------------------------------------------
# 見出し経路 (規則2) — L314
# --------------------------------------------------------------------------


def test_the_deep_section_of_project_alpha_carries_its_heading_path(
    sample_vault_copy: Path,
) -> None:
    """L314: H1>H2>H3 の階層が ``heading_path`` としてチャンクに載る。"""
    chunk_settings = rag.ChunkSettings()
    chunks = chunked_sample_vault(sample_vault_copy, chunk_settings)[
        "notes/project-alpha.md"
    ]

    deep = [
        chunk
        for chunk in chunks
        if chunk.heading_path == ("プロジェクトAlpha", "設計", "データモデル")
    ]

    assert deep, [chunk.heading_path for chunk in chunks]
    prefix = " > ".join(("プロジェクトAlpha", "設計", "データモデル"))
    for chunk in deep:
        assert chunk.embed_text.startswith(prefix + "\n\n")
        assert chunk.embed_text.endswith(chunk.body)
        assert prefix not in chunk.body
        assert "データモデル >" not in chunk.body


def test_the_note_title_is_never_repeated_in_the_heading_path(
    sample_vault_copy: Path,
) -> None:
    """H1 がノートタイトルと同じ場合、``heading_path`` で重複させない。

    Obsidian では H1 をタイトルと同じ文字列にする書き方が一般的で、そのまま
    並べると ``プロジェクトAlpha > プロジェクトAlpha > 設計`` になる。情報が
    増えないのに接頭辞が伸び、本文に使える上限がその分だけ削られる。
    """
    chunks = chunked_sample_vault(sample_vault_copy)["notes/project-alpha.md"]

    for chunk in chunks:
        assert chunk.heading_path[0] == "プロジェクトAlpha"
        pairs = zip(chunk.heading_path, chunk.heading_path[1:], strict=False)
        assert all(left != right for left, right in pairs), chunk.heading_path
    assert max(len(chunk.heading_path) for chunk in chunks) == 3


def test_a_repeated_heading_further_down_the_path_is_kept() -> None:
    """離れた位置の同名見出しは畳まない (文脈が違うため)。"""
    text = "# 設計\n\n概要です。\n\n## 概要\n\n本文。\n\n### 設計\n\n詳細。\n"
    note = rag.parse_note("notes/repeat.md", text)

    paths = [chunk.heading_path for chunk in rag.chunk_note(note, rag.ChunkSettings())]

    assert ("repeat", "設計", "概要", "設計") in paths


def test_a_note_without_headings_uses_only_the_title(
    sample_vault_copy: Path,
) -> None:
    """見出しが 1 つも無いノートの ``heading_path`` は長さ 1。"""
    chunks = chunked_sample_vault(sample_vault_copy)["notes/no-heading.md"]

    assert chunks
    assert all(chunk.heading_path == ("no-heading",) for chunk in chunks)


def test_content_before_the_first_heading_becomes_its_own_chunk() -> None:
    """最初の見出しより前の本文にも文脈が付く (捨てない、規則2)。"""
    note = rag.parse_note("notes/lead.md", "前書きの段落です。\n\n# 見出し\n\n本文。\n")

    chunks = rag.chunk_note(note, rag.ChunkSettings())

    assert chunks[0].heading_path == ("lead",)
    assert chunks[0].body == "前書きの段落です。"
    assert chunks[1].heading_path == ("lead", "見出し")


def test_headings_inside_code_fences_do_not_split_a_chunk(
    sample_vault_copy: Path,
) -> None:
    """D-34: フェンス内の ``# 行`` はセクション境界にならない。"""
    chunks = chunked_sample_vault(sample_vault_copy)["notes/code-fence.md"]

    paths = [chunk.heading_path for chunk in chunks]
    fenced = [chunk for chunk in chunks if "# 見出しに見える行" in chunk.body]

    assert ("code-fence", "コードフェンス", "見出しに見える行") not in paths
    assert len(fenced) == 1, "フェンスが途中で分断されています"
    assert "```bash" in fenced[0].body and "```" in fenced[0].body


# --------------------------------------------------------------------------
# 空チャンクを作らない (規則5)
# --------------------------------------------------------------------------


def test_empty_and_whitespace_only_notes_produce_no_chunks(
    sample_vault_copy: Path,
) -> None:
    """空ノート・空白のみのノートは 0 チャンク。"""
    chunked = chunked_sample_vault(sample_vault_copy)

    assert chunked["notes/empty.md"] == ()
    assert chunked["notes/whitespace-only.md"] == ()


def test_no_chunk_is_empty_or_whitespace_only(sample_vault_copy: Path) -> None:
    """どのチャンクの ``body`` / ``embed_text`` も空白だけにならない。"""
    chunked = chunked_sample_vault(sample_vault_copy)

    for relpath, chunks in chunked.items():
        for chunk in chunks:
            assert chunk.body.strip(), relpath
            assert chunk.embed_text.strip(), relpath
            assert chunk.body == chunk.body.strip(), relpath


# --------------------------------------------------------------------------
# E25 / D-32 guard: トークン近似は決定論的で設定可能
# --------------------------------------------------------------------------


def test_token_estimate_is_deterministic_and_configurable(
    sample_vault_copy: Path,
) -> None:
    """D-32 guard: 外部トークナイザもランタイムも使わない近似器であること。

    (i) 同じ入力からは常に同じ値が出る (ネットワークにも時刻にも依存しない)
    (ii) CJK と非 CJK を**別々の係数**で数える
    (iii) 係数を変えると値が変わり、少なくとも 1 ノートのチャンク境界も変わる
          (E25)

    係数を無視して固定値を返す実装にすると (ii)(iii) が落ちる。
    """
    # (i) 決定論: 同じ入力からは何度呼んでも同じ値が出る
    text = "日本語のテキストと ASCII text が混ざった行。"
    assert len({tokens(text) for _ in range(5)}) == 1

    # (ii) 文字種別で数える (CJK 5 文字 = 5 / ASCII 8 文字 = 2 トークン)
    assert tokens("あいうえお") == 5
    assert tokens("abcdefgh") == 2
    assert tokens("あいうabcd") == 4
    assert tokens("") == 0

    # (iii) 係数が効く (固定値を返す実装にすると、ここが落ちる)
    assert tokens("あ" * 100) == 100
    assert tokens("あ" * 100, cjk=2.0) == 50
    assert tokens("a" * 100) == 25
    assert tokens("a" * 100, other=2.0) == 50

    # E25: 係数を変えると少なくとも 1 ノートのチャンク境界が変わる
    baseline = rag.ChunkSettings()
    coarser = dataclasses.replace(baseline, cjk_chars_per_token=2.0)
    before = chunked_sample_vault(sample_vault_copy, baseline)
    after = chunked_sample_vault(sample_vault_copy, coarser)
    changed = [
        relpath
        for relpath in before
        if [chunk.body for chunk in before[relpath]]
        != [chunk.body for chunk in after[relpath]]
    ]

    assert changed, "cjk_chars_per_token を変えてもチャンク境界が動かない"


def test_character_boundary_fallback_shares_the_token_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-32 guard: 文字境界フォールバックは近似式を独自に持たない。

    以前は ``_pack_characters`` 内のクロージャが ``estimate_tokens`` と同じ式を
    独立に再実装しており、近似式 (``_tokens_from_counts``) を差し替えても
    文字境界フォールバックの分割数だけ追随しなかった (round-10 レビュー)。
    ``_tokens_from_counts`` を差し替えると、段落/行/文字のどの経路の分割数も
    連動して変わることを確かめる。
    """
    long_line = "あ" * 600
    note = rag.parse_note("notes/long-line.md", f"# 長い行\n\n{long_line}\n")
    chunk_settings = rag.ChunkSettings(max_tokens=240)

    baseline_chunks = rag.chunk_note(note, chunk_settings)
    original_formula = chunker_module._tokens_from_counts

    def doubled(
        cjk_count: int,
        total_count: int,
        *,
        cjk_chars_per_token: float,
        ascii_chars_per_token: float,
    ) -> int:
        return 2 * original_formula(
            cjk_count,
            total_count,
            cjk_chars_per_token=cjk_chars_per_token,
            ascii_chars_per_token=ascii_chars_per_token,
        )

    monkeypatch.setattr(chunker_module, "_tokens_from_counts", doubled)
    doubled_chunks = rag.chunk_note(note, chunk_settings)

    assert len(doubled_chunks) > len(baseline_chunks), (
        len(baseline_chunks),
        len(doubled_chunks),
    )
    # 文字を 1 つも落としていないこと (規則3)。
    assert "".join(chunk.body for chunk in doubled_chunks) == long_line


def test_max_tokens_changes_the_chunk_boundaries(sample_vault_copy: Path) -> None:
    """E24: ``chunk.max_tokens`` を 240→80 にするとチャンク総数が増える。"""
    wide = rag.ChunkSettings(max_tokens=240)
    narrow = rag.ChunkSettings(max_tokens=80)

    wide_chunks = chunked_sample_vault(sample_vault_copy, wide)
    narrow_chunks = chunked_sample_vault(sample_vault_copy, narrow)
    wide_total = sum(len(chunks) for chunks in wide_chunks.values())
    narrow_total = sum(len(chunks) for chunks in narrow_chunks.values())

    assert narrow_total > wide_total, (wide_total, narrow_total)
    for settings, chunked in ((wide, wide_chunks), (narrow, narrow_chunks)):
        for chunks in chunked.values():
            for chunk in chunks:
                assert estimate(chunk.embed_text, settings) <= settings.max_tokens


# --------------------------------------------------------------------------
# chunk_id と付随メタデータ
# --------------------------------------------------------------------------


def test_chunk_ids_are_stable_and_unique_across_the_vault(
    sample_vault_copy: Path,
) -> None:
    """``chunk_id`` は再実行で完全一致し、vault 全体で一意。"""
    first = chunked_sample_vault(sample_vault_copy)
    second = chunked_sample_vault(sample_vault_copy)

    first_ids = [chunk.chunk_id for chunks in first.values() for chunk in chunks]
    second_ids = [chunk.chunk_id for chunks in second.values() for chunk in chunks]

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert "notes/project-alpha.md#0000" in first_ids
    assert "notes/日本語 ファイル名.md#0000" in first_ids


def test_ordinals_are_contiguous_within_a_note(sample_vault_copy: Path) -> None:
    """``ordinal`` はノート内で 0 から連番、``chunk_id`` はその 4 桁表現。"""
    for chunks in chunked_sample_vault(sample_vault_copy).values():
        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
        for chunk in chunks:
            assert chunk.chunk_id == f"{chunk.relpath}#{chunk.ordinal:04d}"


def test_chunks_carry_the_note_tags_and_links(sample_vault_copy: Path) -> None:
    """タグとリンクはノート単位の値をそのまま各チャンクに載せる。"""
    notes = parsed_sample_notes(sample_vault_copy)
    chunked = chunked_sample_vault(sample_vault_copy)

    for relpath, chunks in chunked.items():
        for chunk in chunks:
            assert chunk.tags == notes[relpath].tags
            assert chunk.links == notes[relpath].links
    assert chunked["notes/tags-and-links.md"][0].tags == ("レシピ", "料理/和食")


# --------------------------------------------------------------------------
# 設定の誤りは静かに壊さず ConfigError にする
# --------------------------------------------------------------------------


def test_a_heading_path_that_fills_the_whole_budget_is_a_config_error() -> None:
    """見出し経路だけで上限を使い切る設定は ``ConfigError``。

    本文を 1 文字も落とさずに上限も守ろうとすると「1 文字ずつのチャンクの山」に
    なる。索引としては壊れているので、静かに作らず落とす。メッセージには
    相対パスと数値だけを載せ、見出し文字列 (ノート本文の一部) は載せない。
    """
    note = rag.parse_note("notes/deep.md", "# とても長い見出しの文字列です\n\n本文。\n")

    with pytest.raises(ConfigError) as excinfo:
        rag.chunk_note(note, rag.ChunkSettings(max_tokens=5))

    message = str(excinfo.value)
    assert "notes/deep.md" in message
    assert "max_tokens" in message
    assert "とても長い見出し" not in message


def test_chunking_never_touches_the_filesystem(sample_vault_copy: Path) -> None:
    """チャンカは純関数である (入力は ParsedNote と ChunkSettings だけ)。"""
    note = parsed_sample_notes(sample_vault_copy)["notes/project-alpha.md"]
    settings = rag.ChunkSettings()

    before = sorted(path.name for path in (sample_vault_copy / "notes").iterdir())
    chunks = rag.chunk_note(note, settings)
    after = sorted(path.name for path in (sample_vault_copy / "notes").iterdir())

    assert before == after
    assert chunks == rag.chunk_note(note, settings)
