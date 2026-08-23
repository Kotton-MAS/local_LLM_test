"""追跡対象ファイルに実 vault の痕跡が入らないことを機械的に守る。

このリポジトリは PUBLIC で、利用者の実 vault は個人ノートである。
``vaults/sample.toml`` は「実 vault を索引するときは ``vaults/local.toml``
を作る」と案内しており、その ``[vault] dir`` には実 vault の絶対パス
(利用者名を含む) が入る。

仕様書 §7 リスク3 の緩和策は「コミット前に ``git status`` で確認する」と
いう**人手の手順**だったが、キットの SubagentStop 自動コミットはその手順を
飛び越えて追加する。しかも自動コミットの機密パターン (``.env`` / ``*.key``
等) は ``local.toml`` に一致しない。

人手の手順で守る代わりに、``.gitignore`` をホワイトリスト方式にしたうえで
「その方式が実際に効いているか」をここで固定する。公開は不可逆なので、
守り方を散文ではなくテストに置く。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rag import load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent

# 実 vault を指す設定ファイルとして現実的に作られる名前。
# vaults/ 配下は sample 以外を追跡しないので、いずれも無視されるはず。
UNTRACKED_VAULT_PATHS: tuple[str, ...] = (
    "vaults/local.toml",
    "vaults/local2.toml",
    "vaults/my-vault.toml",
    "vaults/private/notes/secret.md",
    "vaults/sample-backup/notes/x.md",
)

# 合成 vault は公開可能なので追跡し続けなければならない。
# ここが無視されると受け入れ検証の入力が CI から消える。
TRACKED_VAULT_PATHS: tuple[str, ...] = (
    "vaults/sample.toml",
    "vaults/sample/notes/empty.md",
    "vaults/sample/.obsidian/app.json",
    "vaults/sample/attachments/pixel.png",
)


def _is_ignored(relpath: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relpath],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.parametrize("relpath", UNTRACKED_VAULT_PATHS, ids=UNTRACKED_VAULT_PATHS)
def test_a_real_vault_config_is_never_tracked(relpath: str) -> None:
    """実 vault を指しうるパスが .gitignore で無視されること。

    列挙方式 (無視するものを並べる) だと ``local2.toml`` のような派生を
    取りこぼす。ホワイトリスト方式なら取りこぼしは「無視されすぎる」方向に
    出るので、公開事故にはならない。
    """
    assert _is_ignored(relpath), (
        f"{relpath} が .gitignore で無視されていない。実 vault の絶対パス "
        f"(利用者名を含む) が PUBLIC リポジトリへ自動コミットされ得る"
    )


@pytest.mark.parametrize("relpath", TRACKED_VAULT_PATHS, ids=TRACKED_VAULT_PATHS)
def test_the_sample_vault_stays_tracked(relpath: str) -> None:
    """ホワイトリストが厳しすぎて合成 vault まで落としていないこと。

    無視する側だけを検査すると「vaults/ を丸ごと無視する」で通ってしまい、
    受け入れ検証の入力が CI から消えたことに気づけない。
    """
    assert not _is_ignored(relpath), (
        f"{relpath} が無視されている。合成 vault は受け入れ検証の入力であり "
        f"追跡対象でなければならない"
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relpath],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert tracked.returncode == 0, f"{relpath} が git の追跡対象に無い"


def _tracked_and_staged_files() -> list[str]:
    """追跡済み + 「未追跡だが無視もされていない」ファイルを返す。

    ``git ls-files`` だけだと blob が作られた後しか検出できない。キットの
    自動コミットは SubagentStop で、テストを回す Stop フックより**先**に
    走るため、実 vault のパスを含むファイルはまず 1 回コミットされてから
    検出されることになる。その時点で ``git rm`` では消えず、履歴の書き換えが
    必要になる。未追跡ファイルも見ることで、blob が作られる前の窓で気づける。
    """
    names: list[str] = []
    for args in (
        ["git", "ls-files", "-z"],
        ["git", "ls-files", "-z", "-o", "--exclude-standard"],
    ):
        out = subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
        names.extend(n for n in out.split("\0") if n)
    return names


def _real_home_identifiers() -> tuple[str, ...]:
    """この環境の「本物の利用者」を指す文字列を集める。

    許可リスト方式 (``testuser`` / ``someone`` を許す) は採らない。合成名は
    増え続け、許可リストを育てるうちに本物を1つ通してしまう。代わりに
    **いま実行している利用者の識別子だけ**を検出対象にする。コミットしようと
    しているのはその人なので、狙いと検出範囲が一致する。

    副次的な利点として、テスト内の ``/home/testuser`` のような意図的な
    合成名に反応しない (実際 round-10 の時点で 3 ファイル 5 箇所ある)。
    """
    home = Path.home()
    # **パス成分として**照合する。素の利用者名で探すと、短い一般語の
    # 利用者名 (CI の runner など) が harness/runner.py への言及に誤反応する
    # (実際に CI で 7 件の偽陽性を出した)。利用者名の露出はパスの形でしか
    # 起きないので、パスの形だけを見れば十分。
    candidates = {str(home), f"/home/{home.name}", f"/Users/{home.name}"}
    return tuple(sorted(c for c in candidates if len(c) >= len("/home/") + 2))


def test_no_tracked_file_exposes_the_real_user_identity() -> None:
    """追跡対象ファイルに、この環境の利用者名・ホームパスが現れないこと。

    ``.gitignore`` はファイル単位でしか守れない。コード・テスト・**ドキュメント**
    に実パスを書いてしまう経路が別に残るため、内容側も検査する。

    以前はここで ``docs/`` を丸ごと除外していた。規則の説明文が実パスに言及
    するためだったが、それは穴だった。実際に仕様書の 2 箇所で規則の文言その
    ものが利用者名を公開しており、``origin/main`` には無い = 次の PR で新規に
    公開される状態だった (round-10 で発見)。``docs/`` は planner / doc-writer が
    自動生成する場所で、エラー出力を貼る形で実パスが入りやすく、除外は広がる
    方向にしか動かない。ディレクトリ単位の除外はやめた。

    規則を説明したいときは ``/home/<user>/...`` のようなプレースホルダで書く。
    """
    identifiers = _real_home_identifiers()
    if not identifiers:
        pytest.skip("利用者識別子を決められない環境")

    offenders: list[str] = []
    for name in _tracked_and_staged_files():
        path = REPO_ROOT / name
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # バイナリ・読めないものは対象外
        for lineno, line in enumerate(text.splitlines(), start=1):
            for identifier in identifiers:
                if identifier in line:
                    offenders.append(f"{name}:{lineno}")
                    break
    assert not offenders, (
        "追跡対象ファイルに実行環境の利用者名かホームパスが含まれている。"
        "PUBLIC リポジトリなので公開は不可逆。規則を説明したい場合は "
        f"/home/<user>/... のようなプレースホルダで書くこと: {offenders[:5]}"
    )


def test_no_symlink_is_tracked() -> None:
    """追跡対象にシンボリックリンクが 1 件も無いこと。

    git は symlink を mode 120000 の blob として保存し、**その中身はリンク先の
    パス文字列そのもの**である。``vaults/sample/`` はホワイトリストで再包含
    されているため、そこに実 vault へのリンクを 1 本置くだけで絶対パスが
    blob として記録される。しかも内容検査は ``read_text`` がディレクトリへの
    リンクで ``IsADirectoryError`` になり、握り潰されて見えない (round-10)。

    現在 0 件なので、増えたときだけ落ちる。
    """
    out = subprocess.run(
        ["git", "ls-files", "-s"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    links = [
        ln.split("\t", 1)[-1] for ln in out.splitlines() if ln.startswith("120000")
    ]
    assert not links, (
        "追跡対象にシンボリックリンクがある。リンク先の絶対パスが blob として "
        f"PUBLIC リポジトリに保存される: {links}"
    )


def test_the_sample_vault_config_points_at_the_synthetic_vault() -> None:
    """``vaults/sample.toml`` の vault が合成 vault を指し続けること。

    ``sample.toml`` はホワイトリストで**追跡対象のまま**なので、案内どおり
    ``local.toml`` を作らずにこのファイルの ``dir`` を実 vault へ書き換える
    のが現実的な経路になる (round-10)。その場合、利用者名を含まない
    ``~/ドキュメント/...`` 形式でも vault のフォルダ構成が公開される。

    合成 vault の設定は変わらないので、値を固定しても偽陽性が出ない。
    """
    settings = load_settings(REPO_ROOT / "vaults" / "sample.toml")
    assert settings.vault_dir == (REPO_ROOT / "vaults" / "sample").resolve(), (
        f"vaults/sample.toml が合成 vault 以外を指している: {settings.vault_dir}"
    )
