"""`.claude/decisions.yaml` の本文に書いたテスト参照が実在することを守る。

## なぜこのファイルが要るか

`check_decisions.py` が実在を検証するのは `guard_test` フィールドだけで、
決定ごとに 1 本しか持てない。ところが D-30 のように「構造・静的・動的・設定」
の 4 層で規則を強制する決定では、1 本では規則の全体を守れない。

round-9 の指摘 (F-9-008) はまさにそれで、対応として **rule 本文に「この層は
どのテストが守っているか」を列挙した**。読み手には有用だが、列挙されたテスト名は
`check_decisions.py` も Stop フックも見ないため、改名・削除しても何も落ちない。
実測で、本文に現れるテスト参照 36 件のうち実走対象は guard_test の 30 件だけ
だった。

CLAUDE.md は「再発している事象への対策をプロンプト層に置かない。2 回目以降は
hook / スキーマ / テストへ層を下げる」と定めている。F-9-008 は「層が guard_test
に載っていない」事象の 1 回目で、その対策を散文に置いたのが 2 回目にあたる。

そこで散文を消すのではなく、**散文を機械検証の入力に変える**。決定本文に
テスト名を書くことがそのまま拘束になるので、列挙を増やすほど守りが厚くなる。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DECISIONS = REPO_ROOT / ".claude" / "decisions.yaml"

# 決定の rule / rationale に書かれる pytest の node id。
_TEST_REFERENCE = re.compile(r"tests/[A-Za-z0-9_/]+\.py::test_[A-Za-z0-9_]+")


def _referenced_node_ids() -> tuple[str, ...]:
    text = DECISIONS.read_text(encoding="utf-8")
    return tuple(sorted(set(_TEST_REFERENCE.findall(text))))


def test_the_decisions_file_actually_references_tests() -> None:
    """参照が 0 件なら、この検査自体が空回りしている。

    正規表現の取りこぼしや decisions.yaml の書式変更で参照を 1 件も拾えなく
    なると、以下の検査は「違反 0 件」で緑になる。何も守っていない状態が
    緑で通るのを防ぐ。
    """
    references = _referenced_node_ids()
    assert len(references) >= 10, (
        f"decisions.yaml から拾えたテスト参照が {len(references)} 件しかない。"
        "正規表現が書式に追随できていない可能性がある"
    )


@pytest.mark.parametrize(
    "node_id", _referenced_node_ids(), ids=lambda n: n.split("::")[-1]
)
def test_each_referenced_test_names_an_existing_file(node_id: str) -> None:
    """参照先のテストが実在すること。

    ``guard_test`` に載らない層 (D-30 の静的検査・設定検査など) は、この
    検査だけが改名・削除から守っている。1 件でも壊れれば、その層はもう誰も
    守っていない。

    pytest をサブプロセスで起動して収集を試す案は採らない。テストの中で
    テストランナーを起動する形は環境変数の受け渡しで壊れやすく (実際
    ``uv`` を見失って落ちた)、得られる保証は「ファイルに ``def <名前>(``
    がある」とほぼ同じだった。
    """
    relpath, _, test_name = node_id.partition("::")
    path = REPO_ROOT / relpath
    assert path.is_file(), f"{node_id} の参照先 {relpath} が存在しない"
    assert f"def {test_name}(" in path.read_text(encoding="utf-8"), (
        f"{relpath} に {test_name} の定義が見つからない"
    )
