"""ハーネス (L3) のレイアウトと層の境界の固定。

このファイルが固定する性質は 3 つある。

1. **``harness`` は ``llmkit`` のサブモジュールを直接 import しない。**
   L3 が使ってよいのは ``llmkit`` が再エクスポートする公開シンボルだけで、
   ``from llmkit.client import _HttpChatClient`` のような内部参照を許すと
   層の境界が散文の主張に戻る (仕様書 §8)。
2. **``harness.__all__`` はサブモジュールの ``__all__`` の和集合と一致する。**
   ``llmkit/__init__.py`` と同じ方針 (F-1-002)。``cli`` は L4 の入口であって
   公開 API ではないため和集合から除く (仕様書 §9 決定41)。
3. **``llmkit`` は ``harness`` を import しない。**
   逆向きの参照が 1 つでも生まれると L2 が L3 に依存し、``llmkit`` 単体で
   完結しているという前提 (D-06 / test_llmkit_importable_from_repo_root) が崩れる。

いずれも AST で見る (文字列一致だと docstring の説明文を違反と誤検出する)。
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import harness
import harness.cli

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "harness"
PACKAGE_DIR = REPO_ROOT / "llmkit"

#: L3 を構成するサブモジュール。``cli`` は公開 API ではないため含めない
#: (``llmkit`` の ``_SUBMODULE_NAMES`` が ``cli`` を含まないのと同じ扱い)。
_SUBMODULE_NAMES = (
    "gpu",
    "records",
    "report",
    "runner",
    "suite",
)


def imported_module_names(module_path: Path) -> list[str]:
    """そのファイルが import しているモジュール名を列挙する。"""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


# --------------------------------------------------------------------------
# (i) harness -> llmkit は公開 API 経由だけ
# --------------------------------------------------------------------------


def test_no_harness_module_imports_an_llmkit_submodule() -> None:
    """``harness/*.py` のどれも ``llmkit.<submodule>`` を import しない。"""
    module_paths = sorted(HARNESS_DIR.glob("*.py"))
    assert len(module_paths) >= len(_SUBMODULE_NAMES) + 1, module_paths

    offenders: list[str] = []
    for module_path in module_paths:
        offenders += [
            f"{module_path.name}: {name}"
            for name in imported_module_names(module_path)
            if name.startswith("llmkit.")
        ]

    assert not offenders, f"llmkit サブモジュールの直接 import: {offenders}"


def test_the_harness_actually_uses_the_llmkit_public_api() -> None:
    """上の検査が「そもそも llmkit を使っていない」で通っていないことの確認。"""
    users = [
        module_path.name
        for module_path in sorted(HARNESS_DIR.glob("*.py"))
        if "llmkit" in imported_module_names(module_path)
    ]

    assert len(users) >= 2, users


# --------------------------------------------------------------------------
# (ii) harness.__all__ = サブモジュールの __all__ の和集合 (cli を除く)
# --------------------------------------------------------------------------


def test_harness_all_matches_the_union_of_submodule_all() -> None:
    """公開シンボルの追加漏れ (再エクスポート忘れ) を機械的に検出する。"""
    union: set[str] = set()
    for name in _SUBMODULE_NAMES:
        module = importlib.import_module(f"harness.{name}")
        module_all = getattr(module, "__all__", None)
        assert module_all is not None, f"harness.{name} に __all__ がありません"
        union.update(module_all)

    assert set(harness.__all__) == union
    assert len(harness.__all__) == len(set(harness.__all__)), "重複エクスポート"
    for name in harness.__all__:
        assert hasattr(harness, name), name


def test_the_cli_is_deliberately_not_part_of_the_public_api() -> None:
    """§9 決定41: ``cli`` は再エクスポートしない (L4 の入口であるため)。"""
    assert set(harness.cli.__all__), "harness.cli に __all__ がありません"
    assert set(harness.cli.__all__).isdisjoint(harness.__all__)
    assert "cli" not in _SUBMODULE_NAMES


def test_every_harness_submodule_is_covered_by_the_union_check() -> None:
    """サブモジュールを足したのに ``_SUBMODULE_NAMES`` へ入れ忘れると落ちる。"""
    discovered = {
        module_path.stem
        for module_path in HARNESS_DIR.glob("*.py")
        if module_path.stem != "__init__"
    }

    assert discovered == {*_SUBMODULE_NAMES, "cli"}


# --------------------------------------------------------------------------
# (iii) llmkit -> harness は 0 件
# --------------------------------------------------------------------------


def test_llmkit_never_imports_the_harness() -> None:
    """L2 が L3 に依存していないこと (依存の向きは一方向)。"""
    offenders: list[str] = []
    for module_path in sorted(PACKAGE_DIR.glob("*.py")):
        offenders += [
            f"{module_path.name}: {name}"
            for name in imported_module_names(module_path)
            if name == "harness" or name.startswith("harness.")
        ]

    assert not offenders, f"llmkit から harness への import: {offenders}"
