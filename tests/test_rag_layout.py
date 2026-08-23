"""RAG 層 (L3) のレイアウトと層の境界の固定。

``tests/test_harness_layout.py`` と同型の 3 つ (L3→llmkit は公開 API 経由だけ /
``__all__`` は和集合 / llmkit→L3 は 0 件) に加えて、``rag/`` 固有の境界を 3 つ
固定する。

4. **``rag/`` は推論ランタイムに直接触れない** (D-25 guard,
   ``test_rag_never_talks_to_the_runtime_directly``)。埋め込みの HTTP は
   ``llmkit.embeddings`` にあり、import 検査だけでは「httpx を直接使う」経路を
   塞げないため、HTTP メソッド呼び出しとエンドポイント文字列も見る。
5. **``rag`` と ``harness`` は互いを import しない**。同じ L3 の 2 本が
   依存し合うと、どちらかを単独で動かせなくなる。
6. **vault に触れるのは ``rag/vault.py`` だけ** (D-30 の構造層,
   ``test_only_the_vault_module_reads_the_vault``)。

いずれも **AST で見る**。文字列一致だと docstring の説明文 (このファイル自身も
``.post`` や ``/embeddings`` という語を含む) を違反と誤検出する (決定12・43)。
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from conftest import write_rag_settings

import harness
import rag

REPO_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = REPO_ROOT / "rag"
HARNESS_DIR = REPO_ROOT / "harness"
PACKAGE_DIR = REPO_ROOT / "llmkit"

#: L3 (rag) を構成するサブモジュール。3b で ``store`` / ``indexer`` / ``cli`` が
#: 加わる。``cli`` は L4 の入口であって公開 API ではないため、加わっても
#: ``_SUBMODULE_NAMES`` には入れない (``harness`` と同じ扱い)。
_SUBMODULE_NAMES = (
    "chunker",
    "parser",
    "settings",
    "vault",
)

#: HTTP クライアントのメソッド名。``dict.get`` もここに一致するため、``rag/`` では
#: ``Mapping.get`` を使わず ``in`` + 添字で書く (誤検出を許して検査を鈍らせるより、
#: 書き方を 1 つに固定するほうが境界が強い)。
_HTTP_METHOD_NAMES = frozenset({"get", "post", "request"})

#: ランタイムのエンドポイントを表す文字列。docstring は対象外。
_ENDPOINT_LITERALS = ("/embeddings", "/chat/completions", "/api/chat")

#: vault の中身を読む・列挙する API。``rag/vault.py`` だけが使ってよい。
_FILESYSTEM_READ_NAMES = frozenset(
    {
        "open",
        "read_text",
        "read_bytes",
        "iterdir",
        "glob",
        "rglob",
        "walk",
        "scandir",
        "listdir",
    }
)

#: 例外: ``rag/settings.py`` は**設定ファイル自身**を読む。読む対象は呼び出し元が
#: 渡した TOML のパスだけで、vault 相対パスを受け取る関数を 1 つも持たない。
_FILESYSTEM_READ_ALLOWANCES = {"settings.py": frozenset({"read_text"})}


def module_paths() -> list[Path]:
    return sorted(RAG_DIR.glob("*.py"))


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


def called_attribute_names(tree: ast.AST) -> list[str]:
    """``x.y(...)`` の ``y`` を列挙する (呼び出しのみ。属性参照は含めない)。"""
    return [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


def non_docstring_string_literals(tree: ast.AST) -> list[str]:
    """docstring を除いた文字列リテラルを列挙する。"""
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


# --------------------------------------------------------------------------
# (i) rag -> llmkit は公開 API 経由だけ
# --------------------------------------------------------------------------


def test_no_rag_module_imports_an_llmkit_submodule() -> None:
    """``rag/*.py`` のどれも ``llmkit.<submodule>`` を import しない。"""
    paths = module_paths()
    assert len(paths) >= len(_SUBMODULE_NAMES) + 1, paths

    offenders: list[str] = []
    for module_path in paths:
        offenders += [
            f"{module_path.name}: {name}"
            for name in imported_module_names(module_path)
            if name.startswith("llmkit.")
        ]

    assert not offenders, f"llmkit サブモジュールの直接 import: {offenders}"


def test_the_rag_package_actually_uses_the_llmkit_public_api() -> None:
    """上の検査が「そもそも llmkit を使っていない」で通っていないことの確認。"""
    users = [
        module_path.name
        for module_path in module_paths()
        if "llmkit" in imported_module_names(module_path)
    ]

    assert len(users) >= 2, users


# --------------------------------------------------------------------------
# (ii) rag.__all__ = サブモジュールの __all__ の和集合
# --------------------------------------------------------------------------


def test_rag_all_matches_the_union_of_submodule_all() -> None:
    """公開シンボルの追加漏れ (再エクスポート忘れ) を機械的に検出する。"""
    union: set[str] = set()
    for name in _SUBMODULE_NAMES:
        module = importlib.import_module(f"rag.{name}")
        module_all = getattr(module, "__all__", None)
        assert module_all is not None, f"rag.{name} に __all__ がありません"
        union.update(module_all)

    assert set(rag.__all__) == union
    assert len(rag.__all__) == len(set(rag.__all__)), "重複エクスポート"
    for name in rag.__all__:
        assert hasattr(rag, name), name


def test_every_rag_submodule_is_covered_by_the_union_check() -> None:
    """サブモジュールを足したのに ``_SUBMODULE_NAMES`` へ入れ忘れると落ちる。"""
    discovered = {
        module_path.stem
        for module_path in RAG_DIR.glob("*.py")
        if module_path.stem != "__init__"
    }

    assert discovered == set(_SUBMODULE_NAMES)


# --------------------------------------------------------------------------
# (iii) llmkit -> rag は 0 件 / rag <-> harness も 0 件
# --------------------------------------------------------------------------


def test_llmkit_never_imports_the_rag_package() -> None:
    """L2 が L3 に依存していないこと (依存の向きは一方向)。"""
    offenders: list[str] = []
    for module_path in sorted(PACKAGE_DIR.glob("*.py")):
        offenders += [
            f"{module_path.name}: {name}"
            for name in imported_module_names(module_path)
            if name == "rag" or name.startswith("rag.")
        ]

    assert not offenders, f"llmkit から rag への import: {offenders}"


def test_rag_and_harness_never_import_each_other() -> None:
    """同じ L3 の 2 本が依存し合わないこと (どちらも単独で動く)。"""
    offenders: list[str] = []
    for module_path in module_paths():
        offenders += [
            f"rag/{module_path.name}: {name}"
            for name in imported_module_names(module_path)
            if name == "harness" or name.startswith("harness.")
        ]
    for module_path in sorted(HARNESS_DIR.glob("*.py")):
        offenders += [
            f"harness/{module_path.name}: {name}"
            for name in imported_module_names(module_path)
            if name == "rag" or name.startswith("rag.")
        ]

    assert not offenders, f"L3 どうしの相互 import: {offenders}"
    assert set(rag.__all__).isdisjoint(harness.__all__), "公開シンボル名の衝突"


# --------------------------------------------------------------------------
# (iv) D-25 guard: rag は推論ランタイムに直接触れない
# --------------------------------------------------------------------------


def test_rag_never_talks_to_the_runtime_directly() -> None:
    """D-25 guard: ``rag/`` に HTTP 呼び出しもエンドポイント文字列も無い。

    埋め込みは ``llmkit.embeddings`` (L2) 経由でしか呼べない。ここが破られると
    例外翻訳表が 2 か所に分岐し、L3 が「ランタイムを叩くもの」と「叩かないもの」に
    分裂して既存の境界検査が意味を失う (仕様書 §3 論点1)。
    """
    offenders: list[str] = []
    for module_path in module_paths():
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        offenders += [
            f"{module_path.name}: .{name}()"
            for name in called_attribute_names(tree)
            if name in _HTTP_METHOD_NAMES
        ]
        offenders += [
            f"{module_path.name}: import {name}"
            for name in imported_module_names(module_path)
            if name.split(".")[0] in {"httpx", "requests", "urllib", "http", "socket"}
        ]
        offenders += [
            f"{module_path.name}: {literal!r}"
            for literal in non_docstring_string_literals(tree)
            for endpoint in _ENDPOINT_LITERALS
            if endpoint in literal
        ]

    assert not offenders, f"rag から推論ランタイムへの直接経路: {offenders}"


# --------------------------------------------------------------------------
# (v) D-30 の構造層: vault に触れるのは rag/vault.py だけ
# --------------------------------------------------------------------------


def test_only_the_vault_module_reads_the_vault() -> None:
    """D-30 (構造): vault の読み取りを 1 モジュールに閉じる。

    閉じていれば「書き込み API が無い」ことの静的検査 (``test_rag_vault.py``)
    が vault へのすべての経路をカバーできる。散らばると、検査していない
    モジュールから書き込む経路が開く。
    """
    offenders: list[str] = []
    for module_path in module_paths():
        if module_path.name == "vault.py":
            continue
        allowed = _FILESYSTEM_READ_ALLOWANCES.get(module_path.name, frozenset())
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for name in called_attribute_names(tree):
            if name in _FILESYSTEM_READ_NAMES and name not in allowed:
                offenders.append(f"{module_path.name}: .{name}()")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "open"
            ):
                offenders.append(f"{module_path.name}: open()")

    assert not offenders, f"rag/vault.py 以外からの vault 読み取り: {offenders}"


def test_the_vault_module_is_the_one_that_actually_reads(
    sample_vault_copy: Path,
) -> None:
    """上の検査が「誰も読んでいない」で通っていないことの確認。"""
    settings_path = write_rag_settings(sample_vault_copy.parent)
    settings = rag.load_settings(settings_path)

    assert list(rag.iter_vault_files(settings)), "合成 vault から 1 件も読めていません"


# --------------------------------------------------------------------------
# D-08 の走査範囲を rag/ にも広げる
# --------------------------------------------------------------------------


def test_rag_schemas_use_pydantic_dataclasses_not_basemodel() -> None:
    """D-08: ``pydantic.BaseModel`` を継承しない (``tests/test_layout.py`` と同趣旨)。

    継承するとその定義行が mypy の ``disallow_any_explicit`` に触れ、
    ``# type: ignore`` を撒く以外に通す手段が無くなる。
    """
    offenders: list[str] = []
    for module_path in module_paths():
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                name = base.attr if isinstance(base, ast.Attribute) else None
                if isinstance(base, ast.Name):
                    name = base.id
                if name == "BaseModel":
                    offenders.append(f"rag/{module_path.name}::{node.name}")

    assert not offenders, f"BaseModel を継承しているクラス: {offenders}"
