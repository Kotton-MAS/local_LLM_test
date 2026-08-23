"""パッケージレイアウトと公開 API の固定 (D-06)。

このファイルは 2 つの guard_test を含む:

- ``test_llmkit_importable_from_repo_root`` … D-06
  (llmkit はリポジトリ直下のフラットレイアウトに置き、src/ にしない)
- ``test_schemas_use_pydantic_dataclasses_not_basemodel`` … D-08
  (スキーマは pydantic dataclass で定義し、BaseModel を継承しない)
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import llmkit

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "llmkit"
HARNESS_DIR = REPO_ROOT / "harness"

#: L2 を構成するサブモジュール。それぞれが自分の公開 API を ``__all__`` で
#: 宣言しており、llmkit.__all__ はこれらの和集合と一致するべき (F-1-002)。
_SUBMODULE_NAMES = (
    "bootstrap",
    "catalog",
    "client",
    "config",
    "errors",
    "manifest",
    "vram",
)


# --------------------------------------------------------------------------
# guard_test (D-06)
# --------------------------------------------------------------------------


def test_llmkit_importable_from_repo_root() -> None:
    """D-06 guard: リポジトリルートから追加設定なしで ``import llmkit`` が通る。

    ``PYTHONPATH`` を空にした別プロセスで確かめる (親プロセスの sys.path に
    依存しないことの確認 = フラットレイアウトが効いていることの確認)。
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""

    completed = subprocess.run(
        [sys.executable, "-c", "import llmkit, sys; sys.stdout.write(llmkit.__file__)"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout) == PACKAGE_DIR / "__init__.py"


def test_package_is_not_under_a_src_layout() -> None:
    assert (PACKAGE_DIR / "__init__.py").is_file()
    assert not (REPO_ROOT / "src").exists()
    assert Path(llmkit.__file__ or "").parent == PACKAGE_DIR


def test_pyproject_has_no_build_system_section() -> None:
    """D-06 の前提: 非パッケージプロジェクトのまま pythonpath で解決している。"""
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)

    assert "build-system" not in pyproject
    assert pyproject["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]


def test_public_api_is_reexported_from_the_package_root() -> None:
    """L3 は ``llmkit`` の公開シンボルだけを import すれば足りる。"""
    expected = {
        "AppConfig",
        "ChatClient",
        "ChatMessage",
        "ChatResult",
        "ConfigError",
        "LlmkitError",
        "OpenAICompatibleClient",
        "RuntimeUnavailableError",
        "VramBudgetExceededError",
        "load_config",
    }

    assert expected <= set(llmkit.__all__)
    for name in llmkit.__all__:
        assert hasattr(llmkit, name), name


def test_public_api_matches_the_union_of_submodule_all() -> None:
    """F-1-002: llmkit.__all__ が全サブモジュールの __all__ の和集合と一致する。

    ``bootstrap`` / ``estimate_profile`` / ``ModelSpec`` 等、L3 (Phase 2/3) や
    L4 (cli.py) が起動・見積り・カタログ照会に使う入口が ``llmkit`` の
    トップレベルから欠けたまま放置される (= サブモジュール直接 import を
    強いる) ことを機械的に防ぐ。サブモジュール側で ``__all__`` に新しい公開
    シンボルを追加したのにここへの再エクスポートを忘れると、このテストが落ちる。
    """
    union: set[str] = set()
    for name in _SUBMODULE_NAMES:
        module = importlib.import_module(f"llmkit.{name}")
        module_all = getattr(module, "__all__", None)
        assert module_all is not None, f"llmkit.{name} に __all__ がありません"
        union.update(module_all)

    assert set(llmkit.__all__) == union
    assert len(llmkit.__all__) == len(set(llmkit.__all__)), "重複エクスポートがあります"
    for name in llmkit.__all__:
        assert hasattr(llmkit, name), name


# --------------------------------------------------------------------------
# guard_test (D-08)
# --------------------------------------------------------------------------


def test_schemas_use_pydantic_dataclasses_not_basemodel() -> None:
    """D-08 guard: pydantic.BaseModel を継承しない。

    継承するとその定義行が mypy の ``disallow_any_explicit`` に触れ、
    ``# type: ignore`` を撒く以外に通す手段が無くなる。

    走査範囲は L2 (``llmkit/``) と L3 (``harness/``) の両方。ハーネスも
    スイート TOML を pydantic で検証しており、同じ制約の下にある。
    """
    offenders: list[str] = []
    module_paths = sorted(PACKAGE_DIR.glob("*.py")) + sorted(HARNESS_DIR.glob("*.py"))
    assert len(module_paths) > len(list(PACKAGE_DIR.glob("*.py"))), (
        "harness/ が走査対象から外れています"
    )
    for module_path in module_paths:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                name = base.attr if isinstance(base, ast.Attribute) else None
                if isinstance(base, ast.Name):
                    name = base.id
                if name == "BaseModel":
                    offenders.append(
                        f"{module_path.parent.name}/{module_path.name}::{node.name}"
                    )

    assert not offenders, f"BaseModel を継承しているクラス: {offenders}"


def test_mypy_strictness_is_not_relaxed() -> None:
    """D-08 の前提 (mypy 設定) が緩められていないこと。"""
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        mypy_settings = tomllib.load(stream)["tool"]["mypy"]

    assert mypy_settings["strict"] is True
    assert mypy_settings["disallow_any_explicit"] is True
