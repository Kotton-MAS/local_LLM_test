"""テスト方針そのものの検証 (D-02)。

このファイルは 1 つの guard_test を含む:

- ``test_live_tests_are_skipped_by_default`` … D-02
  (実ランタイムに接続するテストは ``@pytest.mark.live`` を付け、``--run-live``
  を指定したときだけ実行する)

「テストが実 HTTP を出さない」ことは、個々のテストの書き方ではなく仕組みで
担保する。ここが落ちたら、CI (Ollama 無し・ネットワーク制限あり) が壊れる。
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest
from conftest import LIVE_MARKER, RUN_LIVE_OPTION

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
CONFTEST_SOURCE = (TESTS_DIR / "conftest.py").read_text(encoding="utf-8")

LIVE_SAMPLE = """
import pytest


@pytest.mark.live
def test_needs_a_real_runtime() -> None:
    raise AssertionError("live テストが既定で実行された")


def test_runs_without_a_runtime() -> None:
    assert True
"""


# --------------------------------------------------------------------------
# guard_test (D-02)
# --------------------------------------------------------------------------


def test_live_tests_are_skipped_by_default(pytester: pytest.Pytester) -> None:
    """D-02 guard: live マーカー付きテストは既定でスキップされる。

    実際の ``tests/conftest.py`` をそのまま入れ子の pytest に読ませて検証する
    (方針の写しではなく本物の実装を測る)。
    """
    pytester.makeconftest(CONFTEST_SOURCE)
    pytester.makepyfile(test_live_sample=LIVE_SAMPLE)

    result = pytester.runpytest("-q")

    result.assert_outcomes(passed=1, skipped=1)


def test_live_tests_run_only_when_run_live_is_given(
    pytester: pytest.Pytester,
) -> None:
    """``--run-live`` を渡したときだけ live テストが実際に走る。"""
    pytester.makeconftest(CONFTEST_SOURCE)
    pytester.makepyfile(test_live_sample=LIVE_SAMPLE)

    result = pytester.runpytest("-q", RUN_LIVE_OPTION)

    # 走れば AssertionError で落ちる = 既定のスキップが効いていた証拠。
    result.assert_outcomes(passed=1, failed=1)


def test_live_marker_is_registered_in_pyproject() -> None:
    """マーカーは pyproject.toml にも登録する (未知マーカー警告を出さない)。"""
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)

    markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]

    assert any(str(marker).startswith(f"{LIVE_MARKER}:") for marker in markers)


# --------------------------------------------------------------------------
# 実ネットワーク遮断
# --------------------------------------------------------------------------


def test_real_network_access_is_blocked_in_tests() -> None:
    """live 以外のテストからのソケット接続は機械的に禁止されている。"""
    import socket

    with pytest.raises(AssertionError, match="実ネットワーク"):
        socket.create_connection(("localhost", 11434), timeout=0.01)


def imported_modules(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_every_http_test_module_uses_mock_transport() -> None:
    """httpx を import するテストは必ず MockTransport 経由であること。"""
    offenders: list[str] = []
    for module_path in sorted(TESTS_DIR.glob("test_*.py")):
        source = module_path.read_text(encoding="utf-8")
        if "httpx" not in imported_modules(ast.parse(source)):
            continue
        if "MockTransport" in source or "RecordingTransport" in source:
            continue
        offenders.append(module_path.name)

    assert not offenders, f"MockTransport を使っていない httpx テスト: {offenders}"


def test_no_test_module_defines_its_own_llmkit_env_fixture() -> None:
    """環境変数の隔離は conftest.py の autouse fixture 1 つに集約する。"""
    duplicates: list[str] = []
    for module_path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_isolate_llmkit_env":
                duplicates.append(module_path.name)

    assert not duplicates, f"conftest.py と重複した fixture: {duplicates}"
