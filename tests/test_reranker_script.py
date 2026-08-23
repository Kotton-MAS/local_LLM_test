"""リランカー起動スクリプトの契約テスト (決定 D-17)。

``scripts/start-reranker.sh`` は llama.cpp の ``llama-server`` を起動する。
llama-server もモデルもリポジトリ外 (``~/.local``) に置くため、パスは
**すべて環境変数で上書きできなければならない**。上書きできることを散文で
主張するだけでは守られないので、``--dry-run`` の出力を掃引して固定する。

``--dry-run`` が存在チェックを行わないのは、llama-server が入っていない CI
でもこの配線を検証できるようにするため (D-17)。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "start-reranker.sh"

# 既定値。スクリプトの ${VAR:-既定} と対応する。
DEFAULT_PORT = "8081"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_CTX_SIZE = "2048"
DEFAULT_NGL = "99"

# (環境変数名, 差し替える値, 既定の出力に現れる文字列)
# RERANKER_HOST の例は 0.0.0.0 (全公開) ではなく 192.0.2.10 (TEST-NET-1、
# RFC 5737 の文書用アドレス) にする。llama-server には認証機構が無いため、
# テストの上書き例が「0.0.0.0 は想定内の使い方」と読めてしまわないため (F-6-003)。
OVERRIDE_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "LLAMA_SERVER_BIN",
        "/opt/custom/llama-server",
        ".local/opt/llama.cpp/llama-server",
    ),
    ("RERANKER_MODEL", "/data/other-reranker.gguf", "bge-reranker-v2-m3-Q6_K.gguf"),
    ("RERANKER_PORT", "9999", DEFAULT_PORT),
    ("RERANKER_HOST", "192.0.2.10", DEFAULT_HOST),
    ("RERANKER_NGL", "0", DEFAULT_NGL),
    ("RERANKER_CTX_SIZE", "8192", DEFAULT_CTX_SIZE),
)

# スクリプトが ${VAR:-既定} で参照する6変数。テスト実行者のシェルにこれらが
# 既に export されていても既定値前提のテストが影響を受けないよう、
# _dry_run() は毎回これらを明示的に除去してから baseline を組む (F-6-006)。
_SCRIPT_ENV_VARS: tuple[str, ...] = tuple(case[0] for case in OVERRIDE_CASES)


def _dry_run(env_overrides: dict[str, str] | None = None) -> str:
    """``--dry-run`` を実行し、解決されたコマンド行を返す。"""
    env = dict(os.environ)
    for var in _SCRIPT_ENV_VARS:
        env.pop(var, None)
    # HOME 依存の既定値を安定させる (テスト実行者の HOME に依存しないため)。
    env["HOME"] = "/home/testuser"
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"--dry-run は存在チェックをせず exit 0 のはず: {result.stderr}"
    )
    return result.stdout.strip()


def test_launch_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"{SCRIPT} が存在しない"
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} に実行ビットが立っていない"


def test_dry_run_prints_a_single_command_line_without_checking_files() -> None:
    """llama-server が無い環境 (CI) でも exit 0 でコマンド行を出す (D-17)。"""
    out = _dry_run(
        {
            "LLAMA_SERVER_BIN": "/nonexistent/llama-server",
            "RERANKER_MODEL": "/nonexistent/model.gguf",
        }
    )

    assert len(out.splitlines()) == 1, f"1 行であるべき: {out!r}"
    assert "--reranking" in out
    assert "/nonexistent/llama-server" in out
    assert "/nonexistent/model.gguf" in out


def test_dry_run_uses_the_documented_defaults() -> None:
    out = _dry_run()

    assert ".local/opt/llama.cpp/llama-server" in out
    assert "bge-reranker-v2-m3-Q6_K.gguf" in out
    assert f"--host {DEFAULT_HOST}" in out
    assert f"--port {DEFAULT_PORT}" in out
    assert f"--n-gpu-layers {DEFAULT_NGL}" in out
    assert f"--ctx-size {DEFAULT_CTX_SIZE}" in out


@pytest.mark.parametrize(
    ("var", "value", "default_marker"),
    OVERRIDE_CASES,
    ids=[case[0] for case in OVERRIDE_CASES],
)
def test_launch_script_paths_are_overridable_by_environment(
    var: str, value: str, default_marker: str
) -> None:
    """D-17 guard: 6 項目すべてが環境変数で上書きできる。

    1 項目でも ``${VAR:-既定}`` からハードコードに変わると、その項目の
    ケースだけが落ちる。
    """
    baseline = _dry_run()
    assert default_marker in baseline, (
        f"既定の出力に {default_marker!r} が現れるはず: {baseline!r}"
    )

    overridden = _dry_run({var: value})

    assert value in overridden, f"{var} の上書きが反映されていない: {overridden!r}"
    assert default_marker not in overridden, (
        f"{var} を上書きしたのに既定値 {default_marker!r} が残っている: {overridden!r}"
    )


def test_launch_script_never_escalates_privileges_or_touches_systemd() -> None:
    """スクリプトが不可逆な操作・権限昇格を行わないことを固定する。"""
    body = SCRIPT.read_text(encoding="utf-8")

    for forbidden in ("sudo", "systemctl", "rm -rf", "rm -fr"):
        assert forbidden not in body, f"スクリプトに {forbidden!r} が含まれている"


def test_launch_script_defaults_to_loopback() -> None:
    """既定で外部に公開しない (認証機構を持たないため)。"""
    out = _dry_run()

    assert f"--host {DEFAULT_HOST}" in out
    assert DEFAULT_HOST.startswith("127."), "既定はループバックであるべき"


def test_unknown_argument_prints_usage_and_exits_2_without_launching() -> None:
    """F-6-002 guard: 未知の引数 (--help・打ち間違い等) を黙って無視して
    本番起動しない。使い方を stderr に出して exit 2 する。
    """
    env = dict(os.environ)
    for var in _SCRIPT_ENV_VARS:
        env.pop(var, None)
    env["HOME"] = "/home/testuser"
    # 起動パスに落ちても存在しないバイナリで確実に失敗させる (誤って
    # 実プロセスを起動しないための保険)。
    env["LLAMA_SERVER_BIN"] = "/nonexistent/llama-server"

    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2, (
        f"未知の引数は exit 2 であるべき: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert "使い方" in result.stderr
    assert result.stdout == ""
