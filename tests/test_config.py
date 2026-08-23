"""設定層 (llmkit/config.py) のテスト。実 I/O は一時ファイルの読み書きのみ。"""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmkit.config import AppConfig, RuntimeConfig, load_config
from llmkit.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"
EXTERNAL_CONFIG = REPO_ROOT / "configs" / "external_openai.toml"

VALID_TOML = """
[runtime]
kind = "ollama"
base_url = "http://localhost:11434/v1"
api_key_env = "LLMKIT_TEST_KEY"
timeout_s = 120.0
is_local = true

[generation]
model = "qwen3-14b"
context_tokens = 16384
temperature = 0.7
top_p = 0.9
max_output_tokens = 1024
seed = 0

[vram]
budget_gib = 16.0
runtime_overhead_gib = 0.8
active_profile = "rag_default"

[profiles.rag_default]
generation = "qwen3-14b"
embedding = "ruri-v3-310m"
reranker = "bge-reranker-v2-m3"
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_default_config_returns_app_config() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert isinstance(config, AppConfig)
    assert config.runtime.kind == "ollama"
    assert config.runtime.base_url == "http://localhost:11434/v1"
    assert config.runtime.is_local is True
    assert config.generation.model == "qwen3-14b"
    assert config.generation.context_tokens == 16384
    # 実測でオフロードが始まらない増分の上限 (D-12)。カード容量ではない
    assert config.vram.budget_gib == pytest.approx(14.0)
    assert config.vram.runtime_overhead_gib == pytest.approx(0.8)
    assert config.vram.active_profile == "rag_default"
    assert set(config.profiles) == {
        "rag_default",
        "long_context",
        "lightweight",
        "oversized",
    }
    assert config.profiles["rag_default"].model_ids() == (
        "qwen3-14b",
        "ruri-v3-310m",
        "bge-reranker-v2-m3",
    )
    assert config.profiles["long_context"].model_ids() == ("gpt-oss-20b",)


def test_load_external_config_with_same_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMKIT_API_KEY", "sk-test-external")

    config = load_config(EXTERNAL_CONFIG)

    assert isinstance(config, AppConfig)
    assert config.runtime.kind == "openai_compatible"
    assert config.runtime.is_local is False
    assert config.runtime.base_url.startswith("https://")
    assert config.api_key.get_secret_value() == "sk-test-external"


def test_shipped_configs_contain_no_api_key_value() -> None:
    """configs/*.toml に api_key そのものを書けないことを固定する (D-05)。"""
    for path in (DEFAULT_CONFIG, EXTERNAL_CONFIG):
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        assert "api_key" not in raw
        assert "api_key" not in raw["runtime"]
        assert raw["runtime"]["api_key_env"] == "LLMKIT_API_KEY"


@pytest.mark.parametrize(
    ("broken_toml", "expected_key"),
    [
        pytest.param(
            VALID_TOML.replace("[generation]", "[generation]\nunknown_knob = 1", 1),
            "unknown_knob",
            id="unknown-key",
        ),
        pytest.param(
            VALID_TOML.replace("timeout_s = 120.0", 'timeout_s = "fast"', 1),
            "timeout_s",
            id="type-mismatch",
        ),
        pytest.param(
            VALID_TOML.replace(
                'base_url = "http://localhost:11434/v1"',
                'base_url = "localhost:11434"',
                1,
            ),
            "base_url",
            id="base-url-not-a-url",
        ),
        pytest.param(
            VALID_TOML.replace("context_tokens = 16384", "context_tokens = 0", 1),
            "context_tokens",
            id="non-positive-context",
        ),
        pytest.param(
            VALID_TOML.replace("context_tokens = 16384", "context_tokens = -1", 1),
            "context_tokens",
            id="negative-context",
        ),
        pytest.param(
            VALID_TOML.replace(
                'active_profile = "rag_default"', 'active_profile = "missing"', 1
            ),
            "active_profile",
            id="unknown-active-profile",
        ),
        # --------------------------------------------------------------
        # F-1-029: Field 制約 (ge/le/gt) の境界値。context_tokens 以外の
        # 数値制約フィールドすべてに同じ検証パターンを揃える。
        # --------------------------------------------------------------
        pytest.param(
            VALID_TOML.replace("temperature = 0.7", "temperature = 2.1", 1),
            "temperature",
            id="temperature-above-max",
        ),
        pytest.param(
            VALID_TOML.replace("temperature = 0.7", "temperature = -0.1", 1),
            "temperature",
            id="temperature-below-min",
        ),
        pytest.param(
            VALID_TOML.replace("top_p = 0.9", "top_p = 0.0", 1),
            "top_p",
            id="top-p-at-zero-is-excluded",
        ),
        pytest.param(
            VALID_TOML.replace("top_p = 0.9", "top_p = 1.1", 1),
            "top_p",
            id="top-p-above-max",
        ),
        pytest.param(
            VALID_TOML.replace("max_output_tokens = 1024", "max_output_tokens = 0", 1),
            "max_output_tokens",
            id="max-output-tokens-zero",
        ),
        pytest.param(
            VALID_TOML.replace("max_output_tokens = 1024", "max_output_tokens = -1", 1),
            "max_output_tokens",
            id="max-output-tokens-negative",
        ),
        pytest.param(
            VALID_TOML.replace("budget_gib = 16.0", "budget_gib = 0.0", 1),
            "budget_gib",
            id="budget-gib-zero",
        ),
        pytest.param(
            VALID_TOML.replace("budget_gib = 16.0", "budget_gib = -1.0", 1),
            "budget_gib",
            id="budget-gib-negative",
        ),
        pytest.param(
            VALID_TOML.replace(
                "runtime_overhead_gib = 0.8", "runtime_overhead_gib = -0.1", 1
            ),
            "runtime_overhead_gib",
            id="runtime-overhead-gib-negative",
        ),
        pytest.param(
            VALID_TOML.replace("timeout_s = 120.0", "timeout_s = 0.0", 1),
            "timeout_s",
            id="timeout-s-zero",
        ),
        # --------------------------------------------------------------
        # F-1-029: NonEmptyStr (min_length=1) 制約。同じ Field 制約クラスの
        # 境界値が数値フィールドしかテストされていなかったので合わせて埋める。
        # --------------------------------------------------------------
        pytest.param(
            VALID_TOML.replace(
                'api_key_env = "LLMKIT_TEST_KEY"', 'api_key_env = ""', 1
            ),
            "api_key_env",
            id="api-key-env-empty",
        ),
        pytest.param(
            VALID_TOML.replace('model = "qwen3-14b"', 'model = ""', 1),
            "model",
            id="generation-model-empty",
        ),
        pytest.param(
            VALID_TOML.replace('generation = "qwen3-14b"', 'generation = ""', 1),
            "generation",
            id="profile-generation-empty",
        ),
        pytest.param(
            VALID_TOML.replace(
                'active_profile = "rag_default"', 'active_profile = ""', 1
            ),
            "active_profile",
            id="active-profile-empty",
        ),
        # --------------------------------------------------------------
        # F-1-016: base_url の userinfo (認証情報埋め込み) 経由の秘密混入。
        # --------------------------------------------------------------
        pytest.param(
            VALID_TOML.replace(
                'base_url = "http://localhost:11434/v1"',
                'base_url = "http://admin:sup3rs3cret@localhost:11434/v1"',
                1,
            ),
            "base_url",
            id="base-url-with-inline-credentials",
        ),
        # --------------------------------------------------------------
        # F-2-007: base_url のクエリ・フラグメント経由の秘密混入 (CWE-532)。
        # userinfo だけでなく ?key=... / #... でも秘密を書けてしまっていた
        # 取りこぼしを塞ぐ (query/fragment はパス連結後に不正な URL になる
        # ため正当性の観点でも不要)。
        # --------------------------------------------------------------
        pytest.param(
            VALID_TOML.replace(
                'base_url = "http://localhost:11434/v1"',
                'base_url = "http://localhost:11434/v1?key=sup3rs3cret"',
                1,
            ),
            "base_url",
            id="base-url-with-secret-in-query",
        ),
        pytest.param(
            VALID_TOML.replace(
                'base_url = "http://localhost:11434/v1"',
                'base_url = "http://localhost:11434/v1#sup3rs3cret"',
                1,
            ),
            "base_url",
            id="base-url-with-secret-in-fragment",
        ),
    ],
)
def test_invalid_config_raises_config_error_naming_the_key(
    tmp_path: Path, broken_toml: str, expected_key: str
) -> None:
    path = write_config(tmp_path, broken_toml)

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    assert expected_key in str(excinfo.value)


@pytest.mark.parametrize(
    ("valid_toml", "field_getter", "expected"),
    [
        pytest.param(
            VALID_TOML.replace("temperature = 0.7", "temperature = 0.0", 1),
            lambda c: c.generation.temperature,
            0.0,
            id="temperature-at-min-inclusive-boundary",
        ),
        pytest.param(
            VALID_TOML.replace("temperature = 0.7", "temperature = 2.0", 1),
            lambda c: c.generation.temperature,
            2.0,
            id="temperature-at-max-inclusive-boundary",
        ),
        pytest.param(
            VALID_TOML.replace("top_p = 0.9", "top_p = 1.0", 1),
            lambda c: c.generation.top_p,
            1.0,
            id="top-p-at-max-inclusive-boundary",
        ),
        pytest.param(
            VALID_TOML.replace(
                "runtime_overhead_gib = 0.8", "runtime_overhead_gib = 0.0", 1
            ),
            lambda c: c.vram.runtime_overhead_gib,
            0.0,
            id="runtime-overhead-gib-at-min-inclusive-boundary",
        ),
    ],
)
def test_inclusive_boundary_values_are_accepted_as_valid(
    tmp_path: Path,
    valid_toml: str,
    field_getter: Callable[[AppConfig], float],
    expected: float,
) -> None:
    """F-2-010: le/ge の閉端値そのものが有効な設定として受理されることを確認する。

    F-1-029 で追加した境界値テストは範囲外 (2.1, -0.1, 1.1 等) の拒否のみを
    検証しており、閉端の値 (0.0, 2.0, 1.0) を `le`/`ge` から `lt`/`gt` に
    書き換えても検知できないミューテーション耐性の穴があった (実測確認済み)。
    ここでは閉端値そのものが有効な設定として読み込めることを確認し、
    包含側の境界演算子を固定する。top_p の下限 (`gt=0.0`) は
    ``top-p-at-zero-is-excluded`` が既に排他側を固定済みなのでここでは扱わない。
    """
    path = write_config(tmp_path, valid_toml)

    config = load_config(path)

    assert field_getter(config) == pytest.approx(expected)


def test_broken_toml_syntax_raises_config_error(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[runtime\nkind = 'ollama'\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    assert str(path) in str(excinfo.value)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(tmp_path / "nope.toml")

    assert "nope.toml" in str(excinfo.value)


def test_inline_api_key_is_rejected(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        VALID_TOML.replace("[runtime]", '[runtime]\napi_key = "sk-leaked"', 1),
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    assert "runtime.api_key" in str(excinfo.value)
    assert "sk-leaked" not in str(excinfo.value)


def test_base_url_with_inline_credentials_does_not_leak_the_secret(
    tmp_path: Path,
) -> None:
    """F-1-016 (CWE-522): base_url の userinfo 経由の秘密混入を入口で拒否する。

    D-05 の「api_key は設定ファイルに書けない」に加え、base_url 経由でも
    平文の資格情報を書けないようにする。例外メッセージにも秘密自体を含めない。
    """
    secret = "sup3rs3cret"
    path = write_config(
        tmp_path,
        VALID_TOML.replace(
            'base_url = "http://localhost:11434/v1"',
            f'base_url = "http://admin:{secret}@localhost:11434/v1"',
            1,
        ),
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    assert "base_url" in message
    assert secret not in message


def test_runtime_config_rejects_userinfo_in_base_url_directly() -> None:
    """load_config を経由しない直接構築でも同じ検証が効く (フィールドが要)。

    reviewer-security の repro_command (F-1-016) の再現: 直接構築だと
    :class:`~llmkit.errors.ConfigError` ではなく pydantic の生の
    ``ValidationError`` になる (``load_config`` だけが ``ConfigError`` へ
    翻訳し、``input`` を落としてメッセージを組み立てる。
    ``_format_validation_error`` 参照)。いずれの経路でも base_url は
    受理されないことがここでの主張であり、pydantic 自身の raw
    ``ValidationError`` 文字列表現に入力値が残ることは pydantic の一般的な
    挙動でありここでは対象外 (秘密が漏れないことの保証は
    ``load_config`` が返す ``ConfigError`` に対してのみ行う。
    ``test_base_url_with_inline_credentials_does_not_leak_the_secret`` 参照)。
    """
    with pytest.raises(ValidationError):
        RuntimeConfig(base_url="http://admin:sup3rs3cret@10.0.0.9:11434/v1")


def test_local_config_loads_without_api_key_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLMKIT_TEST_KEY", raising=False)
    path = write_config(tmp_path, VALID_TOML)

    config = load_config(path)

    assert config.api_key.get_secret_value() == ""


def test_remote_config_without_api_key_env_raises_naming_env_var_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLMKIT_TEST_KEY", raising=False)
    path = write_config(
        tmp_path, VALID_TOML.replace("is_local = true", "is_local = false", 1)
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    assert "LLMKIT_TEST_KEY" in message
    assert "is_local" in message


def test_api_key_never_appears_in_repr_or_str(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-super-secret-value"
    monkeypatch.setenv("LLMKIT_TEST_KEY", secret)
    path = write_config(tmp_path, VALID_TOML)

    config = load_config(path)

    assert config.api_key.get_secret_value() == secret
    assert secret not in repr(config)
    assert secret not in str(config)
    assert secret not in repr(config.api_key)
    assert "LLMKIT_TEST_KEY" in repr(config)


def test_config_is_frozen(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_TOML))
    attribute = "context_tokens"

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config.generation, attribute, 1)
