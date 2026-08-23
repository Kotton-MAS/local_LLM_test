"""比較スイート (harness/suite.py) の読み込みと実効設定の導出のテスト。

このファイルの中心は ``apply_case`` である。モデル ID は設定の 2 か所
(``generation.model`` / ``profiles[active].generation``) に現れ、参照先が違う。
片方だけ差し替えても何のエラーも出ないまま「20B に投げているのに 14B の VRAM
見積りを記録した比較結果」が生成されるため、**両方が同時に動くこと**を
掃引テストで固定する (D-19)。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from conftest import DEFAULT_CONFIG

from harness.suite import (
    ComparisonSuite,
    ModelCase,
    apply_case,
    load_suite,
    suite_sha256,
)
from llmkit import (
    AppConfig,
    ConfigError,
    estimate_profile,
    load_config,
    resolve_profile,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_MODULE = REPO_ROOT / "harness" / "suite.py"

VALID_SUITE = """\
[suite]
id = "ja_basic_test"
description = "テスト用スイート"
warmup_runs = 2

[[models]]
model_id = "qwen3-14b"

[[models]]
model_id = "gpt-oss-20b"
profile = "long_context"
context_tokens = 32768
temperature = 0.0
top_p = 0.5
max_output_tokens = 512
seed = 7

[[prompts]]
id = "summarize"
text = "次の文章を3行で要約してください。"
system = "あなたは日本語の要約アシスタントです。"
tags = ["summary", "ja"]

[[prompts]]
id = "translate"
text = "次の文を英語に訳してください。"
"""


def write_suite(directory: Path, text: str, *, name: str = "suite.toml") -> Path:
    destination = directory / name
    destination.write_text(text, encoding="utf-8")
    return destination


@pytest.fixture
def base_config() -> AppConfig:
    return load_config(DEFAULT_CONFIG)


@pytest.fixture
def suite(tmp_path: Path) -> ComparisonSuite:
    return load_suite(write_suite(tmp_path, VALID_SUITE))


# --------------------------------------------------------------------------
# load_suite
# --------------------------------------------------------------------------


def test_load_suite_reads_every_declared_field(suite: ComparisonSuite) -> None:
    assert suite.suite.id == "ja_basic_test"
    assert suite.suite.description == "テスト用スイート"
    assert suite.suite.warmup_runs == 2

    assert [case.model_id for case in suite.models] == ["qwen3-14b", "gpt-oss-20b"]
    first, second = suite.models
    assert first.profile is None
    assert first.context_tokens is None
    assert second.profile == "long_context"
    assert second.context_tokens == 32768
    assert second.temperature == pytest.approx(0.0)
    assert second.top_p == pytest.approx(0.5)
    assert second.max_output_tokens == 512
    assert second.seed == 7

    assert suite.prompt_ids == ("summarize", "translate")
    assert suite.prompts[0].tags == ("summary", "ja")
    assert suite.prompts[0].system == "あなたは日本語の要約アシスタントです。"
    assert suite.prompts[1].system is None
    assert suite.prompts[1].tags == ()


def test_warmup_runs_defaults_to_one(tmp_path: Path) -> None:
    """D-22: ウォームアップの既定は 1 回。書かなくても 0 にはならない。"""
    text = VALID_SUITE.replace("warmup_runs = 2\n", "")
    loaded = load_suite(write_suite(tmp_path, text))

    assert loaded.suite.warmup_runs == 1


def test_warmup_runs_may_be_zero(tmp_path: Path) -> None:
    """0 は「コールド実行そのものを測る」正当な指定として通す。"""
    text = VALID_SUITE.replace("warmup_runs = 2", "warmup_runs = 0")

    assert load_suite(write_suite(tmp_path, text)).suite.warmup_runs == 0


INVALID_SUITES: tuple[tuple[str, str, str], ...] = (
    (
        "path_traversal_in_id",
        VALID_SUITE.replace('id = "ja_basic_test"', 'id = "../escape"'),
        "suite.id",
    ),
    (
        "duplicate_prompt_ids",
        VALID_SUITE.replace('id = "translate"', 'id = "summarize"'),
        "prompts",
    ),
    (
        "unknown_key",
        VALID_SUITE.replace(
            'description = "テスト用スイート"',
            'description = "テスト用スイート"\nwarmup_run = 2',
        ),
        "warmup_run",
    ),
    (
        "negative_warmup_runs",
        VALID_SUITE.replace("warmup_runs = 2", "warmup_runs = -1"),
        "warmup_runs",
    ),
)


@pytest.mark.parametrize(
    ("label", "text", "expected_key"),
    INVALID_SUITES,
    ids=[label for label, _, _ in INVALID_SUITES],
)
def test_invalid_suites_raise_config_error_naming_the_key(
    tmp_path: Path, label: str, text: str, expected_key: str
) -> None:
    """不正なスイートは ``ConfigError`` になり、メッセージに該当キー名が出る。"""
    with pytest.raises(ConfigError) as excinfo:
        load_suite(write_suite(tmp_path, text, name=f"{label}.toml"))

    assert expected_key in str(excinfo.value)


def test_suite_id_rejects_path_separators(tmp_path: Path) -> None:
    """``id`` は ``results/<id>/`` のディレクトリ名になるため区切り文字を弾く。"""
    for bad_id in ("ja/basic", "ja\\\\basic", "..", "JA_BASIC", "ja basic"):
        text = VALID_SUITE.replace('id = "ja_basic_test"', f'id = "{bad_id}"')
        with pytest.raises(ConfigError) as excinfo:
            load_suite(write_suite(tmp_path, text, name="bad.toml"))
        assert "suite.id" in str(excinfo.value)


SUITE_WITHOUT_MODELS = """\
[suite]
id = "empty"

[[prompts]]
id = "p1"
text = "こんにちは"
"""

SUITE_WITHOUT_PROMPTS = """\
[suite]
id = "empty"

[[models]]
model_id = "qwen3-14b"
"""


@pytest.mark.parametrize(
    ("label", "text", "expected_key"),
    [
        ("no_models", SUITE_WITHOUT_MODELS, "models"),
        ("no_prompts", SUITE_WITHOUT_PROMPTS, "prompts"),
    ],
    ids=["no_models", "no_prompts"],
)
def test_empty_models_or_prompts_are_rejected(
    tmp_path: Path, label: str, text: str, expected_key: str
) -> None:
    """1 件も対象が無いスイートは、黙って空の比較結果を出さずに落とす。"""
    with pytest.raises(ConfigError) as excinfo:
        load_suite(write_suite(tmp_path, text, name=f"{label}.toml"))

    assert expected_key in str(excinfo.value)


def test_missing_suite_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_suite(tmp_path / "does_not_exist.toml")

    assert "does_not_exist.toml" in str(excinfo.value)


def test_broken_toml_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_suite(write_suite(tmp_path, "[suite\nid = 'x'", name="broken.toml"))

    assert "TOML" in str(excinfo.value)


# --------------------------------------------------------------------------
# suite_sha256
# --------------------------------------------------------------------------


def test_suite_sha256_is_reproducible_and_content_sensitive(tmp_path: Path) -> None:
    first = write_suite(tmp_path, VALID_SUITE, name="a.toml")
    same = write_suite(tmp_path, VALID_SUITE, name="b.toml")
    changed = write_suite(
        tmp_path,
        VALID_SUITE.replace("次の文を英語に訳してください。", "次の文を仏語に訳して。"),
        name="c.toml",
    )

    assert suite_sha256(first) == suite_sha256(same)
    assert suite_sha256(changed) != suite_sha256(first)
    assert len(suite_sha256(first)) == 64


def test_suite_sha256_on_a_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        suite_sha256(tmp_path / "missing.toml")


# --------------------------------------------------------------------------
# apply_case — D-19 guard
# --------------------------------------------------------------------------

#: (model_id, 期待 served_name, rag_default プロファイルでの見積り合計 GiB)。
#: 合計は llmkit/catalog.py の実測較正値から導かれる:
#:   qwen3-14b   7.81 + 0.57 + 0.28 + 0.1575*16 + 0.8 = 11.98
#:   gpt-oss-20b 11.32 + 0.57 + 0.28 + 0.0244*16 + 0.8 = 13.36
#:   qwen3-8b    4.18 + 0.57 + 0.28 + 0.1425*16 + 0.8 = 8.11
MODEL_CASES: tuple[tuple[str, str, float], ...] = (
    ("qwen3-14b", "qwen3:14b-q4_K_M", 11.98),
    ("gpt-oss-20b", "gpt-oss:20b", 13.36),
    ("qwen3-8b", "qwen3:8b-q4_K_M", 8.11),
)


@pytest.mark.parametrize(
    ("model_id", "served_name", "total_gib"),
    MODEL_CASES,
    ids=[model_id for model_id, _, _ in MODEL_CASES],
)
def test_apply_case_keeps_the_request_model_and_the_profile_model_identical(
    base_config: AppConfig, model_id: str, served_name: str, total_gib: float
) -> None:
    """D-19 guard: モデル ID の 2 か所が常に一致する。

    ``generation.model`` はリクエストの ``model`` に、
    ``profiles[active].generation`` は VRAM 見積りと実行マニフェストに届く。
    片方だけを差し替える実装では ``total_gib`` の掃引が落ちる。
    """
    applied = apply_case(base_config, ModelCase(model_id=model_id))

    assert applied.generation.model == model_id
    assert applied.active_profile().generation == model_id
    assert (
        applied.generation.model
        == applied.profiles[applied.vram.active_profile].generation
    )

    profile = resolve_profile(applied)
    assert profile.model_ids[0] == model_id
    assert profile.generation.served_name == served_name

    estimate = estimate_profile(applied)
    assert model_id in estimate.weights_gib
    assert estimate.total_gib == pytest.approx(total_gib, abs=0.01)


def test_apply_case_moves_the_vram_estimate_off_the_base_model(
    base_config: AppConfig,
) -> None:
    """変異検証の的: プロファイル側を差し替えないと見積りがベース値のまま残る。"""
    baseline = estimate_profile(base_config)
    applied = estimate_profile(
        apply_case(base_config, ModelCase(model_id="gpt-oss-20b"))
    )

    assert baseline.total_gib == pytest.approx(11.98, abs=0.01)
    assert "qwen3-14b" in baseline.weights_gib
    assert "qwen3-14b" not in applied.weights_gib
    assert applied.total_gib != pytest.approx(baseline.total_gib, abs=0.01)


def test_apply_case_switches_the_active_profile(base_config: AppConfig) -> None:
    applied = apply_case(
        base_config, ModelCase(model_id="gpt-oss-20b", profile="long_context")
    )

    assert applied.vram.active_profile == "long_context"
    assert applied.active_profile().generation == "gpt-oss-20b"
    assert resolve_profile(applied).model_ids == ("gpt-oss-20b",)
    # 対象外のプロファイルには触らない。
    assert applied.profiles["rag_default"].generation == "qwen3-14b"


def test_apply_case_keeps_base_values_for_omitted_fields(
    base_config: AppConfig,
) -> None:
    applied = apply_case(base_config, ModelCase(model_id="qwen3-8b"))

    assert applied.generation.context_tokens == base_config.generation.context_tokens
    assert applied.generation.temperature == base_config.generation.temperature
    assert applied.generation.top_p == base_config.generation.top_p
    assert (
        applied.generation.max_output_tokens == base_config.generation.max_output_tokens
    )
    assert applied.generation.seed == base_config.generation.seed
    assert applied.runtime == base_config.runtime
    assert applied.vram.budget_gib == base_config.vram.budget_gib


def test_apply_case_applies_every_generation_override(base_config: AppConfig) -> None:
    applied = apply_case(
        base_config,
        ModelCase(
            model_id="qwen3-8b",
            context_tokens=8192,
            temperature=0.1,
            top_p=0.4,
            max_output_tokens=512,
            seed=42,
        ),
    )

    assert applied.generation.context_tokens == 8192
    assert applied.generation.temperature == pytest.approx(0.1)
    assert applied.generation.top_p == pytest.approx(0.4)
    assert applied.generation.max_output_tokens == 512
    assert applied.generation.seed == 42
    # context_tokens は KV 見積りに効く: 4.18 + 0.57 + 0.28 + 0.1425*8 + 0.8
    assert estimate_profile(applied).total_gib == pytest.approx(6.97, abs=0.01)


def test_apply_case_treats_zero_as_an_override_not_as_unset(
    base_config: AppConfig,
) -> None:
    """``seed = 0`` / ``temperature = 0.0`` は falsy だが正当な上書き値。"""
    changed_base = apply_case(base_config, ModelCase(model_id="qwen3-14b", seed=99))
    applied = apply_case(
        changed_base, ModelCase(model_id="qwen3-14b", seed=0, temperature=0.0)
    )

    assert applied.generation.seed == 0
    assert applied.generation.temperature == pytest.approx(0.0)


def test_apply_case_does_not_mutate_the_base_config(base_config: AppConfig) -> None:
    """ベース設定は共有される。適用が破壊的だと 2 モデル目以降が汚れる。"""
    before_model = base_config.generation.model
    before_profile = base_config.profiles["rag_default"].generation
    before_active = base_config.vram.active_profile

    apply_case(base_config, ModelCase(model_id="gpt-oss-20b", profile="lightweight"))

    assert base_config.generation.model == before_model
    assert base_config.profiles["rag_default"].generation == before_profile
    assert base_config.vram.active_profile == before_active
    assert base_config.profiles["lightweight"].generation == "qwen3-8b"


def test_apply_case_rejects_a_profile_missing_from_the_base_config(
    base_config: AppConfig,
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        apply_case(base_config, ModelCase(model_id="qwen3-8b", profile="nope"))

    message = str(excinfo.value)
    assert "models[].profile" in message
    assert "nope" in message


# --------------------------------------------------------------------------
# 層の境界
# --------------------------------------------------------------------------


def test_suite_module_imports_only_the_llmkit_public_api() -> None:
    """``harness.suite`` は ``llmkit`` のサブモジュールを直接 import しない。

    L3 が使ってよいのは ``llmkit`` が再エクスポートする公開シンボルだけである
    (``from llmkit.config import AppConfig`` のような直接参照を許すと、層の境界が
    散文の主張に戻る)。
    """
    tree = ast.parse(SUITE_MODULE.read_text(encoding="utf-8"))
    offenders = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("llmkit.")
    ]
    offenders += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("llmkit.")
    ]

    assert not offenders, f"llmkit サブモジュールの直接 import: {offenders}"
