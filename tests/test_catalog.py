"""モデルカタログ (llmkit/catalog.py) のテスト。

カタログは VRAM 見積りの唯一の出典 (D-01) であり、数値はゴールデン値として
ここでロックする。数値を書き換えると tests/test_vram.py の帯テストも落ちる。
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from types import MappingProxyType

import pytest

from llmkit.catalog import (
    MODEL_CATALOG,
    ModelRole,
    ModelSpec,
    get_model_spec,
    known_model_ids,
    resolve_model_spec,
)
from llmkit.errors import ConfigError

_ROLES: tuple[ModelRole, ...] = ("generation", "embedding", "reranker")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATHS = (
    REPO_ROOT / "configs" / "default.toml",
    REPO_ROOT / "configs" / "external_openai.toml",
)

# 仕様書 §4 T2「カタログ初期値」表と 1:1 で対応するゴールデン値。
GOLDEN_ROWS: tuple[tuple[str, str, str, float, float], ...] = (
    ("qwen3-14b", "generation", "Q4_K", 9.0, 0.10),
    ("gpt-oss-20b", "generation", "MXFP4", 11.5, 0.004),
    ("qwen3-8b", "generation", "Q4_K", 4.5, 0.08),
    ("ruri-v3-310m", "embedding", "fp16", 0.7, 0.0),
    ("ruri-reranker", "reranker", "fp16", 0.8, 0.0),
)


def test_catalog_contains_exactly_the_documented_models() -> None:
    assert known_model_ids() == tuple(sorted(row[0] for row in GOLDEN_ROWS))


@pytest.mark.parametrize(
    ("model_id", "role", "quantization", "weights_gib", "kv_gib_per_1k_tokens"),
    GOLDEN_ROWS,
    ids=[row[0] for row in GOLDEN_ROWS],
)
def test_catalog_golden_values_are_locked(
    model_id: str,
    role: str,
    quantization: str,
    weights_gib: float,
    kv_gib_per_1k_tokens: float,
) -> None:
    spec = get_model_spec(model_id)

    assert spec.role == role
    assert spec.quantization == quantization
    assert spec.weights_gib == pytest.approx(weights_gib)
    assert spec.kv_gib_per_1k_tokens == pytest.approx(kv_gib_per_1k_tokens)


@pytest.mark.parametrize("model_id", [row[0] for row in GOLDEN_ROWS])
def test_every_spec_declares_a_source_note_and_served_name(model_id: str) -> None:
    spec = get_model_spec(model_id)

    assert spec.served_name
    assert spec.max_context_tokens > 0
    # Phase 0 実測前であることが読み取れること (仕様書 §4 T2 / リスク2)
    assert "(仮)" in spec.source_note


def test_documented_context_limits_match_requirements() -> None:
    """要件書 L86-L92「Qwen3-14B は 16k」「gpt-oss-20B は 128k」の再現。"""
    assert get_model_spec("qwen3-14b").max_context_tokens == 16384
    assert get_model_spec("gpt-oss-20b").max_context_tokens == 131072


def test_get_model_spec_raises_config_error_naming_the_unknown_id() -> None:
    with pytest.raises(ConfigError) as excinfo:
        get_model_spec("does-not-exist")

    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "qwen3-14b" in message


def test_model_spec_is_frozen() -> None:
    spec = get_model_spec("qwen3-14b")
    attribute = "weights_gib"

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(spec, attribute, 99.0)


def test_catalog_mapping_is_read_only() -> None:
    """カタログを実行時に書き換えられないこと (見積りの再現性の前提)。"""
    assert isinstance(MODEL_CATALOG, MappingProxyType)


def test_every_model_referenced_by_shipped_configs_exists() -> None:
    """configs/*.toml が参照するモデルがすべてカタログにあること。"""
    for path in CONFIG_PATHS:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        referenced = {raw["generation"]["model"]}
        for profile in raw["profiles"].values():
            referenced.update(
                value for key, value in profile.items() if isinstance(value, str)
            )
        assert referenced <= set(known_model_ids()), path


def test_model_spec_declares_all_documented_fields() -> None:
    field_names = {field.name for field in dataclasses.fields(ModelSpec)}

    assert field_names == {
        "model_id",
        "served_name",
        "role",
        "quantization",
        "weights_gib",
        "kv_gib_per_1k_tokens",
        "max_context_tokens",
        "source_note",
    }


# --------------------------------------------------------------------------
# resolve_model_spec (F-1-001): 外部 API 向けの passthrough 解決
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "is_local", [True, False], ids=["is_local=true", "is_local=false"]
)
def test_resolve_model_spec_returns_catalog_entry_when_registered(
    is_local: bool,
) -> None:
    """カタログ登録済みモデルは is_local に関わらずカタログの値をそのまま返す。"""
    spec = resolve_model_spec("qwen3-14b", is_local=is_local)

    assert spec is get_model_spec("qwen3-14b")


def test_resolve_model_spec_raises_config_error_for_unregistered_model_when_local() -> (
    None
):
    """ローカル実行では未登録モデルを passthrough で通さない (D-01 の前提を守る)。"""
    with pytest.raises(ConfigError) as excinfo:
        resolve_model_spec("gpt-4o-mini", is_local=True)

    message = str(excinfo.value)
    assert "gpt-4o-mini" in message
    assert "qwen3-14b" in message


def test_resolve_model_spec_returns_passthrough_for_unregistered_remote_model() -> None:
    """外部 API (is_local=false) では未登録モデルでも ConfigError にならない。

    要件書 L144 の3項目のうち「ローカルと外部 API の切り替え」軸の再現
    (F-1-001)。「ランタイムの差し替え (vLLM 等)」軸は対象外 (D-09、
    ``test_resolve_model_spec_raises_config_error_for_unregistered_model_when_local``
    参照)。VRAM 見積りに寄与しないよう weights_gib/kv は 0.0 にする。
    """
    spec = resolve_model_spec("gpt-4o-mini", is_local=False)

    assert spec.model_id == "gpt-4o-mini"
    assert spec.served_name == "gpt-4o-mini"
    assert spec.role == "generation"
    assert spec.weights_gib == pytest.approx(0.0)
    assert spec.kv_gib_per_1k_tokens == pytest.approx(0.0)
    assert spec.max_context_tokens > 0
    assert "passthrough" in spec.source_note
    assert "gpt-4o-mini" not in known_model_ids()  # カタログ自体は書き換わらない


@pytest.mark.parametrize("role", _ROLES, ids=_ROLES)
def test_resolve_model_spec_passthrough_respects_requested_role(
    role: ModelRole,
) -> None:
    spec = resolve_model_spec("external-only-model", is_local=False, role=role)

    assert spec.role == role


def test_resolve_model_spec_passthrough_is_frozen_like_catalog_entries() -> None:
    spec = resolve_model_spec("gpt-4o-mini", is_local=False)
    attribute = "weights_gib"

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(spec, attribute, 99.0)
