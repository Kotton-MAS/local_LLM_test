"""モデルカタログ (llmkit/catalog.py) のテスト。

カタログは VRAM 見積りの唯一の出典 (D-01) であり、数値はゴールデン値として
ここでロックする。数値を書き換えると ``test_catalog_reproduces_phase0_measurements``
(Phase 0 実測 8 点の再現) と tests/test_vram.py のプロファイル値テストも落ちる。

このファイルは 1 つの guard_test を含む:

- ``test_unavailable_models_declare_their_unavailability`` … D-14
  (取得不能なモデルは仮値である旨と理由を source_note に明記する)
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
    REPO_ROOT / "configs" / "ollama_openai_compat.toml",
)

# 仕様書 (較正版) §4 T1「カタログを Phase 0 実測値で較正する」表と 1:1 で
# 対応するゴールデン値。生成 3 モデルは実測フィット値、ruri 2 モデルは未実測の仮値。
GOLDEN_ROWS: tuple[tuple[str, str, str, float, float], ...] = (
    ("qwen3-14b", "generation", "Q4_K", 7.81, 0.1575),
    ("gpt-oss-20b", "generation", "MXFP4", 11.32, 0.0244),
    ("qwen3-8b", "generation", "Q4_K", 4.18, 0.1425),
    ("ruri-v3-310m", "embedding", "fp16", 0.7, 0.0),
    ("ruri-reranker", "reranker", "fp16", 0.8, 0.0),
)

#: docs/phase0-vram-measurements.md の実測表のうち 100% GPU に収まった 8 点。
#: (model_id, num_ctx, VRAM 増分 GiB)。増分は「重み + KV + オーバーヘッド」の合計。
PHASE0_MEASUREMENTS: tuple[tuple[str, int, float], ...] = (
    ("qwen3-14b", 4096, 9.24),
    ("qwen3-14b", 16384, 11.13),
    ("qwen3-14b", 32768, 13.65),
    ("qwen3-8b", 8192, 6.12),
    ("qwen3-8b", 16384, 7.26),
    ("qwen3-8b", 32768, 9.54),
    ("gpt-oss-20b", 32768, 12.90),
    ("gpt-oss-20b", 65536, 13.68),
)

#: 実測は合計値のみのため、重みと分離するために置いたオーバーヘッドの仮定
#: (configs/*.toml の vram.runtime_overhead_gib と同じ値)。
FIT_OVERHEAD_GIB = 0.8

#: 較正済み (実測フィット) のモデルと、取得不能で未実測のままのモデル。
CALIBRATED_MODEL_IDS = ("qwen3-14b", "gpt-oss-20b", "qwen3-8b")
UNAVAILABLE_MODEL_IDS = ("ruri-v3-310m", "ruri-reranker")


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
    assert spec.source_note


@pytest.mark.parametrize("model_id", CALIBRATED_MODEL_IDS, ids=CALIBRATED_MODEL_IDS)
def test_calibrated_models_cite_the_phase0_measurements(model_id: str) -> None:
    """較正済みモデルは実測日と出典を書き、仮値マーカーを残さない。"""
    source_note = get_model_spec(model_id).source_note

    assert "実測 2026-08-22" in source_note
    assert "docs/phase0-vram-measurements.md" in source_note
    assert "(仮)" not in source_note


@pytest.mark.parametrize("model_id", UNAVAILABLE_MODEL_IDS, ids=UNAVAILABLE_MODEL_IDS)
def test_unavailable_models_declare_their_unavailability(model_id: str) -> None:
    """D-14 guard: 取得できなかったモデルは仮値であることと理由を明示する。

    Phase 0 で ruri 2 モデルは GGUF 非提供のため ``ollama pull`` できなかった。
    カタログからは消さず (プロファイル定義と要件書の対応が切れるため)、較正済みの
    3 モデルと同じ体裁で置かないことをここで固定する。
    """
    source_note = get_model_spec(model_id).source_note

    assert "(仮)" in source_note
    assert "取得不能" in source_note
    assert "Phase 3" in source_note
    assert "実測 2026-08-22" not in source_note


@pytest.mark.parametrize(
    ("model_id", "context_tokens", "measured_gib"),
    PHASE0_MEASUREMENTS,
    ids=[f"{model_id}@{tokens}" for model_id, tokens, _ in PHASE0_MEASUREMENTS],
)
def test_catalog_reproduces_phase0_measurements(
    model_id: str, context_tokens: int, measured_gib: float
) -> None:
    """E11: カタログ値が docs/phase0-vram-measurements.md の実測 8 点を再現する。

    ``weights_gib`` か ``kv_gib_per_1k_tokens`` を書き換えると、この再現テストと
    ``GOLDEN_ROWS`` の両方が落ちる (較正値が実測から静かに乖離しないため)。
    """
    spec = get_model_spec(model_id)

    estimated_gib = (
        spec.weights_gib
        + spec.kv_gib_per_1k_tokens * (context_tokens / 1024)
        + FIT_OVERHEAD_GIB
    )

    assert estimated_gib == pytest.approx(measured_gib, abs=0.02)


def test_context_limits_match_phase0_measurements() -> None:
    """実測で 100% GPU が確認できた範囲を最大コンテキストとして持つ。

    ``qwen3-14b`` は 32,768 まで実測済み (要件書 L86-L92 の「16k」より広い)。
    ``gpt-oss-20b`` はモデル自体の上限 131,072 を持つが、GPU 常駐上限は
    65,536〜98,303 の間であり、そちらは vram.budget_gib で表現する (D-12)。
    """
    assert get_model_spec("qwen3-14b").max_context_tokens == 32768
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


def test_shipped_configs_never_contain_an_api_key_value() -> None:
    """出荷する設定ファイルに api_key の「値」を書かない (D-05)。

    ``configs/*.toml`` はコミット対象であり、環境変数「名」だけを
    ``api_key_env`` に書く。設定ファイルを 1 つ増やしたときに検査から漏れない
    よう、判定は ``CONFIG_PATHS`` (出荷設定の一覧) をそのまま回す。
    """
    for path in CONFIG_PATHS:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        assert "api_key" not in raw, path
        assert "api_key" not in raw["runtime"], path
        assert raw["runtime"]["api_key_env"] == "LLMKIT_API_KEY", path


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
