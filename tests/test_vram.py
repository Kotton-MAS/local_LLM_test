"""VRAM プロファイル見積り (llmkit/vram.py) のテスト。

このファイルは 3 つの guard_test を含む:

- ``test_estimate_is_pure_and_needs_no_gpu`` … D-01 (静的テーブルのみで見積もる)
- ``test_all_vram_values_are_gib``           … D-03 (単位は GiB に統一する)
- ``test_estimate_reproduces_the_configuration1_coresidency_measurement`` … D-16
  (別プロセスが確保する分も合算し、構成1 の同居実測を再現する)
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import socket
import subprocess
from pathlib import Path

import pytest
from conftest import write_config_variant

from llmkit.catalog import ModelSpec, get_model_spec
from llmkit.config import AppConfig, GenerationParams, VramConfig, load_config
from llmkit.errors import ConfigError, VramBudgetExceededError
from llmkit.vram import (
    ResolvedProfile,
    VramEstimate,
    check_budget,
    estimate_profile,
    estimate_resolved_profile,
    resolve_profile,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"
EXTERNAL_CONFIG = REPO_ROOT / "configs" / "external_openai.toml"

#: 較正後の出荷プロファイル見積り (GiB)。出典は docs/phase0-vram-measurements.md
#: の実測フィット値であり、要件書の帯ではない。
#: (プロファイル名, context_tokens, 合計 GiB, 既定予算 14.0 に収まるか)
SHIPPED_PROFILE_ESTIMATES: tuple[tuple[str, int, float, bool], ...] = (
    ("rag_default", 16384, 11.98, True),
    ("long_context", 65536, 13.68, True),
    ("long_context", 131072, 15.24, False),
    ("oversized", 131072, 16.09, False),
)

#: 実測でオフロードが始まらない増分の上限として設定した既定予算 (D-12)。
DEFAULT_BUDGET_GIB = 14.0

# テスト専用のダミー。予算ガード本体の検証をカタログ値のキャリブレーションから切り離す。
HUGE_MODEL = ModelSpec(
    model_id="test-huge-20gib",
    served_name="test-huge:20gib",
    role="generation",
    serving_runtime="ollama",
    quantization="none",
    weights_gib=20.0,
    kv_gib_per_1k_tokens=0.0,
    max_context_tokens=4096,
    source_note="テスト専用のダミー。カタログには登録しない",
)
SYNTHETIC_PROFILE = ResolvedProfile(name="synthetic_oversized", models=(HUGE_MODEL,))


@pytest.fixture
def config() -> AppConfig:
    return load_config(DEFAULT_CONFIG)


def with_vram(
    config: AppConfig,
    *,
    budget_gib: float | None = None,
    runtime_overhead_gib: float | None = None,
) -> AppConfig:
    """``[vram]`` の一部だけを差し替えた設定を返す (他は同一)。"""
    vram = VramConfig(
        budget_gib=config.vram.budget_gib if budget_gib is None else budget_gib,
        runtime_overhead_gib=(
            config.vram.runtime_overhead_gib
            if runtime_overhead_gib is None
            else runtime_overhead_gib
        ),
        active_profile=config.vram.active_profile,
    )
    return dataclasses.replace(config, vram=vram)


def with_context_tokens(config: AppConfig, context_tokens: int) -> AppConfig:
    return dataclasses.replace(
        config,
        generation=dataclasses.replace(
            config.generation, context_tokens=context_tokens
        ),
    )


# --------------------------------------------------------------------------
# プロファイル解決
# --------------------------------------------------------------------------


def test_resolve_profile_defaults_to_active_profile(config: AppConfig) -> None:
    profile = resolve_profile(config)

    assert profile.name == "rag_default"
    assert profile.model_ids == ("qwen3-14b", "ruri-v3-310m", "bge-reranker-v2-m3")
    assert profile.generation.model_id == "qwen3-14b"


def test_resolve_unknown_profile_raises_config_error(config: AppConfig) -> None:
    with pytest.raises(ConfigError) as excinfo:
        resolve_profile(config, "no_such_profile")

    assert "no_such_profile" in str(excinfo.value)


def test_resolve_profile_raises_config_error_for_unregistered_model_when_local(
    config: AppConfig,
) -> None:
    """カタログ未登録モデルは is_local=true のプロファイルでは通さない (D-01)。"""
    local_external = dataclasses.replace(
        config,
        generation=dataclasses.replace(config.generation, model="gpt-4o-mini"),
        profiles={
            **config.profiles,
            "rag_default": dataclasses.replace(
                config.profiles["rag_default"], generation="gpt-4o-mini"
            ),
        },
    )

    with pytest.raises(ConfigError) as excinfo:
        resolve_profile(local_external, "rag_default")

    assert "gpt-4o-mini" in str(excinfo.value)


def test_resolve_profile_allows_passthrough_model_for_remote_runtime(
    config: AppConfig,
) -> None:
    """F-1-001: is_local=false ならカタログ未登録モデルでも解決できる。

    見積りには寄与しないため (weights_gib=kv_gib_per_1k_tokens=0.0)、合計は
    embedding/reranker の重みと overhead のみになる。
    """
    remote_external = dataclasses.replace(
        config,
        runtime=dataclasses.replace(config.runtime, is_local=False),
        generation=dataclasses.replace(config.generation, model="gpt-4o-mini"),
        profiles={
            **config.profiles,
            "rag_default": dataclasses.replace(
                config.profiles["rag_default"], generation="gpt-4o-mini"
            ),
        },
    )

    profile = resolve_profile(remote_external, "rag_default")

    assert profile.generation.model_id == "gpt-4o-mini"
    assert profile.generation.served_name == "gpt-4o-mini"
    assert profile.generation.weights_gib == pytest.approx(0.0)

    estimate = estimate_resolved_profile(profile, remote_external, context_tokens=16384)
    expected_embedding_reranker = get_model_spec("ruri-v3-310m").weights_gib + (
        get_model_spec("bge-reranker-v2-m3").weights_gib
    )
    assert estimate.weights_total_gib == pytest.approx(expected_embedding_reranker)
    assert estimate.kv_cache_gib == pytest.approx(0.0)


def test_resolve_profile_marks_catalog_models_external_for_shipped_external_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-6-001 掃引: 出荷済み外部 API 構成 (``configs/external_openai.toml``) は
    カタログ登録済みモデルであっても ``serving_runtime`` を ``"external"`` に
    差し替える。

    ``rag_default`` の生成・埋め込み・リランカーはいずれもカタログ登録済み
    (``qwen3-14b`` / ``ruri-v3-310m`` / ``bge-reranker-v2-m3``) であり、
    どれか 1 つでもカタログ値 (``ollama`` / ``llama_cpp_server``) のまま
    漏れるとこのテストが落ちる。``weights_gib`` はカタログ値のまま維持され、
    VRAM 見積り (``test_estimate_follows_the_documented_formula`` 等) に
    影響しないことも併せて固定する。
    """
    monkeypatch.setenv("LLMKIT_API_KEY", "sk-test-do-not-leak-0123456789")
    external_config = load_config(EXTERNAL_CONFIG)
    profile = resolve_profile(external_config)

    assert profile.model_ids == ("qwen3-14b", "ruri-v3-310m", "bge-reranker-v2-m3")
    assert {spec.serving_runtime for spec in profile.models} == {"external"}

    for spec in profile.models:
        catalog_spec = get_model_spec(spec.model_id)
        assert spec.weights_gib == pytest.approx(catalog_spec.weights_gib)
        assert spec.kv_gib_per_1k_tokens == pytest.approx(
            catalog_spec.kv_gib_per_1k_tokens
        )


# --------------------------------------------------------------------------
# 見積り式
# --------------------------------------------------------------------------


def test_estimate_follows_the_documented_formula(config: AppConfig) -> None:
    """weights の総和 + KV + overhead という式そのものを固定する。"""
    estimate = estimate_profile(config, "rag_default", context_tokens=16384)

    expected_weights = sum(
        get_model_spec(model_id).weights_gib
        for model_id in ("qwen3-14b", "ruri-v3-310m", "bge-reranker-v2-m3")
    )
    expected_kv = get_model_spec("qwen3-14b").kv_gib_per_1k_tokens * (16384 / 1024)

    assert estimate.weights_total_gib == pytest.approx(expected_weights)
    assert estimate.kv_cache_gib == pytest.approx(expected_kv)
    assert estimate.runtime_overhead_gib == pytest.approx(0.8)
    assert estimate.total_gib == pytest.approx(expected_weights + expected_kv + 0.8)
    assert estimate.weights_gib == {
        "qwen3-14b": pytest.approx(7.81),
        "ruri-v3-310m": pytest.approx(0.57),
        "bge-reranker-v2-m3": pytest.approx(0.28),
    }


@pytest.mark.parametrize(
    ("profile_name", "context_tokens", "expected_total_gib", "expected_within_budget"),
    SHIPPED_PROFILE_ESTIMATES,
    ids=[f"{name}@{tokens}" for name, tokens, _, _ in SHIPPED_PROFILE_ESTIMATES],
)
def test_shipped_profiles_match_phase0_measurements(
    config: AppConfig,
    profile_name: str,
    context_tokens: int,
    expected_total_gib: float,
    expected_within_budget: bool,
) -> None:
    """E7: 出荷プロファイルの見積り合計を実測由来の値で固定する。

    出典は要件書の帯ではなく docs/phase0-vram-measurements.md の実測フィット
    (``long_context @32768 = 12.90`` と ``@65536 = 13.68`` は実測値と一致する)。
    """
    estimate = estimate_profile(config, profile_name, context_tokens=context_tokens)

    assert estimate.budget_gib == pytest.approx(DEFAULT_BUDGET_GIB)
    assert estimate.total_gib == pytest.approx(expected_total_gib, abs=0.005)
    assert estimate.within_budget is expected_within_budget


def test_context_tokens_change_estimate(config: AppConfig) -> None:
    """E2: context_tokens を増やすと見積りが単調増加する (増分が 0 でない)。"""
    contexts = (4096, 8192, 16384, 32768, 65536, 131072)
    totals = [
        estimate_profile(config, "rag_default", context_tokens=ctx).total_gib
        for ctx in contexts
    ]

    for smaller, larger in itertools.pairwise(totals):
        assert larger > smaller

    small = estimate_profile(config, "rag_default", context_tokens=4096)
    large = estimate_profile(config, "rag_default", context_tokens=131072)
    kv_rate = get_model_spec("qwen3-14b").kv_gib_per_1k_tokens
    assert large.total_gib - small.total_gib == pytest.approx(
        kv_rate * ((131072 - 4096) / 1024)
    )


def test_context_tokens_from_config_is_used_when_not_overridden(
    config: AppConfig,
) -> None:
    """generation.context_tokens が既定値として見積りに配線されていること。"""
    changed = with_context_tokens(config, 32768)

    assert estimate_profile(changed, "rag_default").context_tokens == 32768
    assert (
        estimate_profile(changed, "rag_default").total_gib
        > estimate_profile(config, "rag_default").total_gib
    )


def test_overhead_changes_estimate(config: AppConfig) -> None:
    """E5: runtime_overhead_gib を 0.8 -> 2.0 にすると合計が厳密に 1.2 増える。"""
    baseline = estimate_profile(config, "rag_default", context_tokens=16384)
    raised = estimate_profile(
        with_vram(config, runtime_overhead_gib=2.0), "rag_default", context_tokens=16384
    )

    assert baseline.runtime_overhead_gib == pytest.approx(0.8)
    assert raised.runtime_overhead_gib == pytest.approx(2.0)
    assert raised.total_gib - baseline.total_gib == pytest.approx(1.2)


# --------------------------------------------------------------------------
# 構成1 の同居実測 (D-16 guard / E14)
# --------------------------------------------------------------------------

#: docs/phase0-vram-measurements.md「構成1 の同居実測 (2026-08-23)」の 4 段階。
#: いずれも nvidia-smi が返すデバイス全体の使用量 (GiB 累計) であり、
#: プロセス単位の値ではない。
MEASURED_IDLE_GIB = 0.56
MEASURED_PLUS_RERANKER_GIB = 0.84
MEASURED_PLUS_GENERATION_GIB = 11.96
MEASURED_PLUS_EMBEDDING_GIB = 12.53

#: 予算判定の対象となる「増分」。アイドル分は含めない (D-12)。
MEASURED_INCREMENT_GIB = MEASURED_PLUS_EMBEDDING_GIB - MEASURED_IDLE_GIB


def test_estimate_reproduces_the_configuration1_coresidency_measurement(
    config: AppConfig,
) -> None:
    """D-16 guard / E14: 構成1 の同居実測 (2026-08-23) を見積りが再現する。

    - アイドル 0.56 GiB は増分予算に含めない (D-12: budget_gib は「実測で
      オフロードが始まらない**増分**の上限」)。
    - リランカーは Ollama ではなく llama-server (別プロセス) が確保するが、
      判定対象は nvidia-smi が返すデバイス全体の使用量でプロセス境界と無関係
      なため、単純に合算する (D-16)。``llmkit/vram.py`` の式は変更しない。
    - 埋め込み・リランカーの ``weights_gib`` は純粋な重みではなく、自ランタイムの
      オーバーヘッドを含む実測 VRAM 増分そのもの。片方でも旧仮値 (0.7 / 0.8) に
      戻すと、合計と成分の両方でこのテストが落ちる。
    """
    reranker_increment = MEASURED_PLUS_RERANKER_GIB - MEASURED_IDLE_GIB
    generation_increment = MEASURED_PLUS_GENERATION_GIB - MEASURED_PLUS_RERANKER_GIB
    embedding_increment = MEASURED_PLUS_EMBEDDING_GIB - MEASURED_PLUS_GENERATION_GIB

    # 実測 4 段が自己整合している (段ごとの増分の和 = 全体の増分)
    assert reranker_increment + generation_increment + embedding_increment == (
        pytest.approx(MEASURED_INCREMENT_GIB)
    )

    estimate = estimate_profile(config, "rag_default", context_tokens=16384)

    # 合計: 予測 11.98 vs 実測増分 11.97
    assert estimate.total_gib == pytest.approx(MEASURED_INCREMENT_GIB, abs=0.02)

    # 成分ごとにも一致する (合計だけ合わせた偶然の一致ではない)
    assert estimate.weights_gib["bge-reranker-v2-m3"] == pytest.approx(
        reranker_increment, abs=0.005
    )
    assert estimate.weights_gib["ruri-v3-310m"] == pytest.approx(
        embedding_increment, abs=0.005
    )
    generation_total = (
        estimate.weights_gib["qwen3-14b"]
        + estimate.kv_cache_gib
        + estimate.runtime_overhead_gib
    )
    assert generation_total == pytest.approx(generation_increment, abs=0.02)


# --------------------------------------------------------------------------
# 予算判定
# --------------------------------------------------------------------------


def test_check_budget_returns_estimate_when_within_budget(config: AppConfig) -> None:
    estimate = check_budget(resolve_profile(config, "rag_default"), config)

    assert estimate.within_budget
    assert estimate.excess_gib == pytest.approx(0.0)


def test_synthetic_oversized_profile_raises_with_full_breakdown(
    config: AppConfig,
) -> None:
    """予算ガード本体の検証。カタログのキャリブレーションに依存させない。"""
    with pytest.raises(VramBudgetExceededError) as excinfo:
        check_budget(SYNTHETIC_PROFILE, config, context_tokens=16384)

    error = excinfo.value
    expected_total = 20.0 + 0.8
    assert error.profile_name == "synthetic_oversized"
    assert error.total_gib == pytest.approx(expected_total)
    assert error.budget_gib == pytest.approx(14.0)
    assert error.excess_gib == pytest.approx(expected_total - 14.0)
    assert error.kv_cache_gib == pytest.approx(0.0)
    assert error.runtime_overhead_gib == pytest.approx(0.8)
    assert error.weights_gib == {"test-huge-20gib": pytest.approx(20.0)}

    message = str(error)
    assert "synthetic_oversized" in message  # プロファイル名
    assert "20.80" in message  # 合計値
    assert "14.00" in message  # 予算値
    assert "6.80" in message  # 超過量


def test_budget_threshold_changes_verdict(config: AppConfig) -> None:
    """E4: 同一プロファイル・同一カタログで budget_gib だけを変えると判定が反転する。

    較正後の rag_default @16384 の見積りは 11.98 GiB (= 8.66 + 2.52 + 0.8)。
    既定予算 14.0 では収まり、見積り値の直下 11.5 に下げると 0.48 GiB 超過する。
    """
    profile = resolve_profile(config, "rag_default")
    baseline = estimate_resolved_profile(profile, config, context_tokens=16384)
    assert baseline.total_gib == pytest.approx(11.98)

    generous = with_vram(config, budget_gib=14.0)
    strict = with_vram(config, budget_gib=11.5)

    assert check_budget(profile, generous, context_tokens=16384).within_budget

    with pytest.raises(VramBudgetExceededError) as excinfo:
        check_budget(profile, strict, context_tokens=16384)

    assert excinfo.value.excess_gib == pytest.approx(0.48)


def test_budget_reflects_the_measured_gpu_resident_ceiling(config: AppConfig) -> None:
    """D-12 guard: 既定予算が「実測でオフロードが始まらない増分の上限」である。

    実測 (docs/phase0-vram-measurements.md) では gpt-oss-20b 単体は num_ctx=65,536
    (合計 13.68 GiB) まで 100% GPU、num_ctx=98,304 (見積り 14.46 GiB) で CPU
    オフロードが発生した。GPU 常駐上限はモデル単位の定数ではなく
    ``vram.budget_gib`` の予算判定で表現する。
    """
    profile = resolve_profile(config, "long_context")

    assert config.vram.budget_gib == pytest.approx(DEFAULT_BUDGET_GIB)

    # 100% GPU だった点は通る
    resident = check_budget(profile, config, context_tokens=65536)
    assert resident.total_gib == pytest.approx(13.68, abs=0.005)
    assert resident.within_budget

    # オフロードが観測された点、およびそれより大きい点は停止側になる
    for context_tokens, expected_total_gib in ((98304, 14.46), (131072, 15.24)):
        with pytest.raises(VramBudgetExceededError) as excinfo:
            check_budget(profile, config, context_tokens=context_tokens)
        assert excinfo.value.total_gib == pytest.approx(expected_total_gib, abs=0.005)


def test_budget_check_is_skipped_for_remote_runtimes(config: AppConfig) -> None:
    """runtime.is_local=false なら超過プロファイルでも例外を出さない。"""
    remote = dataclasses.replace(
        config, runtime=dataclasses.replace(config.runtime, is_local=False)
    )

    estimate = check_budget(SYNTHETIC_PROFILE, remote, context_tokens=16384)

    assert estimate.total_gib > estimate.budget_gib
    assert estimate.within_budget is False


# --------------------------------------------------------------------------
# guard_test (D-01 / D-03)
# --------------------------------------------------------------------------


def test_estimate_is_pure_and_needs_no_gpu(
    config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-01: 見積りは静的テーブルのみで行い、GPU・外部プロセス・通信に触れない。"""
    forbidden_modules = frozenset(
        {
            "subprocess",
            "socket",
            "httpx",
            "requests",
            "pynvml",
            "torch",
            "urllib",
            "urllib.request",
            "os",
            "shutil",
        }
    )
    for module_name in ("vram", "catalog"):
        tree = ast.parse(
            (REPO_ROOT / "llmkit" / f"{module_name}.py").read_text(encoding="utf-8")
        )
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        clash = imported & forbidden_modules
        assert not clash, (
            f"llmkit/{module_name}.py が {sorted(clash)} を import している"
        )

    def explode(*args: object, **kwargs: object) -> object:
        message = "見積りが外部リソースに触れた"
        raise AssertionError(message)

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(subprocess, "Popen", explode)
    monkeypatch.setattr(socket, "socket", explode)
    monkeypatch.setattr(socket, "create_connection", explode)

    first = estimate_profile(config, "rag_default", context_tokens=16384)
    second = estimate_profile(config, "rag_default", context_tokens=16384)

    assert first.total_gib == second.total_gib
    assert first == second


def test_all_vram_values_are_gib(tmp_path: Path) -> None:
    """D-03: VRAM に関する数値は GiB に統一し、GB / MB / MiB を混在させない。"""
    non_gib_units = ("_gb", "_mb", "_mib", "_kb", "_kib", "byte", "gigabyte")
    float_fields = {
        ModelSpec: {"weights_gib", "kv_gib_per_1k_tokens"},
        VramConfig: {"budget_gib", "runtime_overhead_gib"},
        VramEstimate: {
            "weights_total_gib",
            "kv_cache_gib",
            "runtime_overhead_gib",
            "total_gib",
            "budget_gib",
        },
    }

    for cls, expected_float_fields in float_fields.items():
        names = {field.name for field in dataclasses.fields(cls)}
        assert expected_float_fields <= names, cls
        for name in names:
            assert not any(unit in name for unit in non_gib_units), (cls, name)
        for name in expected_float_fields:
            assert "gib" in name, (cls, name)

    # 派生値も GiB
    assert "gib" in "excess_gib"
    estimate = estimate_profile(load_config(DEFAULT_CONFIG), "rag_default")
    assert isinstance(estimate.excess_gib, float)

    # 要件書の「16GB」はコード上 16.0 GiB として読める (GB へ換算しない)
    written_as_16 = write_config_variant(
        tmp_path, {"budget_gib = 14.0": "budget_gib = 16.0"}
    )
    assert load_config(written_as_16).vram.budget_gib == pytest.approx(16.0)

    # 出荷設定の予算はカード容量 16376 MiB = 15.99 GiB を超えない。
    # GB と取り違えて 17.17 を書くとここで落ちる。
    assert load_config(DEFAULT_CONFIG).vram.budget_gib <= 15.99

    # 生成パラメータ側にメモリ単位のフィールドを紛れ込ませない
    for field in dataclasses.fields(GenerationParams):
        assert not any(unit in field.name for unit in non_gib_units)
