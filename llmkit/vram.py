"""VRAM プロファイルの解決・見積り・予算判定。

見積りは **純関数** であり、GPU・``nvidia-smi``・ネットワーク・ファイルに一切
触れない (D-01)。判定はモデルをロードする「前」に行う必要があるため、実測ベースでは
原理的に成立しない。単位はすべて GiB (D-03)。

見積り式 (仕様書 §4 T2)::

    estimate = Σ(model.weights_gib for model in profile.models)
             + generation.kv_gib_per_1k_tokens * (context_tokens / 1024)
             + vram.runtime_overhead_gib
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from llmkit.catalog import ModelRole, ModelSpec, resolve_model_spec
from llmkit.config import AppConfig
from llmkit.errors import ConfigError, VramBudgetExceededError

logger = logging.getLogger(__name__)

__all__ = [
    "ResolvedProfile",
    "VramEstimate",
    "check_budget",
    "estimate_profile",
    "estimate_resolved_profile",
    "resolve_profile",
]

_TOKENS_PER_KILO = 1024.0


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    """プロファイル名を :class:`ModelSpec` の並びに解決した結果。"""

    name: str
    models: tuple[ModelSpec, ...]

    @property
    def generation(self) -> ModelSpec:
        """生成モデル。KV キャッシュはこのモデルの値で見積もる。"""
        for spec in self.models:
            if spec.role == "generation":
                return spec
        msg = f"プロファイル '{self.name}' に生成モデルが含まれていません"
        raise ConfigError(
            msg, remediation="[profiles.*] の generation にモデル ID を指定してください"
        )

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(spec.model_id for spec in self.models)


@dataclass(frozen=True, slots=True)
class VramEstimate:
    """VRAM 使用量の見積り内訳。値はすべて GiB (D-03)。"""

    profile_name: str
    context_tokens: int
    weights_gib: Mapping[str, float]
    weights_total_gib: float
    kv_cache_gib: float
    runtime_overhead_gib: float
    total_gib: float
    budget_gib: float

    @property
    def excess_gib(self) -> float:
        """予算超過量。予算内なら 0.0。"""
        return max(0.0, self.total_gib - self.budget_gib)

    @property
    def within_budget(self) -> bool:
        return self.total_gib <= self.budget_gib

    def summary(self) -> str:
        """ログ・例外メッセージ用の 1 行サマリ。"""
        return (
            f"プロファイル '{self.profile_name}': 想定 VRAM 使用量 "
            f"{self.total_gib:.2f} GiB / 予算 {self.budget_gib:.2f} GiB "
            f"(重み {self.weights_total_gib:.2f} + KV {self.kv_cache_gib:.2f} + "
            f"オーバーヘッド {self.runtime_overhead_gib:.2f}, "
            f"context_tokens={self.context_tokens})"
        )


def resolve_profile(
    config: AppConfig, profile_name: str | None = None
) -> ResolvedProfile:
    """プロファイル名をカタログ上の :class:`ModelSpec` 並びへ解決する。

    Args:
        config: 読み込み済み設定。
        profile_name: 省略時は ``config.vram.active_profile``。

    Raises:
        ConfigError: プロファイル未定義、または参照モデルがカタログに無い場合。
    """
    name = profile_name if profile_name is not None else config.vram.active_profile
    profile = config.profiles.get(name)
    if profile is None:
        known = ", ".join(sorted(config.profiles)) or "(なし)"
        msg = f"プロファイル '{name}' は設定に定義されていません。定義済み: {known}"
        raise ConfigError(
            msg, remediation="[profiles.<名前>] を追加するか名前を修正してください"
        )
    is_local = config.runtime.is_local
    declared: tuple[tuple[str | None, ModelRole], ...] = (
        (profile.generation, "generation"),
        (profile.embedding, "embedding"),
        (profile.reranker, "reranker"),
    )
    slots: tuple[tuple[str, ModelRole], ...] = tuple(
        (model_id, role) for model_id, role in declared if model_id
    )
    return ResolvedProfile(
        name=name,
        models=tuple(
            resolve_model_spec(model_id, is_local=is_local, role=role)
            for model_id, role in slots
        ),
    )


def estimate_resolved_profile(
    profile: ResolvedProfile,
    config: AppConfig,
    *,
    context_tokens: int | None = None,
) -> VramEstimate:
    """解決済みプロファイルの VRAM 使用量を見積もる (純関数)。"""
    tokens = (
        context_tokens
        if context_tokens is not None
        else config.generation.context_tokens
    )
    weights = MappingProxyType(
        {spec.model_id: spec.weights_gib for spec in profile.models}
    )
    weights_total = sum(weights.values())
    kv_cache = profile.generation.kv_gib_per_1k_tokens * (tokens / _TOKENS_PER_KILO)
    overhead = config.vram.runtime_overhead_gib
    return VramEstimate(
        profile_name=profile.name,
        context_tokens=tokens,
        weights_gib=weights,
        weights_total_gib=weights_total,
        kv_cache_gib=kv_cache,
        runtime_overhead_gib=overhead,
        total_gib=weights_total + kv_cache + overhead,
        budget_gib=config.vram.budget_gib,
    )


def estimate_profile(
    config: AppConfig,
    profile_name: str | None = None,
    *,
    context_tokens: int | None = None,
) -> VramEstimate:
    """プロファイル名から直接 VRAM 使用量を見積もる (解決 + 見積り)。"""
    profile = resolve_profile(config, profile_name)
    return estimate_resolved_profile(profile, config, context_tokens=context_tokens)


def check_budget(
    profile: ResolvedProfile,
    config: AppConfig,
    *,
    context_tokens: int | None = None,
) -> VramEstimate:
    """見積りが VRAM 予算に収まっているか判定する。

    ``runtime.is_local`` が false のときは判定をスキップする
    (外部 API はローカル VRAM を使わないため)。

    Raises:
        VramBudgetExceededError: ローカル実行かつ予算を超過した場合。
    """
    estimate = estimate_resolved_profile(profile, config, context_tokens=context_tokens)
    if not config.runtime.is_local:
        logger.debug(
            "runtime.is_local=false のため VRAM 予算判定をスキップします: %s",
            estimate.summary(),
        )
        return estimate
    if estimate.within_budget:
        return estimate

    msg = (
        f"プロファイル '{estimate.profile_name}' の想定 VRAM 使用量 "
        f"{estimate.total_gib:.2f} GiB が予算 {estimate.budget_gib:.2f} GiB を "
        f"{estimate.excess_gib:.2f} GiB 超過しています ({estimate.summary()})"
    )
    raise VramBudgetExceededError(
        msg,
        remediation=(
            "より小さいプロファイルへ切り替えるか、generation.context_tokens を"
            "減らすか、vram.budget_gib を実機の VRAM に合わせて見直してください"
        ),
        profile_name=estimate.profile_name,
        weights_gib=estimate.weights_gib,
        kv_cache_gib=estimate.kv_cache_gib,
        runtime_overhead_gib=estimate.runtime_overhead_gib,
        total_gib=estimate.total_gib,
        budget_gib=estimate.budget_gib,
        excess_gib=estimate.excess_gib,
    )
