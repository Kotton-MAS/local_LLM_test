"""モデルカタログ (静的テーブル)。

VRAM 見積りの唯一の出典はこの静的テーブルであり、実行時に ``nvidia-smi`` や
実測値を参照しない (D-01)。数値はすべて **GiB** で持つ (D-03)。

生成 3 モデルの値は ``docs/phase0-vram-measurements.md`` (2026-08-22 実測) の
線形フィット結果で較正済み。埋め込み・リランカーは取得不能のため `(仮)` のまま
残す (``source_note`` に理由を明記する)。更新するときは ``source_note`` に実測日と
出典を書くこと。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from llmkit.errors import ConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_CATALOG",
    "ModelRole",
    "ModelSpec",
    "get_model_spec",
    "known_model_ids",
    "resolve_model_spec",
]

ModelRole = Literal["generation", "embedding", "reranker"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """1 モデルの静的メタデータ。

    Attributes:
        model_id: 設定ファイルが参照する論理名。
        served_name: 推論ランタイムに送る実名 (例 ``qwen3:14b-q4_K_M``)。
        role: プロファイル内での役割。
        quantization: 量子化方式。
        weights_gib: 重みが占める VRAM (GiB)。
        kv_gib_per_1k_tokens: コンテキスト 1024 トークンあたりの KV キャッシュ (GiB)。
        max_context_tokens: モデルが受け付ける最大コンテキスト長。
        source_note: 数値の根拠 (要件書の該当行 / 仮置きである旨)。
    """

    model_id: str
    served_name: str
    role: ModelRole
    quantization: str
    weights_gib: float
    kv_gib_per_1k_tokens: float
    max_context_tokens: int
    source_note: str


_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="qwen3-14b",
        served_name="qwen3:14b-q4_K_M",
        role="generation",
        quantization="Q4_K",
        weights_gib=7.81,
        kv_gib_per_1k_tokens=0.1575,
        max_context_tokens=32768,
        source_note=(
            "実測 2026-08-22 / docs/phase0-vram-measurements.md。"
            "num_ctx=4096/16384/32768 の 3 点 (すべて 100% GPU) を線形フィットし、"
            "オーバーヘッド 0.8 GiB を仮定して重みを分離した"
        ),
    ),
    ModelSpec(
        model_id="gpt-oss-20b",
        served_name="gpt-oss:20b",
        role="generation",
        quantization="MXFP4",
        weights_gib=11.32,
        kv_gib_per_1k_tokens=0.0244,
        max_context_tokens=131072,
        source_note=(
            "実測 2026-08-22 / docs/phase0-vram-measurements.md。"
            "num_ctx=32768/65536 の 2 点 (100% GPU) を線形フィットし、"
            "オーバーヘッド 0.8 GiB を仮定して重みを分離した。"
            "GPU 常駐上限は 65,536〜98,303 の間 (98,304 で CPU オフロード)"
        ),
    ),
    ModelSpec(
        model_id="qwen3-8b",
        served_name="qwen3:8b-q4_K_M",
        role="generation",
        quantization="Q4_K",
        weights_gib=4.18,
        kv_gib_per_1k_tokens=0.1425,
        max_context_tokens=32768,
        source_note=(
            "実測 2026-08-22 / docs/phase0-vram-measurements.md。"
            "num_ctx=8192/16384/32768 の 3 点 (すべて 100% GPU) を線形フィットし、"
            "オーバーヘッド 0.8 GiB を仮定して重みを分離した"
        ),
    ),
    ModelSpec(
        model_id="ruri-v3-310m",
        served_name="hf.co/cl-nagoya/ruri-v3-310m",
        role="embedding",
        quantization="fp16",
        weights_gib=0.7,
        kv_gib_per_1k_tokens=0.0,
        max_context_tokens=8192,
        source_note=(
            "(仮) 取得不能: GGUF 非提供のため ollama pull できない "
            "(Phase 0 実測で確認)。weights_gib は 310M パラメータ x 2 byte "
            "からの未実測の仮値。方式は Phase 3 で決定する"
        ),
    ),
    ModelSpec(
        model_id="ruri-reranker",
        served_name="hf.co/cl-nagoya/ruri-reranker-large",
        role="reranker",
        quantization="fp16",
        weights_gib=0.8,
        kv_gib_per_1k_tokens=0.0,
        max_context_tokens=512,
        source_note=(
            "(仮) 取得不能: GGUF 非提供のため ollama pull できない "
            "(Phase 0 実測で確認。Ollama にリランキング API 自体が無い)。"
            "重み・served_name・最大コンテキストはいずれも未実測の仮値。"
            "方式は Phase 3 で決定する"
        ),
    ),
)

MODEL_CATALOG: Mapping[str, ModelSpec] = MappingProxyType(
    {spec.model_id: spec for spec in _SPECS}
)


def known_model_ids() -> tuple[str, ...]:
    """カタログに登録されている model_id を昇順で返す。"""
    return tuple(sorted(MODEL_CATALOG))


def get_model_spec(model_id: str) -> ModelSpec:
    """model_id から :class:`ModelSpec` を引く。

    Raises:
        ConfigError: カタログに存在しない model_id を指定した場合。
            (ランタイム側にモデルが無い場合の ``ModelNotFoundError`` とは別物で、
            こちらは設定またはカタログの記述漏れが原因。)
    """
    spec = MODEL_CATALOG.get(model_id)
    if spec is None:
        msg = (
            f"モデル '{model_id}' は llmkit/catalog.py に登録されていません。"
            f"登録済み: {', '.join(known_model_ids())}"
        )
        raise ConfigError(
            msg,
            remediation=(
                "設定の model 名を修正するか、llmkit/catalog.py に ModelSpec を"
                "追加してください"
            ),
        )
    return spec


_PASSTHROUGH_QUANTIZATION = "unknown (passthrough)"
# 外部 API 側の実際の上限は分からないため広く確保する。VRAM 見積りには
# 使わない (weights_gib=kv_gib_per_1k_tokens=0.0) ため、大きくしても
# D-01 (静的テーブルが唯一の出典) には影響しない。
_PASSTHROUGH_MAX_CONTEXT_TOKENS = 1_048_576


def resolve_model_spec(
    model_id: str,
    *,
    is_local: bool,
    role: ModelRole = "generation",
) -> ModelSpec:
    """model_id を :class:`ModelSpec` に解決する (``get_model_spec`` の拡張版)。

    カタログ登録済みならそのまま返す (``is_local`` に関わらず同じ挙動)。
    カタログ未登録の場合:

    - ``is_local=True`` (ローカル VRAM を使う実行) では ``get_model_spec`` と
      同じ :class:`~llmkit.errors.ConfigError` を送出する。ローカル実行では
      VRAM 見積りの正確性が要件そのものであり、未登録モデルを黙って通すと
      D-01 (静的テーブルが唯一の出典) の前提が崩れるため。
    - ``is_local=False`` (外部 API) では、カタログを経由しない passthrough な
      :class:`ModelSpec` を合成して返す。外部 API はローカル VRAM を使わず
      ``check_budget`` も判定をスキップするため、weights_gib と
      kv_gib_per_1k_tokens を 0.0 にしても D-01 とは衝突しない (要件書 L144
      の3項目のうち「ローカルと外部 API の切り替え」軸を設定変更のみで
      可能にするための経路)。``served_name`` は ``model_id`` をそのまま使う
      (外部 OpenAI 互換 API は設定上のモデル名がそのまま送出名になる運用を
      想定)。

      注意: 「ランタイムの差し替え (vLLM 等)」軸は対象外 (D-09)。カタログ
      登録済み model_id 間の切替は設定変更のみで可能だが、ランタイム固有の
      served_name (例: vLLM に渡す実モデル名) を設定側から上書きする手段は
      Phase 1 には無い。ローカル実行で未登録 model_id を渡すと下の
      ``is_local=True`` 分岐どおり :class:`~llmkit.errors.ConfigError` になる。

    Args:
        model_id: 設定ファイルが参照する論理名。
        is_local: ``runtime.is_local``。
        role: passthrough で合成する :class:`ModelSpec` の役割
            (カタログ登録済みモデルには影響しない)。

    Raises:
        ConfigError: ``is_local=True`` でカタログに存在しない model_id の場合。
    """
    spec = MODEL_CATALOG.get(model_id)
    if spec is not None:
        return spec
    if is_local:
        return get_model_spec(model_id)
    logger.debug(
        "カタログ未登録の model_id '%s' を is_local=false のため "
        "passthrough で解決します (role=%s)",
        model_id,
        role,
    )
    return ModelSpec(
        model_id=model_id,
        served_name=model_id,
        role=role,
        quantization=_PASSTHROUGH_QUANTIZATION,
        weights_gib=0.0,
        kv_gib_per_1k_tokens=0.0,
        max_context_tokens=_PASSTHROUGH_MAX_CONTEXT_TOKENS,
        source_note=(
            "passthrough: カタログ未登録の外部 API 用モデル (is_local=false)。"
            "VRAM 見積りには寄与しない (weights_gib=kv_gib_per_1k_tokens=0.0)"
        ),
    )
