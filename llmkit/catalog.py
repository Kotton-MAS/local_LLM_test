"""モデルカタログ (静的テーブル)。

VRAM 見積りの唯一の出典はこの静的テーブルであり、実行時に ``nvidia-smi`` や
実測値を参照しない (D-01)。数値はすべて **GiB** で持つ (D-03)。

生成 3 モデルの値は ``docs/phase0-vram-measurements.md`` (2026-08-22 実測) の
線形フィット結果で較正済み。埋め込み・リランカーは 2026-08-23 の第2次実測で較正した
(D-14 撤回)。更新するときは ``source_note`` に実測日と出典を書くこと。

``serving_runtime`` は「そのモデルを載せるサーバプロセス」を表すモデル単位の静的
メタデータであり、``config.RuntimeKind`` (接続単位のチャット送出経路) とも
``client.ApiStyle`` (ワイヤプロトコル) とも直交する (D-15)。Phase 1 ではディスパッチに
使わず、実行マニフェストに記録するだけである。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from llmkit.errors import ConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_CATALOG",
    "ModelRole",
    "ModelSpec",
    "ServingRuntime",
    "get_model_spec",
    "known_model_ids",
    "resolve_model_spec",
]

ModelRole = Literal["generation", "embedding", "reranker"]

#: そのモデルを実際に載せるサーバプロセス (D-15)。既定値は与えない。
#: 既定を ``"ollama"`` にすると、将来追加されるモデルが黙って Ollama 扱いになり、
#: 2026-08-23 に検出した誤り (Ollama にリランキング API が無い) を再生産する。
ServingRuntime = Literal["ollama", "llama_cpp_server", "external"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """1 モデルの静的メタデータ。

    Attributes:
        model_id: 設定ファイルが参照する論理名。
        served_name: 推論ランタイムに送る実名 (例 ``qwen3:14b-q4_K_M``)。
        role: プロファイル内での役割。
        serving_runtime: このモデルを載せるサーバプロセス (D-15)。
        quantization: 量子化方式。
        weights_gib: このモデルが占める VRAM (GiB)。生成モデルは重みのみ
            (KV とランタイムのオーバーヘッドは見積り式が別に足す) だが、埋め込み・
            リランカーは自ランタイムのオーバーヘッドを含む実測 VRAM 増分そのもの
            (D-16)。
        kv_gib_per_1k_tokens: コンテキスト 1024 トークンあたりの KV キャッシュ (GiB)。
        max_context_tokens: モデルが受け付ける最大コンテキスト長。
        source_note: 数値の根拠 (要件書の該当行 / 仮置きである旨)。
    """

    model_id: str
    served_name: str
    role: ModelRole
    serving_runtime: ServingRuntime
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
        serving_runtime="ollama",
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
        serving_runtime="ollama",
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
        serving_runtime="ollama",
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
        served_name="hf.co/Targoyle/ruri-v3-310m-GGUF",
        role="embedding",
        serving_runtime="ollama",
        quantization="unknown (GGUF)",
        weights_gib=0.57,
        kv_gib_per_1k_tokens=0.0,
        max_context_tokens=8192,
        source_note=(
            "実測 2026-08-23 / docs/phase0-vram-measurements.md。"
            "元リポジトリ (sentence-transformers 形式) は ollama pull できないが、"
            "有志の GGUF 変換版なら Ollama で扱える (336 MB、埋め込み"
            "エンドポイントが 768 次元を返す)。"
            "weights_gib=0.57 は純粋な重みではなく、Ollama にロードしたときの"
            "実測 VRAM 増分そのもの (自ランタイムのオーバーヘッド込み、D-16)。"
            "量子化方式は配布側のタグから未確認のため unknown (GGUF) とする。"
            "max_context_tokens=8192 はモデルカード由来で未実測"
        ),
    ),
    ModelSpec(
        model_id="bge-reranker-v2-m3",
        served_name="bge-reranker-v2-m3-Q6_K.gguf",
        role="reranker",
        serving_runtime="llama_cpp_server",
        quantization="Q6_K",
        weights_gib=0.28,
        kv_gib_per_1k_tokens=0.0,
        max_context_tokens=8192,
        source_note=(
            "実測 2026-08-23 / docs/phase0-vram-measurements.md。"
            "要件書第一候補の Ruri Reranker は GGUF 非提供のため、要件書が代替と"
            "して挙げる BGE Reranker v2-m3 を採用した。Ollama にはリランキング "
            "API が無い (native・OpenAI 互換のどちらの経路でも 404) ため "
            "llama.cpp の llama-server (Vulkan) を別ポートで併走させる (D-15)。"
            "weights_gib=0.28 は純粋な重みではなく、llama-server 自身の"
            "オーバーヘッドを含む実測 VRAM 増分そのもの (D-16)。"
            "測定条件は --ctx-size 2048 / --n-gpu-layers 99 の 1 点のみで、"
            "ctx-size を上げるとこの値は過小評価になる。"
            "max_context_tokens=8192 は BGE Reranker v2-m3 系列の上限として"
            "記載したもので未実測"
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

    カタログ登録済みの場合:

    - ``is_local=True`` ではカタログの :class:`ModelSpec` をそのまま返す。
    - ``is_local=False`` (外部 API) では ``serving_runtime`` だけ ``"external"``
      に差し替えて返す (``weights_gib`` / ``kv_gib_per_1k_tokens`` を含む他の
      フィールドはカタログ値のまま維持し、VRAM 見積りの挙動は変えない)。
      これをしないと、外部 API 実行のマニフェストに『ローカルの Ollama /
      llama-server が載せている』という偽の主張が残る (F-6-001、D-15 追記)。

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
        if is_local:
            return spec
        # 外部 API 実行では、カタログ登録済みモデルであっても実際に載せて
        # いるのはローカルの Ollama / llama-server ではない (F-6-001)。
        # weights_gib / kv_gib_per_1k_tokens はカタログ値のまま維持する
        # (D-16 の合算式・VRAM 見積りの挙動を変えないため)。
        return replace(spec, serving_runtime="external")
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
        serving_runtime="external",
        quantization=_PASSTHROUGH_QUANTIZATION,
        weights_gib=0.0,
        kv_gib_per_1k_tokens=0.0,
        max_context_tokens=_PASSTHROUGH_MAX_CONTEXT_TOKENS,
        source_note=(
            "passthrough: カタログ未登録の外部 API 用モデル (is_local=false)。"
            "VRAM 見積りには寄与しない (weights_gib=kv_gib_per_1k_tokens=0.0)"
        ),
    )
