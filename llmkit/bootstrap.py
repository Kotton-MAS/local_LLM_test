"""起動シーケンス (設定 → 検査 → ログ → マニフェスト → クライアント)。

仕様書 §4 T4 の順序をそのまま 1 関数に集約する::

    1. 設定ロード
    2. プロファイル解決
    3. VRAM 見積り
    4. INFO ログに想定使用量を出力      (要件書 L295)
    5. 予算超過なら WARNING + 例外で停止 (要件書 L296 / D-04)
    6. 実行マニフェスト書き出し
    7. ChatClient を返す

5 で停止する場合、クライアントは生成されず HTTP は 1 バイトも発行されない。
「警告のみで継続」しないのは、継続すると実行時 OOM になり原因が特定しにくい
エラーに化けるため (D-04)。

入口は 2 つある。1 (設定ロード) を含む :func:`bootstrap` と、読み込み済みの
:class:`~llmkit.config.AppConfig` を受け取って 2 以降だけを行う
:func:`bootstrap_from_config` である。後者は L3 が ``dataclasses.replace`` で
設定を上書きしてから起動する経路 (Phase 2 のモデル比較) のために分けたもので、
:func:`bootstrap` は前者に ``load_config`` を足しただけの薄いラッパである。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from llmkit.catalog import resolve_model_spec
from llmkit.client import (
    ChatClient,
    api_style_for,
    create_chat_client,
    endpoint_url_for,
)
from llmkit.config import AppConfig, load_config
from llmkit.errors import VramBudgetExceededError
from llmkit.manifest import RunManifest, build_manifest, write_manifest
from llmkit.vram import (
    ResolvedProfile,
    VramEstimate,
    check_budget,
    estimate_resolved_profile,
    resolve_profile,
)

logger = logging.getLogger(__name__)

__all__ = ["BootstrapResult", "bootstrap", "bootstrap_from_config"]


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """起動シーケンスの成果物。

    ``client`` が仕様書の言う「返す ChatClient」本体で、残りは CLI と Phase 2 が
    再実行条件を表示・記録するために必要な副産物。
    """

    config: AppConfig
    profile: ResolvedProfile
    estimate: VramEstimate
    manifest: RunManifest
    manifest_path: Path | None
    client: ChatClient
    endpoint_url: str
    served_name: str


def bootstrap(
    config_path: Path,
    *,
    profile_name: str | None = None,
    http_client: httpx.Client | None = None,
    output_dir: Path | None = None,
    write_manifest_file: bool = True,
) -> BootstrapResult:
    """設定ファイルから推論クライアントを起動する。

    Args:
        config_path: TOML 設定ファイル。
        profile_name: 省略時は ``vram.active_profile``。
        http_client: 注入する ``httpx.Client``。テストは ``MockTransport`` を渡す。
        output_dir: マニフェストの出力先。省略時は ``outputs/runs/``。
        write_manifest_file: False ならマニフェストを組み立てるがファイルに書かない。

    Raises:
        ConfigError: 設定・プロファイル・カタログの解決に失敗した場合。
        VramBudgetExceededError: 想定 VRAM 使用量が予算を超えた場合。
    """
    return bootstrap_from_config(
        load_config(config_path),
        config_path,
        profile_name=profile_name,
        http_client=http_client,
        output_dir=output_dir,
        write_manifest_file=write_manifest_file,
    )


def bootstrap_from_config(
    config: AppConfig,
    config_path: Path,
    *,
    profile_name: str | None = None,
    http_client: httpx.Client | None = None,
    output_dir: Path | None = None,
    write_manifest_file: bool = True,
) -> BootstrapResult:
    """読み込み済みの :class:`AppConfig` から推論クライアントを起動する。

    :func:`bootstrap` の本体。設定ファイルを読む前に L3 が
    ``dataclasses.replace`` で値を差し替えられるよう、ロードと起動を分けている
    (Phase 2 のモデル比較ハーネスがモデルごとに N 回起動する経路)。

    Warning:
        ``config`` が実行時に上書きされた場合、``manifest.config_sha256`` は
        **ベース設定ファイル (``config_path``) の内容ハッシュ**であり、実際に
        使われた実効値の同一性を表さない。上書き後の 2 実行はモデルが違っても
        同じ ``config_sha256`` を持つ。実効値の同一性は L3 の
        ``run_fingerprint`` が担う (D-20)。

    Args:
        config: 読み込み済み (かつ必要なら上書き済み) の設定。
        config_path: ``config`` の出所となったベース設定ファイル。
            ``config_sha256`` の計算にのみ使う。
        profile_name: 省略時は ``vram.active_profile``。
        http_client: 注入する ``httpx.Client``。テストは ``MockTransport`` を渡す。
        output_dir: マニフェストの出力先。省略時は ``outputs/runs/``。
        write_manifest_file: False ならマニフェストを組み立てるがファイルに書かない。

    Raises:
        ConfigError: プロファイル・カタログの解決に失敗した場合。
        VramBudgetExceededError: 想定 VRAM 使用量が予算を超えた場合。
    """
    profile = resolve_profile(config, profile_name)
    estimate = estimate_resolved_profile(profile, config)

    logger.info("起動前 VRAM 見積り: %s", estimate.summary())

    try:
        check_budget(profile, config)
    except VramBudgetExceededError as exc:
        logger.warning(
            "VRAM 予算を超過したため起動を中止します"
            " (推論リクエストは発行しません): %s",
            exc,
        )
        raise

    manifest = build_manifest(config, profile, estimate, config_path)
    manifest_path = (
        write_manifest(manifest, output_dir) if write_manifest_file else None
    )

    # ChatClient は Protocol であり endpoint_url / served_name を宣言しないため、
    # 表示・記録用の 2 値はクライアントからではなく設定から導出する
    # (マニフェストの api_style / endpoint_url と同じ出典にそろえる)。
    api_style = api_style_for(config.runtime.kind)
    endpoint_url = endpoint_url_for(config.runtime.base_url, api_style)
    served_name = resolve_model_spec(
        config.generation.model, is_local=config.runtime.is_local
    ).served_name

    client = create_chat_client(config, http_client=http_client)
    logger.info(
        "推論クライアントを初期化しました: api_style=%s url=%s model=%s",
        api_style,
        endpoint_url,
        served_name,
    )
    return BootstrapResult(
        config=config,
        profile=profile,
        estimate=estimate,
        manifest=manifest,
        manifest_path=manifest_path,
        client=client,
        endpoint_url=endpoint_url,
        served_name=served_name,
    )
