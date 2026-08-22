"""実行マニフェスト (再現可能ログ)。

要件書「ログ: 実行日時・モデル・量子化・全パラメータ」を満たすため、1 回の起動で
使われた設定・プロファイル・VRAM 見積り・実行環境を JSON 1 ファイルに固定する。

- **実際に使われた設定値を写す**。雛形をハードコードしない (仕様書 §5 有効性観点 E8)。
- ``config_sha256`` は設定ファイルの内容ハッシュ。同一設定なら一致し、1 文字でも
  変われば変わる = 再現性の担保。
- api_key の値は書かない。``runtime.api_key_env`` (環境変数「名」) だけを書く (D-05)。
- 出力先は ``outputs/runs/`` (gitignore 済み)。
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform as platform_module
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llmkit.catalog import ModelRole
from llmkit.config import AppConfig
from llmkit.errors import ConfigError
from llmkit.vram import ResolvedProfile, VramEstimate

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "SCHEMA_VERSION",
    "ManifestGeneration",
    "ManifestModelEntry",
    "ManifestProfile",
    "ManifestRuntime",
    "ManifestVram",
    "RunManifest",
    "build_manifest",
    "compute_config_sha256",
    "manifest_filename",
    "write_manifest",
]

SCHEMA_VERSION = "1"
DEFAULT_OUTPUT_DIR = Path("outputs/runs")

# ISO8601 (started_at_utc) はコロンを含みファイル名に使えないため、
# ファイル名だけは同じ時刻を詰めた表記にする。
_FILENAME_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True, slots=True)
class ManifestModelEntry:
    """プロファイルを構成する 1 モデルの記録。"""

    model_id: str
    served_name: str
    role: ModelRole
    quantization: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "served_name": self.served_name,
            "role": self.role,
            "quantization": self.quantization,
        }


@dataclass(frozen=True, slots=True)
class ManifestProfile:
    """使用した VRAM プロファイル。"""

    name: str
    models: tuple[ManifestModelEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "models": [model.to_dict() for model in self.models],
        }


@dataclass(frozen=True, slots=True)
class ManifestVram:
    """VRAM 見積りの内訳。単位はすべて GiB (D-03)。"""

    weights_gib: Mapping[str, float]
    weights_total_gib: float
    kv_cache_gib: float
    runtime_overhead_gib: float
    total_gib: float
    budget_gib: float
    within_budget: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "weights_gib": dict(self.weights_gib),
            "weights_total_gib": self.weights_total_gib,
            "kv_cache_gib": self.kv_cache_gib,
            "runtime_overhead_gib": self.runtime_overhead_gib,
            "total_gib": self.total_gib,
            "budget_gib": self.budget_gib,
            "within_budget": self.within_budget,
        }


@dataclass(frozen=True, slots=True)
class ManifestGeneration:
    """生成パラメータの全記録。Phase 2 の再現実行はここだけを見れば足りる。"""

    model: str
    context_tokens: int
    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "context_tokens": self.context_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ManifestRuntime:
    """接続先ランタイムの記録。``api_key_env`` は環境変数「名」のみ (D-05)。"""

    kind: str
    base_url: str
    is_local: bool
    timeout_s: float
    api_key_env: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "base_url": self.base_url,
            "is_local": self.is_local,
            "timeout_s": self.timeout_s,
            "api_key_env": self.api_key_env,
        }


@dataclass(frozen=True, slots=True)
class RunManifest:
    """1 回の起動を再現するために必要な情報の全体。"""

    schema_version: str
    run_id: str
    started_at_utc: str
    profile: ManifestProfile
    vram: ManifestVram
    generation: ManifestGeneration
    runtime: ManifestRuntime
    config_path: str
    config_sha256: str
    python_version: str
    platform: str

    def to_dict(self) -> dict[str, object]:
        """JSON 化できる素の辞書に変換する。"""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "started_at_utc": self.started_at_utc,
            "profile": self.profile.to_dict(),
            "vram": self.vram.to_dict(),
            "generation": self.generation.to_dict(),
            "runtime": self.runtime.to_dict(),
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "python_version": self.python_version,
            "platform": self.platform,
        }

    def to_json(self) -> str:
        """永続化・アサート双方で使う JSON 文字列表現。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    def filename(self) -> str:
        """``{started_at}-{run_id}.json`` 形式のファイル名。"""
        return manifest_filename(self.started_at_utc, self.run_id)


def manifest_filename(started_at_utc: str, run_id: str) -> str:
    """ISO8601 の開始時刻と run_id からファイル名を組み立てる。"""
    moment = datetime.fromisoformat(started_at_utc)
    return f"{moment.strftime(_FILENAME_TIMESTAMP_FORMAT)}-{run_id}.json"


def compute_config_sha256(config_path: Path) -> str:
    """設定ファイルの内容ハッシュ (再現性の担保)。

    Raises:
        ConfigError: 設定ファイルを読めない場合。
    """
    try:
        return hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError as exc:
        msg = f"設定ファイルを読めないためハッシュを計算できません: {config_path}"
        raise ConfigError(
            msg, remediation=f"パスが正しいか確認してください ({exc.strerror})"
        ) from exc


def build_manifest(
    config: AppConfig,
    profile: ResolvedProfile,
    estimate: VramEstimate,
    config_path: Path,
    *,
    run_id: str | None = None,
    started_at: datetime | None = None,
) -> RunManifest:
    """実行マニフェストを組み立てる。

    値はすべて引数の ``config`` / ``estimate`` から取る (E8: 雛形を固定値で
    埋めない)。``run_id`` / ``started_at`` はテストが固定できるよう注入可能。
    """
    moment = started_at if started_at is not None else datetime.now(UTC)
    generation = config.generation
    runtime = config.runtime
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id if run_id is not None else uuid.uuid4().hex[:12],
        started_at_utc=moment.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        profile=ManifestProfile(
            name=profile.name,
            models=tuple(
                ManifestModelEntry(
                    model_id=spec.model_id,
                    served_name=spec.served_name,
                    role=spec.role,
                    quantization=spec.quantization,
                )
                for spec in profile.models
            ),
        ),
        vram=ManifestVram(
            weights_gib=dict(estimate.weights_gib),
            weights_total_gib=estimate.weights_total_gib,
            kv_cache_gib=estimate.kv_cache_gib,
            runtime_overhead_gib=estimate.runtime_overhead_gib,
            total_gib=estimate.total_gib,
            budget_gib=estimate.budget_gib,
            within_budget=estimate.within_budget,
        ),
        generation=ManifestGeneration(
            model=generation.model,
            context_tokens=generation.context_tokens,
            temperature=generation.temperature,
            top_p=generation.top_p,
            max_output_tokens=generation.max_output_tokens,
            seed=generation.seed,
        ),
        runtime=ManifestRuntime(
            kind=runtime.kind,
            base_url=runtime.base_url,
            is_local=runtime.is_local,
            timeout_s=runtime.timeout_s,
            api_key_env=runtime.api_key_env,
        ),
        config_path=str(config_path),
        config_sha256=compute_config_sha256(config_path),
        python_version=platform_module.python_version(),
        platform=platform_module.platform(),
    )


def write_manifest(manifest: RunManifest, output_dir: Path | None = None) -> Path:
    """マニフェストを ``{output_dir}/{started_at}-{run_id}.json`` に書き出す。

    Raises:
        ConfigError: 出力先ディレクトリに書けない場合。
    """
    directory = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
    destination = directory / manifest.filename()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        destination.write_text(manifest.to_json() + "\n", encoding="utf-8")
    except OSError as exc:
        msg = f"実行マニフェストを書き出せません: {destination}"
        raise ConfigError(
            msg,
            remediation=(
                f"出力先ディレクトリの権限を確認してください ({exc.strerror})"
            ),
        ) from exc
    logger.info("実行マニフェストを書き出しました: %s", destination)
    return destination
