"""``llmkit.bootstrap_from_config`` (読み込み済み設定からの起動) のテスト。

``bootstrap`` は ``load_config`` を足しただけの薄いラッパであり、既存の
``tests/test_bootstrap.py`` がその経路を測っている。ここで測るのは分割で新しく
できた入口、すなわち **L3 が上書きした ``AppConfig`` を渡して起動する経路**である。

実 HTTP は発行しない (D-02)。マニフェストの書き出し先は必ず ``tmp_path``。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import pytest
from conftest import DEFAULT_CONFIG, RecordingTransport

import llmkit
from llmkit import (
    ChatMessage,
    VramBudgetExceededError,
    bootstrap,
    bootstrap_from_config,
    load_config,
)


def override_model(config: llmkit.AppConfig, model_id: str) -> llmkit.AppConfig:
    """``generation.model`` と対象プロファイルの両方を差し替える (D-19 と同じ形)。

    ``harness.suite.apply_case`` を import せずに手で書いているのは、このテストが
    測るのが ``bootstrap_from_config`` 側 (上書き済み設定を起動できること) であり、
    L3 の適用ロジックではないため。
    """
    profiles = dict(config.profiles)
    active = config.vram.active_profile
    profiles[active] = dataclasses.replace(profiles[active], generation=model_id)
    return dataclasses.replace(
        config,
        generation=dataclasses.replace(config.generation, model=model_id),
        profiles=profiles,
    )


# --------------------------------------------------------------------------
# 委譲 (既存シグネチャの挙動を変えていないこと)
# --------------------------------------------------------------------------


def test_bootstrap_delegates_to_bootstrap_from_config(tmp_path: Path) -> None:
    """同じ設定ファイルに対し、2 つの入口が同じ起動結果を返す。"""
    transport = RecordingTransport()

    with transport.client() as http_client:
        through_path = bootstrap(
            DEFAULT_CONFIG, http_client=http_client, output_dir=tmp_path / "runs"
        )
        through_config = bootstrap_from_config(
            load_config(DEFAULT_CONFIG),
            DEFAULT_CONFIG,
            http_client=http_client,
            output_dir=tmp_path / "runs",
        )

    assert through_config.config == through_path.config
    assert through_config.profile == through_path.profile
    assert through_config.estimate == through_path.estimate
    assert through_config.endpoint_url == through_path.endpoint_url
    assert through_config.served_name == through_path.served_name
    assert through_config.manifest.config_sha256 == through_path.manifest.config_sha256
    assert transport.call_count == 0, "起動だけで HTTP を発行してはならない"


def test_bootstrap_from_config_is_exported_from_the_package_root() -> None:
    """L3 は ``from llmkit import bootstrap_from_config`` だけで足りる。"""
    assert "bootstrap_from_config" in llmkit.__all__
    assert llmkit.bootstrap_from_config is bootstrap_from_config


# --------------------------------------------------------------------------
# 上書き済み設定からの起動
# --------------------------------------------------------------------------


def test_overridden_config_reaches_both_the_request_and_the_vram_estimate(
    tmp_path: Path,
) -> None:
    """上書きしたモデルがリクエスト・マニフェスト・VRAM 見積りの全部に届く。"""
    transport = RecordingTransport()
    overridden = override_model(load_config(DEFAULT_CONFIG), "gpt-oss-20b")

    with transport.client() as http_client:
        result = bootstrap_from_config(
            overridden,
            DEFAULT_CONFIG,
            http_client=http_client,
            output_dir=tmp_path / "runs",
        )
        result.client.chat([ChatMessage(role="user", content="やあ")])

    payload: object = json.loads(transport.requests[0].content)
    assert isinstance(payload, dict)

    assert payload["model"] == "gpt-oss:20b"
    assert result.served_name == "gpt-oss:20b"
    assert result.profile.model_ids[0] == "gpt-oss-20b"
    assert set(result.estimate.weights_gib) == {
        "gpt-oss-20b",
        "ruri-v3-310m",
        "bge-reranker-v2-m3",
    }
    assert result.estimate.total_gib == pytest.approx(13.36, abs=0.01)
    assert result.manifest.profile.models[0].model_id == "gpt-oss-20b"


def test_config_sha256_does_not_follow_runtime_overrides(tmp_path: Path) -> None:
    """D-20 の存在理由: 上書きしても ``config_sha256`` はベースファイルのまま。

    ``config_sha256`` は「どの設定ファイルを起点にしたか」しか表さない。実効値の
    同一性はここでは表せないため、L3 が ``run_fingerprint`` を別に持つ。
    """
    transport = RecordingTransport()
    base = load_config(DEFAULT_CONFIG)

    with transport.client() as http_client:
        baseline = bootstrap_from_config(
            base,
            DEFAULT_CONFIG,
            http_client=http_client,
            output_dir=tmp_path / "runs",
        )
        overridden = bootstrap_from_config(
            override_model(base, "qwen3-8b"),
            DEFAULT_CONFIG,
            http_client=http_client,
            output_dir=tmp_path / "runs",
        )

    assert overridden.manifest.config_sha256 == baseline.manifest.config_sha256
    assert overridden.manifest.profile.models[0].model_id == "qwen3-8b"
    assert baseline.manifest.profile.models[0].model_id == "qwen3-14b"


def test_bootstrap_from_config_accepts_a_profile_name(tmp_path: Path) -> None:
    transport = RecordingTransport()

    with transport.client() as http_client:
        result = bootstrap_from_config(
            load_config(DEFAULT_CONFIG),
            DEFAULT_CONFIG,
            profile_name="lightweight",
            http_client=http_client,
            output_dir=tmp_path / "runs",
        )

    assert result.profile.name == "lightweight"
    assert result.estimate.total_gib == pytest.approx(7.26, abs=0.01)


def test_bootstrap_from_config_aborts_over_budget_without_http(tmp_path: Path) -> None:
    """D-04 の性質が新しい入口でも成立する: 予算超過で HTTP 0 回。"""
    transport = RecordingTransport()
    base = load_config(DEFAULT_CONFIG)
    over_budget = dataclasses.replace(
        base,
        generation=dataclasses.replace(base.generation, context_tokens=131072),
        vram=dataclasses.replace(base.vram, active_profile="oversized"),
    )

    with (
        transport.client() as http_client,
        pytest.raises(VramBudgetExceededError) as excinfo,
    ):
        bootstrap_from_config(
            over_budget,
            DEFAULT_CONFIG,
            http_client=http_client,
            output_dir=tmp_path / "runs",
        )

    assert excinfo.value.profile_name == "oversized"
    assert transport.requests == []
    assert not (tmp_path / "runs").exists()


def test_bootstrap_from_config_can_skip_writing_the_manifest_file(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport()

    with transport.client() as http_client:
        result = bootstrap_from_config(
            load_config(DEFAULT_CONFIG),
            DEFAULT_CONFIG,
            http_client=http_client,
            output_dir=tmp_path / "runs",
            write_manifest_file=False,
        )

    assert result.manifest_path is None
    assert not (tmp_path / "runs").exists()
    assert result.manifest.config_sha256


def test_repeated_bootstraps_share_one_http_client(tmp_path: Path) -> None:
    """モデルを替えて N 回起動しても、注入した httpx.Client を使い回せる。"""
    transport = RecordingTransport()
    base = load_config(DEFAULT_CONFIG)
    served: list[str] = []

    with transport.client() as http_client:
        for model_id in ("qwen3-14b", "qwen3-8b", "gpt-oss-20b"):
            result = bootstrap_from_config(
                override_model(base, model_id),
                DEFAULT_CONFIG,
                http_client=http_client,
                output_dir=tmp_path / "runs",
            )
            result.client.chat([ChatMessage(role="user", content="やあ")])
            served.append(result.served_name)

    assert served == ["qwen3:14b-q4_K_M", "qwen3:8b-q4_K_M", "gpt-oss:20b"]
    assert transport.call_count == 3
    sent = [json.loads(request.content)["model"] for request in transport.requests]
    assert sent == served
    assert isinstance(transport.requests[0], httpx.Request)
