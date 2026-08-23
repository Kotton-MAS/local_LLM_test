"""実行マニフェスト (llmkit/manifest.py) のテスト。

このファイルは 1 つの guard_test を含む:

- ``test_manifest_and_logs_never_contain_api_key`` … D-05
  (api_key の値はマニフェスト・ログに出さない。環境変数名だけを出す)

実 HTTP は発行しない (D-02)。書き出し先は必ず ``tmp_path`` で、リポジトリの
``outputs/`` には触れない。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx
import pytest

from llmkit.bootstrap import bootstrap
from llmkit.catalog import ModelSpec
from llmkit.client import ChatMessage
from llmkit.config import AppConfig, GenerationParams, load_config
from llmkit.errors import ConfigError
from llmkit.manifest import (
    SCHEMA_VERSION,
    RunManifest,
    build_manifest,
    compute_config_sha256,
    write_manifest,
)
from llmkit.vram import ResolvedProfile, estimate_resolved_profile, resolve_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"
EXTERNAL_CONFIG = REPO_ROOT / "configs" / "external_openai.toml"
COMPAT_CONFIG = REPO_ROOT / "configs" / "ollama_openai_compat.toml"

API_KEY = "sk-test-do-not-leak-0123456789"

# 仕様書 §4 T4 が列挙したキー一覧。これが期待値集合であり、
# 実装がキーを 1 つ落とすと test_manifest_covers_every_specified_key が落ちる。
EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "started_at_utc",
        "profile",
        "vram",
        "generation",
        "runtime",
        "config_path",
        "config_sha256",
        "python_version",
        "platform",
    }
)
EXPECTED_SECTION_KEYS: Mapping[str, frozenset[str]] = {
    "profile": frozenset({"name", "models"}),
    "vram": frozenset(
        {
            "weights_gib",
            "weights_total_gib",
            "kv_cache_gib",
            "runtime_overhead_gib",
            "total_gib",
            "budget_gib",
            "within_budget",
        }
    ),
    "generation": frozenset(
        {
            "model",
            "context_tokens",
            "temperature",
            "top_p",
            "max_output_tokens",
            "seed",
        }
    ),
    "runtime": frozenset(
        {
            "kind",
            "base_url",
            "is_local",
            "timeout_s",
            "api_key_env",
            "api_style",
            "endpoint_url",
        }
    ),
}
EXPECTED_MODEL_KEYS = frozenset(
    {"model_id", "served_name", "role", "serving_runtime", "quantization"}
)


def make_manifest(config: AppConfig, config_path: Path = DEFAULT_CONFIG) -> RunManifest:
    """設定から実行マニフェストを 1 つ組み立てる。"""
    profile = resolve_profile(config)
    estimate = estimate_resolved_profile(profile, config)
    return build_manifest(config, profile, estimate, config_path)


def manifest_dict(config: AppConfig) -> dict[str, object]:
    payload: object = json.loads(make_manifest(config).to_json())
    assert isinstance(payload, dict)
    return {str(key): value for key, value in payload.items()}


def section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload[name]
    assert isinstance(value, dict), name
    return {str(key): nested for key, nested in value.items()}


def assert_manifest_shape(payload: Mapping[str, object]) -> None:
    """仕様書のキー一覧を満たしているかを 1 か所で判定する。"""
    assert set(payload) == set(EXPECTED_TOP_LEVEL_KEYS)
    for name, expected in EXPECTED_SECTION_KEYS.items():
        assert set(section(payload, name)) == set(expected), name
    models = section(payload, "profile")["models"]
    assert isinstance(models, list)
    assert models, "profile.models が空では構成モデルを再現できない"
    for model in models:
        assert isinstance(model, dict)
        assert set(model) == set(EXPECTED_MODEL_KEYS)


def removal_paths() -> tuple[tuple[str, ...], ...]:
    """「1 つ消すと落ちる」ことを確かめるためのキー経路一覧。"""
    paths: list[tuple[str, ...]] = [(key,) for key in sorted(EXPECTED_TOP_LEVEL_KEYS)]
    for name, keys in EXPECTED_SECTION_KEYS.items():
        paths.extend((name, key) for key in sorted(keys))
    paths.extend(("profile", "models", key) for key in sorted(EXPECTED_MODEL_KEYS))
    return tuple(paths)


def without(payload: Mapping[str, object], path: tuple[str, ...]) -> dict[str, object]:
    """指定経路のキーを 1 つだけ削った複製を返す。"""
    broken: dict[str, object] = copy.deepcopy(dict(payload))
    current: object = broken
    for key in path[:-1]:
        assert isinstance(current, dict)
        nested: object = current[key]
        if isinstance(nested, list):
            nested = nested[0]
        current = nested
    assert isinstance(current, dict)
    del current[path[-1]]
    return broken


# --------------------------------------------------------------------------
# キー網羅
# --------------------------------------------------------------------------


def test_manifest_covers_every_specified_key() -> None:
    """仕様書 §4 T4 のキー一覧が漏れなく存在する。"""
    assert_manifest_shape(manifest_dict(load_config(DEFAULT_CONFIG)))


@pytest.mark.parametrize(
    "path", removal_paths(), ids=[".".join(path) for path in removal_paths()]
)
def test_removing_any_specified_key_fails_the_coverage_check(
    path: tuple[str, ...],
) -> None:
    """キーを 1 つ消すと網羅テストが落ちる (期待値集合が効いていることの証明)。"""
    broken = without(manifest_dict(load_config(DEFAULT_CONFIG)), path)

    with pytest.raises(AssertionError):
        assert_manifest_shape(broken)


def test_manifest_schema_version_and_run_id_are_present() -> None:
    payload = manifest_dict(load_config(DEFAULT_CONFIG))

    assert payload["schema_version"] == SCHEMA_VERSION
    run_id = payload["run_id"]
    assert isinstance(run_id, str)
    assert run_id


def test_started_at_is_iso8601_utc() -> None:
    started_at = manifest_dict(load_config(DEFAULT_CONFIG))["started_at_utc"]

    assert isinstance(started_at, str)
    assert started_at.endswith("Z")
    assert "T" in started_at


# --------------------------------------------------------------------------
# 配線 (E8): マニフェストは「実際に使われた設定」を写す
# --------------------------------------------------------------------------


GenerationVariant = Callable[[GenerationParams], GenerationParams]

E8_CASES: tuple[
    tuple[str, GenerationVariant, GenerationVariant, object, object], ...
] = (
    (
        "temperature",
        lambda g: dataclasses.replace(g, temperature=0.1),
        lambda g: dataclasses.replace(g, temperature=1.3),
        0.1,
        1.3,
    ),
    (
        "top_p",
        lambda g: dataclasses.replace(g, top_p=0.5),
        lambda g: dataclasses.replace(g, top_p=0.95),
        0.5,
        0.95,
    ),
    (
        "context_tokens",
        lambda g: dataclasses.replace(g, context_tokens=4096),
        lambda g: dataclasses.replace(g, context_tokens=16384),
        4096,
        16384,
    ),
    (
        "max_output_tokens",
        lambda g: dataclasses.replace(g, max_output_tokens=128),
        lambda g: dataclasses.replace(g, max_output_tokens=2048),
        128,
        2048,
    ),
    (
        "seed",
        lambda g: dataclasses.replace(g, seed=7),
        lambda g: dataclasses.replace(g, seed=12345),
        7,
        12345,
    ),
    (
        "model",
        lambda g: dataclasses.replace(g, model="qwen3-14b"),
        lambda g: dataclasses.replace(g, model="qwen3-8b"),
        "qwen3-14b",
        "qwen3-8b",
    ),
)


@pytest.mark.parametrize(
    ("field", "variant_a", "variant_b", "expected_a", "expected_b"),
    E8_CASES,
    ids=[case[0] for case in E8_CASES],
)
def test_manifest_reflects_actual_config(
    field: str,
    variant_a: GenerationVariant,
    variant_b: GenerationVariant,
    expected_a: object,
    expected_b: object,
) -> None:
    """E8: 設定を変えるとマニフェストの該当値も変わる (雛形のハードコードでない)。"""
    base = load_config(DEFAULT_CONFIG)
    config_a = dataclasses.replace(base, generation=variant_a(base.generation))
    config_b = dataclasses.replace(base, generation=variant_b(base.generation))

    assert expected_a != expected_b, "掃引ケースが同値では配線を検出できない"
    assert section(manifest_dict(config_a), "generation")[field] == expected_a
    assert section(manifest_dict(config_b), "generation")[field] == expected_b


def test_manifest_generation_covers_all_generation_params() -> None:
    """generation セクションが GenerationParams の全フィールドを写している。"""
    generation = load_config(DEFAULT_CONFIG).generation
    fields = {field.name for field in dataclasses.fields(generation)}

    assert set(EXPECTED_SECTION_KEYS["generation"]) == fields


# --------------------------------------------------------------------------
# 配線 (E9): runtime.kind が api_style と送信先を決める
# --------------------------------------------------------------------------


def test_runtime_kind_changes_the_recorded_api_style_and_endpoint() -> None:
    """E9: ``runtime.kind`` を変えると api_style と endpoint_url の両方が変わる。

    ``kind`` が「マニフェストに記録されるだけの飾り」に戻ると (F-2-003 の再発)、
    2 つの kind で同じ api_style / endpoint_url が記録され、このテストが落ちる。
    """
    base = load_config(DEFAULT_CONFIG)
    assert base.runtime.kind == "ollama"
    compat = dataclasses.replace(
        base, runtime=dataclasses.replace(base.runtime, kind="openai_compatible")
    )

    native_runtime = section(manifest_dict(base), "runtime")
    compat_runtime = section(manifest_dict(compat), "runtime")

    assert native_runtime["kind"] == "ollama"
    assert native_runtime["api_style"] == "ollama_native"
    assert native_runtime["endpoint_url"] == "http://localhost:11434/api/chat"

    assert compat_runtime["kind"] == "openai_compatible"
    assert compat_runtime["api_style"] == "openai_compatible"
    assert (
        compat_runtime["endpoint_url"] == "http://localhost:11434/v1/chat/completions"
    )

    # base_url は 1 文字も変えていない。変わったのは kind から導出される 2 値だけ。
    assert native_runtime["base_url"] == compat_runtime["base_url"]
    assert native_runtime["api_style"] != compat_runtime["api_style"]
    assert native_runtime["endpoint_url"] != compat_runtime["endpoint_url"]


def test_shipped_configs_record_the_route_they_select() -> None:
    """設定ファイルの差し替えだけで記録される経路が変わる (E9 の出荷設定版)。"""
    native = section(manifest_dict(load_config(DEFAULT_CONFIG)), "runtime")
    compat = section(manifest_dict(load_config(COMPAT_CONFIG)), "runtime")

    assert native["endpoint_url"] == "http://localhost:11434/api/chat"
    assert compat["endpoint_url"] == "http://localhost:11434/v1/chat/completions"


def test_manifest_vram_matches_the_estimate() -> None:
    config = load_config(DEFAULT_CONFIG)
    estimate = estimate_resolved_profile(resolve_profile(config), config)
    vram = section(manifest_dict(config), "vram")

    assert vram["total_gib"] == pytest.approx(estimate.total_gib)
    assert vram["budget_gib"] == pytest.approx(estimate.budget_gib)
    assert vram["kv_cache_gib"] == pytest.approx(estimate.kv_cache_gib)
    assert vram["weights_total_gib"] == pytest.approx(estimate.weights_total_gib)
    assert vram["within_budget"] is True


def test_manifest_profile_lists_every_model_of_the_profile() -> None:
    config = load_config(DEFAULT_CONFIG)
    profile = section(manifest_dict(config), "profile")
    models = profile["models"]
    assert isinstance(models, list)

    assert profile["name"] == "rag_default"
    assert [model["model_id"] for model in models] == [
        "qwen3-14b",
        "ruri-v3-310m",
        "bge-reranker-v2-m3",
    ]
    assert models[0]["served_name"] == "qwen3:14b-q4_K_M"
    assert models[0]["quantization"] == "Q4_K"


def recorded_serving_runtimes(
    config: AppConfig, models: tuple[ModelSpec, ...], name: str
) -> dict[str, str]:
    """指定した ModelSpec 並びでマニフェストを組み、載せ先の記録を取り出す。"""
    profile = ResolvedProfile(name=name, models=models)
    estimate = estimate_resolved_profile(profile, config)
    payload: object = json.loads(
        build_manifest(config, profile, estimate, DEFAULT_CONFIG).to_json()
    )
    assert isinstance(payload, dict)
    recorded = section({str(k): v for k, v in payload.items()}, "profile")["models"]
    assert isinstance(recorded, list)
    result: dict[str, str] = {}
    for entry in recorded:
        assert isinstance(entry, dict)
        model_id = entry["model_id"]
        serving_runtime = entry["serving_runtime"]
        assert isinstance(model_id, str)
        assert isinstance(serving_runtime, str)
        result[model_id] = serving_runtime
    return result


def test_serving_runtime_change_is_visible_in_the_manifest() -> None:
    """E15: ``ModelSpec.serving_runtime`` を変えるとマニフェストの記録も変わる。

    ``serving_runtime`` を追加しただけでマニフェストに配線しないと、この属性は
    Phase 3 まで誰にも観測されない飾りになる (D-15 が ``source_note`` に散文で
    書く案を退けた理由が消える)。出荷設定では「リランカーだけ別プロセス」が
    記録され、全件を同じ値に潰すとここが落ちる。
    """
    config = load_config(DEFAULT_CONFIG)
    profile = resolve_profile(config)

    shipped = recorded_serving_runtimes(config, profile.models, profile.name)

    assert shipped == {
        "qwen3-14b": "ollama",
        "ruri-v3-310m": "ollama",
        "bge-reranker-v2-m3": "llama_cpp_server",
    }

    # 掃引: カタログ側の値を変えると記録側も追随する (雛形の固定値ではない)
    swept = recorded_serving_runtimes(
        config,
        tuple(
            dataclasses.replace(spec, serving_runtime="external")
            for spec in profile.models
        ),
        profile.name,
    )

    assert set(swept.values()) == {"external"}
    assert swept != shipped


# --------------------------------------------------------------------------
# 再現性 (config_sha256)
# --------------------------------------------------------------------------


def test_config_sha256_is_content_addressed(tmp_path: Path) -> None:
    """同じ内容なら一致し、1 文字変えると変わる。"""
    original = DEFAULT_CONFIG.read_text(encoding="utf-8")
    same = tmp_path / "same.toml"
    same.write_text(original, encoding="utf-8")
    changed = tmp_path / "changed.toml"
    changed.write_text(
        original.replace("temperature = 0.7", "temperature = 0.8"), encoding="utf-8"
    )

    assert compute_config_sha256(same) == compute_config_sha256(DEFAULT_CONFIG)
    assert compute_config_sha256(changed) != compute_config_sha256(DEFAULT_CONFIG)


# --------------------------------------------------------------------------
# 永続化
# --------------------------------------------------------------------------


def test_write_manifest_creates_json_file_under_output_dir(tmp_path: Path) -> None:
    manifest = make_manifest(load_config(DEFAULT_CONFIG))
    output_dir = tmp_path / "runs"

    destination = write_manifest(manifest, output_dir)

    assert destination.parent == output_dir
    assert re.fullmatch(
        rf"\d{{8}}T\d{{6}}Z-{re.escape(manifest.run_id)}\.json", destination.name
    ), destination.name
    payload: object = json.loads(destination.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert_manifest_shape({str(key): value for key, value in payload.items()})


def test_write_manifest_does_not_touch_repository_outputs(tmp_path: Path) -> None:
    """テストは必ず tmp_path に書く (リポジトリの outputs/ を汚さない)。"""
    manifest = make_manifest(load_config(DEFAULT_CONFIG))

    destination = write_manifest(manifest, tmp_path / "runs")

    assert REPO_ROOT not in destination.parents


# --------------------------------------------------------------------------
# guard_test (D-05)
# --------------------------------------------------------------------------


def test_manifest_and_logs_never_contain_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D-05 guard: api_key の値はマニフェスト JSON にもログにも現れない。

    現れてよいのは ``api_key_env`` (環境変数「名」) だけ。
    """
    monkeypatch.setenv("LLMKIT_API_KEY", API_KEY)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3:14b-q4_K_M",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    with (
        caplog.at_level(logging.DEBUG, logger="llmkit"),
        httpx.Client(transport=httpx.MockTransport(handler)) as http_client,
    ):
        result = bootstrap(
            EXTERNAL_CONFIG,
            http_client=http_client,
            output_dir=tmp_path / "runs",
        )
        # 実際に 1 往復させ、リクエスト組み立て時のログも検査対象に含める。
        result.client.chat([ChatMessage(role="user", content="ping")])

    assert result.manifest_path is not None
    manifest_json = result.manifest_path.read_text(encoding="utf-8")

    assert API_KEY not in manifest_json
    assert API_KEY not in result.manifest.to_json()
    assert API_KEY not in caplog.text
    assert "LLMKIT_API_KEY" in manifest_json
    payload: object = json.loads(manifest_json)
    assert isinstance(payload, dict)
    assert section(payload, "runtime")["api_key_env"] == "LLMKIT_API_KEY"


def test_base_url_with_inline_credentials_cannot_reach_the_manifest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """D-05 guard 拡張 (F-1-016): base_url 経由の秘密混入も 4 経路すべてで防ぐ。

    ``runtime.base_url`` は ``ManifestRuntime.base_url`` としてマニフェストに
    そのまま書かれる (D-05 の伏字化対象は api_key のみ)。そのため base_url に
    userinfo (user:pass@) を含められると、マニフェスト JSON・起動ログ・CLI
    出力・例外メッセージの 4 経路すべてに平文で複製される。config 層の
    バリデータ (``RuntimeConfig._validate_base_url``) が ``bootstrap()`` より
    前で ``ConfigError`` を送出するため、そもそもマニフェストが組み立てられる
    ところまで到達しないことを固定する。
    """
    secret = "sup3rs3cret"
    config_path = tmp_path / "malicious.toml"
    config_path.write_text(
        DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
            'base_url = "http://localhost:11434/v1"',
            f'base_url = "http://admin:{secret}@localhost:11434/v1"',
            1,
        ),
        encoding="utf-8",
    )

    with (
        caplog.at_level(logging.DEBUG, logger="llmkit"),
        pytest.raises(ConfigError) as excinfo,
    ):
        bootstrap(config_path, output_dir=tmp_path / "runs")

    assert secret not in str(excinfo.value)
    assert secret not in caplog.text
    assert not (tmp_path / "runs").exists(), "マニフェストが書き出されてはいけない"


@pytest.mark.parametrize(
    ("mutated_base_url", "case_id"),
    [
        pytest.param("http://localhost:11434/v1?key=sup3rs3cret", "query", id="query"),
        pytest.param(
            "http://localhost:11434/v1#sup3rs3cret", "fragment", id="fragment"
        ),
    ],
)
def test_base_url_with_query_or_fragment_secret_cannot_reach_the_manifest(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mutated_base_url: str,
    case_id: str,
) -> None:
    """D-05 guard 拡張 (F-2-007): base_url のクエリ・フラグメント経由の秘密混入。

    F-1-016 は userinfo (user:pass@) のみを塞いでおり、``?key=SECRET`` や
    ``#SECRET`` の形で秘密を書けてしまう取りこぼしが round 2 で見つかった
    (CWE-532)。``test_base_url_with_inline_credentials_cannot_reach_the_manifest``
    と同じ構造で、クエリ・フラグメントのケースを追加して回帰を防ぐ。
    """
    secret = "sup3rs3cret"
    config_path = tmp_path / f"malicious-{case_id}.toml"
    config_path.write_text(
        DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
            'base_url = "http://localhost:11434/v1"',
            f'base_url = "{mutated_base_url}"',
            1,
        ),
        encoding="utf-8",
    )

    with (
        caplog.at_level(logging.DEBUG, logger="llmkit"),
        pytest.raises(ConfigError) as excinfo,
    ):
        bootstrap(config_path, output_dir=tmp_path / "runs")

    assert secret not in str(excinfo.value)
    assert secret not in caplog.text
    assert not (tmp_path / "runs").exists(), "マニフェストが書き出されてはいけない"
