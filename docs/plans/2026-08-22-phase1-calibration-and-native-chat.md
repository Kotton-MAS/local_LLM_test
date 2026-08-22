# Phase 1 追補: Phase 0 実測にもとづくカタログ較正とネイティブ `/api/chat` 経路の追加

> 前提資料: `docs/phase0-vram-measurements.md` (実測記録) / `docs/plans/2026-08-22-phase1-inference-client-l2.md` (元仕様書) / `.claude/decisions.yaml` (D-01〜D-09)
> 位置づけ: 既存 `llmkit` への **追加変更**。新規パッケージは作らない。
>
> **承認済み (2026-08-22)**: Q-1 = `budget_gib` を 16.0 → **14.0** に下げる / Q-2 = 要件書の訂正を**今回のサイクルに含める**。

## 1. ゴール

Phase 0 実測を唯一の出典として、`llmkit` の「設定したコンテキスト長が実機に反映される」ことと「VRAM 見積りが実測と一致する」ことを同時に成立させる。

## 2. 現状認識

| パス | 内容 |
|---|---|
| `llmkit/client.py:L170-L448` | `OpenAICompatibleClient`。`/chat/completions` 固定。`options.num_ctx` をボディに載せるが実機で無視される |
| `llmkit/client.py:L318-L402` | `_raise_for_error_status`。**OpenAI のエラースキーマを一切パースしておらず**、HTTP ステータス + 本文小文字部分一致のみで判定している |
| `llmkit/client.py:L95-L114` | `ChatResult`。時間指標は `latency_s` のみ。prompt/eval 分離の受け皿が無い (F-1-006) |
| `llmkit/catalog.py:L59-L126` | 静的テーブル。全 5 件の `source_note` が `(仮)` |
| `llmkit/vram.py:L133-L216` | 見積り式と予算判定。式そのものは変更不要 |
| `llmkit/bootstrap.py:L103` | `OpenAICompatibleClient` を直接 new。`runtime.kind` を見ていない (F-2-003) |
| `llmkit/manifest.py:L133-L150` | `ManifestRuntime`。`kind` は記録するが、それがどの経路を選んだかは記録されない |
| `tests/conftest.py:L35-L48,L162-L163` | `SUCCESS_PAYLOAD` (OpenAI 形状) と `_default_handler`。6 モジュールが import している |

### 既存の慣習で守るべきもの

1. **ゴールデン値は 1 か所でロックする**: カタログ数値は `tests/test_catalog.py::GOLDEN_ROWS` が、プロファイル合計は `tests/test_vram.py` が固定する。
2. **例外は必ず `LlmkitError` 系に翻訳し `remediation` を付ける**。ランタイム固有の型を公開 API に出さない。
3. **テストは `httpx.MockTransport` で完結する** (D-02)。autouse fixture がソケットを機械的に塞いでいる。

### 影響範囲

`llmkit/{catalog,client,bootstrap,manifest,cli,__init__}.py`、`configs/*.toml`、`tests/` の 8 ファイル、`.claude/decisions.yaml`、`docs/`。
`llmkit/{config,errors,vram}.py` は**変更しない**。`main.py` / `Makefile` / CI 定義も変更しない。

## 3. 前提・制約

### ハード制約

- 既存の 218 テストを壊さない。**テストを通すためにアサーションを緩めない**。期待値の変更は §4 に変更前→変更後を明記した行だけを許す。
- mypy `strict` + `disallow_any_explicit`。`Any` 明示禁止。`pydantic.BaseModel` を継承しない (D-08)。
- ruff `T20` (`print()` 禁止)。
- テストは実 HTTP を出さない (D-02)。ネイティブ経路も `MockTransport` のみで検証する。
- 推論ランタイム固有の**型**を公開 API に露出させない (`OllamaNativeClient` という**クラス名**は許容)。
- **既存の公開シグネチャを変更しない**: `OpenAICompatibleClient.__init__` / `.chat` / `.served_name` / `.endpoint_url`、`ChatClient` Protocol、`build_manifest(...)`、`bootstrap(...)`、`BootstrapResult`。追加のみ許す。
- 埋め込み・リランカーの推論呼び出しは実装しない (Phase 3)。

### ソフト制約

- ネイティブ実装は `llmkit/client.py` 内に置く (新規モジュールを作らず `tests/test_layout.py::_SUBMODULE_NAMES` の手動更新を避ける)。800 行を超えるなら分割を提案してよい。
- オーバーヘッド分離仮定 (0.8 GiB) は据え置く。実測は合計値のみのため。
- 追加依存は入れない。

## 4. タスク分解

**実行順序 T1 → T2 → T3 → T4 → T5 を厳守する。** T1+T2 完了時点で `make ci` が緑になることを確認してから T3 に進む (§7 リスク3)。

### T1: カタログを Phase 0 実測値で較正する

| model_id | weights_gib | kv_gib_per_1k | max_context_tokens | 根拠 |
|---|---|---|---|---|
| `qwen3-14b` | 9.0 → **7.81** | 0.10 → **0.1575** | 16384 → **32768** | 実測 3 点 |
| `qwen3-8b` | 4.5 → **4.18** | 0.08 → **0.1425** | 32768 (据置) | 実測 3 点 |
| `gpt-oss-20b` | 11.5 → **11.32** | 0.004 → **0.0244** | 131072 (据置) | 実測 2 点 |
| `ruri-v3-310m` | 0.7 (据置) | 0.0 | 8192 (据置) | **未実測** |
| `ruri-reranker` | 0.8 (据置) | 0.0 | 512 (据置) | **未実測** |

- `source_note` を書き分ける。生成 3 モデルは `(仮)` を外し `実測 2026-08-22 / docs/phase0-vram-measurements.md` とフィットに使った num_ctx を書く。`gpt-oss-20b` には「GPU 常駐上限は 65,536〜98,303 の間 (98,304 で CPU オフロード)」を追記。
- ruri 2 モデルは `(仮)` を残し **`取得不能`** と「GGUF 非提供のため `ollama pull` できない。方式は Phase 3 で決定」を明記 (D-14)。
- `tests/test_catalog.py` に **`test_catalog_reproduces_phase0_measurements`** を新設。GPU 内に収まった 8 点を parametrize し `weights_gib + kv*(ctx/1024) + 0.8` が実測と `abs=0.02` で一致することを検証する。
- `test_documented_context_limits_match_requirements` → `test_context_limits_match_phase0_measurements` に改名し `qwen3-14b == 32768` へ。
- `tests/test_vram.py::test_estimate_follows_the_documented_formula` の `weights_gib` 期待値 `9.0 → 7.81`。
- `tests/test_client_errors.py::test_context_length_400_maps_to_context_length_error`: 要求値を `32768 → 65536`、期待文字列を `"65536"` と `"32768"` に (「要求値と上限が別の値」の意図は維持)。

**受け入れ基準**: 実測 8 点 (`qwen3-14b` 4096→9.24 / 16384→11.13 / 32768→13.65、`qwen3-8b` 8192→6.12 / 16384→7.26 / 32768→9.54、`gpt-oss-20b` 32768→12.90 / 65536→13.68) がすべて緑。値を 1 つ書き換えると本テストと `GOLDEN_ROWS` の**両方**が落ちる (E11)。

**触るファイル**: `llmkit/catalog.py`, `tests/test_catalog.py`, `tests/test_vram.py`, `tests/test_client_errors.py` / **所要**: M

> **実装時の追記 (T1 実装者 / 2026-08-22)** — 仕様に書かれていなかった判断:
>
> 1. `tests/test_catalog.py::test_every_spec_declares_a_source_note_and_served_name` は全 5 モデルに
>    `"(仮)" in source_note` を課していた。較正で生成 3 モデルから `(仮)` が外れるため、この 1 本を
>    **2 本に分割**した (緩和ではなく、モデル群ごとに強い主張へ再アンカーした):
>    `test_calibrated_models_cite_the_phase0_measurements` (較正 3 モデル: `実測 2026-08-22` と
>    `docs/phase0-vram-measurements.md` を含み `(仮)` を含まない) と、
>    `test_unavailable_models_declare_their_unavailability` (ruri 2 モデル: `(仮)` / `取得不能` / `Phase 3` を含み
>    `実測 2026-08-22` を含まない)。後者は **D-14 の guard_test** であり、T5 で決定を追記する時点で実在する。
>    元の 1 本は `served_name` / `max_context_tokens > 0` / `source_note` が空でないことの検査として残す。
> 2. `tests/test_catalog.py` の実測再現テストが使うオーバーヘッド 0.8 GiB は `FIT_OVERHEAD_GIB` として
>    明示した (実測は合計値のみで、重みとの分離はこの仮定に依存するため)。
> 3. §4 T4 の期待値更新表のうち **理由が「T1 較正」の 3 行** (`test_bootstrap.py:74` `12.90→12.63` /
>    `:96-97` `6.58→7.26` / `:334` `12.90→12.63`) は T1 の変更が直接の原因なので **T1 で更新した**。
>    「T1+T2 完了時点で `make ci` が緑」を成立させるため。T4 に残る期待値更新は 4 行 (`:224` / `test_manifest.py:84` /
>    `test_acceptance_phase1.py:46-68`、および `:76` は T2 で実施済み)。

### T2: VRAM 予算を実測値にそろえ、`budget_gib=12.0` の回避を撤去する

- `configs/default.toml` と `configs/external_openai.toml` の `budget_gib` を **16.0 → 14.0**。
  根拠: 実測で 13.68 GiB は 100% GPU、14.46 GiB 相当 (num_ctx=98,304) で CPU オフロード。14.0 はその区間の内側。カード容量 16 GiB は「増分」の上限ではない (アイドルで 0.60 GiB を使用)。
- `tests/test_vram.py`:
  - `REQUIREMENTS_BAND_GIB` と `test_documented_profiles_match_requirements_table` を廃し **`test_shipped_profiles_match_phase0_measurements`** に置換 (出典を要件書の帯から実測へ)。
  - `test_budget_threshold_changes_verdict`: baseline `12.9 → 12.63`、generous `16.0 → 14.0`、strict `12.5 → 12.0` (excess 0.63)。元仕様書 §4 T2 の「⚠ 仕様との食い違い」注記は**解消済みとして削除**。
  - `test_synthetic_oversized_profile_raises_with_full_breakdown`: budget `16.0 → 14.0`、期待 `"16.00" → "14.00"` / `"4.80" → "6.80"`。合計 `"20.80"` は据置。
  - `test_all_vram_values_are_gib` (D-03 guard) の末尾を差し替え: (a) `budget_gib = 16.0` と書いた設定が 16.0 として読めること、(b) `configs/default.toml` の `budget_gib <= 15.99` (GB と取り違えて 17.17 を書いたら落ちる)。
  - **`test_budget_reflects_the_measured_gpu_resident_ceiling`** を新設 (D-12 guard)。`long_context @65536 = 13.68` が収まり `@98304 = 14.46` と `@131072 = 15.24` が超過側になること。
- **回避の撤去**: `budget_gib = 12.0` を使う 4 箇所を、予算を触らず `active_profile = "oversized"` かつ `context_tokens = 131072` (見積り 16.74、予算 14.0 を 2.74 超過) に置換する。
  対象: `test_bootstrap.py::test_oversized_profile_warns_and_aborts_without_http` (D-04 guard、`budget_gib == approx(14.0)` に更新) / `test_cli_doctor_exits_1_and_makes_no_request_when_budget_exceeded` / `test_acceptance_phase1.py::test_l296_oversized_profile_warns_and_stops` / `test_lowering_the_budget_alone_flips_bootstrap_from_ok_to_abort` (ok 側 `14.0 → 17.0`、ng 側は既定のまま)。
- `test_remote_runtime_skips_the_budget_guard` の置換元文字列 `"budget_gib = 16.0"` → `"budget_gib = 14.0"`。

**受け入れ基準**: `rg "budget_gib = 12\.0"` が**ヒット 0 件**。`test_shipped_profiles_match_phase0_measurements` が `rag_default @16384 = 12.63` (収まる) / `long_context @65536 = 13.68` (収まる) / `long_context @131072 = 15.24` (**超過**) / `oversized @131072 = 16.74` (超過) を固定。D-04 guard が **既定の `budget_gib` のまま**通る。

**触るファイル**: `configs/*.toml`, `tests/test_vram.py`, `tests/test_bootstrap.py`, `tests/test_acceptance_phase1.py` / **所要**: M

> **実装時の追記 (T2 実装者 / 2026-08-22)** — 仕様に書かれていなかった判断:
>
> 1. **`tests/test_config.py::test_load_default_config_returns_app_config` の
>    `budget_gib` 期待値を `16.0 → 14.0` に更新した。** この 1 行は出荷設定の実値を読む再アンカーであり、
>    更新しないと `make ci` が緑にならない。§5 テスト観点の「変えない: `test_config.py`」に対する
>    唯一の例外で、同ファイルの他の行 (`VALID_TOML` 内の `budget_gib = 16.0` を含む) は変更していない。
> 2. `test_shipped_profiles_match_phase0_measurements` は 4 点を parametrize し、合計を `abs=0.005`
>    (小数第 2 位まで一意) で固定した。判定 (`within_budget`) も同時に固定している。
> 3. `test_all_vram_values_are_gib` (D-03 guard) は「`budget_gib = 16.0` と書いた設定が 16.0 として読める」を
>    conftest の `write_config_variant` で作った一時設定で検証する。このため `tests/test_vram.py` が
>    conftest ヘルパを import するようになった (署名に `tmp_path` を追加。guard の関数名は不変)。
> 4. `test_lowering_the_budget_alone_flips_bootstrap_from_ok_to_abort` は ok/ng の差を `budget_gib` だけに
>    するため、**両方**に `context_tokens = 131072` と `active_profile = "oversized"` を与え、ok 側のみ
>    `budget_gib = 17.0` にした (ng 側は既定 14.0 のまま)。
> 5. `configs/*.toml` の `budget_gib` 行の直上に、14.0 の根拠 (実測 13.68 は 100% GPU / 14.46 相当で
>    オフロード / カード容量は増分の上限ではない) をコメントで書いた。設定値だけを見た人が
>    「16 GiB のカードなのに 14.0」を誤って戻さないため。

### T3: ネイティブ `/api/chat` クライアントと `runtime.kind` 分岐を実装する

**抽象の置き方**: `ChatClient` Protocol は**変更しない**。共有の抽象基底 `_HttpChatClient` に「HTTP 送信・接続エラー翻訳・ステータス/本文からの例外翻訳・api_key ヘッダ・所有権と close」を集約し、`OpenAICompatibleClient` と新設 `OllamaNativeClient` の 2 具象がボディ組み立てと応答パースだけを実装する。**`runtime.kind` の分岐はファクトリ `create_chat_client` 1 か所だけ**に置く (内部分岐案は採らない。クラス名と挙動が乖離し、既存 3 テストファイルの回帰検出力が落ちるため)。これにより F-2-003 (`kind` が飾り) が解消される。

追加する要素:

- `ApiStyle = Literal["ollama_native", "openai_compatible"]`
- `api_style_for(kind) -> ApiStyle`
- `endpoint_url_for(base_url, style) -> str` — `openai_compatible` は `{base_url}/chat/completions` (現状と同一)、`ollama_native` は **`base_url` 末尾の `/v1` を 1 セグメントだけ除去**して `/api/chat` を付ける (`http://localhost:11434/v1` → `http://localhost:11434/api/chat`)。設定キーは増やさない (D-11)。
- `OllamaNativeClient` — ボディは `{model, messages, stream: false, options: {num_ctx, temperature, top_p, num_predict, seed}}` の**トップレベル 4 キー厳密**。`stream` の省略は NDJSON を招くため必須。`max_output_tokens` → `options.num_predict`。
- **応答パース** (pydantic dataclass + `TypeAdapter`、`extra="ignore"`): 必須 = `model` / `message.content` / `done`。任意 = `done_reason` / `prompt_eval_count` / `prompt_eval_duration`(ns) / `eval_count` / `eval_duration`(ns)。`done is False` は `UpstreamError`。`finish_reason = done_reason or "stop"`。
  **任意にする理由**: Ollama はプロンプトキャッシュヒット時に `prompt_eval_*` を返さない。必須にすると正常な生成が `UpstreamError` になる。D-07 の趣旨は `ChatTimings` が `None` で欠測を明示することで満たす (0 で埋めない)。
- **エラー翻訳**: 既存 `_raise_for_error_status` は OpenAI のエラースキーマをパースしておらず、HTTP ステータス + 本文の小文字部分一致だけで判定している。したがってネイティブ形式にもそのまま成立する。**この関数を基底クラスへ移して両経路で共有**する。ネイティブ固有の追加は 1 点: **HTTP 200 かつ本文トップレベルに非空の `error` がある場合**も同じ翻訳表に流す。JSON デコード失敗時の `remediation` は「`stream=false` を送っているか / `base_url` がネイティブ API を指しているか」に差し替える。
- **`ChatTimings`** (frozen): `prompt_eval_count` / `prompt_eval_seconds` / `eval_count` / `eval_seconds` (すべて `| None`)、property `prompt_tokens_per_second` / `eval_tokens_per_second` (欠測・0 秒なら `None`)。`ChatResult` に**末尾に既定値つきで** `timings: ChatTimings | None = None` を追加。ns → 秒は `/ 1e9`。互換経路は常に `None`。
- `create_chat_client(config, *, http_client=None) -> ChatClient`
- `client.__all__` に `ApiStyle` / `ChatTimings` / `OllamaNativeClient` / `api_style_for` / `create_chat_client` / `endpoint_url_for` を追加。
- `tests/conftest.py`: `NATIVE_SUCCESS_PAYLOAD` を追加し `_default_handler` を「`request.url.path` が `/api/chat` で終わるならネイティブ形状」に分岐。**`SUCCESS_PAYLOAD` の名前と中身は変更しない** (6 モジュールが import 済み)。

**受け入れ基準**:
- `endpoint_url_for` が **HTTP を 1 バイトも出さずに**検証できる純関数であること。
- `test_native_request_body_contains_exactly_the_expected_keys`: トップレベルが `{model, messages, stream, options}` に厳密一致、`stream is False`、`options` が 5 キーに厳密一致。
- **ネイティブ版パラメータ配線テスト**: 6 項目すべてを parametrize し、設定を変えるとボディの対応パスが変わる (互換経路の `test_param_wiring.py` と同じ 6 項目・別のパス)。
- **エラー翻訳の等価性**: 5 経路について同じ本文・ステータスをネイティブに流すと**互換経路と同じ例外型**が出る。加えて `200 + {"error": ...}` が正しく翻訳される。
- `RuntimeUnavailableError` のメッセージに `runtime.base_url` (`endpoint_url` ではない) と `ollama serve` が含まれる (既存 `test_l297` を壊さないため)。
- `test_native_result_exposes_prompt_and_eval_timings`: `prompt_eval_count=11 / prompt_eval_duration=550_000_000 / eval_count=64 / eval_duration=1_000_000_000` で `prompt_tokens_per_second == approx(20.0)` かつ `eval_tokens_per_second == approx(64.0)`。`prompt_eval_count` が**欠けた**応答では `UpstreamError` にならず `None` (0.0 ではない)。
- `test_runtime_kind_selects_the_client_implementation`: **設定 1 行の違いだけで**実装と送信先が反転する。
- 既存の `tests/test_client.py` / `test_client_errors.py` / `test_param_wiring.py` は **1 行も変更せずに通る** (T1 の `max_context_tokens` 由来の 1 箇所を除く)。

**触るファイル**: `llmkit/client.py`, `tests/test_client_native.py` (新規), `tests/test_client_factory.py` (新規), `tests/conftest.py` / **所要**: L

> **実装時の追記 (T3 実装者 / 2026-08-23)** — 仕様に書かれていなかった判断:
>
> 1. **`client.__all__` への追加は T4 へ送った。** T3 だけで `__all__` を増やすと
>    `tests/test_layout.py::test_public_api_matches_the_union_of_submodule_all`
>    (llmkit.__all__ = 全サブモジュール __all__ の和集合) が落ち、T3 完了時点で `make ci` が
>    緑にならないため。`llmkit/__init__.py` の再エクスポートは T4 の担当なので、**T4 で
>    `client.__all__` への 6 シンボル追加と `__init__.py` の再エクスポートを同時に**入れる。
>    追加対象: `ApiStyle` / `ChatTimings` / `OllamaNativeClient` / `api_style_for` /
>    `create_chat_client` / `endpoint_url_for`。
> 2. **`create_chat_client` の戻り型は仕様どおり `ChatClient` (Protocol)** にした。Protocol は
>    `chat` しか宣言していないため、**T4 の bootstrap は戻り値から `endpoint_url` /
>    `served_name` を読めない**。`endpoint_url_for(config.runtime.base_url,
>    api_style_for(config.runtime.kind))` と `resolve_model_spec(...)` で設定から直接導出するか、
>    具象型で受けること (Protocol の変更は §3 ハード制約で禁止)。
> 3. **HTTP 200 + 本文 `error` の翻訳で、コンテキスト長超過の分岐も通す。** 元の翻訳表は
>    コンテキスト長超過だけ `status == 400` で門番していた。200 + `error` はステータスで
>    判別できないため `_raise_translated_error(..., from_body_error=True)` を渡し、
>    この経路に限り 400 門番を迂回する。互換経路の呼び出しは既定 (`False`) のままで挙動不変。
>    メッセージも専用文言 (「応答本文でエラーを報告しました (HTTP 200, ...)」) にした。
> 4. **`error` が空文字・空 dict の場合は正常応答として扱う** (仕様の「非空の `error`」の境界)。
>    `test_native_empty_error_field_does_not_break_a_successful_response` で固定した。
> 5. **ネイティブ経路の `TokenUsage` は欠測を 0 で埋める。** `TokenUsage` は 3 フィールドとも
>    `int` (公開シグネチャは変更不可) のため `prompt_eval_count` / `eval_count` が無い応答では
>    0 を入れる。欠測かどうかの識別は `ChatTimings` の `None` が担う (D-13 の趣旨はそちらで満たす)。
> 6. `ChatTimings` の 4 フィールドに既定値は付けない (構築時に 4 スロットすべてを明示させる)。
>    `_HttpChatClient.__enter__` の戻り型は基底へ移す都合で `Self` にした
>    (`OpenAICompatibleClient` では従来どおり同クラスに解決されるため公開シグネチャは実質不変)。
> 7. `tests/conftest.py` には `NATIVE_SUCCESS_PAYLOAD` に加えて **`NATIVE_CHAT_PATH` 定数**を
>    追加し、`_default_handler` の分岐条件をテスト側から参照できるようにした。
>    `NATIVE_SUCCESS_PAYLOAD` の内容は `SUCCESS_PAYLOAD` と同じ生成結果 (content「テスト応答」/
>    prompt 11 tok / eval 7 tok) をネイティブ形状で表したもの。
> 8. 検証: ネイティブ 6 項目のミューテーション (設定参照 → 既定値の定数) で
>    `test_native_generation_params_reach_request_body` の対応ケースのみが 6/6 落ちることを実測。
>    加えて D-10 / D-11 / D-13 の guard_test と「`stream` 省略」「200+error フック除去」も
>    それぞれ変異で落ちることを確認した。

### T4: bootstrap / CLI / マニフェストを新経路へ配線する

- `llmkit/bootstrap.py`: `OpenAICompatibleClient(...)` → `create_chat_client(...)`。`BootstrapResult` のフィールド構成は不変。
- `llmkit/manifest.py`: `ManifestRuntime` に `api_style: str` と `endpoint_url: str` を追加し `to_dict()` に含める。`build_manifest` のシグネチャは不変。
- `llmkit/cli.py`: `_report_startup` に `API 経路` の 1 行を追加。
- `llmkit/__init__.py`: T3 で追加したシンボルを再エクスポート。
- `configs/ollama_openai_compat.toml` を新設 (`kind = "openai_compatible"` / `base_url = "http://localhost:11434/v1"` / `is_local = true`)。**同一の Ollama に対して 2 経路を A/B 比較する**ための設定。`tests/test_catalog.py::CONFIG_PATHS` にも追加。

**既存テストの期待値更新 (すべて再アンカーであり緩和ではない)**:

| ファイル:行 | 変更前 | 変更後 | 理由 |
|---|---|---|---|
| `test_bootstrap.py:74` | `"12.90"` | `"12.63"` | T1 較正 |
| `test_bootstrap.py:76` | `"16.00"` | `"14.00"` | T2 予算 |
| `test_bootstrap.py:96-97` | `6.58` | `7.26` | T1 較正 |
| `test_bootstrap.py:224` | `.../v1/chat/completions` | `http://localhost:11434/api/chat` | T3 `kind="ollama"` |
| `test_bootstrap.py:334` | `"12.90"` | `"12.63"` | T1 較正 |
| `test_manifest.py:84` | 5 キー | `+ {api_style, endpoint_url}` | 記録項目追加 |
| `test_acceptance_phase1.py:46-68` | `OpenAICompatibleClient` を直接 new | `create_chat_client` 経由 | 受け入れ条件1 を**実際に使う経路**で測るため |

- `tests/test_manifest.py` に **`test_runtime_kind_changes_the_recorded_api_style_and_endpoint`** を追加 (E9)。

**受け入れ基準**: `make ci` が緑。`test_layout.py::test_public_api_matches_the_union_of_submodule_all` が**修正なしで**通る。`configs/default.toml` で `doctor` の送信先が `/api/chat`、`configs/ollama_openai_compat.toml` で `/v1/chat/completions`。変更した期待値が上表の 7 行**だけ**であること。

**触るファイル**: `llmkit/{bootstrap,manifest,cli,__init__}.py`, `configs/ollama_openai_compat.toml`, `tests/test_{bootstrap,manifest,acceptance_phase1,catalog}.py` / **所要**: M

> **実装時の追記 (T4 実装者 / 2026-08-23)** — 仕様に書かれていなかった判断:
>
> 1. **bootstrap の `endpoint_url` / `served_name` は設定から導出する。** `create_chat_client` の
>    戻り型 `ChatClient` (Protocol) は `chat` しか宣言しないため (T3 追記2)、
>    `endpoint_url_for(config.runtime.base_url, api_style_for(config.runtime.kind))` と
>    `resolve_model_spec(config.generation.model, is_local=config.runtime.is_local).served_name`
>    で求める。`BootstrapResult` のフィールド構成は不変。起動ログには `api_style=` を追加した。
> 2. **`llmkit/manifest.py` が `llmkit/client.py` を import する** (`api_style_for` /
>    `endpoint_url_for`)。`api_style` / `endpoint_url` の出典を bootstrap の表示値と 1 か所に
>    そろえるため。循環 import は起きない (client は manifest を import しない)。
>    `ManifestRuntime` の 2 フィールドの型は仕様どおり `str` (`ApiStyle` を公開スキーマに
>    露出させない)。
> 3. **CLI の追加行の書式は `API 経路   : {api_style} (runtime.kind={kind})`** とし、
>    `接続先` 行の直前に置いた。値は `result.manifest.runtime` から読む (表示とマニフェストで
>    出典が分岐しないため)。
> 4. **`configs/ollama_openai_compat.toml` は `configs/default.toml` と非コメント行の差が
>    `kind` の 1 行だけ**になるようにした (A/B 比較の対照条件を 1 変数に保つため)。
>    `is_local = true` (ローカル Ollama なので VRAM 予算ガードは有効)。この経路が
>    `options.num_ctx` を無視することをファイル冒頭のコメントに明記した。
> 5. **§5 安全性観点「新設 `configs/ollama_openai_compat.toml` に `api_key` キーが無い」の
>    テストは `tests/test_catalog.py::test_shipped_configs_never_contain_an_api_key_value`
>    として追加した。** 本来の置き場所である `tests/test_config.py` は §5 テスト観点で
>    「変えない」と指定されているため。判定は `CONFIG_PATHS` (出荷設定の一覧) をそのまま回す
>    ので、設定ファイルを増やしたときに検査から漏れない。
> 6. **E9 の出荷設定版として `tests/test_manifest.py::test_shipped_configs_record_the_route_they_select`
>    を追加した** (仕様が要求する `test_runtime_kind_changes_the_recorded_api_style_and_endpoint`
>    は `dataclasses.replace` で `kind` だけを掃引する版)。受け入れ基準の
>    「設定ファイルの差し替えだけで切り替わる」を CLI 実行なしで固定するため。
> 7. `llmkit/__init__.py` の冒頭 docstring を「OpenAI 互換推論クライアント層」から
>    「推論クライアント層 (経路は `runtime.kind` で選ぶ)」に直した。再エクスポートの追加で
>    パッケージの説明が事実と食い違うため。
> 8. 検証: `build_manifest` の `api_style` を定数に固定する変異と `endpoint_url` を
>    `runtime.base_url` に置き換える変異で、それぞれ E9 の 2 本が落ちることを実測した。
>    実機 (Ollama 0.32.15) で `doctor` を 2 設定で実行し、送信先が `/api/chat` と
>    `/v1/chat/completions` に切り替わることを確認した (後者は `ollama ps` の CONTEXT が
>    4096 に戻り、互換経路が `num_ctx` を無視するという Phase 0 実測を再現した)。

### T5: 意図的な決定の記録と、実測に反する文書の訂正

- `.claude/decisions.yaml` に D-10〜D-14 を追記 (**T3/T4 で guard_test が実在してから**)。
- `docs/localllmrequirements.md` を実測に合わせて訂正:
  - 「20B(MXFP4) は 128k コンテキストまで VRAM 内で完結」→ 実測で 65,536 まで。98,304 で CPU オフロード。
  - 実測速度 (14B 84.3 → 61.5 t/s、20B 57.5 → 134.3 t/s、8B 104.6 t/s を追記)。
  - 構成表に実測合計 (構成1 = 12.63 / 構成2 = 13.68 @65k / 構成3 = 16.74 @128k) を併記。
  - 「OpenAI 互換 API を全層の共通境界とする」に注記: **層の共通境界は `ChatClient` Protocol (L2 内部) であり、ワイヤプロトコルではない**。ローカル Ollama ではネイティブ `/api/chat` を使う (D-10)。
- `docs/phase0-vram-measurements.md` の「→ 設計判断が必要。未解決。」を本仕様書への参照と D-10 に差し替える。
- `docs/next-pr-candidates.md`: Phase 0 課題表の MEDIUM 2 件と INFO 1 件を「対応済み (本 PR)」に更新。INFO の「`max_context_tokens` をプロファイル単位で分けるべきか」は **不要と結論** (D-12: 常駐上限は同居モデルに依存するのでモデル単位の定数では表せない)。
- `README.md`: 2 経路の使い分けと 3 つの設定ファイルの意味を追記。

**受け入れ基準**: `check_decisions.py` が通り D-10〜D-14 の guard_test 5 本が実在して緑。`test_l298` が通る (**要件書を編集する際 L294-L298 より上の行数を増減させない**か、増減させたら `ACCEPTANCE_MAP` を同時に直す)。要件書に「128k が VRAM 内完結」が残っていない。

**触るファイル**: `.claude/decisions.yaml`, `docs/{localllmrequirements,phase0-vram-measurements,next-pr-candidates}.md`, `README.md` / **所要**: S

> **実装時の追記 (T5 実装者 / 2026-08-23)** — 仕様に書かれていなかった判断:
>
> 1. **要件書は行数を 1 行も増減させずに訂正した** (355 行のまま)。`ACCEPTANCE_MAP` が参照する
>    L294-L298 を動かさないため、追記はすべて既存行の書き換えに収めている。訂正箇所は
>    L64 (VRAM 表) / L74-L76 (構成表) / L81 (設計方針) / L92-L94 (採用モデル表) / L144 (共通境界の注記) の 9 行。
> 2. **構成表には「較正後見積り」と「実測」を書き分けた。** 構成1 / 構成3 は埋め込み・リランカーが
>    取得不能で同居実測ができていないため (D-14)、12.63 / 16.74 を実測と書くと出典が偽になる。
>    実測値として書いたのは構成2 の 13.68 GiB @64k と 14B 単体の 11.13 GiB @16k のみ。
> 3. **採用モデル表の「コンテキスト」列も訂正した** (20B: 128k → 64k、8B: `—` → 32k)。仕様は速度の
>    訂正しか挙げていないが、同じ表の隣接セルに「128k (VRAM 内完結)」が残ると §4 T5 の受け入れ基準
>    (「128k が VRAM 内完結」が残っていない) を満たさないため。
> 4. **`docs/phase0-vram-measurements.md` は「→ 未解決」の 1 行だけでなく直前の段落も現在形から過去形に
>    直した** (「現状では反映されません」が事実と食い違うため)。埋め込み・リランカーの節には D-14 への
>    参照を 1 行足した。
> 5. **`docs/next-pr-candidates.md` には「本 PR で併せて解消した既存 finding」の注記を足した**
>    (`F-1-006` → D-13 / `F-2-003` → D-10)。43 件の一覧そのものは書き換えていない (出典が
>    `triage.json` であり、再スキャンで再生成される表のため)。
> 6. **`README.md` には「2 つの API 経路」節を新設した** (`runtime.kind` の表 + どちらを使うべきか)。
>    あわせて `budget_gib = 14.0` が「カード容量ではなく増分の上限」であることを 1 行で書いた
>    (設定値だけを見た人が 16.0 に戻さないため。configs 側のコメントと同じ趣旨)。
> 7. 検証: `python3 ~/.claude/scripts/check_decisions.py .claude/decisions.yaml` が
>    「14 件の決定、全てに guard_test あり」で通り、D-10〜D-14 の guard_test 5 本を個別実行して
>    すべて緑であることを確認した。`make ci` も緑 (311 passed)。

## 5. 評価軸

### 機能観点
要件書 L294-L298 の 5 条件を引き続き 1:1 で測る。**L294 は今回からファクトリ経由**になり「実際に使われる経路」で測られる。新規の機能的主張は 1 つ: 「`kind = "ollama"` のとき `context_tokens` が `options.num_ctx` としてネイティブ `/api/chat` に届く」。MockTransport でボディまで、実機反映は live テスト。

### 性能観点
`uv run pytest` が **10 秒未満**。VRAM 見積りは引き続き純関数 (`test_estimate_is_pure_and_needs_no_gpu` を変更しない)。実測 t/s の記録は Phase 2 の担当。

### 安全性観点
api_key が config repr / ログ / マニフェスト JSON / 例外メッセージに平文で出ない — **ネイティブ経路の 5 失敗パターンでも**検証する。`UpstreamError` にボディ全文を載せない (ネイティブの `{"error": ...}` 本文も**マーカー判定にのみ使う**)。新設 `configs/ollama_openai_compat.toml` に `api_key` キーが無い。`_forbid_real_network` を変更しない。

### テスト観点
新規: `tests/test_client_native.py`, `tests/test_client_factory.py`。
変えない: `test_main.py`, `test_client.py`, `test_param_wiring.py`, `test_layout.py`, `test_testing_policy.py`, `test_config.py`。
カバレッジ 85% 以上を維持。ネイティブのエラー翻訳 5 分岐 + `200 + error` 分岐がすべて実行されること。

### ★ 有効性観点

既存 E1〜E8 を維持したうえで:

| # | 掃引する値 | 変わるべき出力 | テスト |
|---|---|---|---|
| E9 | `runtime.kind` | 送信先 URL / ボディ形状 / `manifest.runtime.api_style` | `test_client_factory.py::test_runtime_kind_selects_the_client_implementation` + `test_manifest.py::test_runtime_kind_changes_the_recorded_api_style_and_endpoint` |
| E10 | `generation.*` 6 項目 | **ネイティブ**ボディの `options.num_ctx` / `options.num_predict` 等 | `test_client_native.py::test_native_generation_params_reach_request_body` |
| E11 | カタログの `weights_gib` / `kv_gib_per_1k_tokens` | 実測 8 点の再現テストが落ちる | `test_catalog.py::test_catalog_reproduces_phase0_measurements` |
| E12 | `vram.budget_gib` (既定 14.0) | `long_context @65536` は通り `@131072` は停止 | `test_vram.py::test_budget_reflects_the_measured_gpu_resident_ceiling` |
| E13 | ネイティブ応答の `prompt_eval_*` / `eval_*` | `ChatTimings` の 2 つの t/s、欠測時は `None` | `test_client_native.py::test_native_result_exposes_prompt_and_eval_timings` |

**E9 と E11 が「値を変えても出力が変わらない」状態で通る実装は不合格。** E9 が死ぬと `kind` はまた飾りに戻り (F-2-003 の再発)、E11 が死ぬと較正値が実測から静かに乖離する。

## 6. 意図的な決定 (`.claude/decisions.yaml` に追記)

```yaml
- id: D-10
  rule: "runtime.kind がクライアント実装を選ぶ。kind='ollama' はネイティブ /api/chat、kind='openai_compatible' は /v1/chat/completions を使う。分岐は create_chat_client の 1 か所だけに置く"
  rationale: "Phase 0 実測で /v1/chat/completions は options.num_ctx を無視することを確認した (16384 / 32768 を送っても ollama ps の CONTEXT は 4096)。コンテキスト長を設定で切り替えるという Phase 1 の前提が OpenAI 互換経路では成立しない。これにより元仕様書 §8 Q-B の決定 (1)『OpenAI 互換のみ』を変更する。層の共通境界はワイヤプロトコルではなく ChatClient Protocol とする。D-09 は変更しない"
  guard_test: "tests/test_client_factory.py::test_runtime_kind_selects_the_client_implementation"

- id: D-11
  rule: "ネイティブ経路のエンドポイントは runtime.base_url の末尾 /v1 を 1 セグメントだけ除去して /api/chat を連結して導出する。ネイティブ URL 用の設定キーを追加しない"
  rationale: "既存の base_url は OpenAI 互換の /v1 を指す前提で configs とテストに固定されており、キーを増やすと『設定のみで切替』の軸が増えて D-09 の整理と衝突する。導出が合わない構成 (ゲートウェイ等) は kind='openai_compatible' を選べば従来どおり動く"
  guard_test: "tests/test_client_native.py::test_native_endpoint_url_is_derived_from_base_url"

- id: D-12
  rule: "GPU 常駐上限はモデル単位の定数ではなく vram.budget_gib (プロファイル単位の予算判定) で表現する。budget_gib はカード容量ではなく『実測でオフロードが始まらない増分の上限』であり、本機の既定は 14.0 GiB"
  rationale: "実測で gpt-oss-20b 単体は合計 13.68 GiB (num_ctx=65536) まで 100% GPU、num_ctx=98304 (見積り 14.46 GiB) で CPU オフロードが発生した。上限は同居する埋め込み・リランカーの有無に依存するため model 単位の max_context_tokens では表せない。カード 16376 MiB = 15.99 GiB のうちアイドルで 0.60 GiB を使用しており、増分の予算 16.0 は原理的に大きすぎる。要件書 L296 の『16GB を超えたら停止』は、より早く停止する本設定でも満たされる"
  guard_test: "tests/test_vram.py::test_budget_reflects_the_measured_gpu_resident_ceiling"

- id: D-13
  rule: "ChatResult は prompt/eval を分離した速度指標を ChatTimings (任意フィールド、既定 None) として持つ。ネイティブ応答が prompt_eval_count 等を返さない場合は 0 で埋めず None にする。OpenAI 互換経路では常に None"
  rationale: "要件書 L303 (Phase 2 受け入れ) が『プロンプト処理速度』を要求し、F-1-006 が受け皿の不在を指摘している。これを返すのはネイティブ応答だけで、その実装を今行っている。0 で埋めると『プロンプトキャッシュヒットで未計測』と『実際に 0 トークン』が区別できず、Phase 2 の比較が静かに壊れる (D-07 と同じ趣旨)"
  guard_test: "tests/test_client_native.py::test_native_result_exposes_prompt_and_eval_timings"

- id: D-14
  rule: "ruri-v3-310m / ruri-reranker はカタログに残すが、source_note に『取得不能』と理由を明記し、weights_gib は未実測の仮値のままとする。代替方式の決定は Phase 3 着手時まで行わない"
  rationale: "Phase 0 実測で両モデルは GGUF を持たず ollama pull できないことが判明した (Ollama にはリランキング API 自体が無い)。カタログから消すと構成1/構成3 のプロファイル定義と要件書の対応が切れ、予算ガードの実証も失われる。一方で仮値を実測値と同じ体裁で置くと較正済みの 3 モデルと区別が付かなくなる"
  guard_test: "tests/test_catalog.py::test_unavailable_models_declare_their_unavailability"
```

## 7. 想定リスク (これが起きたら止まって相談)

1. **ネイティブ応答の欠損フィールドが想定より多い**
   `done_reason` / `prompt_eval_*` を任意にする判断は Phase 0 の観測とキャッシュヒット時の既知挙動にもとづくが、MockTransport では実応答の実態を検証できない。**`live` マーカー付きテストを 1 本用意し、実機で 1 回でも `UpstreamError` が出たら止めて相談する。**

2. **`budget_gib=14.0` が要件書 L296 と読み合わせて否決される**
   → **承認済み (Q-1 = 14.0)**。D-12 に根拠を記録する。

3. **較正とクライアント変更を同時に入れて、落ちたテストの原因が切り分けられなくなる**
   期待値の書き換えが 7 行 + 帯テスト 1 本と広範囲。**T1+T2 を先に完了させ `make ci` が緑になることを確認してから T3 に進む。T1 の途中で `llmkit/client.py` を触り始めたら止める。**

## 参考: 較正後の見積り値一覧 (overhead 0.8 込み、単位 GiB)

| プロファイル | 構成 | ctx=16,384 | ctx=32,768 | ctx=65,536 | ctx=131,072 |
|---|---|---|---|---|---|
| `rag_default` | 14b + ruri + reranker | **12.63** | 15.15 | — | 30.27 |
| `long_context` | gpt-oss-20b | 12.51 | 12.90 (実測一致) | **13.68** (実測一致) | **15.24** |
| `lightweight` | 8b | **7.26** (実測一致) | 9.54 (実測一致) | — | — |
| `oversized` | 20b + ruri + reranker | 14.01 | 14.40 | 15.18 | **16.74** |

予算 14.0 での判定: `rag_default @16384` 収まる / `long_context @65536` 収まる (余裕 0.32) / `long_context @131072` 1.24 超過 / `oversized @131072` 2.74 超過。
