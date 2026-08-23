# Phase 2: モデル比較ハーネス（用途 C）— 仕様書とタスクリスト

> 前提資料: `docs/localllmrequirements.md`（用途 C / Phase 2 受け入れ条件 / 非機能ログ要件）/ `.claude/decisions.yaml`（D-01〜D-17）/ `docs/phase0-vram-measurements.md` / `docs/plans/` 既存3本
> 位置づけ: **L3 の新規トップレベルパッケージ `harness/`**。`llmkit/`（L2）へは**追加のみ**（公開シグネチャ変更なし）。Phase 3（RAG）のコードは書かない。
> ブランチ: `feat/phase2-harness`（Phase 1 マージ後に作成済み）
>
> **承認済み (2026-08-23)**: Q3 = 速度・再現条件のみ自動化（品質採点機構は作らない）/ VRAM は見積り + `nvidia-smi` 実測の両方を記録（D-01 維持、実測は L3 の責務）/ Q-1 = `max_output_tokens=512`・プロンプト8問 / Q-2 = `suites/ja_basic.toml` + `--config` 別指定 / Q-3 = `qwen3-14b` / `gpt-oss-20b` / `qwen3-8b`

## 1. ゴール

プロンプト集と対象モデルリスト（単一 TOML）を与えると、モデル×プロンプトの応答と速度・VRAM・再現条件を `results/` 配下に永続化する CLI を作る。

## 2. 現状認識

| パス | 内容 | Phase 2 での使い方 |
|---|---|---|
| `llmkit/bootstrap.py` | `bootstrap(config_path: Path, ...)` → `BootstrapResult` | **入口が Path 固定**。モデルを切り替えて N 回起動する経路が無い（F-1-004） |
| `llmkit/client.py` | `ChatResult.latency_s` / `tokens_per_second`（壁時計、欠測を 0.0 で表す）/ `measured_tokens_per_second`（`float \| None`）/ `ChatTimings` | 速度指標の出典 |
| `llmkit/manifest.py` | `RunManifest`（`config_sha256` / `generation` 全項目 / `profile.models[]` / `vram` 内訳） | 再現条件の記録。実効値がそのまま写る（E8） |
| `llmkit/vram.py` | `estimate_resolved_profile`（純関数、GPU に触れない、D-01） | 見積り側。実測とは別列で記録 |
| `llmkit/config.py` | `AppConfig`（frozen pydantic dataclass）、`ProfileConfig.generation` | オーバーライド対象が **2 か所** |
| `tests/conftest.py` | `_forbid_real_network` / `RecordingTransport` / `tmp_config` | ハーネスのテストもこれに乗る |
| `tests/test_acceptance_phase1.py` | `ACCEPTANCE_MAP` が要件書 L294-L298 を**行番号で**参照 | **要件書の行数を1行も増減させてはならない** |
| `.gitignore` | `outputs/` は ignore 済み。`results/` の記載は**無い** | `results/` はコミット対象にできる |
| `pyproject.toml` | 依存は `httpx` / `pydantic` のみ。YAML なし。`pythonpath = ["."]` | 新規トップレベルパッケージは設定変更なしでテスト・型検査対象になる |

### ★ 見落とすと壊れる構造（本仕様の中核）

**モデル ID は設定の2か所に現れ、参照する先が違う。**

- `config.generation.model` → `resolve_model_spec` → **リクエストの `model` フィールド**
- `config.profiles[active].generation` → `resolve_profile` → **VRAM 見積りと実行マニフェストの `profile.models[]`**

`configs/default.toml` では両方 `qwen3-14b` で一致しているため、**片方だけ差し替えても何もエラーにならない**。メインセッションが実測で確認済み:

```
generation.model だけ gpt-oss-20b に差し替えた場合:
  リクエストに送る model = gpt-oss:20b
  VRAM 見積りの内訳      = {'qwen3-14b': 7.81, ...}   ← 14B の値
  見積り合計             = 11.98 GiB                  ← 正しくは 13.36

両方を差し替えた場合:
  リクエストに送る model = gpt-oss:20b
  VRAM 見積りの内訳      = {'gpt-oss-20b': 11.32, ...}
  見積り合計             = 13.36 GiB
```

**「20B に投げているのに 14B の VRAM 見積りを記録した比較結果」が静かに生成される。** ハーネスの中核はここを塞ぐことにある（→ D-19）。

### 既存の慣習で守るべきもの

1. **欠測は `None`。0 で埋めない**（D-07 / D-13 / `_speed_summary` の前例）
2. **スキーマは pydantic dataclass + `TypeAdapter`**。`BaseModel` を継承しない（D-08）
3. **ゴールデン値は1か所でロックし、アサーションは緩めず「より強い主張へ再アンカー」する**

### 影響範囲

新規: `harness/`（7ファイル）/ `suites/`（1ファイル）/ `results/`（成果物）/ `tests/` 8ファイル。
変更: `llmkit/bootstrap.py`（**追加のみ**）/ `llmkit/__init__.py`（再エクスポート）/ `tests/test_layout.py`（走査範囲の拡張）/ `docs/` / `README.md` / `.claude/decisions.yaml`。
**不変: `llmkit/{config,client,vram,catalog,manifest,errors,cli}.py` / `main.py` / `Makefile` / `.github/workflows/` / `pyproject.toml` / `uv.lock`。**

## 3. 前提・制約

### ハード制約

- 既存 **331 テストを壊さない**。**アサーションを緩めない**
- mypy `strict` + `disallow_any_explicit`。`Any` 明示禁止。`# type: ignore` 0 個。`pydantic.BaseModel` を継承しない（D-08）
- ruff `T20`（`print()` 禁止）。CLI 出力は `sys.stdout.write` / `sys.stderr.write` か logging
- **テストは実 HTTP を 1 バイトも出さない**（D-02）。実ランタイム接続は `@pytest.mark.live` で既定スキップ。CI（GPU 無し・Ollama 無し・`nvidia-smi` 無し）で全件緑
- **`llmkit/` に `nvidia-smi` / `subprocess` / GPU 参照を持ち込まない**（D-01 維持）
- `llmkit/` の既存公開シグネチャを変更しない（**追加のみ**）
- **`docs/localllmrequirements.md` の行数を1行も増減させない**
- **新規依存パッケージを追加しない**（`uv lock --check` が無変更で通ること）
- Phase 3（RAG）のコードを書かない。埋め込み・リランカーの推論呼び出しをしない
- 秘密情報を `results/` に書かない。`api_key` の値・`nvidia-smi` のプロセス一覧・ユーザー名を記録しない

### ソフト制約

- 集計統計は中央値（`statistics.median`）。平均はコールド実行の外れ値に引きずられる
- Markdown の応答本文は先頭 400 文字 + 折りたたみ。全文は JSONL 側に持つ
- ケース単位の失敗は記録して次のモデルへ進む。CLI は全ケース失敗時のみ exit 1

### 論点への判定

**`llmkit` の公開 API で足りるか → 足りない。`bootstrap_from_config` を追加する。**

| 案 | 判定 | 根拠 |
|---|---|---|
| モデルごとに一時 TOML を書き出して `bootstrap(path)` | **却下** | `config_sha256` が tmp ファイルのハッシュになり、`results/` に残る再現条件が「消えたファイル」を指す。TOML 生成＝設定スキーマの二重実装 |
| `bootstrap` に `overrides` 引数を足す | 却下 | 既存シグネチャに任意の上書き表が生えると D-09 と混線し、L2 が L3 の都合を知る |
| **`bootstrap_from_config(config, config_path, *, ...)` を追加し `bootstrap` をその薄いラッパにする** | **採用** | 追加のみ。既存の呼び出し側は 1 文字も変わらない。オーバーライドは `dataclasses.replace` で L3 が行い、L2 は「渡された `AppConfig` を起動する」だけになる |

帰結として **`config_sha256` は「ベース設定ファイルの同一性」しか表さなくなる**。実効値の同一性は `run_fingerprint` が担う（→ D-20）。この2つを混同すると、モデルの違う2実行が同じハッシュを持ったまま受け入れ条件が通ってしまう。

**プロンプト集の形式 → 単一の TOML スイートファイル。** YAML は依存追加が必要。JSONL は `[[models]]` とプロンプト集を1ファイルに収められず「入力の同一性」を1つの sha256 で表せない。出力の生ログは要件書どおり JSONL（→ D-18）。

**速度指標 → 代表値は `measured_tokens_per_second`（eval 優先）。3値すべてを別列で記録し、欠測は `null`**（→ D-21）。

**ロード順バイアス → ウォームアップ既定1回 + model-major 実行 + 実行順序の全件記録**（→ D-22）。ウォームアップ結果は**捨てない**（`phase="warmup"` で JSONL に残す）。捨てるとコールド/ウォームの差自体が観測できなくなる。

## 4. タスク分解

**実行順序 T1 → T2 → T3 → T4 → T5 を厳守。** 各タスク完了時に `make ci` が緑になることを確認してから次へ進む。

### T1: `bootstrap_from_config` の追加とスイートスキーマ・実効設定の導出

**`llmkit/bootstrap.py`（追加のみ）**

- `bootstrap_from_config(config: AppConfig, config_path: Path, *, profile_name=None, http_client=None, output_dir=None, write_manifest_file=True) -> BootstrapResult` を新設
- `bootstrap(config_path, ...)` は `load_config` して委譲するだけにする。**引数名・順序・既定値・戻り値・例外は一切変えない**
- docstring に「`config` が実行時に上書きされた場合、`manifest.config_sha256` はベースファイルのハッシュであり実効値の同一性を表さない（実効値の同一性は L3 の `run_fingerprint` が担う、D-20）」を明記
- `__all__` と `llmkit/__init__.py` に追加

**`harness/suite.py`（新規）**

| セクション | フィールド | 必須 | 備考 |
|---|---|---|---|
| `[suite]` | `id`（`[a-z0-9_-]+`）, `description`, `warmup_runs`（int ≥ 0、既定 1） | id | `id` は `results/<id>/` のディレクトリ名。**パス区切り・`..` を拒否** |
| `[[models]]` | `model_id`, `profile`, `context_tokens`, `temperature`, `top_p`, `max_output_tokens`, `seed` | `model_id` | 省略項目はベース `AppConfig` の値 |
| `[[prompts]]` | `id`, `text`, `system`（任意）, `tags`（任意） | `id` / `text` | `id` はスイート内で一意 |

- `load_suite(path) -> ComparisonSuite` と `suite_sha256(path) -> str`
- **`apply_case(base, case) -> AppConfig`**: `generation` の6項目と、**対象プロファイルの `generation` フィールドの両方**を同時に差し替え、`vram.active_profile` を対象プロファイルに揃える（D-19）
- 例外は `llmkit.ConfigError` を再利用（新しい例外階層を作らない）

**受け入れ基準**
- `bootstrap_from_config` を追加した状態で **既存 331 テストが1件も修正なしで緑**
- `apply_case` の戻り値で `config.generation.model == config.profiles[active].generation` が**常に**成立（3モデルで parametrize）
- `generation.model` **だけ**を replace する経路が存在しない（`rg` で `apply_case` 内1か所に閉じている）
- 不正スイート4ケース（`id` に `../` / prompts の id 重複 / 未知キー / `warmup_runs = -1`）が `ConfigError` になりメッセージに該当キー名を含む
- `import harness.suite` が `llmkit` のサブモジュールを直接 import していない
- `uv lock --check` が無変更で通る

### T2: VRAM 実測プローブ（`nvidia-smi` を L3 に隔離）

**`harness/gpu.py`（新規）**
- `GpuMemory`（frozen dataclass）: `name`, `used_mib`, `total_mib`
- `VramProbe` Protocol: `read() -> GpuMemory | None`
- `NvidiaSmiProbe`: コマンド列と実行関数を**コンストラクタで注入可能**に
- 失敗時（未インストール / 非0終了 / パース失敗 / タイムアウト）は**例外を上に投げず `None` を返す**。DEBUG ログ1行にとどめ、warn は実行につき1回だけ
- **記録するのは GPU 名・used/total MiB のみ**。`--query-compute-apps` は使わない。生出力をログにも記録にも出さない
- タイムアウト既定5秒

**受け入れ基準**
- `rg -n "nvidia-smi|subprocess|GPUtil|pynvml" llmkit/` が **0件**
- `rg -n "nvidia-smi" harness/` が `harness/gpu.py` の中だけ
- フェイク注入で (a) 正常 → `GpuMemory` (b) `FileNotFoundError` → `None` (c) 非0終了 → `None` (d) 壊れた CSV → `None` (e) タイムアウト → `None` の5ケースが緑。**いずれも例外が外に出ない**
- `test_estimate_is_pure_and_needs_no_gpu`（D-01 guard）が**無修正で**緑
- 実プロセスを起動するテストが0件

### T3: ランナー・レコード・再現フィンガープリント

**`harness/records.py`（新規）** — 1実行 = 1レコード

| グループ | フィールド |
|---|---|
| 同一性 | `schema_version`, `run_id`, `run_fingerprint`, `sequence_index`, `case_index` |
| 条件 | `model_id`, `served_name`, `quantization`, `serving_runtime`, `profile_name`, `context_tokens`, `temperature`, `top_p`, `max_output_tokens`, `seed` |
| 区分 | `phase`（`"warmup"` / `"measure"`）, `prompt_id`, `prompt_sha256` |
| 応答 | `response_text`, `finish_reason`, `prompt_tokens`, `completion_tokens`, `total_tokens` |
| 速度 | `latency_s`, `eval_tokens_per_second`, `prompt_tokens_per_second`, `wallclock_tokens_per_second`, `generation_tokens_per_second`, `generation_tokens_per_second_source`（`"eval"`/`"wallclock"`/`null`） |
| VRAM | `vram_estimate_gib`, `vram_budget_gib`, `vram_used_mib`, `vram_total_mib`, `vram_increment_gib` |
| 失敗 | `error`（`None` または `{"type", "message"}`） |

- `wallclock_tokens_per_second` は `latency_s > 0 and completion_tokens > 0` のときだけ値を持つ。**`ChatResult.tokens_per_second` の 0.0 をそのまま書かない**

**`harness/runner.py`（新規）**
- `plan_run(suite, base_config, config_path) -> RunPlan`: HTTP を1バイトも出さずに全ケースの実効 `AppConfig`・VRAM 見積り・予算判定・`run_fingerprint` を解決する（`--dry-run` の実体）
- `run_suite(plan, *, probe, http_client=None, results_dir, clock=None, run_id=None) -> RunResult`
  - 実行順序は **model-major**（モデル1つにつき全プロンプトを連続実行）
  - 各モデル: `bootstrap_from_config` → ウォームアップ ×`warmup_runs` → プローブで `vram_used_mib` 取得 → 全プロンプトを `phase="measure"` で実行
  - 起動直後に1回プローブし `vram_idle_mib` を記録。`vram_increment_gib = (used - idle) / 1024`
  - `LlmkitError` はケース単位で捕捉し記録して次へ。`VramBudgetExceededError` はスキップし理由を記録（**HTTP は発行されない**、D-04）
- `run_fingerprint`: 正規化 JSON（`sort_keys=True`, `ensure_ascii=False`）の sha256
  - `suite_sha256` / `config_sha256` / `schema_version`
  - 全ケースの `RunManifest.to_dict()` から **`run_id` と `started_at_utc` を除いたもの**
  - 全プロンプトの `(id, sha256(text), sha256(system or ""))`
  - `warmup_runs`
  - **含めない**: 応答テキスト、レイテンシ、速度、`vram_used_mib`、実行時刻

**受け入れ基準**
- **同一入力の2回実行で `run_fingerprint` が一致**（応答内容と latency は毎回異なる状態で検証）
- **フィンガープリント13要素の掃引テスト**が全件で「変わる」を示す（`model_id` / `context_tokens` / `temperature` / `top_p` / `max_output_tokens` / `seed` / `profile` / `warmup_runs` / プロンプト本文 / プロンプト追加 / `runtime.base_url` / `runtime.kind` / `vram.budget_gib`）
- **`model_id` だけを変えた2実行で、`config_sha256` は一致し `run_fingerprint` は異なる**（D-20 の存在理由をテストで固定）
- `model_id` を変えると (a) リクエストの `model` (b) マニフェストの `profile.models[0].model_id` (c) `vram.weights_gib` のキー (d) `vram_estimate_gib` が**すべて**変わる（D-19 guard）
- `warmup_runs` を 0→1→2 と変えると HTTP 回数と `phase="warmup"` のレコード数が対応して変わり、**`measure` の件数と集計値は変わらない**（D-22 guard）
- `timings` を返さない応答で `eval_*` が `null`、`source == "wallclock"` になる。`completion_tokens = 0` で代表値が `null` になり **0.0 にならない**（D-21 guard）
- 1モデル目が `ModelNotFoundError` でも 2・3モデル目が完走し `error.type` が記録される
- 予算超過ケースの HTTP 発行回数が 0

### T4: 出力（Markdown 表 + JSONL + run.json）と CLI とスイート実体

**`harness/report.py`（新規）** — 出力先 `results/<suite_id>/<YYYYmmddTHHMMSSZ>-<fingerprint[:12]>/`

| ファイル | 内容 |
|---|---|
| `report.md` | 再現条件ヘッダ / **モデル比較表** / プロンプトごとの応答セクション（先頭400文字 + 折りたたみ） |
| `records.jsonl` | 1行1レコード（warmup 含む）。`sequence_index` 昇順 |
| `run.json` | 再現条件の構造化版 + 集計値 |
| `manifests/*.json` | `llmkit` の実行マニフェスト（モデルごと1本） |

**モデル比較表の列**（受け入れ条件2 に直結。列名は変えない）:
`model_id` / `served_name` / `quantization` / `設定コンテキスト長` / `生成速度 t/s (中央値)` / `速度出典` / `プロンプト処理速度 t/s (中央値)` / `壁時計速度 t/s (中央値)` / `VRAM 見積り GiB` / `VRAM 実測増分 GiB` / `測定数 n/N`

- 集計は `phase == "measure"` かつ `error is None` のみ。**`None` は集計から除外し `n/N` を必ず併記**
- 欠測セルは `—`。数値 0 を書かない

**`harness/cli.py`（新規）** — `python -m harness.cli run --suite suites/ja_basic.toml --config configs/default.toml [--dry-run] [--models a,b] [--limit N]`

- `--dry-run`: 実行計画を stdout に出して **HTTP 0 回で exit 0**
- 出力は `sys.stdout.write` / `sys.stderr.write` のみ

**`suites/ja_basic.toml`（新規）** — 日本語プロンプト**8問**、`max_output_tokens = 512`。要約 / 言い換え / 箇条書き整形 / 用語説明 / 短文翻訳 / 構造化抽出 / 指示追従 / 曖昧な依頼への対応。**機密・個人情報を一切含めない**。対象は `qwen3-14b` / `gpt-oss-20b` / `qwen3-8b`。

**受け入れ基準**
- `report.md` のヘッダ行に「生成速度」「プロンプト処理速度」「設定コンテキスト長」の3文字列が含まれる
- 表の行数がモデル数と一致し、`model_id` がスイートの宣言順に並ぶ
- 速度が全件 `None` のケースでセルが `—` で `0.0` / `0` を含まない
- `records.jsonl` の各行のキー集合が `RECORD_KEYS` と**完全一致**
- `run.json` から `run_fingerprint` を再計算すると記録値と一致する
- `--dry-run` が **Ollama 無しで exit 0**、HTTP 0回、`run_fingerprint` を出力
- `--models qwen3-8b` で対象が1件になり `run_fingerprint` がフル実行時と**異なる**
- プローブの戻り値を変えると実測列と `vram_used_mib` が変わり、`None` なら `—` / `null`（D-23 guard）
- `rg -n "print\(" harness/` が 0件

### T5: 実機実行・`results/` へのコミット・受け入れテスト・決定と文書

**(a) 実機実行**
1. `ollama stop` で全モデルをアンロードし `nvidia-smi` でアイドルを確認
2. `--dry-run` で予算判定を先に確認
3. 本実行 → `results/ja_basic/<...>/` が生成される
4. **同じコマンドをもう一度実行**し `run_fingerprint` が一致することを確認して報告（2回目はコミットしない）

**(b) `results/` の扱い**
- `.gitignore` は変更しない。`outputs/` の行の直下にコメント1行だけ追記して違いを明示
- コミットするのは **1実行分のみ**
- コミット前に `rg -n "sk-|Bearer |password|token" results/` が0件、`run.json` / マニフェストに `api_key` の値が無く `api_key_env` だけがあることを確認して報告

**(c) `tests/test_acceptance_phase2.py`（新規、成果物コミット後に追加）**

要件書 **L302-L305** の4条件を4関数に1対1対応させる。

| 行 | テスト関数 | 検証内容 |
|---|---|---|
| L302 | `test_l302_suite_and_model_list_produce_a_comparison_file` | MockTransport で実行し3ファイルが生成される |
| L303 | `test_l303_output_contains_speeds_and_context_length` | `report.md` に3指標の列が存在し値が入る |
| L304 | `test_l304_rerunning_the_same_input_reproduces_the_conditions` | 2回実行で `run_fingerprint` 一致、`run.json` の再現条件セクションが完全一致 |
| L305 | `test_l305_japanese_comparison_results_are_committed` | `results/` を走査し **3つ以上の相異なる `model_id`**、日本語（CJK）プロンプト1問以上、`measure` レコードに非 `null` の `generation_tokens_per_second` と `vram_used_mib` |

- **要件書の行を参照する検査は `- [ ] ` ではなく `- [` で始まることを見る**（Phase 2 完了時にチェックを付けても落ちないように）

**(d) `tests/test_layout.py` への追加（緩和ではなく拡張）**
- D-08 guard の走査範囲に `harness/*.py` を追加
- `tests/test_harness_layout.py`（新規）: (i) `harness` が `llmkit` の**サブモジュール**を import していない (ii) `harness/__init__.py` の `__all__` が和集合と一致 (iii) `llmkit/` が `harness` を import していない

**(e) `.claude/decisions.yaml`**: D-18〜D-24 を追記（**guard_test が実在してから**）

**(f) 文書**
- `docs/localllmrequirements.md`: **行数を1行も増減させない**。用途C の `(仮)` を確定内容に、Q3 行を「解決済み」に、L302-L305 のチェックボックスを `- [x]` に
- `docs/next-pr-candidates.md`: F-1-004 を「対応済み（本 PR）」に
- `README.md`: 「モデル比較ハーネス（Phase 2）」節を新設。決定の参照範囲を `D-01〜D-24` に。Project Structure に `harness/` `suites/` `results/` を追加

**受け入れ基準**
- `results/` 配下に3モデル以上・日本語プロンプトの比較結果が **1実行分コミットされている**
- `check_decisions.py` が「24件」で通り、D-18〜D-24 の guard_test 7本を**個別実行して緑**
- `test_l298`（Phase 1）が**無修正で**緑
- 実機2回実行で `run_fingerprint` が一致したことを実測値付きで報告
- **実測 VRAM と見積りの差を報告する**。差が 2.0 GiB 以上のケースがあれば §7 リスク1 に従い停止して相談
- `make ci` が緑。`uv lock --check` が無変更

## 5. 評価軸

### 機能観点
要件書 L302-L305 の4条件が4関数に1対1対応。L302-L304 は MockTransport で決定論的に、L305 はコミット済み成果物の静的検査で測る。

### 性能観点

| 指標 | 期待値 |
|---|---|
| `uv run pytest` 全体 | **15秒未満**（既存目安12秒 + ハーネス分。超えたら報告）|
| `--dry-run` | 1秒未満、HTTP 0回、プロセス生成0回 |
| VRAM 見積り | 引き続き純関数（`test_estimate_is_pure_and_needs_no_gpu` 無修正で緑）|
| 実機比較実行 | 3モデル × 8プロンプト + ウォームアップ3 = 27リクエストが完走 |

### 安全性観点
- `rg -n "nvidia-smi|subprocess" llmkit/` が **0件**
- `rg -n "def embed|def rerank|/v1/embeddings|/v1/rerank" llmkit/ harness/` が **0件**
- `results/` に `api_key` の値・GPU プロセス一覧・ユーザー名が入っていない
- D-04 guard（予算超過で HTTP 0回）がハーネス経路でも成立
- `harness/` から `llmkit` サブモジュールへの直接 import が0件、`llmkit/` から `harness` への import が0件

### ★ 有効性観点

既存 E1〜E16 を維持したうえで:

| # | 掃引する値 | 変わるべき出力 | テスト |
|---|---|---|---|
| **E17** | `models[].model_id` | リクエストの `model` / マニフェストの `profile.models[0]` / `vram.weights_gib` のキー / `vram_estimate_gib` / `report.md` の行 の**すべて** | `test_harness_runner.py::test_model_override_changes_both_the_request_and_the_vram_estimate` |
| **E18** | `context_tokens` | `options.num_ctx` / 「設定コンテキスト長」列 / `run_fingerprint` | `test_harness_runner.py::test_context_tokens_reach_the_request_the_report_and_the_fingerprint` |
| **E19** | 生成パラメータ4種 | リクエストボディ / マニフェストの `generation` / `run_fingerprint` | `test_harness_runner.py::test_suite_generation_overrides_reach_the_request_and_the_manifest` |
| **E20** | `warmup_runs` | HTTP 回数と `warmup` レコード数。**集計値と `measure` 件数は変わらない** | `test_harness_runner.py::test_warmup_runs_are_recorded_but_excluded_from_aggregates` |
| **E21** | 応答の `eval_count` / `eval_duration` の有無 | `generation_tokens_per_second`（値と source）/ `n/N`。**欠測時に 0 が現れない** | `test_harness_metrics.py::test_missing_timings_are_recorded_as_null_and_excluded_from_aggregates` |
| **E22** | VRAM プローブの戻り値 | `vram_used_mib` / `report.md` の実測列 | `test_harness_gpu.py::test_vram_probe_result_is_visible_in_the_report` |
| **E23** | フィンガープリント入力の13要素 | `run_fingerprint` | `test_harness_runner.py::test_every_reproduction_field_changes_the_fingerprint` |

**E17 が「`generation.model` だけ変えても VRAM 見積りが変わらない」状態で通る実装は不合格。** それが本仕様が防ごうとしている唯一最大の欠陥である。

**E21 が「欠測を 0.0 として集計する」実装で通ってはならない。**

**変異検証を必須とする**:
1. `apply_case` からプロファイル側の差し替えを外す → E17 が落ちる
2. 欠測時に `ChatResult.tokens_per_second`（0.0 になり得る）をそのまま記録する → E21 が落ちる
3. `run_fingerprint` からマニフェストの `generation` を除く → E18・E19・E23 が落ちる
4. `run_fingerprint` に `started_at_utc` を含める → 「2回実行で一致」が落ちる
5. ウォームアップを集計に含める → E20 が落ちる
6. `NvidiaSmiProbe` の失敗時に例外を送出する → gpu テストの (b)(c)(d)(e) が落ちる

## 6. 意図的な決定（`.claude/decisions.yaml` に追記）

D-18〜D-24 を追記する。内容は planner 出力のとおり（`rule` / `rationale` / `guard_test` の3点セット）:

- **D-18**: 入力は単一 TOML スイート。YAML/JSONL は入力に採らない。出力の生ログのみ JSONL。新規依存を追加しない
- **D-19**: `apply_case` が `generation.model` と対象プロファイルの `generation` を**同時に**差し替える。片方だけの経路を作らない
- **D-20**: 再現性の判定は `config_sha256` ではなく `run_fingerprint`。`run_id` と `started_at_utc` を除外する
- **D-21**: 代表値は `measured_tokens_per_second`（eval 優先）+ 出典ラベル。3値を別カラムで残す。欠測は `null`、集計から除外し `n/N` を併記
- **D-22**: ウォームアップ既定1回。結果は捨てず `phase="warmup"` で残し集計からのみ除外。model-major 実行と実行順序の記録
- **D-23**: `nvidia-smi` は `harness/gpu.py` だけ。注入可能。失敗は例外を出さず `None`。プロセス一覧を取らない
- **D-24**: `results/` はコミット対象（`outputs/` は ignore のまま）。スイートに機密・個人情報を書かない

## 7. 想定リスク（これが起きたら止まって相談）

1. **3モデルの逐次実行で VRAM 実測が汚れる。**
   `OLLAMA_MAX_LOADED_MODELS` の既定は 2 で、`qwen3:14b`（約11 GiB）と `gpt-oss:20b`（約12 GiB）は同居できない。退避のタイミング次第で `vram_used_mib` が前のモデル分を含む。**実測増分と見積りの差が 2.0 GiB 以上のケースが出たら止めて相談する**。ハーネスから `ollama` コマンドを呼んで自動アンロードする案は**採らない**（外部プロセスへの副作用が増え D-23 の隔離方針と衝突）。

2. **`results/` に応答テキストをコミットすることで想定外の内容が入る。**
   プロンプトは管理できるが応答は制御できない。**応答に個人情報・認証情報らしき文字列・不適切な生成物が含まれていたらコミットせず止めて相談する**。

3. **`bootstrap` の分割が既存331テストに波及する。**
   委譲に切り替えた結果、ログの出力順・マニフェスト書き出しのタイミング・例外の送出順が1つでも変わると既存テストが落ちる。**既存テストを1行でも修正したくなったら、その時点で止めて相談する**（分割方法が誤っているサイン）。

## 8. ファイル構成

```
harness/                    # L3。トップレベル (D-06 と同じフラットレイアウト)
├── __init__.py             # 公開 API の再エクスポート + __all__
├── suite.py                # ComparisonSuite / ModelCase / PromptSpec / load_suite / apply_case
├── gpu.py                  # GpuMemory / VramProbe(Protocol) / NvidiaSmiProbe  ← nvidia-smi はここだけ
├── records.py              # RunRecord / RunSummary / 集計 (median, n/N)
├── runner.py               # RunPlan / plan_run / run_suite / run_fingerprint
├── report.py               # report.md / records.jsonl / run.json の書き出し
└── cli.py                  # python -m harness.cli run [--dry-run]

suites/ja_basic.toml        # 日本語 8 問 × 3 モデル、max_output_tokens=512
results/ja_basic/<ts>-<fp12>/{report.md, records.jsonl, run.json, manifests/*.json}
```

**`llmkit/harness/` にしない理由**: 「L3 は `llmkit` の公開シンボルのみを使う」という制約は、**L3 が別パッケージであって初めて機械検証できる**。同一パッケージ内に置くと `from llmkit.client import _HttpChatClient` のような内部参照が構文上いつでも書けてしまい、境界が散文の主張に戻る。

**`test_layout.py::_SUBMODULE_NAMES` への波及**: `harness/` は llmkit のサブモジュールではないため変更不要。ただし `llmkit/bootstrap.py` の `__all__` に `bootstrap_from_config` を足すため `llmkit/__init__.py` への再エクスポートが必須（等号検査のため忘れると落ちる）。D-08 guard の走査範囲は `harness/*.py` にも広げる。

**`pyproject.toml` は変更不要**: `pythonpath = ["."]` で `import harness` が解決し、mypy の `exclude` は `.venv` のみなので `harness/` は自動的に strict の対象。新規依存が無いため `uv.lock` も無変更。

## 9. 実装時に決めたこと (T1 / T2 実装者による追記、2026-08-23)

仕様書に書かれていなかった選択を実装者が決めた箇所。次の周の reviewer / fixer が
読むのはこの節であり、実装コードのコメントではない。**T3 以降はここに追記していく。**

### T1 (`bootstrap_from_config` / `harness/suite.py`)

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | `bootstrap_from_config(config, config_path, *, ...)` の `config_path` は**第2位置引数**にした (キーワード専用にしない) | `config` と `config_path` は常にセットで意味を成す。片方だけ渡せる形にすると「上書き前のファイルを指すハッシュ」であることが呼び出し側から見えにくくなる |
| 2 | `[suite]` セクションを表す型として `SuiteMeta` を追加した (§8 の3型 + 1) | TOML のセクション構造と 1:1 に対応させると、pydantic のエラー位置がそのまま `suite.warmup_runs` のようなキー名になる。手で平坦化するとキー名の対応付けを自前で書くことになる |
| 3 | `[[models]]` / `[[prompts]]` は**最低1件必須** (`min_length=1`) | 0 件のスイートは何もエラーを出さずに空の比較結果を書き出す。T4 の `--models` フィルタ結果が 0 件になる場合は `load_suite` を経由しないため、この制約とは独立に T4 側で扱う必要がある |
| 4 | `ModelCase` の数値制約はベース `GenerationParams` と同じ範囲にそろえた (`temperature` 0.0〜2.0 / `top_p` (0.0, 1.0] / `context_tokens`・`max_output_tokens` は正整数) | スイート側だけ緩いと、`apply_case` の後 (= 起動直前) に pydantic の検証エラーが出る。入口で落とす方が該当キー名を示せる |
| 5 | `apply_case` は `case.profile` がベース設定に無いとき `ConfigError` にする。メッセージに `models[].profile` を含める | `vram.active_profile` に存在しない名前を書くと `AppConfig` の再検証が pydantic の `ValidationError` を投げ、llmkit の例外階層の外に出る。入口で翻訳する |
| 6 | 上書き値の採否は `is None` で判定する (`or` を使わない) | `seed = 0` / `temperature = 0.0` は正当な上書き値。真偽値で判定すると黙ってベース値に戻る (D-21 と同じ「0 と欠測を区別する」趣旨) |
| 7 | `apply_case` はベース `AppConfig` を破壊しない (`profiles` を dict コピーしてから差し替える) | ベース設定は全モデルで共有される。破壊的だと 2 モデル目以降が前のモデルの汚染を受ける |
| 8 | 公開シンボルは §8 の列挙より広い (`SuiteMeta` / `CommandResult` / `CommandRunner` / `run_command` / `DEFAULT_NVIDIA_SMI_COMMAND` / `DEFAULT_TIMEOUT_S`) | T5 の `harness/__init__.py` の `__all__` = サブモジュールの `__all__` の和集合という等号検査に合わせるため、注入点と既定値を公開側に置いた |

### T2 (`harness/gpu.py`)

| # | 決めたこと | 理由 |
|---|---|---|
| 9 | 複数 GPU が刺さっている場合は**先頭 (index 0) の1台だけ**を読む。合算しない | 対象は単一 GPU 機。合算すると「どのカードに載ったか」が読めなくなり、`vram_increment_gib` の意味が壊れる |
| 10 | 失敗として捕捉する例外は `OSError` と `subprocess.SubprocessError` の2系統 | `FileNotFoundError` (未インストール) / `PermissionError` / `TimeoutExpired` を型を数え上げずに覆える。捕捉漏れが `run_suite` の途中でハーネスごと落とす事故を防ぐ |
| 11 | 失敗ログには**失敗の分類文だけ**を出す。生出力・例外メッセージ・コマンド全体を出さない | ドライバのエラーメッセージやパス経由でホスト名・ユーザー名が比較結果のログへ混入し得る (CLAUDE.md のログ出力ルール) |
| 12 | 「`llmkit/` が GPU に触れていない」の機械検査は `rg` の全文一致ではなく **AST ベース** (docstring を除外し、import と非 docstring 文字列リテラルだけを見る) にした | §4 T2 受け入れ基準の `rg -n "nvidia-smi\|subprocess\|GPUtil\|pynvml" llmkit/` は**本タスク着手前 (base ref) の時点ですでに2件ヒットする**。`llmkit/vram.py:3` と `llmkit/catalog.py:3` の docstring に「実行時に `nvidia-smi` を参照**しない**」という D-01 の説明文が書かれているため。この2ファイルは本 PR の不変対象であり編集できない。全文一致はコメントに書いた違反を検出できず説明文を違反と誤検出するため、実行されるコードを見る検査に置き換えた (`tests/test_harness_gpu.py::test_llmkit_never_touches_the_gpu_or_spawns_processes`、変異検証つき) |

### T3 (`harness/records.py` / `harness/runner.py`)

| # | 決めたこと | 理由 |
|---|---|---|
| 13 | `plan_run(suite, suite_path, base_config, config_path)` とし、**`suite_path` を第2位置引数**に加えた (§4 T3 の記載は `plan_run(suite, base_config, config_path)`) | `run_fingerprint` の入力に `suite_sha256` が要るが、スイートのハッシュはファイルからしか計算できない。T1 の決定1 と同じ形 (中身とその出所を必ず対で渡す) にそろえた |
| 14 | `RunPlan` にベース `AppConfig` を持たせた | `run_suite` が `http_client` を省略されたときに自分で `httpx.Client` を作る必要があり、`runtime.timeout_s` の出典が要る。ケース側の設定は上書き済みで「ベースの接続設定」を表さない |
| 15 | `run_suite` は**自分で生成した `httpx.Client` だけを閉じる**。注入されたものは閉じない | `ChatClient` Protocol に `close()` が無いため、`BootstrapResult.client` からは閉じられない。所有権の規則を `llmkit._HttpChatClient` と同じにした |
| 16 | フィンガープリントは `fingerprint_inputs()` (素の辞書) と `fingerprint_digest()` に分け、両方を公開した | ハッシュ値だけを `run.json` に残すと「何が変わったから変わったのか」が追えない。T4 の受け入れ基準「`run.json` から `run_fingerprint` を再計算すると一致する」も入力が残っていて初めて成立する |
| 17 | 計画段階のマニフェストの `run_id` / `started_at_utc` は**固定値にしない** (実際の uuid と現在時刻のまま) | 当初は `--dry-run` の決定性のため固定値を入れたが、**変異検証で「`started_at_utc` をフィンガープリントに含める」を入れても 18 件全部が緑のままだった**。固定すると除外規定が働いていない実装と区別できない。除外を効かせる (= テストで守れる) には、入力側が実行のたびに変わっている必要がある。計画の決定性は固定値ではなく除外そのものが担保する |
| 18 | `warmup_runs` がプロンプト数を超える場合、ウォームアップは**先頭から巡回**して同じプロンプトを再利用する | 「ウォームアップ回数」はロード状態を作るための回数であって、追加のプロンプト集ではない。足りない分を計測用プロンプトから借りると、計測前に一部プロンプトだけ余分にキャッシュが温まる |
| 19 | VRAM はモデルごとに 2 回だけ読む (起動直後 = idle、ウォームアップ後 = 計測値)。**計測値はそのモデルの全レコード (warmup 含む) に同じ値を載せる** | プロンプトごとに読むと `nvidia-smi` の起動コストが計測時間に混ざる。値はモデル単位の事実なので、レコード単位で持たせても意味が増えない |
| 20 | `vram_increment_gib` が負でも 0 に丸めない | 負値は「前のモデルが退避された」という実態 (§7 リスク1 でまさに疑う対象)。0 に潰すと観測できなくなる |
| 21 | 起動に失敗したモデル (予算超過を含む) は、**プロンプト数と同じ件数の失敗レコード** (`phase="measure"`, `error` 付き) を残す。warmup レコードは 0 件 | 比較表の `n/N` の N をモデル間で比較可能に保つため。行ごと空にすると「速いから測定数が少ない」のか「そもそも走っていない」のかが表から読めない |
| 22 | 1 プロンプトの失敗ではそのモデルの残りのプロンプトを打ち切らない (全プロンプトを試して記録する) | 「ケース単位の失敗は記録して次へ進む」(§3 ソフト制約) の単位をプロンプトに取った。モデル単位で打ち切ると、たまたま 1 問だけ失敗した比較が丸ごと欠測になる |
| 23 | レコードは **素の frozen dataclass + `to_dict()`** (pydantic dataclass にしない)。`RECORD_KEYS` は `dataclasses.fields()` から導出する | ここは外部入力を検証する層ではなく自分で組み立てて書き出す層。`llmkit/manifest.py` と同じ形にそろえた (D-08 は満たす) |
| 24 | 1 モデルの中で速度の出典が混在したら、集計の出典ラベルは `"mixed"` にする | 分母の違う値 (eval と壁時計) の中央値を、片方のラベルで代表させると実測 1.8 倍の乖離 (F-4-004) が表の上で消える |
| 25 | 集計の VRAM 実測値は、そのモデルの計測レコードのうち**最初の非 null** を採る | モデル単位で 1 回しか読んでいない値なので中央値を取る意味がない |
| 26 | E18 のテスト名は `test_context_tokens_reach_the_request_the_record_and_the_fingerprint` にした (§5 の表は `..._the_report_...`) | T3 時点で `report.md` は存在しない。**T4 はこのテストに「設定コンテキスト長」列の検査を足し、§5 の名前へ戻すこと** |
| 27 | フィンガープリントの 13 要素の掃引は、**スイートファイルも設定ファイルも書き換えず** in-memory の `dataclasses.replace` で振る | ファイルを書き換えると `suite_sha256` / `config_sha256` が変わり、フィンガープリントは何を入力に含めていても変わってしまう (= 何も測れないテストになる)。ハーネスが実行時にやっているのも in-memory 上書きであり、D-20 が想定している状況そのもの |
| 28 | **`--models` フィルタで対象が 0 件になる経路は T3 では扱わない** (T4 の宿題)。実測: `dataclasses.replace(suite, models=())` は llmkit の例外階層の**外**にある生の `pydantic_core.ValidationError` を投げる (`models: Tuple should have at least 1 item`) | `plan_run` はスイートを受け取るだけで絞り込みを知らない。T4 は**フィルタ結果が空になった時点で** `ConfigError` (メッセージに `--models` と該当値を含む) に翻訳すること。空のまま `ComparisonSuite` を組み立てると利用者に pydantic の内部エラーがそのまま出る |
| 29 | §4 T4 の受け入れ基準 `rg -n "print\(" harness/` は**そのままでは常に 3 件ヒットする** (`run_fingerprint(` / `def fingerprint(` が部分一致するため)。単語境界付き (`rg -n "\bprint\(" harness/` = 0 件) で検査すること | T2 の決定12 と同じ「全文一致の機械検査が説明文や別語を誤検出する」型の問題。`print()` 禁止の実効的な担保は ruff `T20` であり、`make lint` が既に強制している |

### T4 (`harness/report.py` / `harness/cli.py` / `suites/ja_basic.toml`)

| # | 決めたこと | 理由 |
|---|---|---|
| 30 | 「測定数 n/N」の **n は生成速度の代表値が得られた件数、N は `attempted` (試行件数)**。`MetricSummary.coverage` (n/成功件数) は表に使わない | §4 T4 の定義「測定できた件数/試行件数」に合わせた。`coverage` は起動に失敗したモデルで `0/0` になり、「速いから測定数が少ない」のか「そもそも走っていない」のかが表から読めない (決定21 と同じ趣旨) |
| 31 | 「欠測時に `0.0` / `0` が現れない」の機械検査は**欠測セルそのものに限定**する。レポート全文の検査はしない | 全文検査は原理的に不可能。`測定数 n/N` は `0/8` が**正しい表示**であり、`10.06` のような正当な数値も部分文字列 `0.0` を含む。決定12・決定29 と同じ「全文一致の機械検査が正当な内容を誤検出する」型の問題 (`tests/test_harness_report.py::test_unmeasurable_cells_are_em_dashes_and_never_zero`、変異検証つき) |
| 32 | 応答本文は**引用ブロック (`> `) で書き、コードフェンスで囲まない**。表のセルにも入れない | 応答自身が ``` を含むと文書構造が壊れる。応答は制御できない (§7 リスク2)。表のセルに入れると `|` 1 個で列がずれる |
| 33 | 応答セクションの各モデル見出しの下に、1 行の事実サマリ (生成速度 / 完了トークン / `finish_reason`) を付ける | 応答の比較中に速度を見るために JSONL と往復させない。欠測は表と同じ `—` |
| 34 | 起動・生成に失敗したケースは「空の応答」ではなく **`- 失敗: <例外型>` + メッセージの引用**として書く | 空欄だと「モデルが何も返さなかった」と読める。決定21 で失敗レコードを残した意味が Markdown 側で消える |
| 35 | `run.json` の `reproduction` キーに `fingerprint_inputs()` の戻り値**そのもの**を載せる | §4 T4 受け入れ基準「`run.json` から `run_fingerprint` を再計算すると記録値と一致する」の実現方法。決定16 (入力を残す) の帰結。`fingerprint_digest(run_json["reproduction"]) == run_json["run_fingerprint"]` が成立する |
| 36 | **`--limit N` は「プロンプトを先頭 N 問に絞る」**と定義した (モデル数ではない)。`N < 1` は `ConfigError` (メッセージに `--limit` を含む) | §4 T4 は `--limit N` の対象を書いていない。ウォームアップ回数はロード状態を作る回数 (決定18) なので絞る対象にならず、モデルは `--models` が担う。残るのはプロンプトだけ |
| 37 | `--models` / `--limit` の絞り込みは **`run_fingerprint` を変える** (フル実行と一致しない) | 絞り込みは入力そのものの変更である。一致させると「8 問中 1 問だけ回した結果」と「8 問回した結果」が同じ再現条件を名乗る (D-20 の存在理由そのもの) |
| 38 | `--models` は**未知の `model_id` を黙って無視せず `ConfigError`** にする (結果が 0 件でなくても) | 打ち間違いを無視すると「指定したはずのモデルの行が表から抜けたまま完走する」。決定28 が要求する「0 件で `ConfigError`」も同じ分岐で満たす |
| 39 | CLI に `--results-dir` を**足さない**。出力先の差し替えは `main(..., results_root=, probe=, clock=)` のキーワード引数で行う | §4 T4 のオプション列を増やさずに、テストが `results/` へ 1 バイトも書かずに全経路を回せるようにする。`llmkit/cli.py` の `http_client` / `stdout` / `stderr` と同じ注入形にそろえた |
| 40 | 出力ディレクトリ名の時刻は **CLI が決め、同じ値を `run_suite(clock=...)` に渡す** | ディレクトリ名の `<YYYYmmddTHHMMSSZ>` と `run.json` の `started_at_utc` を食い違わせないため。帰結として、**同一秒に同一フィンガープリントで 2 回走らせると同じディレクトリへ上書きする** (T5 の 2 回目の実行は数分後になるため実害はない) |
| 41 | `harness/__init__.py` は `report` を再エクスポートするが **`cli` は再エクスポートしない** | `llmkit/__init__.py` と `test_layout.py::_SUBMODULE_NAMES` が `cli` を含まないのと同じ扱い (CLI は L4 の入口であって公開 API ではない)。**T5 の `tests/test_harness_layout.py` で `__all__` の和集合を検査する際は `cli` を除外すること** |
| 42 | E22 のテストは §5 の表どおり `tests/test_harness_gpu.py::test_vram_probe_result_is_visible_in_the_report` に置いた (`harness.report` を import する) | §5 が場所と名前を指定しており、T5 が D-23 の `guard_test` にこの名前を書く。決定26 (E18 の名前を §5 に戻す) と同じ理由で、テスト名は仕様側を正とする |
| 43 | `print()` 禁止の機械検査は **AST ベース** (`ast.Call` の関数名が `print`) にした。`rg` は使わない | 決定29 の帰結。決定29 が示した `rg -n "\bprint\(" harness/` も**まだ 1 件ヒットする**: `harness/cli.py` の docstring に「``print()`` ではなく `sys.stdout` へ書く」という規約の説明があるため (`llmkit/cli.py` と同じ文面)。全文一致では説明文と違反を区別できない。`tests/test_harness_cli.py::test_no_harness_module_calls_print` が**呼び出し**を見る (変異検証つき)。実効的な強制は ruff `T20` が担い、この検査はそれが外れていないことを見る |
| 44 | `suites/ja_basic.toml` の内容 (8 問 / 3 モデル / `max_output_tokens=512` / 日本語 / 機密らしき文字列なし) をテストで固定した | 承認済みの値 (Q-1 / Q-3) と D-24 が、`results/` にコミットする前に守られていることを機械で見る。T5 の L305 (日本語 CJK プロンプト 1 問以上) の前提でもある |

**T5 への申し送り (guard_test の候補)**

| 決定 | 実在する guard_test |
|---|---|
| D-19 | `tests/test_harness_runner.py::test_model_override_changes_both_the_request_and_the_vram_estimate` |
| D-20 | `tests/test_harness_runner.py::test_same_input_reproduces_the_fingerprint_while_config_sha256_alone_does_not` |
| D-21 | `tests/test_harness_report.py::test_unmeasurable_cells_are_em_dashes_and_never_zero` |
| D-22 | `tests/test_harness_runner.py::test_warmup_runs_are_recorded_but_excluded_from_aggregates` |
| D-23 | `tests/test_harness_gpu.py::test_vram_probe_result_is_visible_in_the_report` |
| D-24 | `tests/test_harness_cli.py::test_the_shipped_suite_carries_no_credential_like_strings` |

### 不具合修正 (2026-08-23、実機実行を受けたフィクサーによる追記)

実機実行 `results/ja_basic/20260823T024941Z-324de92cc334/` で、2 モデル目以降の
`vram_increment_gib` が意味を成さない値 (`gpt-oss-20b` で `1.37`、`qwen3-8b` で
`-5.24`) になっていることが判明した。原因は `harness/runner.py` がアイドル基準
(`idle = self._probe.read()`) を**モデルごとのループ内**で測り直しており、2 モデル
目以降の増分が「そのモデルの起動直後」ではなく「前のモデルがロード済みの状態」
からの差になっていたため (本仕様書冒頭の要求「起動直後に1回プローブし
`vram_idle_mib` を記録」に反する実装不具合であり、§7 リスク1 のモデル退避汚染とは
別の原因)。固定アイドル (844 MiB) を基準に取り直すと見積りと誤差 0.00〜0.01 で
一致し、静的テーブル (D-01) 自体は正しいことを確認済み。

**修正内容:**

- `harness/runner.py`: アイドル基準 (`probe.read()`) を `run_suite` の先頭で
  **1 回だけ**測り、`_SuiteRunner` の全ケースへ注入する形に変更 (モデルごとの
  ループからは削除)。`RunResult` に `vram_idle_mib: int | None` を追加
- `harness/report.py`: `build_run_json` の出力に `vram_idle_mib`
  (トップレベルキー) を追加。仕様書冒頭の「`vram_idle_mib` を `run.json` に記録」
  を満たす
- `harness/records.py`: `VramReading` の docstring を、`idle` が実行全体で
  1 回だけ測られ全モデルで共有される値であることが分かるように修正 (計算式・
  型は変更なし)

**訂正: 決定19 (T3, 上記) の記述は不正確だった。** 「VRAM はモデルごとに 2 回
だけ読む (起動直後 = idle、ウォームアップ後 = 計測値)」の**前半 (idle) は誤り**。
正しくは「idle は実行全体で 1 回だけ読み、ウォームアップ後の計測値のみモデルごと
に読む」。決定19 の後半 (計測値はモデル単位でよい) は変更なし。

**追加した guard_test:**

- `tests/test_harness_runner.py::test_idle_baseline_is_measured_once_and_shared_across_all_models`
  — プローブが呼ばれるたびに異なる値を返すフェイクを注入し、全モデルの
  レコードが同じ `idle` を基準にしていることを固定する (この不具合の再発防止)
- `tests/test_harness_report.py::test_the_run_json_records_the_shared_idle_baseline`
  — `run.json` の `vram_idle_mib` と、それを基準にした各モデルの
  `vram_increment_gib` を固定する
- `tests/test_harness_report.py::test_the_run_json_idle_is_null_when_no_gpu_probe_is_available`
  — プローブ不在 (CI 等) でも `vram_idle_mib` が `null` のまま完走することを固定する

### T5 (c)〜(f) (受け入れテスト / レイアウト検査 / 決定 / 文書、2026-08-23)

実機実行と成果物のコミット ((a)(b)) はメインセッションが完了済み。以下は (c)〜(f) の
実装者が仕様に書かれていない選択をした箇所。

| # | 決めたこと | 理由 |
|---|---|---|
| 45 | L302-L304 は**出荷スイート `suites/ja_basic.toml` と `configs/default.toml` をそのまま入力**にし、`harness.cli.main` 経由で回す (テスト専用の小さいスイートを新設しない) | 受け入れ条件は「プロンプト集とモデルリストを**与えると**生成される」であり、実際に出荷する入力で測るのが条件に最も近い。テスト専用スイートを別に書くと、出荷スイートが壊れていても受け入れテストは緑のままになる。書き出し先・プローブ・時刻はすべて注入する (決定39) ため `results/` には 1 バイトも書かない |
| 46 | **「4 条件 ↔ 4 関数」の対応検査は 5 つ目の関数を作らず**、各テストの先頭で `assert_is_acceptance_line(番号)` を呼ぶ形にした | Phase 2 には Phase 1 の L298 (「上記に対する自動テストが存在しパスする」) にあたる条件が無い。5 関数目を足すと 4 対 4 の対応そのものが崩れる。`ACCEPTANCE_MAP` は対応表として残し、docstring の冒頭に `要件書 LNNN:` を書く体裁は `test_acceptance_phase1.py` と同じ |
| 47 | 参照行の検査は **`- [` で始まること**だけを見る (`- [ ] ` 固定にしない)。`test_l298` (Phase 1) の `- [ ] ` 固定は**変更しない** | Phase 2 完了と同時に `- [x]` へ変わるため。Phase 1 側を触ると「無修正で緑」(§4 T5 受け入れ基準) が検証できなくなる |
| 48 | L305 の**日本語判定は `report.md` の「プロンプト:」直後の引用ブロック**に CJK 文字があることで見る | `records.jsonl` / `run.json` はプロンプト本文を持たず `prompt_sha256` しか無い (D-20 の入力設計の帰結)。コミット済み成果物の中でプロンプト本文の出典は `report.md` だけ |
| 49 | L305 は「条件を満たす実行が **1 つ以上ある**」を検査する (全ディレクトリが満たすことは求めない) | 将来 `--models` で絞った実行を追加でコミットしても受け入れ条件は壊れない。恒真になっていないことは**変異検証**で確認済み (閾値を 3→4 モデルにすると落ちる) |
| 50 | D-08 guard の走査範囲拡張では、**`harness/` が実際に走査対象に入っていること自体をアサート**した (`module_paths` が `llmkit/` だけの件数より多い) | glob が空を返しても素通りする検査になるため。範囲の「拡張」が事実であることを機械で見る |
| 51 | **D-18 の `guard_test` は `tests/test_harness_suite.py::test_load_suite_reads_every_declared_field`** を採った (§9 の申し送り表に D-18 の候補が無かった) | 単一 TOML に `[suite]` / `[[models]]` / `[[prompts]]` が同居し全フィールドが読めることを固定しており、入力を YAML/JSONL へ移すと必ず落ちる |
| 52 | **D-24 の `guard_test` は申し送り表どおり**機密文字列の検査を採り、「`results/` はコミット対象」側は `tests/test_acceptance_phase2.py::test_l305_japanese_comparison_results_are_committed` が実質の guard になる | 1 決定 1 guard のため、両方は書けない。`results/` を ignore した瞬間に落ちるのは L305 の方だが、機密混入の方が不可逆な事故であるため guard に採った (L305 の存在は本節に記録する) |
| 53 | **README の「実行マニフェスト」節の記述を訂正**した (「同じ設定なら `config_sha256` が一致するため Phase 2 の比較実験の再現条件になる」→ 実効値の同一性は `run_fingerprint` が担う) | §4 T5(f) は README の**新設節**しか指示していないが、放置すると同じ README の中に D-20 と矛盾する記述が残る。Phase 1 時点では正しかった記述が Phase 2 で偽になった箇所 |
| 54 | `docs/next-pr-candidates.md` の F-1-004 は**行を削除せず** severity と id を打ち消し線にし、内容セルの末尾に対応済みの追記をした | 同ファイルの「Phase 0 実測で判明した課題」節が既に採っている体裁 (`~~MEDIUM~~` + 「**対応済み (本 PR)**」) にそろえた。行を消すと過去のレビュー結果との対応が追えなくなる |
| 55 | L304 の 2 回実行は、**応答本文・完了トークン数・所要時間を実行ごとにずらすハンドラ**を使う (`varying_handler(offset=...)`) | 2 回とも同じ応答を返すと「再現された」が恒真になる。実際、当初は両実行で同じ応答列になっており、`response_text` の差分アサーションが落ちて発覚した |
