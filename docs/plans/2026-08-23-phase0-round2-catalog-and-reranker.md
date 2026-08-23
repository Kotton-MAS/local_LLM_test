# Phase 1 追補 2: Phase 0 残り2項目の実機完了をカタログ・VRAM 見積り・ドキュメントへ反映する

> 前提資料: `docs/phase0-vram-measurements.md`（第2次実測を追記済み）/ `docs/plans/2026-08-22-phase1-calibration-and-native-chat.md`（直前の追補）/ `.claude/decisions.yaml`（D-01〜D-14）
> 位置づけ: 既存 `llmkit` への**追加変更**。現在の PR #1 に含める。新規パッケージは作らない。
> 推論呼び出し（embed / rerank）は **Phase 3 のまま。今回は実装しない。**
>
> **承認済み (2026-08-23)**: Q-1 = D-14 は id 維持で「撤回」に書き換え / Q-2 = `ModelSpec.serving_runtime` を既定値なしで追加 / Q-3 = `weights_gib` に実測 VRAM 増分そのものを入れる。

## 1. ゴール

Phase 0 で実測済みとなった埋め込み・リランカー・構成1同居の 3 値をカタログと文書に反映し、「別ランタイムで動くモデル」をメタデータとして識別可能にする。

## 2. 現状認識

| パス | 内容 |
|---|---|
| `llmkit/catalog.py` | `ModelSpec`（8 フィールド、frozen+slots、標準 dataclass）。`ruri-v3-310m` (0.7) / `ruri-reranker` (0.8) の `source_note` に「取得不能」「(仮)」 |
| `llmkit/vram.py` | 見積り式 `Σweights + kv(generation) + runtime_overhead_gib`。**プロセス単位の前提は入っていない** |
| `llmkit/manifest.py` | `ManifestModelEntry`（4 キー）。モデル単位メタデータの唯一の出力先 |
| `llmkit/config.py` | `RuntimeKind = Literal["ollama","openai_compatible"]` — **接続単位**のチャット送出経路 |
| `llmkit/client.py` | `ApiStyle = Literal["ollama_native","openai_compatible"]` — **ワイヤプロトコル**。`RuntimeKind` から導出 |
| `tests/test_catalog.py` | `GOLDEN_ROWS` / `PHASE0_MEASUREMENTS` / `UNAVAILABLE_MODEL_IDS` / フィールド集合の完全一致テスト |
| `tests/test_vram.py` | `SHIPPED_PROFILE_ESTIMATES`（`rag_default@16384=12.63` / `oversized@131072=16.74`） |

### 既存の慣習で守るべきもの

1. **ゴールデン値は1か所でロックする**（カタログ数値とプロファイル合計）
2. **アサーションは緩めず「より強い主張へ再アンカー」する**（前 PR で `source_note` テストを2本に分割した前例に従う）
3. **公開シンボルを増やしたら `llmkit/__init__.py` に再エクスポートする**（`test_public_api_matches_the_union_of_submodule_all` が和集合の等号で検査）

### 影響範囲

`llmkit/{catalog,manifest,__init__}.py` / `configs/*.toml` 3本 / `tests/` 5ファイル + 新規1本 / `scripts/`（新設）/ `docs/` 3本 / `README.md` / `.claude/decisions.yaml`。
**`llmkit/{config,client,vram,bootstrap,cli,errors}.py` は変更しない。**

### 較正の計算（メインセッションが独立に検算済み）

実測（`docs/phase0-vram-measurements.md`）:

| 段階 | 累計 | 増分 |
|---|---|---|
| アイドル | 0.56 | — |
| + リランカー (llama-server) | 0.84 | 0.28 |
| + `qwen3:14b` @16384 | 11.96 | 11.12 |
| + 埋め込み | 12.53 | 0.57 |

**アイドルを除いた増分合計 = 12.53 − 0.56 = 11.97 GiB**（0.28+11.12+0.57 = 11.97 で自己整合）

較正後カタログによる `rag_default @16384` の見積り:

```
qwen3-14b weights 7.81 + 埋め込み 0.57 + リランカー 0.28
+ KV 0.1575×16 = 2.52 + overhead 0.80 = 11.98 GiB
```

**予測 11.98 vs 実測増分 11.97 → 差 0.01 GiB。** 成分ごとにも一致する:

| 成分 | カタログ | 実測増分 | 差 |
|---|---|---|---|
| 生成 (7.81 + 2.52 + 0.80) | 11.13 | 11.12 | 0.01 |
| 埋め込み | 0.57 | 0.57 | 0.00 |
| リランカー | 0.28 | 0.28 | 0.00 |

**既存較正値との整合**: 生成の 11.13 は Phase 0 単体実測と一致しており、同居によって生成モデルの占有量は変わらない。既存 3 モデルの値と `runtime_overhead_gib=0.8` は**一切変更しない**。

**他プロファイルへの波及**:

| プロファイル | 変更前 | 変更後 | 予算 14.0 の判定 |
|---|---|---|---|
| `rag_default @16384` | 12.63 | **11.98** | 収まる（余裕 2.02） |
| `oversized @131072` | 16.74 | **16.09** | **超過 2.09**（D-04 guard は既定予算のまま維持） |
| `long_context` / `lightweight` | 変更なし | 変更なし | 変更なし |

## 3. 前提・制約

### ハード制約

- 既存 **313 テストを壊さない**。**アサーションを緩めない**。期待値の変更は §4 に「変更前 → 変更後」を明記した行のみ許す
- mypy `strict` + `disallow_any_explicit`。`Any` 明示禁止。`pydantic.BaseModel` を継承しない（D-08）
- ruff `T20`（`print()` 禁止）。テストは実 HTTP を出さない（D-02）
- **埋め込み・リランカーの推論呼び出しを実装しない**（Phase 3）。`llmkit/` に `embed()` / `rerank()` を追加しない
- `main.py` / `Makefile` / `.github/workflows/` を変更しない
- 既存公開シグネチャを変更しない（追加のみ）
- 起動スクリプトのリポジトリ外パスは**すべて環境変数で上書き可能**にする。`systemd` を触らない。`sudo` を書かない
- CI（GPU 無し・llama-server 無し・Ollama 無し）で全テストが緑であること

### 論点への判定

**別プロセス分を合算してよいか → よい。`vram.py` は変更しない。**

`estimate_resolved_profile` には「同一 Ollama プロセス内の合計」という前提がコードにもドキュメントにも入っていない。判定対象は `nvidia-smi` が返すデバイス全体の使用量（プロセス非依存）であり、`budget_gib` は D-12 で「実測でオフロードが始まらない**増分**の上限」と定義済み。これはプロセス境界と無関係な量である。実測でも予測 11.98 と実測 11.97 が 0.01 GiB で一致した。

ただし `runtime_overhead_gib`（0.8）は**生成ランタイム1つ分**として較正された値である。リランカー側の llama-server 自身のオーバーヘッドは実測増分 0.28 に含まれるため、`weights_gib=0.28` は「純粋な重み」ではなく**実測増分そのもの**である（D-16）。

**`ServingRuntime` を新設する理由**（既存 `RuntimeKind` の再利用を退けた根拠）:

| 型 | 所在 | 粒度 | 意味 |
|---|---|---|---|
| `RuntimeKind` | `config.py` | 接続 | 設定が選ぶチャット送出経路 |
| `ApiStyle` | `client.py` | 接続 | ワイヤプロトコル。`RuntimeKind` から導出（D-10） |
| **`ServingRuntime`（新）** | `catalog.py` | **モデル単位・静的** | そのモデルを載せるサーバプロセス |

3つは直交する。リランカーに `openai_compatible` を持たせると「このモデルは chat を話す」という**偽の主張**になる。加えて `catalog.py` が `config.py` を import すると最下層モジュールの独立性（`test_estimate_is_pure_and_needs_no_gpu` が守る性質）が崩れる。

**デフォルト値を付けない。** 既定 `"ollama"` にすると将来追加されるモデルが黙って Ollama 扱いになり、今回検出した誤りを再生産する。

**`model_id` を `ruri-reranker` → `bge-reranker-v2-m3` に改名する。** 実際に動かしているのが BGE なのに `ruri-reranker` を残すと、`model_id: ruri-reranker` / `served_name: bge-reranker-v2-m3-Q6_K.gguf` という**自己矛盾したマニフェスト**が残り Phase 2 の比較記録が読めなくなる。漏れは `test_every_model_referenced_by_shipped_configs_exists` が機械検出する。

埋め込みは `model_id="ruri-v3-310m"` を維持（モデル自体は変わらない）。変わるのは `served_name` のみ。

## 4. タスク分解

**実行順序 T1 → T2 → T3 → T4 → T5 を厳守。** T1+T2 完了時点で `make ci` が緑になることを確認してから T3 に進む。T4（decisions）は guard_test が実在してから。

### T1: カタログを第2次実測で較正し、`ServingRuntime` を導入し、リランカーを改名する

| 項目 | 変更前 | 変更後 |
|---|---|---|
| 埋め込み `served_name` | `hf.co/cl-nagoya/ruri-v3-310m` | `hf.co/Targoyle/ruri-v3-310m-GGUF` |
| 埋め込み `quantization` | `fp16` | GGUF 変換版の実量子化。不明なら `unknown (GGUF)` とし `source_note` に未確認と書く（`fp16` のまま残さない） |
| 埋め込み `weights_gib` | 0.7 | **0.57** |
| 埋め込み `serving_runtime` | — | `"ollama"` |
| リランカー `model_id` | `ruri-reranker` | **`bge-reranker-v2-m3`** |
| リランカー `served_name` | `hf.co/cl-nagoya/ruri-reranker-large` | `bge-reranker-v2-m3-Q6_K.gguf` |
| リランカー `quantization` | `fp16` | **`Q6_K`** |
| リランカー `weights_gib` | 0.8 | **0.28** |
| リランカー `max_context_tokens` | 512 | **8192**（BGE Reranker v2-m3 系列の上限。**未実測であることを明記**） |
| リランカー `serving_runtime` | — | **`"llama_cpp_server"`** |
| 生成 3 モデル | — | 数値は**一切変更しない**。`serving_runtime="ollama"` のみ追加 |

- `ServingRuntime = Literal["ollama", "llama_cpp_server", "external"]` を `catalog.py` に定義し `__all__` と `llmkit/__init__.py` に追加
- `resolve_model_spec` の passthrough 合成は `serving_runtime="external"`
- 埋め込み・リランカーの `source_note` から `(仮)` / `取得不能` / `Phase 3` を**外し**、`実測 2026-08-23 / docs/phase0-vram-measurements.md` と、**`weights_gib` が「そのランタイムのオーバーヘッドを含む実測 VRAM 増分」であること**を書く。リランカーには「要件書第一候補の Ruri Reranker は GGUF 非提供のため、要件書が代替として挙げる BGE Reranker v2-m3 を採用」を書く

**テスト変更**

- `GOLDEN_ROWS`: 埋め込み `0.7 → 0.57`、リランカー行を `("bge-reranker-v2-m3", "reranker", "Q6_K", 0.28, 0.0)` に
- `UNAVAILABLE_MODEL_IDS` と `test_unavailable_models_declare_their_unavailability` を**削除**し、以下2本へ置換（緩和ではなく再アンカー）:
  - `test_no_catalog_entry_claims_to_be_unavailable`（**D-14 撤回の guard**）: 全5モデルの `source_note` に `(仮)` / `取得不能` を含まず `docs/phase0-vram-measurements.md` を含む。かつ `"bge-reranker-v2-m3" in known_model_ids()` と `"ruri-reranker" not in known_model_ids()`
  - `test_phase0_round2_models_cite_the_2026_08_23_measurements`: 埋め込み・リランカーの `source_note` に `実測 2026-08-23` が含まれる
- `test_models_declare_the_runtime_that_serves_them`（**D-15 の guard**）: (a) 生成3+埋め込みが `"ollama"`、リランカーが `"llama_cpp_server"`、(b) passthrough が `"external"`、(c) `set(get_args(ServingRuntime))` が `RuntimeKind` とも `ApiStyle` とも一致しない（3概念が別物であることを型レベルで固定）
- `test_model_spec_declares_all_documented_fields`: 期待集合に `serving_runtime` を追加（9フィールド）
- `test_catalog_reproduces_phase0_measurements`（実測8点）と `PHASE0_MEASUREMENTS` / `CALIBRATED_MODEL_IDS` / `FIT_OVERHEAD_GIB` は**変更しない**

**マニフェスト配線**: `ManifestModelEntry` に `serving_runtime: str` を追加し `to_dict()` に含める。`build_manifest` のシグネチャは不変。`EXPECTED_MODEL_KEYS` に追加（5キー）

**受け入れ基準**

- `rg -n "ruri-reranker" --glob '!docs/plans/**'` が **0件**
- `test_catalog_reproduces_phase0_measurements` の8点が**無修正で**緑（生成3モデルを触っていない証明）
- `serving_runtime` を全件 `"ollama"` に潰すと **D-15 guard と E15 の両方**が落ちる
- `uv run mypy .` が `# type: ignore` 0個で通る

**所要**: L

### T2: プロファイル見積りを再アンカーし、構成1 同居実測の再現テストを追加する

| ファイル | 変更前 → 変更後 | 理由 |
|---|---|---|
| `test_vram.py` `SHIPPED_PROFILE_ESTIMATES` | `rag_default@16384 12.63 → 11.98` / `oversized@131072 16.74 → 16.09` | T1 較正 |
| `test_vram.py` 内訳期待値 | `0.7 / 0.8 → 0.57 / 0.28`（キー名も改名） | T1 較正 |
| `test_vram.py` 閾値テスト | baseline `12.63 → 11.98`、strict `budget_gib 12.0 → 11.5`、excess `0.63 → 0.48` | 見積りが下がり 12.0 では反転しないため |
| `test_bootstrap.py` ログ期待値2箇所 | `"12.63" → "11.98"` | T1 較正 |
| `test_bootstrap.py` docstring | 「見積り 16.74」→「16.09」 | T1 較正 |

`test_budget_reflects_the_measured_gpu_resident_ceiling`（D-12 guard）・`test_lowering_the_budget_alone_flips_bootstrap_from_ok_to_abort`・`test_oversized_profile_warns_and_aborts_without_http`（D-04 guard、16.09 > 14.0 で不変）は**変更しない**。

**新規テスト** `test_estimate_reproduces_the_configuration1_coresidency_measurement`（**D-16 の guard**、E14）:

- 実測4段（0.56 / 0.84 / 11.96 / 12.53）を定数で持ち `MEASURED_INCREMENT_GIB = 12.53 - 0.56` を導出
- `estimate_profile(config, "rag_default", context_tokens=16384).total_gib == approx(11.97, abs=0.02)`
- **成分ごとの一致も検証**: リランカー `approx(0.28)`、埋め込み `approx(0.57)`、生成 `weights + kv + overhead == approx(11.12, abs=0.02)`
- docstring に「アイドル 0.56 GiB は増分予算に含めない（D-12）」「リランカーは別プロセスだがデバイス全体で判定するため合算する（D-16）」を書く

**受け入れ基準**

- 上表の行**だけ**が変更された期待値であること（`git diff` で確認できる形で報告）
- **リランカーを 0.28→0.8、埋め込みを 0.57→0.7 に戻す変異のそれぞれで、このテストが落ちることを実測して報告する**
- `rg "budget_gib = 12\.0"` が引き続き 0件
- `make ci` が緑

**所要**: M

### T3: llama-server 起動スクリプトを追加し、パス解決をテストで固定する

**`scripts/start-reranker.sh`（新規、bash）**

`#!/usr/bin/env bash` + `set -euo pipefail`。環境変数（すべて `${VAR:-既定}` 形式）:

| 変数 | 既定 |
|---|---|
| `LLAMA_SERVER_BIN` | `${HOME}/.local/opt/llama.cpp/llama-server` |
| `RERANKER_MODEL` | `${HOME}/.local/share/llama-models/bge-reranker-v2-m3-Q6_K.gguf` |
| `RERANKER_HOST` | `127.0.0.1` |
| `RERANKER_PORT` | `8081` |
| `RERANKER_NGL` | `99` |
| `RERANKER_CTX_SIZE` | `2048` |

- 起動: `--reranking --host --port --n-gpu-layers --ctx-size --model`
- `--dry-run`: 解決済みコマンド行を1行で標準出力に出して **exit 0**。**存在チェックを行わない**（CI に llama-server が無いため）
- 通常実行時のみバイナリ・モデルの存在を確認し、無ければ **stderr に取得手順を含む対処方法**を出して exit 1
- `sudo` / `systemctl` / `rm` を書かない。パイプ実行を書かない。認証情報を扱わない
- ヘッダコメントに llama.cpp のビルド番号（`b10586` / Vulkan x64）とモデル出典、**`RERANKER_CTX_SIZE` を上げるとカタログの 0.28 GiB が過小評価になる**旨を書く

**`tests/test_reranker_script.py`（新規）**

- `test_launch_script_paths_are_overridable_by_environment`（**D-17 の guard**、E16）: `--dry-run` を2回実行し、既定 env と差し替え env で出力の対応箇所が**すべて変わる**ことを4項目 parametrize で検証
- `test_launch_script_never_escalates_privileges_or_touches_systemd`: 本文に `sudo` / `systemctl` / `rm -rf` が現れない。`RERANKER_HOST` の既定がループバック
- 実行ビットが立っていること

**受け入れ基準**

- `bash scripts/start-reranker.sh --dry-run` が llama-server 未インストール環境で **exit 0** かつコマンド行1行を出力
- **いずれか1つを `${VAR:-既定}` からハードコードに変える変異で、対応するケースが落ちることを実測して報告する**
- `uv run pre-commit run shellcheck --all-files` が緑

**所要**: M

### T4: `.claude/decisions.yaml` の D-14 撤回と D-15〜D-17 の追記

§6 の4エントリを反映する。**T1〜T3 で guard_test が実在してから**着手する。

**受け入れ基準**: `check_decisions.py` が「17 件の決定、全てに guard_test あり」で通り、D-14〜D-17 の guard_test 4本を個別実行して緑

**所要**: S

### T5: 実測に追随していない文書を訂正する

- **`docs/phase0-vram-measurements.md`**: 「余力 3.47 GiB」は **3.46 が正しい**（16376 MiB = 15.9922 GiB、15.99 − 12.53 = 3.46）。メインセッションが検算済み
- **`docs/localllmrequirements.md`**: **行数を1行も増減させない**（`ACCEPTANCE_MAP` が L294-L298 を参照）。訂正対象:
  - 構成1: 「較正後見積り 12.63GiB（未実測）」→「**実測 12.53GiB 累計 / 増分 11.97GiB**（2026-08-23、較正後見積り 11.98 と 0.01 GiB 一致）」
  - 構成3: `16.74` → `16.09`
  - 「リランカーは 12GB 級の民生 GPU で…」→ 実測 0.28 GiB / 50ペア 140ms に置換
  - 「初期採用: `Ruri Reranker`」→ **`BGE Reranker v2-m3` を採用**（Ruri は GGUF 非提供）。「50ペアで 500〜800ms」→ **実測 140ms**
  - **Phase 0 受け入れ条件のチェックボックスを `- [x]` に更新する**（`test_l298` は L294-L298 のみを対象とするため、Phase 0 のブロックは対象外。テストで確認すること）
- **`docs/next-pr-candidates.md`**: Phase 0 課題の HIGH 行を「**対応済み（本 PR）**」に更新し、選択肢 (A)+(C) のハイブリッドを採った（埋め込みは有志 GGUF で Ollama、リランカーは別ランタイム）ことと D-14 撤回を記す
- **`README.md`**: 実装状況表の Phase 0 を「完了（2026-08-23 実測）」に。「Phase 0 は未了です」注記を差し替え。**「リランカーは別ランタイム」節**を新設し `scripts/start-reranker.sh` の手順・環境変数一覧・「Phase 1 では推論呼び出しは未実装（VRAM 見積りにのみ計上）」を書く。Project Structure に `scripts/` を追加。決定の参照範囲を `D-01〜D-17` に更新

**受け入れ基準**: 要件書の行数が不変で `test_l298` が緑。`docs/phase0-vram-measurements.md` に「未達」「未実施」が残っていない

**所要**: S

## 5. 評価軸

### 機能観点

新規の機能的主張は1つ: 「構成1（生成 + 埋め込み + 別ランタイムのリランカー）の VRAM 見積りが実機の同居実測を 0.02 GiB 以内で再現する」。測定は純関数（GPU 不要）。

### 性能観点

- `uv run pytest` が **12秒未満**（T3 で `subprocess` を使うテストが増えるため既存目安 10秒から緩和。超えたら報告）
- VRAM 見積りは引き続き純関数（`test_estimate_is_pure_and_needs_no_gpu` を変更せず、`catalog.py` に禁止 import を持ち込まない）

### 安全性観点

- 起動スクリプトが `sudo` / `systemctl` / `rm -rf` / パイプ実行を含まない（テストで機械検証）。`RERANKER_HOST` の既定がループバック
- **D-04 guard（予算超過で HTTP 0回）が既定 `budget_gib=14.0` のまま維持されること**
- 埋め込み・リランカーの推論コードが混入していないこと（`rg -n "def embed|def rerank|/v1/embeddings|/v1/rerank" llmkit/` が 0件）

### ★ 有効性観点

既存 E1〜E13 を維持したうえで:

| # | 掃引する値 | 変わるべき出力 | テスト |
|---|---|---|---|
| **E14** | 埋め込み / リランカーの `weights_gib` | 構成1 同居実測 11.97 の再現が崩れる | `test_vram.py::test_estimate_reproduces_the_configuration1_coresidency_measurement` |
| **E15** | `ModelSpec.serving_runtime` | マニフェスト JSON の `profile.models[].serving_runtime` | `test_manifest.py::test_serving_runtime_change_is_visible_in_the_manifest` |
| **E16** | 起動スクリプトの4環境変数 | `--dry-run` が出すコマンド行 | `test_reranker_script.py::test_launch_script_paths_are_overridable_by_environment` |

**E15 が「値を変えても出力が変わらない」状態で通る実装は不合格。** `serving_runtime` を追加したのにマニフェストへ配線しないと、この属性は Phase 3 まで誰にも観測されない飾りになり、`source_note` 案を退けた理由が消える。

**変異検証を必須とする**:

1. リランカーの `weights_gib` を 0.8 に戻す → E14 と `GOLDEN_ROWS` の**両方**が落ちる
2. `serving_runtime` を全件 `"ollama"` に潰す → D-15 guard と E15 が落ちる
3. `ManifestModelEntry.to_dict()` から `serving_runtime` を落とす → E15 と `assert_manifest_shape` が落ちる
4. スクリプトの `LLAMA_SERVER_BIN` をハードコードに変える → E16 の該当ケースが落ちる

## 6. 意図的な決定（`.claude/decisions.yaml`）

```yaml
- id: D-14
  rule: "D-14 (ruri-v3-310m / ruri-reranker は取得不能。方式決定は Phase 3) を撤回する。カタログの全モデルは実測に裏付けられ、source_note に『取得不能』『(仮)』を残さない。リランカーは要件書第一候補の Ruri Reranker ではなく BGE Reranker v2-m3 (model_id = bge-reranker-v2-m3) を採用する"
  rationale: "2026-08-23 の実機検証で前提が覆った。埋め込みは有志の GGUF 変換版 hf.co/Targoyle/ruri-v3-310m-GGUF で ollama pull でき /v1/embeddings が 768 次元を返した (VRAM 0.57 GiB)。リランカーは Ollama に rerank API が無い (v0.33.0-rc2 でも /api/rerank・/v1/rerank とも 404) が llama.cpp の llama-server で /v1/rerank が動作した (VRAM 0.28 GiB、50 ペア 140 ms)。Ruri Reranker は GGUF が存在しないため、要件書が代替として挙げる BGE Reranker v2-m3 を採った。旧 D-14 の『Phase 3 まで決定しない』は事実に反する状態になり、残すと fixer が仮値を復活させる根拠になる。id を維持するのは過去の仕様書・PR 本文が D-14 を参照しているため"
  guard_test: "tests/test_catalog.py::test_no_catalog_entry_claims_to_be_unavailable"

- id: D-15
  rule: "『どのサーバプロセスがそのモデルを載せるか』は ModelSpec.serving_runtime (ServingRuntime = Literal['ollama','llama_cpp_server','external']) で表す。既存の config.RuntimeKind / client.ApiStyle を再利用せず、既定値も与えない。Phase 1 ではディスパッチに使わず、実行マニフェストに記録する"
  rationale: "RuntimeKind は接続単位のチャット送出経路、ApiStyle はワイヤプロトコルであり、どちらもモデル単位の『載せ先プロセス』を表さない。リランカーに openai_compatible を持たせると『このモデルは chat を話す』という偽の主張になる。また catalog.py が config.py を import すると最下層モジュールの独立性が崩れる。source_note に散文で書く案は機械識別できず Phase 3 のディスパッチが文字列マッチに退行するため採らない。既定値を与えると将来のモデルが黙って Ollama 扱いになり、まさに今回検出した誤りを再生産する。飾りにしないためマニフェストへ配線し掃引テストで固定する"
  guard_test: "tests/test_catalog.py::test_models_declare_the_runtime_that_serves_them"

- id: D-16
  rule: "VRAM 見積りは別プロセス (llama-server) が確保する分も単純に合算する。llmkit/vram.py の式は変更しない。埋め込み・リランカーの weights_gib は『純粋な重み』ではなく、そのランタイムのオーバーヘッドを含む実測 VRAM 増分そのものを入れる。vram.runtime_overhead_gib (0.8) は生成ランタイム 1 つ分を表す"
  rationale: "予算判定の対象は nvidia-smi が返すデバイス全体の使用量であり、プロセス境界と無関係。budget_gib は D-12 で『実測でオフロードが始まらない増分の上限』と定義済みで、これもプロセス非依存の量である。2026-08-23 の同居実測 (アイドル 0.56 を除く増分 11.97 GiB) に対し較正後見積りは 11.98 GiB で、成分ごとにも一致した (生成 11.13 vs 11.12 / 埋め込み 0.57 / リランカー 0.28)。モデル単位のオーバーヘッド欄を足すと見積り式と ModelSpec の両方が変わるが、実測は合計値しか取れておらず分離の根拠が無い"
  guard_test: "tests/test_vram.py::test_estimate_reproduces_the_configuration1_coresidency_measurement"

- id: D-17
  rule: "llama-server の起動はリポジトリ内の scripts/start-reranker.sh で行い、systemd ユニットを作らない。バイナリ・モデル・ホスト・ポート・GPU レイヤ数・ctx-size はすべて環境変数で上書き可能にし、リポジトリ外の絶対パスをハードコードしない。--dry-run は解決結果を出力して exit 0 する (存在チェックをしない)"
  rationale: "llama.cpp は ~/.local/opt、モデルは ~/.local/share/llama-models に置いた (システムを汚さないユーザースコープ)。この配置は開発者ごとに異なるため既定値としてのみ持ち上書き可能にする。systemd を触ると Ollama 側の管理方式と二重になり、不可逆な変更 (要承認) にもなる。--dry-run を存在チェック無しにするのは llama-server が無い CI で環境変数の配線をテストするため (無いと『上書きできるはず』が散文の主張のままになる)"
  guard_test: "tests/test_reranker_script.py::test_launch_script_paths_are_overridable_by_environment"
```

## 7. 想定リスク（これが起きたら止まって相談）

1. **`RERANKER_CTX_SIZE` を上げると 0.28 GiB が過小評価になる。**
   実測 0.28 GiB は `--ctx-size 2048` / `--n-gpu-layers 99` の1点でしか取れていない。llama-server の VRAM は ctx-size とバッチに依存するため、Phase 3 で長い文書を渡す設計に変えると予算が静かに不足する。**スクリプトの既定を変える提案が出たら止めて相談する**（`source_note` とスクリプトのヘッダに測定条件を明記して警告を残すこと）。

2. **`runtime_overhead_gib` が単一定数であることの限界。**
   0.8 GiB は「生成ランタイム1つ分」として較正されている。将来もう1つ**生成級の**ランタイム（vLLM 等）を同居させるとこの定数では足りない。今回のリランカーは増分 0.28 に自身のオーバーヘッドが含まれるため問題ないが、**プロファイルに2つ目の `role="generation"` を入れる設計が出てきたら止めて相談する**（`ResolvedProfile.generation` が最初の1件しか返さない件とも衝突する）。

3. **`model_id` 改名によりマニフェストの `config_sha256` が変わる。**
   `configs/*.toml` を書き換えるため、本 PR 以前に生成した `outputs/runs/*.json` とはハッシュが一致しなくなる。Phase 1 では比較実験がまだ無いため実害はないと判断したが、**すでに保存済みのマニフェストを再現条件として使っている作業があれば止めて相談する。**

## 参考: 較正後の見積り値一覧（overhead 0.8 込み、単位 GiB。太字が本 PR で変わる値）

| プロファイル | 構成 | ctx=16,384 | ctx=32,768 | ctx=65,536 | ctx=131,072 |
|---|---|---|---|---|---|
| `rag_default` | 14b + 埋め込み + リランカー | **11.98**（旧 12.63、実測増分 11.97） | **14.50**（旧 15.15） | — | **29.62**（旧 30.27） |
| `long_context` | gpt-oss-20b | 12.51 | 12.90（実測一致） | 13.68（実測一致） | 15.24 |
| `lightweight` | 8b | 7.26（実測一致） | 9.54（実測一致） | — | — |
| `oversized` | 20b + 埋め込み + リランカー | **13.36**（旧 14.01） | **13.75**（旧 14.40） | **14.53**（旧 15.18） | **16.09**（旧 16.74） |

予算 14.0 での判定: `rag_default @16384` 収まる（余裕 2.02）/ `long_context @65536` 収まる（余裕 0.32）/ `long_context @131072` 1.24 超過 / `oversized @131072` **2.09 超過（D-04 guard は維持）**。

## 8. 実装時に決めたこと (T1 / T2、仕様に書かれていなかった選択)

implementer が T1・T2 の実装中に自分で決めた事項。次周の reviewer / fixer はここを読む。

### T1

1. **埋め込みの `quantization` は `unknown (GGUF)` を採用した。**
   §4 T1 の表が許した分岐のうち「不明」側を採った。`hf.co/Targoyle/ruri-v3-310m-GGUF` は
   既定タグで pull しており、配布側の量子化タグを確認していないため。推測で `Q8_0` 等を
   書くと `source_note` の「実測」表記と混ざって区別できなくなる。
   → 波及: `GOLDEN_ROWS` の埋め込み行の quantization 列も `fp16 → unknown (GGUF)` に変えた
   (§4 T1 のテスト変更リストは `0.7 → 0.57` しか挙げていなかったが、
   `test_catalog_golden_values_are_locked` が quantization も照合するため必然)。

2. **埋め込みの `max_context_tokens` は 8192 のまま据え置き、`source_note` に
   「モデルカード由来で未実測」と書いた。** §4 T1 の表はリランカーの 512 → 8192 しか
   指示していない。埋め込み側は値を変える根拠 (実測) が無いため据え置き、ただし
   `(仮)` を外す以上「何を根拠に 8192 なのか」が消えるため出所だけ明記した。

3. **`source_note` に `/v1/embeddings` / `/v1/rerank` という文字列を書かない。**
   当初は実測条件として書いていたが、§5 安全性観点の
   `rg -n "def embed|def rerank|/v1/embeddings|/v1/rerank" llmkit/` が 0 件であるべき検査
   なので、散文がヒットすると検査がノイズで死ぬ。事実は
   「埋め込みエンドポイントが 768 次元を返す」「native・OpenAI 互換のどちらの経路でも 404」
   と言い換えて保持した。**この検査は現在 0 件。**

4. **`serving_runtime` は `ModelSpec` / `ManifestModelEntry` とも `role` の直後に置いた。**
   識別子系 → 分類系 → 数値系という既存の並びに合わせただけで、意味は無い
   (JSON は `sort_keys=True` のためキー順に依存しない)。

5. **リランカーの `source_note` に測定条件 (`--ctx-size 2048` / `--n-gpu-layers 99`) を明記した。**
   §7 リスク1 (「ctx-size を上げると 0.28 GiB が過小評価になる」) の警告を、
   T3 のスクリプトヘッダだけでなくカタログ側にも残すため。

### T2

6. **E14 の許容差は「合計 abs=0.02 / リランカー・埋め込み abs=0.005 / 生成 abs=0.02」とした。**
   §4 T2 は成分側の許容差を書いていない。埋め込み・リランカーは実測増分をそのまま
   `weights_gib` に入れているので厳密一致するはずであり、緩めると変異検証1・2
   (0.28 → 0.8 / 0.57 → 0.7) を捕まえられなくなるため合計より狭くした。

7. **E14 は「実測 4 段の増分の和 = 全体の増分」も併せて主張する。**
   定数 4 つを写し間違えたときに、テストが「見積りの誤り」ではなく
   「実測表の写し間違い」として落ちるようにするため。

### T5

8. **要件書の構成表「リランカー」列も書き換えた** (構成1 `BGE/Ruri Reranker` → `BGE Reranker v2-m3`、
   構成3 `Reranker` → `BGE Reranker v2-m3`)。§4 T5 は合計目安の数値しか挙げていないが、
   同じ表の中に「Ruri も候補」と読める記載を残すと、直下の「採用: BGE Reranker v2-m3」(L115) と
   矛盾する。制約「要件書の古い数値をそのまま残さない」の趣旨に合わせて、古い候補表記も消した。

9. **要件書 L117 から Jina Reranker v3 への差し替え検討を「不要」と書き切った。**
   元の文は「レイテンシが問題になる場合は Jina (約 100〜200ms) を検討」だったが、実測 140ms は
   その Jina の想定レンジ内であり、差し替え理由が消滅している。「500〜800ms」を消すだけだと
   検討条件が宙に浮くため、結論まで書いた。**Jina を評価対象から永久に外す意味ではない**
   (Phase 2 の比較ハーネスで再検討する余地は残る)。

10. **`docs/phase0-vram-measurements.md` L107 の「未達だった」を「満たせていなかった」に書き換えた。**
    §4 T5 は「余力 3.47 → 3.46」しか指示していないが、受け入れ基準が「未達」「未実施」が
    残っていないことを要求している。当該行は解決前の経緯を述べた文なので、事実を変えずに
    語のみ差し替えた (直後の「**いずれも解決した。**」は維持)。現在 0 件。

11. **`docs/next-pr-candidates.md` の HIGH 行の severity を `~~HIGH~~` に打ち消した。**
    同じ表の解決済み行 (`~~MEDIUM~~` / `~~INFO~~`) が既に取っている表記に合わせただけ。
    行の削除はしない (再スキャン時の突き合わせに残す、という同ファイル末尾の注記の方針に従う)。

12. **README の「Phase 0 は未了です」注記は「完了。ただしセットアップは各マシンで手動」に差し替えた。**
    単に「完了」とだけ書くと、新しいマシンで clone した人が `ollama pull` 不要と誤読する。
    併せて `doctor` 節の「Phase 0 が未了でも `doctor` は実行できます」も
    「Ollama の導入・`ollama pull` が済んでいないマシンでも」に言い換えた
    (Phase 0 が完了した以上、この条件節は Phase 番号では表せない)。

13. **README の新設節「リランカーは別ランタイム (llama.cpp llama-server)」は
    「2 つの API 経路」節の後・「実行マニフェスト」節の前に置いた。**
    ランタイム/接続経路の話が連続し、その後に記録・テストの話が来る既存の並びに従っただけ。
