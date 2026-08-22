# Phase 1: 推論クライアント層 (L2) 仕様書

> 出典: `docs/localllmrequirements.md` の Phase 1。スコープは L2 のみ。
> 確定済み前提: Phase 0 (Ollama 導入・モデル取得) はユーザーの手動作業 / Q7 ハイブリッド切替は「入れる」。

## 1. ゴール

設定ファイルだけで推論先モデル・エンドポイント・VRAM プロファイルを切り替えられる、Ollama 非依存でテスト可能な OpenAI 互換推論クライアント層 (L2) を実装する。

## 2. 現状認識

### リポジトリ内の関連箇所

| パス | 内容 |
|---|---|
| `main.py:L1-L12` | logger の hello のみ。`logging.basicConfig` は `__main__` ガード内 |
| `tests/test_main.py:L1-L11` | 唯一の既存テスト。`caplog` で INFO ログを検証 |
| `pyproject.toml:L18-L20` | `testpaths=["tests"]`, `pythonpath=["."]`、`[build-system]` **なし**（非パッケージプロジェクト） |
| `pyproject.toml:L26-L38` | ruff select に `T20`（`print()` 禁止が機械強制） |
| `pyproject.toml:L40-L44` | mypy `strict=true` + `disallow_any_explicit=true` |
| `Makefile:L37-L42` | `make ci` = lock-check + lint + fmt-check + type + test |
| `.github/workflows/python-ci.yaml:L26-L31` | `uv sync --locked` 後 `make ci`。**GPU なし・ネットワーク制限あり前提** |
| `.gitignore:L168-L171` | `outputs/` は ignore 済み → 実行マニフェストの出力先に使える |
| `docs/localllmrequirements.md:L60-L83` | VRAM 見積り表・構成1/2/3 表（カタログの唯一の出典） |
| `.claude/decisions.yaml` | **未作成**。`cp ~/.claude/templates/decisions.yaml .claude/decisions.yaml` から起こす |

### 既存の慣習で守るべきもの（3個）

1. **ログ**: モジュール先頭で `logger = logging.getLogger(__name__)`。`basicConfig` はエントリポイントの `__main__` ガード内のみ。`print()` は ruff T20 で機械的に禁止（CLI の人間向け出力も `logging` か `sys.stdout.write` で行う）。
2. **テスト**: `tests/test_{モジュール名}.py`。テスト関数にも型注釈を付ける（`-> None`、fixture は `caplog: pytest.LogCaptureFixture` の形）。
3. **検証**: `make ci` が単一の真実。ローカル / Stop hook / GitHub Actions が同じコマンドを呼ぶ。ここを迂回する検証手段を足さない。

### 影響範囲

新規パッケージの追加のみ。既存の `main.py` / `tests/test_main.py` には**触らない**。`pyproject.toml` は `dependencies` への追加のみ（tool 設定は変更しない）。

## 3. 前提・制約

### ハード制約（絶対に変えない）

- `main.py` と `tests/test_main.py` の既存挙動を変更しない。`test_main_logs_greeting` は通り続けること。
- `pyproject.toml` の `[tool.mypy]` / `[tool.ruff.lint].select` を緩めない。`# type: ignore` / `# noqa` を理由なく追加しない。
- **テストは Ollama 未起動・ネットワーク不通の環境で全件パスすること**。実 HTTP を発行するテストは書かない（書く場合は `@pytest.mark.live` を付け既定でスキップ）。
- Ollama の導入・`ollama pull`・サービス起動をコードやテストから実行しない（Phase 0 はユーザーの手動作業）。
- 推論ランタイム固有の型（`httpx.Response`、生の JSON dict 等）を L2 の公開 API に露出させない。L3 は `llmkit` の公開シンボルのみを import する。
- api_key・トークンを設定ファイル本体・ログ・実行マニフェスト・エラーメッセージに平文で出さない。
- `Makefile` / `.github/workflows/python-ci.yaml` の検証コマンドを変更しない。
- Phase 2（比較ハーネス）/ Phase 3（RAG・埋め込み・リランカーの実処理）のコードを書かない。埋め込み・リランカーは **VRAM プロファイル上の定義（静的メタデータ）としてのみ**扱い、推論呼び出しは実装しない。

### ソフト制約（理由があれば変えてよい）

- パッケージ名は `llmkit`（フラットレイアウト、リポジトリ直下）。
- 設定ファイル形式は TOML（stdlib `tomllib`。YAML 依存と `types-PyYAML` を持ち込まない）。
- 同期実装のみ。async は Phase 2 で並列比較が必要になった時点で追加。
- CLI は stdlib `argparse`（`typer` / `click` を足さない）。
- 追加依存は `httpx` と `pydantic` の2つに留める。

## 4. タスク分解

### T1: パッケージ骨格・設定層・例外階層

- **何をするか**
  - `llmkit/` パッケージを作成し、`pyproject.toml` の `dependencies` に `httpx>=0.28` / `pydantic>=2.9` を追加（`uv add httpx pydantic` → `uv.lock` 更新）。dev グループへの追加は**なし**（`httpx.MockTransport` は httpx 本体に含まれる）。
  - `llmkit/errors.py`: 例外階層と「対処方法つきメッセージ」を定義。
    - `LlmkitError`（基底、`remediation: str` を持つ）
    - `ConfigError` / `VramBudgetExceededError` / `RuntimeUnavailableError` / `ModelNotFoundError` / `OutOfMemoryError` / `ContextLengthError` / `UpstreamError`
  - `llmkit/config.py`: pydantic モデルで設定スキーマを定義し、TOML から読む。api_key は `api_key_env`（環境変数**名**）経由で解決し、値は `pydantic.SecretStr` で保持。
  - `configs/default.toml`（ローカル Ollama）と `configs/external_openai.toml`（外部 API 例）を作成。
- **設定スキーマ（この形状で実装する）**

```toml
[runtime]
kind = "ollama"                       # "ollama" | "openai_compatible"
base_url = "http://localhost:11434/v1"
api_key_env = "LLMKIT_API_KEY"        # 値そのものは書かない。未設定なら空文字で送る
timeout_s = 120.0
is_local = true                       # false のとき VRAM ガードを無効化

[generation]
model = "qwen3-14b"
context_tokens = 16384
temperature = 0.7
top_p = 0.9
max_output_tokens = 1024
seed = 0

[vram]
budget_gib = 16.0
runtime_overhead_gib = 0.8
active_profile = "rag_default"

[profiles.rag_default]                # 要件書 構成1
generation = "qwen3-14b"
embedding = "ruri-v3-310m"
reranker = "ruri-reranker"

[profiles.long_context]               # 要件書 構成2
generation = "gpt-oss-20b"

[profiles.lightweight]
generation = "qwen3-8b"

[profiles.oversized]                  # 要件書 構成3（停止することの実証用に定義を残す）
generation = "gpt-oss-20b"
embedding = "ruri-v3-310m"
reranker = "ruri-reranker"
```

- **触るファイル**: `pyproject.toml`（`dependencies` のみ）, `uv.lock`, `llmkit/__init__.py`, `llmkit/errors.py`, `llmkit/config.py`, `configs/default.toml`, `configs/external_openai.toml`, `tests/test_config.py`
- **受け入れ基準**
  - `load_config(Path("configs/default.toml"))` が `AppConfig` を返し、`configs/external_openai.toml` も同じ関数で読める。
  - 未知キー・型不一致・`base_url` が URL でない・`context_tokens <= 0` のいずれでも `ConfigError` が送出され、メッセージに**該当キー名**が含まれる（スタックトレースのみで終わらない）。
  - `api_key_env` が指す環境変数が未設定でも `is_local=true` なら読み込みが成功する。`is_local=false` かつ未設定なら `ConfigError` になり、メッセージに環境変数名は含むが値は含まない。
  - `repr(config)` / `str(config)` に api_key の平文が現れない（`SecretStr` の性質をテストで固定する）。
  - `uv run mypy .` が `llmkit/` を含めてエラー 0。`Any` を1箇所も明示的に書いていない。
- **想定所要**: M
- **実装時に決めたこと (仕様に書かれていなかった判断)**
  - スキーマは `pydantic.BaseModel` ではなく **pydantic dataclass** (`pydantic.dataclasses.dataclass` + `TypeAdapter`) で定義した。理由: `BaseModel` を継承した時点で mypy が `[explicit-any]` を出し、`disallow_any_explicit = true` を緩めない限り回避できない (ハード制約: pyproject の mypy 設定は変更しない)。リスク3 の「ignore を撒く形になりそうなら型設計が間違っているサイン」に従い、ignore ゼロで済む dataclass 側を採る。
  - pydantic の `strict=True` は使わない (lax モード)。理由: dataclass に `strict=True` を付けると dict 入力自体が `dataclass_exact_type` で拒否され TOML から構築できない。型不一致 (`timeout_s = "fast"` 等) は lax モードでも検出されるため受け入れ基準は満たす。
  - 全設定 dataclass を `frozen=True` にした。理由: 設定は起動時に確定し以後変わらない値であり、実行マニフェスト (T4) の再現性を型で担保するため。
  - `vram.active_profile` が `[profiles.*]` に存在しない場合は `ConfigError` にした。理由: 「未知キー」ではなく参照切れであり、放置すると T2 のプロファイル解決で分かりにくく失敗するため、設定層で該当キー名つきで落とす。
  - 設定ファイル本体に `api_key` / `runtime.api_key` を書いた場合は `ConfigError` にした (値はメッセージに出さない)。理由: D-05 の「値を書ける形にすると事故が起きる」を入口で機械的に塞ぐため。`api_key` は `AppConfig` のフィールドなので `extra = "forbid"` だけでは塞げない。
  - 値域制約を追加した: `timeout_s > 0` / `temperature` は 0.0〜2.0 / `top_p` は (0.0, 1.0] / `max_output_tokens > 0` / `budget_gib > 0` / `runtime_overhead_gib >= 0`。理由: 仕様が明示したのは `context_tokens > 0` のみだが、他も同種の「物理的にあり得ない値」であり、後段 (VRAM 見積り・リクエスト組み立て) で無意味な結果になるため設定層で落とす。
  - `base_url` は末尾のスラッシュを除去して保持する。理由: T3 でパス結合するときに `//` が混入するのを防ぐため。
  - `configs/external_openai.toml` の `generation.model` はカタログ登録済みの `qwen3-14b` のままにした (base_url のみ外部ホストに変更)。理由: `model` はカタログの `model_id` を指す必要があり、カタログ外の名前 (`gpt-4o-mini` 等) を書くと T3 の `served_name` 解決が失敗する。セルフホストの OpenAI 互換エンドポイントへ向ける例として整合する。
  - `LlmkitError.__str__` は `"{message} / 対処: {remediation}"` を返す。理由: 「対処方法つきメッセージ」を CLI (T4) がそのまま人間に見せられるようにするため。
  - `AppConfig.active_profile()` / `ProfileConfig.model_ids()` の 2 ヘルパを追加した。理由: T2 のプロファイル解決と T4 のマニフェスト生成が同じ「宣言順のモデル ID 列」を必要とするため、設定層に 1 箇所だけ置く。

---

### T2: モデルカタログと VRAM プロファイル見積り

- **何をするか**
  - `llmkit/catalog.py`: `ModelSpec`（frozen dataclass または pydantic frozen model）の静的テーブル。フィールド = `model_id`, `served_name`（ランタイムに送る実名。例 `qwen3:14b-q4_K_M`）, `role`（`generation`/`embedding`/`reranker`）, `quantization`, `weights_gib`, `kv_gib_per_1k_tokens`, `max_context_tokens`, `source_note`（要件書の該当行）。
  - `llmkit/vram.py`: プロファイル解決 + 見積り + 予算判定。
  - **見積り式（この式で実装する）**

    ```
    estimate = Σ(model.weights_gib for model in profile.models)
             + generation.kv_gib_per_1k_tokens * (context_tokens / 1024)
             + vram.runtime_overhead_gib
    ```
  - **カタログ初期値（要件書 L60-L83 由来。すべて `(仮)` = Phase 0 実測で更新前提。`source_note` に明記する）**

    | model_id | role | quant | weights_gib | kv_gib_per_1k | 根拠 |
    |---|---|---|---|---|---|
    | `qwen3-14b` | generation | Q4_K | 9.0 | 0.10 | 12〜14B ≈ 8〜9GB |
    | `gpt-oss-20b` | generation | MXFP4 | 11.5 | 0.004 | 20B MXFP4 ≈ 12〜13GB、128k が VRAM 内完結 |
    | `qwen3-8b` | generation | Q4_K | 4.5 | 0.08 | 7〜8B ≈ 4〜5GB |
    | `ruri-v3-310m` | embedding | fp16 | 0.7 | 0.0 | 310M × 2byte |
    | `ruri-reranker` | reranker | fp16 | 0.8 | 0.0 | (仮) |

  - `check_budget(profile, config)` は超過時に `VramBudgetExceededError` を送出。例外には内訳（モデル別重み・KV・オーバーヘッド・合計・予算・超過量）を持たせる。
  - `runtime.is_local == false` のときは予算判定をスキップする（外部 API はローカル VRAM を使わないため）。
- **触るファイル**: `llmkit/catalog.py`, `llmkit/vram.py`, `tests/test_catalog.py`, `tests/test_vram.py`
- **受け入れ基準**
  - `estimate_profile("rag_default", ctx=16384)` が **12.0〜13.5 GiB** に収まる（要件書 構成1「約12〜13GB」の再現）。
  - `estimate_profile("long_context", ctx=131072)` が **12.0〜13.5 GiB** に収まり、予算 16.0 以下と判定される（構成2）。
  - `estimate_profile("oversized", ctx=131072)` の値が構成1・構成2 のいずれよりも**大きい**。
  - **合成プロファイル**（テスト専用に重み 20 GiB のダミー `ModelSpec` を構成）で `check_budget` が `VramBudgetExceededError` を送出し、例外メッセージに「合計値・予算値・超過量・プロファイル名」の4つがすべて含まれる。← 予算ガード本体の検証はカタログ値のキャリブレーションに依存させない。
  - **閾値配線テスト**: `budget_gib` を `13.0` に下げると `rag_default` が超過側に転び、`16.0` に戻すと通る。同一プロファイル・同一カタログで判定だけが反転すること。
  - **KV 配線テスト**: `context_tokens` を 4096 → 131072 に増やすと `qwen3-14b` の見積りが単調増加する（増分が 0 でない）。
  - **overhead 配線テスト**: `runtime_overhead_gib` を 0.8 → 2.0 に変えると見積りが正確に 1.2 増える。
  - カタログの `weights_gib` を1つ書き換えると、上記の帯テストが落ちる（＝ゴールデン値がロックされている）。
  - `is_local=false` の設定では `oversized` プロファイルでも例外が出ない。
- **想定所要**: M
- **⚠ 仕様との食い違い (要確認)**
  - 受け入れ基準「`budget_gib` を **13.0** に下げると `rag_default` が超過側に転ぶ」は、指定されたカタログ値と見積り式では**成立しない**。`rag_default` @ ctx=16384 の見積りは `9.0 + 0.7 + 0.8`(重み) `+ 0.10 * 16`(KV) `+ 0.8`(overhead) = **12.90 GiB** であり、12.90 <= 13.0 のため予算 13.0 では収まってしまう。
  - §7 リスク2「カタログの数値を勝手に大きくしない」に従い、カタログ値は変更していない。代わりに `test_budget_threshold_changes_verdict` の超過側の閾値を **12.5** とした (通る側は仕様どおり 16.0)。「同一プロファイル・同一カタログで budget_gib だけを変えると判定が反転する」という基準の意図は満たしている。
  - Phase 0 の実測でカタログ値が更新され `rag_default` が 13.0 を超えるようになったら、仕様書の 13.0 に戻すこと。
- **実装時に決めたこと (仕様に書かれていなかった判断)**
  - 実シグネチャは `estimate_profile(config, profile_name=None, *, context_tokens=None)` とした。理由: 仕様書の `estimate_profile("rag_default", ctx=16384)` は略記であり、見積りには overhead / budget / profiles 定義 (= `AppConfig`) が必須のため。`profile_name` 省略時は `vram.active_profile`、`context_tokens` 省略時は `generation.context_tokens` を使う (= 設定からの既定値配線)。
  - `estimate_resolved_profile(profile, config, *, context_tokens=None)` を追加した。理由: 予算ガードの検証をテスト専用の合成 `ResolvedProfile` (重み 20 GiB) で行う (受け入れ基準) には、カタログ解決を経由しない入口が必要なため。
  - `resolve_profile` は `ResolvedProfile(name, models: tuple[ModelSpec, ...])` を返し、`generation` は `role == "generation"` の先頭モデルを指す property とした。理由: KV キャッシュの見積りは生成モデルの値だけを使う、という式の前提を型で表すため。
  - `check_budget` は超過しない場合も `VramEstimate` を返す。理由: T4 の起動シーケンスが「INFO ログに想定使用量を出す」「マニフェストに内訳を書く」ために内訳を必要とするため、判定と内訳取得を 2 回計算に分けない。
  - `runtime.is_local = false` のときも見積り自体は計算して返し、**判定だけ**をスキップする。理由: 外部 API でも実行マニフェストに想定値を残せるようにするため。
  - 予算判定の境界は `total_gib <= budget_gib` を「収まる」とした (等号は収まる側)。理由: 予算値は実機 VRAM そのものではなく設定可能な上限であり、ちょうど一致する構成を機械的に弾く理由がないため。
  - カタログに無い `model_id` を参照した場合は `ConfigError` (`ModelNotFoundError` ではない)。理由: `ModelNotFoundError` は §4 T3 のエラー翻訳表でランタイム側 404 (`ollama pull <served_name>` を促す) に割り当て済みであり、原因も対処も異なるため混ぜない。
  - `MODEL_CATALOG` は `MappingProxyType` で公開し実行時に書き換えられないようにした。理由: 見積りの再現性 (D-01) の前提を型・実行時の両方で担保するため。
  - `served_name` と `max_context_tokens` のうち要件書に根拠が無いものは仮置きとし、`source_note` に `(仮)` と「Phase 0 の実測で更新する」を明記した。対象: `qwen3-8b` の 32768、`ruri-v3-310m` の `hf.co/cl-nagoya/ruri-v3-310m` / 8192、`ruri-reranker` の `hf.co/cl-nagoya/ruri-reranker-large` / 512。理由: 推測値を根拠つきの値と区別できないと Phase 0 で更新漏れが起きるため。
  - `VramBudgetExceededError` の内訳はプリミティブ (プロファイル名・モデル別重みの Mapping・KV・overhead・合計・予算・超過量) をキーワード引数で受ける形にした。理由: `errors.py` が `vram.py` の `VramEstimate` を import すると循環 import になるため。

---

### T3: 推論クライアント抽象と OpenAI 互換実装（L）

- **何をするか**
  - `llmkit/client.py`:
    - **公開データ型**: `ChatMessage`（`role`/`content`）, `ChatResult`（`text`, `model`, `finish_reason`, `usage`（prompt/completion/total tokens）, `latency_s`, `tokens_per_second`）。Phase 2 の比較ハーネスがそのまま指標を取れる形にする。
    - **抽象**: `ChatClient` を `typing.Protocol` で定義（`chat(messages: Sequence[ChatMessage]) -> ChatResult`）。L3 はこの Protocol にのみ依存する。
    - **実装**: `OpenAICompatibleClient`。コンストラクタで `httpx.Client` を**注入可能**にする（既定は設定から生成）。これがテスト戦略の要。
  - **リクエストボディ組み立て**（パラメータ配線の中核。ここが仕様の肝）:
    - `model` = `ModelSpec.served_name`（`model_id` ではない）
    - `messages`, `temperature`, `top_p`, `max_tokens`(=`max_output_tokens`), `seed` をトップレベルに載せる
    - `options.num_ctx` = `context_tokens` を `extra_body` 相当としてボディに含める（Ollama ネイティブ拡張。互換エンドポイントが無視しても害はない）
    - `api_key` が空でなければ `Authorization: Bearer ...` ヘッダを付ける
  - **レスポンス解析**: pydantic の strict モデルで受ける。`choices[0].message.content` 欠損・`choices` 空は `UpstreamError`。`json.loads` の戻りを `Any` として扱わない（`object` で受けて pydantic に渡す）。
  - **エラー翻訳表**（この対応で実装する）:

    | 検知条件 | 送出例外 | メッセージに必ず含める要素 |
    |---|---|---|
    | `httpx.ConnectError` / `ConnectTimeout` | `RuntimeUnavailableError` | base_url、`ollama serve` の実行を促す文言、「ランタイムが起動していない可能性」 |
    | HTTP 404、またはボディに `model not found` | `ModelNotFoundError` | 該当 `served_name`、`ollama pull <served_name>` |
    | ボディに `out of memory` / `CUDA` OOM 相当 | `OutOfMemoryError` | プロファイル名、現在の `context_tokens`、より小さいプロファイルへの切替提案 |
    | HTTP 400 かつ context 長超過相当 | `ContextLengthError` | `context_tokens` と `max_context_tokens` |
    | その他 4xx/5xx | `UpstreamError` | ステータスコード（**レスポンスボディ全文は載せない**） |
- **触るファイル**: `llmkit/client.py`, `tests/test_client.py`, `tests/test_client_errors.py`, `tests/test_param_wiring.py`
- **受け入れ基準**
  - すべてのテストが `httpx.MockTransport` で完結し、実ネットワークに一切出ない。テスト実行時に `LLMKIT_*` 環境変数やローカルの Ollama 有無で結果が変わらない。
  - **★ パラメータ配線テスト（`tests/test_param_wiring.py`）**: `model` / `context_tokens` / `temperature` / `top_p` / `max_output_tokens` / `seed` の**6項目すべて**について、設定値を変えると MockTransport が捕捉したリクエストボディの対応フィールドが変わることを parametrize で検証する。1項目でも配線漏れがあれば落ちること。
  - **受け入れ条件1の直接検証**: 呼び出しコードを1行も変えず、config の `generation.model` を `qwen3-14b` → `qwen3-8b` に変えるだけで、送出される `model` フィールドが対応する `served_name` に変わる。
  - **ローカル/外部 API 切替の検証**: `configs/default.toml` と `configs/external_openai.toml` で、リクエスト先 URL と `Authorization` ヘッダの有無が変わる。api_key を設定した場合、ヘッダ値が捕捉できること。
  - エラー翻訳表の各行に1テスト以上が対応し、**送出される例外型**と**メッセージ内の必須要素**の両方をアサートする。
  - どの例外のメッセージにも api_key の値が含まれない（api_key を設定した状態でエラーを起こし、`str(exc)` に平文が無いことを検証）。
  - `ChatResult.tokens_per_second` が `usage.completion_tokens / latency_s` で算出され、`latency_s == 0` でも ZeroDivisionError にならない。
- **想定所要**: L
- **⚠ 仕様との食い違い (実装側に合わせた点)**
  - 「レスポンス解析: pydantic の **strict モデル**で受ける」は `pydantic.BaseModel` / `strict=True` では実現できない。T1 と同じ理由 (`BaseModel` 継承行が mypy `[explicit-any]` になる / dataclass の `strict=True` は dict 入力を拒否する) により、**pydantic dataclass + `TypeAdapter` (lax モード)** で実装した。必須フィールド欠損・型不一致はいずれも検出されるため D-07 の意図 (欠損を黙って通さない) は満たしている。
  - `ChatResult.usage` の型として `TokenUsage` (frozen dataclass) を新設した。仕様は「usage（prompt/completion/total tokens）」としか書いておらず、生 dict を公開 API に出さないためには名前付きの型が必要なため。
  - `tokens_per_second` は保持フィールドではなく `ChatResult` の **property** (算出値) にした。`completion_tokens / latency_s` が常に整合することを型で保証するため。
- **実装時に決めたこと (仕様に書かれていなかった判断)**
  - 応答の必須フィールドは `model` / `choices[].message.content` / `choices[].finish_reason` / `usage.{prompt,completion,total}_tokens` とし、いずれか欠損で `UpstreamError`。理由: D-07 の rationale「速度指標が静かに欠測すると Phase 2 の比較が壊れる」に従い、比較ハーネスが使う値をすべて必須側に置く。
  - 応答モデルは `extra="ignore"` (設定層の `extra="forbid"` と逆)。理由: 上流は `id` / `created` / `system_fingerprint` 等を自由に増やすため、未知キーで落とすと正常な応答が使えなくなる。設定ファイル (人間が書く / 綴り誤りを検出したい) とは要件が異なる。
  - `tokens_per_second` は `latency_s <= 0.0` のとき `0.0` を返す。理由: 計測不能を 0 で表し、ZeroDivisionError を上位に伝播させない (受け入れ基準)。
  - `latency_s` は HTTP の送信直前〜応答受信直後 (`time.perf_counter()`) のみを測り、ボディ組み立てと JSON パースを含めない。理由: Phase 2 が比較したいのは推論ランタイムの応答時間であり、クライアント側の処理時間を混ぜると比較対象がぶれるため。
  - `ChatMessage.role` は `Literal["system", "user", "assistant"]`。理由: 任意文字列を許すとランタイム側で 400 になる綴り誤りを型で防げないため。`tool` ロールは Phase 1 の要件に無いので入れない。
  - `httpx.ConnectError` / `ConnectTimeout` 以外の `httpx.HTTPError` (ReadTimeout・プロトコル違反等) は `UpstreamError` に翻訳し、メッセージには**例外クラス名のみ**を載せる。理由: 翻訳しないと httpx 固有の例外が L3 に漏れ、「ランタイム固有の型を公開 API に露出させない」制約を破るため。
  - エラー翻訳の評価順序は 404 / `model not found` → OOM マーカー → 400 かつ context マーカー → その他 (仕様の表と同順)。本文は**小文字化した部分一致の判定にのみ**使い、本文そのものは例外メッセージに一切載せない。
  - 応答パース失敗時のメッセージには pydantic の `loc` と `msg` だけを載せ、`input` (= 応答本文の断片) は載せない。理由: 「レスポンスボディ全文を載せない」の趣旨を部分文字列にも適用するため。
  - `stream` はリクエストボディに含めない。理由: 仕様が列挙したフィールドのみを送る (OpenAI 互換の既定は非ストリーミング)。
  - 空の `messages` に対するクライアント側ガードは設けず、上流の 400 → `UpstreamError` に委ねる。理由: 仕様に規定がなく、検証を二重に持つと上流の仕様変更時に食い違うため。
  - `httpx.Client` の所有権: **注入されたクライアントは `close()` しない**。自前生成した場合のみ閉じる。`OpenAICompatibleClient` はコンテキストマネージャとして使える。理由: テスト・T4 の CLI が 1 つの `httpx.Client` を使い回せるようにするため。
  - 公開プロパティ `served_name` / `endpoint_url` を追加した。理由: T4 の `doctor` とマニフェストが「実際に送るモデル実名・接続先」を表示する必要があり、内部属性を触らせないため。
  - `llmkit/__init__.py` に `ChatClient` / `ChatMessage` / `ChatResult` / `ChatRole` / `TokenUsage` / `OpenAICompatibleClient` を再エクスポートした (T3 の「触るファイル」一覧外)。理由: ハード制約「L3 は `llmkit` の公開シンボルのみを import する」を成立させるため。
  - テスト側は各ファイルに autouse fixture を置き `LLMKIT_*` 環境変数を除去する。理由: 受け入れ基準「ローカル環境変数で結果が変わらない」を満たすため。共有 fixture の `tests/conftest.py` は T5 の担当なので、T5 でこの fixture に集約してよい。

---

### T4: 起動シーケンスと実行マニフェスト（再現可能ログ）

- **何をするか**
  - `llmkit/manifest.py`: `RunManifest` を生成し JSON で永続化。含めるキー（要件書「ログ: 実行日時・モデル・量子化・全パラメータ」の充足）:
    - `schema_version`, `run_id`, `started_at_utc`（ISO8601 / UTC）
    - `profile`: 名前・構成モデル一覧（`model_id`, `served_name`, `role`, `quantization`）
    - `vram`: 内訳（モデル別重み・KV・overhead・合計・予算・判定）
    - `generation`: `context_tokens` / `temperature` / `top_p` / `max_output_tokens` / `seed` の**全パラメータ**
    - `runtime`: `kind` / `base_url` / `is_local` / `timeout_s` / `api_key_env`（**名前のみ**）
    - `config_path`, `config_sha256`（設定ファイル内容のハッシュ = 再現性の担保）
    - `python_version`, `platform`
  - 出力先は `outputs/runs/{started_at}-{run_id}.json`（`outputs/` は gitignore 済み）。出力先ディレクトリは設定可能。
  - `llmkit/bootstrap.py`: 起動シーケンスを1関数に集約。
    1. 設定ロード → 2. プロファイル解決 → 3. VRAM 見積り → 4. **INFO ログに想定使用量を出力** → 5. 予算超過なら **WARNING ログ + `VramBudgetExceededError` で停止**（クライアントは生成しない、HTTP は一切発行しない）→ 6. マニフェスト書き出し → 7. `ChatClient` を返す
  - `llmkit/cli.py`: `argparse` で2サブコマンド。`print()` は使わない（ruff T20）。
    - `doctor`: 設定ロード + VRAM 見積り + マニフェスト出力 + 疎通確認。失敗時は対処方法つきメッセージを出し**終了コード 1**。
    - `chat "..."`: 一問一答。同じくエラー時は終了コード 1。
- **触るファイル**: `llmkit/manifest.py`, `llmkit/bootstrap.py`, `llmkit/cli.py`, `tests/test_manifest.py`, `tests/test_bootstrap.py`
- **受け入れ基準**
  - **受け入れ条件2の直接検証**: `bootstrap()` 実行時に INFO ログへ想定 VRAM 使用量が出力される。`caplog` で「プロファイル名」「合計 GiB 値」「予算値」の3つが含まれることをアサート。
  - **受け入れ条件3の直接検証**: 予算超過プロファイルで `bootstrap()` を呼ぶと (a) WARNING 以上のログが出て、(b) `VramBudgetExceededError` が送出され、(c) `httpx` の呼び出しが**0回**である（MockTransport のリクエスト数が 0）。
  - **マニフェスト網羅テスト**: 上記キー一覧が漏れなく存在する。キー名の集合を期待値としてアサートし、キーを1つ消すと落ちること。
  - **マニフェスト伏字テスト**: `api_key` の平文がマニフェスト JSON 文字列のどこにも現れない。`api_key_env`（環境変数名）は現れる。
  - **再現性テスト**: 同一設定で2回 `bootstrap()` すると `config_sha256` が一致し、設定を1文字変えると変わる。
  - **マニフェスト配線テスト**: `temperature` を変えるとマニフェストの該当値も変わる（マニフェストが実際に使われた設定を写していることの保証。ハードコードされた雛形でないこと）。
  - CLI の `doctor` を Ollama 未起動相当（MockTransport で ConnectError）で実行すると、終了コードが 1、出力に `ollama serve` と `base_url` が含まれる。
  - `uv run ruff check .` で T20 違反 0。
- **想定所要**: M
- **⚠ 仕様との食い違い (実装側に合わせた点)**
  - 「7. `ChatClient` を返す」に対し、実際は `BootstrapResult` (`config` / `profile` / `estimate` / `manifest` / `manifest_path` / `client` / `endpoint_url` / `served_name`) を返す。仕様の戻り値本体は `.client`。理由: CLI (`doctor`) とマニフェスト表示が見積り・出力先・接続先を必要とし、`ChatClient` だけを返すと呼び出し側で設定ロードと見積りを再実行することになり、再現性 (同じ設定を 2 回読む) が壊れるため。`endpoint_url` / `served_name` を別フィールドにしたのは、`client` の型を Protocol (`ChatClient`) のまま保つため。
  - 予算超過の実証テストは `oversized` プロファイル **かつ `budget_gib` を 12.0 に下げた一時設定**で行う。理由: T2 の「⚠ 仕様との食い違い」と同根で、現在のカタログ値では構成3 (`oversized`) は 13.86 GiB にとどまり 16.0 を超えない。判定に効いているのは「見積り > 予算」という関係そのものであり、受け入れ条件3 の意図 (警告して停止し HTTP を出さない) は満たしている。Phase 0 の実測でカタログ値が上がったら、既定の 16.0 のまま `oversized` を指定するテストに戻すこと。
- **実装時に決めたこと (仕様に書かれていなかった判断)**
  - マニフェストの `generation` セクションに `model` を含めた (仕様の列挙は `context_tokens` / `temperature` / `top_p` / `max_output_tokens` / `seed` のみ)。理由: `generation.model` と `profiles.*.generation` は独立に設定でき、実際にリクエストへ載るのは前者。前者を記録しないと「どのモデルで走ったか」がマニフェストから再現できない。
  - マニフェストの `vram` セクションに `within_budget` (判定結果) を含めた。仕様の「`vram`: 内訳 (…・判定)」の「判定」に対応させたもの。
  - `started_at_utc` は ISO8601 (`2026-08-22T11:13:13.783916Z`) のまま持ち、**ファイル名だけ** `%Y%m%dT%H%M%SZ` の詰め表記にする。理由: ISO8601 のコロンはファイル名に使えない環境がある。
  - `schema_version` は `"1"` (文字列)。理由: 将来 `"1.1"` のような枝番を付けられる余地を残すため。
  - マニフェスト JSON は `indent=2` / `sort_keys=True` / `ensure_ascii=False` で書く。理由: Phase 2 が 2 回の実行条件を `diff` で比較するため、キー順が実行ごとに揺れない形にする。
  - 予算超過で停止した場合、マニフェストは書き出さない (仕様の手順 5 → 6 の順序どおり)。理由: 起動していない実行の記録を残すと、`outputs/runs/` が「実際に走った条件の集合」でなくなる。
  - `bootstrap(..., write_manifest_file=False)` を追加した。理由: マニフェストの内容だけを検査したいテストと、将来のドライラン用。既定は `True` (仕様どおり必ず書く)。
  - `doctor` の疎通確認は、**設定どおりの生成パラメータで** `"ping"` を 1 回送る。軽量な専用パラメータ (`max_output_tokens=1` 等) に差し替えない。理由: `doctor` が答えるべきは「実際に使う設定で通るか」であり、専用パラメータで通しても本番の設定が通る保証にならない。
  - CLI の共通オプションは `--config` / `--profile` / `--output-dir` の 3 つ。`--output-dir` を設けたのは、テストが `tmp_path` に書いてリポジトリの `outputs/` を汚さないようにするため (ハード制約)。
  - CLI の正常出力は stdout、エラーは stderr に出し、終了コードは成功 0 / `LlmkitError` 捕捉 1 の 2 値のみ。理由: 例外の種別で終了コードを分けると CLI の契約が増え、Phase 1 の要件 (「終了コード 1」) を超えるため。
  - `logging.basicConfig` は `llmkit/cli.py` の `__main__` ガード内でのみ呼ぶ (既存 `main.py` と同じ)。`main()` をライブラリとして呼んだ場合はログ設定に触れない。
  - `compute_config_sha256` / `write_manifest` の I/O 失敗は `ConfigError` に翻訳する。理由: 新しい例外型を増やさず、対処方法つきメッセージの形を揃えるため。

---

### T5: 受け入れ条件の機械検証・live マーカー・ドキュメント

- **何をするか**
  - `tests/conftest.py`:
    - `live` マーカーを登録し、`--run-live` オプションが無い限り**自動スキップ**する `pytest_collection_modifyitems` を実装。
    - 共有 fixture: `mock_transport`（リクエスト捕捉つき）, `tmp_config`（TOML を一時ファイルに書き出す）。
  - `pyproject.toml` の `[tool.pytest.ini_options]` に `markers = ["live: 実ランタイムに接続するテスト（既定スキップ）"]` を追加。
  - `tests/test_acceptance_phase1.py`: 要件書 Phase 1 の受け入れ条件5項目に**1対1で対応するテスト関数**を置き、docstring に要件書の該当行番号を書く。
  - `tests/test_layout.py`: `import llmkit` がリポジトリルートから追加設定なしで通ることを固定。
  - `.claude/decisions.yaml` を作成し、§6 の決定を登録（`~/.claude/templates/decisions.yaml` を雛形にする）。
  - `README.md` に Phase 1 の使い方（`uv run python -m llmkit.cli doctor --config configs/default.toml`、Phase 0 が未了でも `doctor` がエラー原因を教えること）を追記。
  - `make ci` を通す。
- **触るファイル**: `tests/conftest.py`, `tests/test_acceptance_phase1.py`, `tests/test_layout.py`, `pyproject.toml`（pytest markers のみ）, `.claude/decisions.yaml`, `README.md`
- **受け入れ基準**
  - `uv run pytest` が **Ollama 未起動・ネットワーク不通の状態で全件パス**（skip は live マーカーのみ）。
  - `make ci` が緑（lock-check / ruff check / ruff format --check / mypy / pytest すべて）。
  - `tests/test_acceptance_phase1.py` に5個のテスト関数があり、それぞれが要件書 L294-L298 の1条件に対応している。
  - 既存の `tests/test_main.py::test_main_logs_greeting` が変更なしで通る。
  - `.claude/decisions.yaml` が `check-decisions.sh` の検証を通る（全エントリの `guard_test` が実在する）。
- **想定所要**: M
- **実装時に決めたこと (仕様に書かれていなかった判断)**
  - `tests/conftest.py` に autouse fixture `_forbid_real_network` を追加した (`socket.socket.connect` / `connect_ex` / `socket.create_connection` を禁止し、`live` マーカー付きテストだけ除外する)。理由: §5 安全性観点「テストが外部ネットワークに出ない」を、個々のテストの書き方ではなく仕組みで担保するため。`httpx.MockTransport` はソケットを開かないので正しいテストは影響を受けない。
  - `tests/conftest.py` に `pytest_plugins = ["pytester"]` を追加した。理由: D-02 の guard_test を「方針の写し」ではなく**実物の `tests/conftest.py` を読ませた入れ子実行**で検証するため。`--run-live` あり/なしの両方を測る。
  - 共有 fixture の実体は `RecordingTransport` クラス (`requests` / `transport` / `client()` / `call_count`) と `mock_transport` / `mock_http_client` fixture。`requests` が空であること自体が「HTTP を 1 回も出していない」証拠になるため、受け入れ条件3 の検証にそのまま使える。
  - `tmp_config` は `write_config_variant(directory, edits, name=...)` を `tmp_path` に束ねたファクトリとして実装した。理由: `bootstrap()` は `Path` しか受け取らないため、設定ファイル経由でしか振れない値 (`budget_gib` / `active_profile` / `is_local`) をテストから掃引するには一時 TOML の書き出しが要る。置換対象が見つからなければその場で落とす (設定書式が変わったのにテストだけ通る事故を防ぐ)。
  - T3 実装者の申し送りに従い、`SUCCESS_PAYLOAD` と `LLMKIT_*` 環境変数除去 fixture を `tests/conftest.py` に集約し、既存テストは `from conftest import ...` で参照する形にした (アサーション内容は変更なし)。`conftest` からの import が pytest / mypy の双方で解決できることは実測で確認済み。集約が崩れていないことは `tests/test_testing_policy.py::test_no_test_module_defines_its_own_llmkit_env_fixture` が固定する。
  - `tests/test_acceptance_phase1.py` の関数名は `test_l294_...` 〜 `test_l298_...` とし、要件書の行番号を名前と docstring の両方に持たせた。L298 (「自動テストが存在しパスする」) のテストは、`make ci` をテスト内から実行すると再帰するため、代わりに (1) 条件 5 個とテスト 5 個の 1 対 1 対応、(2) 各 docstring の行参照、(3) 参照先が要件書のチェックリスト行のままであること、(4) `Makefile` の `ci` ターゲットが `test` を含むこと、の 4 点を検査する。
  - `.claude/decisions.yaml` に **D-08** を追加した (スキーマは pydantic dataclass で定義し `BaseModel` を継承しない)。理由: T1/T3 で実際に効いている判断が仕様書の散文にしか無く、後続の fixer が「一貫性のため」`BaseModel` に戻すと mypy が落ちるところまで行ってしまうため。guard_test は `tests/test_layout.py::test_schemas_use_pydantic_dataclasses_not_basemodel` (AST で基底クラスを検査)。
  - `tests/test_layout.py` は D-06 の guard に加え、レイアウトの前提 (`[build-system]` 非存在 / `pythonpath = ["."]`) と mypy 設定 (`strict` / `disallow_any_explicit`) が緩められていないことも固定する。理由: D-06 / D-08 の rationale はどちらも「その設定であること」に依存しており、設定が変わると決定の根拠が消えるため。
  - `tests/test_testing_policy.py` に、`httpx` を import するテストは必ず MockTransport 経由であることの静的検査を追加した。理由: `live` マーカーの仕組みは「実接続テストを書くときに live を付ける」という運用に依存しており、付け忘れを検出する層が別途要るため。
  - **(round 2 レビュー後に追記)** `.claude/decisions.yaml` に **D-09** を追加した (Phase 1 の「設定のみで切替」はカタログ登録済み model_id 間、および「ローカル/外部 API 切替」軸に限る。ランタイム固有の `served_name` (vLLM 等へ渡す実モデル名) の上書きは対象外)。理由: `resolve_model_spec` の passthrough は「ローカル/外部 API 切替」軸しか救っておらず、要件書 L144 の「ランタイムの差し替え (vLLM 等)」軸は実測で未達 (ローカル実行で vLLM の実モデル名を設定に書いても `ConfigError` になる)。恒久対応 (`[models.<id>]` 設定側上書き表) は Phase 2 の別タスクとし、今回のスコープでは行わない。guard_test は `tests/test_catalog.py::test_resolve_model_spec_raises_config_error_for_unregistered_model_when_local`。`llmkit/catalog.py` の `resolve_model_spec` docstring も「ランタイム差し替え軸は対象外」と明記するよう訂正した。

## 5. 評価軸（Check フェーズへ）

### 機能観点

要件書 L294-L298 の5条件を、`tests/test_acceptance_phase1.py` の5テストで1対1に測る。

1. モデル名変更のみで切替 → config の `generation.model` だけ変えて送出 `model` が変わることを MockTransport で確認
2. プロファイル指定＋想定使用量ログ → `caplog` に合計 GiB が出る
3. 16GB 超過で警告して停止 → WARNING ログ + 例外 + HTTP 発行 0 回
4. ランタイム未起動時の原因特定 → `RuntimeUnavailableError` のメッセージに `base_url` と `ollama serve`
5. 自動テストの存在 → `make ci` が緑

### 性能観点

Phase 1 では実推論を測らない。測るのは以下2点のみ。

- `uv run pytest` の実行時間が **10 秒未満**（実 HTTP・実 GPU に触れていないことの間接指標。超えていたら実接続が混入している疑い）
- VRAM 見積り関数が純関数で、I/O を行わない（GPU / `nvidia-smi` / ネットワークに触れない）

実測 t/s（要件書「30 t/s 以上」）は Phase 0 実測と Phase 2 ハーネスの担当。Phase 1 では `ChatResult.tokens_per_second` を**算出できる形にしておく**ことまでが範囲。

### 安全性観点

- api_key が config の repr / ログ / マニフェスト JSON / 例外メッセージのいずれにも平文で出ない（4箇所すべてにテスト）
- `UpstreamError` にレスポンスボディ全文を載せない（内部実装詳細の漏洩防止。CLAUDE.md セキュリティ原則）
- `configs/*.toml` に api_key の値が書かれていない（`git diff` 確認 + テストで「`api_key` というキーが TOML に存在しないこと」を固定）
- テストが外部ネットワークに出ない（CI の再現性 + 意図しない外部 API 課金の防止）

### テスト観点

- 新規: `tests/test_config.py` / `test_catalog.py` / `test_vram.py` / `test_client.py` / `test_client_errors.py` / `test_param_wiring.py` / `test_manifest.py` / `test_bootstrap.py` / `test_layout.py` / `test_acceptance_phase1.py` / `conftest.py`
- 既存: `tests/test_main.py` は**変更しない**
- カバレッジ: `llmkit/` に対して行カバレッジ 85% 以上を目安（エラー翻訳表の各分岐が通っていることが実質的な基準）

### ★ 有効性観点（パラメータ・閾値が実際に配線されているかの決定論的検証）

このフェーズは「設定で切り替わること」自体が成果物なので、配線テストが仕様の中心。以下を**受け入れ基準に含める**（§4 の各タスクに記載済み）。

| # | 掃引する値 | 変えたときに変わるべき出力 | テスト |
|---|---|---|---|
| E1 | `generation.model` | リクエストボディの `model` | `test_param_wiring.py::test_generation_params_reach_request_body` |
| E2 | `context_tokens` | ボディの `options.num_ctx` **かつ** VRAM 見積り | 同上 + `test_vram.py::test_context_tokens_change_estimate` |
| E3 | `temperature` / `top_p` / `max_output_tokens` / `seed` | ボディの対応フィールド | `test_param_wiring.py`（parametrize） |
| E4 | `vram.budget_gib` | `check_budget` の合否 | `test_vram.py::test_budget_threshold_changes_verdict` |
| E5 | `vram.runtime_overhead_gib` | 見積り合計（差分が厳密に一致） | `test_vram.py::test_overhead_changes_estimate` |
| E6 | `runtime.base_url` / `api_key` | リクエスト先 URL / `Authorization` ヘッダ | `test_client.py::test_local_and_remote_configs_differ` |
| E7 | カタログの `weights_gib` | 構成1/2/3 の見積り帯テストが落ちる | `test_vram.py::test_documented_profiles_match_requirements_table` |
| E8 | 任意の生成パラメータ | マニフェスト JSON の該当値 | `test_manifest.py::test_manifest_reflects_actual_config` |

E1〜E8 のいずれかが「値を変えても出力が変わらない」状態で通ってしまう実装は**不合格**。特に E2 の `num_ctx` と E8 は配線漏れが起きやすく、漏れると Phase 2 の比較実験が丸ごと無意味になる。

## 6. 意図的な決定（`.claude/decisions.yaml` に追記）

```yaml
- id: D-01
  rule: "VRAM 使用量の見積りは llmkit/catalog.py の静的テーブルで行い、nvidia-smi や実測値の参照を行わない"
  rationale: "Phase 0 (実測) が未実施であり、CI 環境には GPU が無い。かつ判定は『モデルをロードする前』に行う必要があるため、実測ベースでは原理的に成立しない。要件書 L60-L83 の表を唯一の出典とする"
  guard_test: "tests/test_vram.py::test_estimate_is_pure_and_needs_no_gpu"

- id: D-02
  rule: "テストは実 HTTP を発行しない。実ランタイムに接続するテストは @pytest.mark.live を付け、--run-live 指定時のみ実行する"
  rationale: "Ollama の導入は実装スコープ外でありユーザーの手動作業。CI (GitHub Actions) には Ollama が無いため、実接続テストは常に赤になる。httpx.MockTransport で全経路を決定論的に検証する"
  guard_test: "tests/test_testing_policy.py::test_live_tests_are_skipped_by_default"

- id: D-03
  rule: "VRAM に関する数値の単位は GiB に統一する。GB と GiB を混在させず、要件書の『16GB』は 16.0 GiB として扱う"
  rationale: "実機 VRAM は 16376 MiB (= 15.99 GiB = 17.17 GB) であり、GB/GiB の取り違えは 7% の判定誤差になる。差分は vram.budget_gib の設定で吸収し、コード内では常に GiB を使う"
  guard_test: "tests/test_vram.py::test_all_vram_values_are_gib"

- id: D-04
  rule: "VRAM 予算超過は WARNING ログを出したうえで VramBudgetExceededError を送出して停止する。警告のみで処理を継続しない"
  rationale: "要件書 L296『16GB を超えるプロファイルを指定した場合、起動時に警告して停止する』。継続すると実行時 OOM になり、原因が特定しにくいエラーに化ける"
  guard_test: "tests/test_bootstrap.py::test_oversized_profile_warns_and_aborts_without_http"

- id: D-05
  rule: "api_key は設定ファイルに直接書かず api_key_env (環境変数名) で参照し、SecretStr で保持する。ログ・実行マニフェスト・例外メッセージには環境変数名のみを出す"
  rationale: "CLAUDE.md セキュリティ原則『認証情報のハードコード禁止』『ログへの認証情報出力禁止』。configs/*.toml はコミット対象のため、値を書ける形にすると事故が起きる"
  guard_test: "tests/test_manifest.py::test_manifest_and_logs_never_contain_api_key"

- id: D-06
  rule: "llmkit パッケージはリポジトリ直下のフラットレイアウトに置き、src/ レイアウトにしない"
  rationale: "既存 pyproject.toml に [build-system] が無く pythonpath=[\".\"] で main.py を解決している。src/ に移すと pytest の pythonpath / mypy 対象 / CI の全てに変更が波及し、テンプレートの検証構成 (make ci) を壊す。パッケージ配布は Phase 1 の要件に無い"
  guard_test: "tests/test_layout.py::test_llmkit_importable_from_repo_root"

- id: D-07
  rule: "推論ランタイムの JSON レスポンスは pydantic の strict モデルで受け、必須フィールドの欠損は UpstreamError にする。dict[str, Any] のまま扱わない"
  rationale: "mypy の disallow_any_explicit = true により Any を明示的に書けない。加えて、欠損を黙って None にすると Phase 2 の速度指標が静かに欠測し、比較結果が壊れる。パースを厳格にして早期に失敗させる"
  guard_test: "tests/test_client.py::test_malformed_response_raises_upstream_error"
```

補足: `check-decisions.sh` が参照先テストの実在を検証するため、`.claude/decisions.yaml` の作成は **T5（該当テストが揃った後）**に行う。T1 時点で作ると hook が落ちる。

## 7. 想定リスク（これが起きたら止まって相談）

1. **Ollama の OpenAI 互換エンドポイントが `num_ctx` を無視する可能性**
   Ollama の `/v1/chat/completions` はコンテキスト長をリクエストで受け付けず、Modelfile 側の設定に従う仕様の可能性がある。これが Phase 0 の実測で確認された場合、「コンテキスト長の設定」という Phase 1 要件を満たすためにネイティブ `/api/chat` への分岐が必要になり、L2 の設計（単一の OpenAI 互換境界）が変わる。**判明した時点で止めて設計判断を仰ぐこと。** Phase 1 では「ボディに載せる・マニフェストに記録する・見積りに使う」までを実装し、実効性の検証は live テストに委ねる。

2. **静的 VRAM テーブルが実測と乖離し、構成3 が 16 GiB を超えない**
   要件書の表は「構成2 ≈ 13GB」「構成3（構成2 + 埋め込み 0.7 + リランカー 0.8）= 16GB 超過」となっており、単純加算では内部的に整合しない（13 + 1.5 ≈ 14.5 GiB）。この差を埋めるために**カタログの数値を勝手に大きくしない**こと。予算ガード本体は合成プロファイルで検証し（T2 受け入れ基準）、構成3 の帯テストは Phase 0 実測後に更新する前提で `(仮)` を明記する。乖離が 1.5 GiB を超えるようなら止めて相談。

3. **mypy `strict` + `disallow_any_explicit` で pydantic / httpx が想定外に詰まる**
   CLAUDE.md は「`# type: ignore` を理由なく使わない」を明記している。回避策として ignore を撒く形になりそうなら、それは型設計が間違っているサイン。ignore を3箇所以上必要とする状況になったら止めて、dataclass + 手書きバリデーションへの切替を含めて相談する。

## 8. 不明点（承認時に確定する3問）

**Q-A: パッケージ名** — (1) `llmkit` ← 推奨 / (2) `locallm` / (3) `local_llm`

**Q-B: `context_tokens` の Phase 1 での扱い**（リスク1 に対応）

- (1) 設定として保持し、VRAM 見積り・マニフェスト記録・リクエストボディへの送出（`options.num_ctx`）まで行い、**実ランタイムでの反映確認は Phase 0 / live テストに委ねる** ← 推奨
- (2) OpenAI 互換エンドポイントとネイティブ `/api/chat` の2実装を最初から用意する

**Q-C: CLI を Phase 1 に含めるか**

- (1) 含める。`doctor` と `chat` の最小2サブコマンド ← 推奨
- (2) ライブラリのみ。CLI は Phase 2 のハーネスと同時に作る

## 付録: ファイル構成案

```
llmkit/                       # フラットレイアウト (D-06)
├── __init__.py               # 公開 API の再エクスポート。L3 はここだけを import する
├── errors.py                 # 例外階層 + 対処方法つきメッセージ           (T1)
├── config.py                 # AppConfig / RuntimeConfig / GenerationParams (T1)
├── catalog.py                # ModelSpec 静的テーブル                      (T2)
├── vram.py                   # プロファイル解決・見積り・予算判定          (T2)
├── client.py                 # ChatClient Protocol + OpenAICompatibleClient (T3)
├── manifest.py               # RunManifest 生成・JSON 永続化・伏字化       (T4)
├── bootstrap.py              # 起動シーケンス (設定→検査→ログ→client)     (T4)
└── cli.py                    # argparse: doctor / chat                     (T4)

configs/
├── default.toml              # ローカル Ollama (構成1 既定)
└── external_openai.toml      # 外部 OpenAI 互換 API 切替の例 (Q7 = 入れる)

tests/
├── conftest.py               # live マーカー既定スキップ + 共有 fixture    (T5)
├── test_main.py              # 既存。変更しない
├── test_config.py            (T1)
├── test_catalog.py           (T2)
├── test_vram.py              (T2)
├── test_client.py            (T3)
├── test_client_errors.py     (T3)
├── test_param_wiring.py      (T3) ★ 有効性テストの中核
├── test_manifest.py          (T4)
├── test_bootstrap.py         (T4)
├── test_layout.py            (T5)
├── test_testing_policy.py    (T5)
└── test_acceptance_phase1.py (T5) 要件書 L294-L298 と 1:1

outputs/runs/                 # 実行マニフェスト (gitignore 済み)
.claude/decisions.yaml        # T5 で作成 (guard_test が実在してから)
main.py                       # 既存。変更しない
```

**レイアウト判断の理由**: 既存 `pyproject.toml` に `[build-system]` が無く、`pythonpath = ["."]` でルートの `main.py` を解決している。`src/` レイアウトにすると `[build-system]` 追加 + `pythonpath` 変更 + mypy 対象の見直しが必要になり、`make ci`（ローカル / Stop hook / GitHub Actions の単一の真実）に波及する。パッケージ配布は Phase 1 の要件に含まれないため、フラットを維持する（→ D-06）。

**追加依存**:

| パッケージ | グループ | 用途 |
|---|---|---|
| `httpx>=0.28` | main (`uv add httpx`) | HTTP クライアント。`MockTransport` がテストに使えるため実 HTTP なしで全経路を検証できる |
| `pydantic>=2.9` | main (`uv add pydantic`) | 設定・レスポンスの strict パース。`py.typed` 提供のため mypy strict と `disallow_any_explicit` に適合する |
| （dev への追加なし） | — | `pytest` / `pytest-cov` / `mypy` / `ruff` は既存。`respx` や `pytest-httpx` は `MockTransport` で代替できるため入れない |

`uv add` 後に `uv lock` が更新され、`make ci` の `uv lock --check` が通ることを T1 の完了条件に含める。
