# local_LLM_test — ローカル LLM 実行基盤

手元の GPU (RTX 4070 Ti SUPER / VRAM 16GB) 上で、外部 API に依存しない LLM 実行環境を作る。
機密性・コスト非依存・学習の3点が狙い。要件は [`docs/localllmrequirements.md`](docs/localllmrequirements.md)。

土台は uv + Docker (Dev Container) の Python プロジェクトテンプレート。

## 実装状況

| Phase | 内容 | 状態 |
|---|---|---|
| **0** | **環境構築と実測 (Ollama 導入・モデル取得・VRAM 実測)** | **完了 (2026-08-23 実測)** |
| **1** | **推論クライアント層 (L2)** — `llmkit/` | **実装済み** |
| **2** | **モデル比較ハーネス (L3)** — `harness/` + `suites/` + `results/` | **完了 (2026-08-23 実機実行)** |
| 3 | RAG パイプライン (Obsidian vault) | 未着手 |
| 4 | チャット UI 接続 (Open WebUI) | 未着手 |

Phase 1 の残課題は [`docs/next-pr-candidates.md`](docs/next-pr-candidates.md) に、
意図的な設計判断は `.claude/decisions.yaml` (D-01〜D-24) にあります。

> **Phase 0 は完了しています** (受け入れ条件 5 項目すべて充足。実測は
> [`docs/phase0-vram-measurements.md`](docs/phase0-vram-measurements.md))。
> ただし Ollama の導入・`ollama pull`・リランカー用 llama-server の配置は
> **各マシンで手動**に行う必要があります (同ドキュメントの「環境の再現手順」)。
> 未セットアップのマシンでも `llmkit` のテストは全件パスし、`doctor` は原因を教えて終了します (後述)。

## 前提条件

このテンプレで生成したプロジェクトは claude-pdca-kit が ~/.claude/ にインストールされていることを前提にしている。

> **Note**: claude-pdca-kit (planner / reviewer 群などのエージェント・スキル) は
> Claude Code の user スコープ (`~/.claude/`) に置かれるため、**Claude Code はホスト側で実行**する。
> Dev Container はテスト・実行用のランタイム環境であり、コンテナ内から Claude Code を
> 起動しても pdca-kit のエージェント・スキルは利用できない。
> コンテナ内で使いたい場合は `compose.yml` のコメントアウトされたマウント設定を参照。

このリポジトリの `.claude/` にはプロジェクト固有のもの (findings スキーマ、
`decisions.yaml`) だけを置く。フック・ガード・reviewer 群をここにコピーしないこと。
プロジェクト側のコピーは user スコープより優先されるため、キット側の改良が
静かに打ち消される。

**エージェントに作業させる前に作業ブランチを切る** (`git switch -c feat/xxx`)。
キットの自動コミットは `main` / `master` では動かない。

## Features

- **uv** によるパッケージ管理
- **Dev Container** による Docker 開発環境 (GPU オプション対応)
- **Ruff** によるリント・フォーマット
- **pytest** によるテスト・カバレッジ計測
- **pre-commit** による自動コード品質チェック
- **GitHub Actions** による CI (lint / format / type check / test)
- **VS Code** 推奨拡張機能・設定同梱

## Getting Started

### 1. テンプレートからプロジェクト作成

GitHub の **Use this template** ボタン、またはクローンして利用します:

```bash
git clone https://github.com/Kotton-MAS/python_dev_template my-project
cd my-project

# プロジェクト名を pyproject.toml の [project] name に合わせて変更する

# 依存関係の同期
uv sync
```

### 2. Dev Container で起動

VS Code で **Dev Containers: Reopen in Container** を実行すると、Docker 環境が自動でビルドされます。

## llmkit (Phase 1: 推論クライアント層)

設定ファイルだけで推論先モデル・エンドポイント・VRAM プロファイルを切り替えられる
推論クライアント層です。仕様は `docs/plans/2026-08-22-phase1-inference-client-l2.md` と、
Phase 0 実測にもとづく追補 `docs/plans/2026-08-22-phase1-calibration-and-native-chat.md`。

### まず `doctor` を実行する

```bash
# 設定の検証 + VRAM 見積り + 実行マニフェスト出力 + ランタイム疎通確認
uv run python -m llmkit.cli doctor --config configs/default.toml

# 一問一答
uv run python -m llmkit.cli chat "日本語で自己紹介して" --config configs/default.toml
```

**Ollama の導入・`ollama pull` が済んでいないマシンでも `doctor` は実行できます。**
そのときは原因と対処方法を出して終了コード 1 で終わります。

```
エラー: 推論ランタイムに接続できません (base_url=http://localhost:11434/v1)。
ランタイムが起動していない可能性があります / 対処: `ollama serve` でランタイムを
起動するか、runtime.base_url が正しいか確認してください
```

VRAM 予算を超えるプロファイルを指定した場合は、**HTTP を 1 回も発行せずに**
警告して停止します (要件書 L296 / 決定 D-04)。

### 2 つの API 経路 (`runtime.kind`)

送出するワイヤプロトコルは `runtime.kind` で決まります。分岐は `create_chat_client` の
1 か所だけにあり、上位層は `ChatClient` Protocol にしか依存しません (決定 D-10)。

| `runtime.kind` | 送信先 | `context_tokens` の反映 |
|---|---|---|
| `ollama` (既定) | ネイティブ `{base_url の /v1 を除去}/api/chat` | **される** (`options.num_ctx`) |
| `openai_compatible` | OpenAI 互換 `{base_url}/chat/completions` | **されない** (Ollama が無視する) |

Phase 0 実測で、Ollama の OpenAI 互換エンドポイントは `options.num_ctx` を無視し
`ollama ps` の CONTEXT が既定の 4096 のままになることを確認しています
(`docs/phase0-vram-measurements.md`)。**ローカル Ollama では `kind = "ollama"` を使ってください。**
ネイティブ側の URL は `runtime.base_url` から導出するため、設定キーは増えません (決定 D-11)。

### 設定ファイル

| ファイル | 用途 |
|---|---|
| `configs/default.toml` | **既定。** ローカル Ollama にネイティブ `/api/chat` で接続 (構成1 = 生成 + 埋め込み + リランカー) |
| `configs/ollama_openai_compat.toml` | 同じローカル Ollama に OpenAI 互換 `/v1/chat/completions` で接続。`default.toml` との差は `kind` の 1 行だけで、2 経路の A/B 比較用 |
| `configs/external_openai.toml` | 外部 OpenAI 互換 API への切り替え例 (`is_local = false` で VRAM 予算ガードは無効) |

- 推論先モデルの切り替えは `[generation] model` の変更**だけ**で済みます (コード変更不要)。
- `vram.budget_gib` の既定は **14.0 GiB** です。カード容量 (16 GiB) ではなく、
  実測で CPU オフロードが始まらない「増分」の上限です (決定 D-12)。
- **api_key の値は設定ファイルに書きません。** 環境変数「名」を `runtime.api_key_env` に書き、
  値は `export LLMKIT_API_KEY=...` で渡します (決定 D-05)。値を直接書くと読み込み時に落ちます。
- VRAM の単位はすべて GiB です (決定 D-03)。

### リランカーは別ランタイム (llama.cpp llama-server)

**Ollama にはリランキング API がありません。** `POST /api/rerank` と `POST /v1/rerank` は
いずれも 404 を返し、最新版 (v0.33.0-rc2) でも未対応です。GGUF 自体は存在しますが、
スコアを返すエンドポイントが無いため Ollama に載せても使えません。
そのため**リランカーだけ llama.cpp の `llama-server` を別ポートで併走**させます (決定 D-17)。

```bash
# 解決したコマンド行を表示するだけ (llama-server が無くても動く。終了コード 0)
scripts/start-reranker.sh --dry-run

# 実際に起動する (フォアグラウンド。既定 127.0.0.1:8081)
scripts/start-reranker.sh
```

リポジトリ外のパスはハードコードせず、すべて環境変数で上書きできます。

| 環境変数 | 既定値 |
|---|---|
| `LLAMA_SERVER_BIN` | `~/.local/opt/llama.cpp/llama-server` |
| `RERANKER_MODEL` | `~/.local/share/llama-models/bge-reranker-v2-m3-Q6_K.gguf` |
| `RERANKER_HOST` | `127.0.0.1` (ループバック) |
| `RERANKER_PORT` | `8081` |
| `RERANKER_NGL` | `99` (GPU オフロードするレイヤ数) |
| `RERANKER_CTX_SIZE` | `2048` |

- **`llama-server` には認証機構がありません。** `RERANKER_HOST` を既定の
  `127.0.0.1` 以外 (例 `0.0.0.0`) にすると、LAN 上の誰でも無認証で
  `/v1/rerank` を叩け、GPU 資源の消費や入力文書の投入が可能になります。
  リモートから使う場合は `RERANKER_HOST` を変えず、SSH ポートフォワード
  (`ssh -L 8081:127.0.0.1:8081 <このホスト>`) を使ってください。
- **CUDA toolkit は不要です。** Vulkan が NVIDIA GPU を認識していれば、配布されている
  プリビルドバイナリ (`llama-b10586-bin-ubuntu-vulkan-x64.tar.gz`) で GPU が使えます。
  ソースビルドも不要です (`vulkaninfo --summary` で認識を確認できます)。
- モデルは `gpustack/bge-reranker-v2-m3-GGUF` の `bge-reranker-v2-m3-Q6_K.gguf` (478 MB)。
  第一候補だった `Ruri Reranker` は GGUF が存在しないため採用していません。
- 実測は **VRAM 0.28 GiB / 50 ペアの再ランキング 140 ms** です
  (`--ctx-size 2048` / `--n-gpu-layers 99` の 1 点。`RERANKER_CTX_SIZE` を上げると
  カタログの `0.28` が過小評価になるため、上げる場合は測り直してください)。
- **Phase 1 では埋め込み・リランカーの推論呼び出しを実装していません。**
  `llmkit` はこの 2 つを **VRAM 見積りに計上するだけ**です (実際に呼ぶのは Phase 3)。
  別プロセスが確保する分も合算します。予算判定の対象が `nvidia-smi` の返す
  デバイス全体の使用量であり、プロセス境界と無関係なためです (決定 D-16)。

### 実行マニフェスト

起動のたびに、使用した設定・プロファイル・VRAM 内訳・設定ファイルの SHA-256 を
`outputs/runs/{開始時刻}-{run_id}.json` に記録します (`outputs/` は gitignore 済み)。
同じ設定ファイルなら `config_sha256` が一致します。ただし比較ハーネス (Phase 2) は
モデルごとに設定を組み替えて起動するため、**実効値の同一性を表すのは `config_sha256` ではなく
`run_fingerprint`** です (決定 D-20。後述の「モデル比較ハーネス」節)。

### テスト

テストは Ollama 未起動・ネットワーク不通でも全件パスします (`httpx.MockTransport` で完結)。
実ランタイムに接続するテストを書く場合は `@pytest.mark.live` を付けてください。既定でスキップされ、
`uv run pytest --run-live` を指定したときだけ実行されます (決定 D-02)。

## モデル比較ハーネス (Phase 2: 比較・計測層)

プロンプト集と対象モデルのリストを 1 つの TOML で与えると、モデル × プロンプトの応答と
速度・VRAM・再現条件を `results/` 配下に書き出す CLI です。仕様は
`docs/plans/2026-08-23-phase2-comparison-harness.md`。

```bash
# 実行計画だけを見る (HTTP を 1 回も発行しない。Ollama が無くても終了コード 0)
uv run python -m harness.cli run --suite suites/ja_basic.toml --config configs/default.toml --dry-run

# 本実行 (results/<suite_id>/<実行時刻>-<run_fingerprint 先頭12桁>/ に書き出す)
uv run python -m harness.cli run --suite suites/ja_basic.toml --config configs/default.toml

# 一部だけ回す (--models は model_id のカンマ区切り、--limit はプロンプトを先頭 N 問に絞る)
uv run python -m harness.cli run --models qwen3-8b --limit 2
```

`--dry-run` は VRAM 見積りと予算判定・予定リクエスト数・`run_fingerprint` を出して終わります。
**比較を実際に回す前に、予算超過で途中から測れなくなる構成を検出できます。**

### スイートの書き方 (`suites/*.toml`)

```toml
[suite]
id = "ja_basic"              # results/<id>/ のディレクトリ名になる (パス区切りは不可)
description = "日本語の基本タスクで比較する"
warmup_runs = 1              # 既定 1。結果は捨てず phase="warmup" として残す (決定 D-22)

[[models]]                   # 比較対象モデルのリスト (最低 1 件)
model_id = "qwen3-14b"       # 省略した項目はベース設定 (--config) の値を使う
profile = "rag_default"
max_output_tokens = 512

[[prompts]]                  # 評価用プロンプト集 (最低 1 件、id はスイート内で一意)
id = "summarize"
text = "次の文章を3行で要約してください。"
system = "あなたは日本語の技術文書を扱うアシスタントです。"  # 任意
tags = ["要約"]                                              # 任意
```

`model_id` を書き替えると、**リクエストの `model` と VRAM 見積りの両方**が同時に切り替わります
(片方だけ変わると「20B に投げているのに 14B の VRAM 見積りを記録した比較結果」が
静かに生成されるため、これは決定 **D-19** として guard_test で固定しています)。

出荷スイート `suites/ja_basic.toml` は日本語 8 問 × 3 モデル (`qwen3-14b` / `gpt-oss-20b` /
`qwen3-8b`)、`max_output_tokens = 512` です。

### 出力 (`results/` はコミット対象、`outputs/` は一時物)

| パス | 内容 | Git |
|---|---|---|
| `results/<suite_id>/<時刻>-<fp12>/report.md` | 再現条件・モデル比較表・プロンプトごとの応答 (先頭 400 文字 + 折りたたみ) | **コミットする** |
| `results/.../records.jsonl` | 1 実行 1 行の生ログ (warmup 含む、応答全文の唯一の出典) | **コミットする** |
| `results/.../run.json` | 再現条件 (`reproduction`) と集計値。ここから `run_fingerprint` を再計算できる | **コミットする** |
| `results/.../manifests/*.json` | `llmkit` の実行マニフェスト (モデルごと 1 本) | **コミットする** |
| `outputs/runs/*.json` | `llmkit` 単体実行のマニフェスト。実行のたびに増える一時物 | gitignore |

`results/` は比較結果の記録そのもの (要件書 Phase 2 の受け入れ条件) なのでコミットします。
`outputs/` は実行ログなので ignore します (決定 **D-24**)。
スイートには機密情報・個人情報を書かないでください。応答本文ごとコミットされます。

### 速度と VRAM の読み方

- 生成速度の代表値は `eval` (ランタイムの実測) を優先し、無ければ壁時計から算出します。
  **どちらを使ったかは「速度出典」列に必ず出ます** (決定 **D-21**)。
- 欠測は `0` ではなく `—` (表) / `null` (JSONL) です。集計から除外し `測定数 n/N` を併記します。
- VRAM は**見積り (`llmkit`) と実測増分 (`nvidia-smi`) の両方**を別列で記録します。
  `nvidia-smi` を呼ぶのは `harness/gpu.py` だけで、失敗しても例外を投げず `None` になります
  (GPU の無い環境でも比較そのものは回ります。決定 **D-23**)。
- アイドル基準は実行全体で 1 回だけ測り、全モデルで共有します。

### 再現条件は `run_fingerprint` で見る

同じ入力で回し直したときに一致するのは `run_fingerprint` です (`config_sha256` **ではありません**)。
ハーネスはモデルごとに設定を組み替えて起動するため、`config_sha256` は「ベース設定ファイルの
同一性」しか表しません。`model_id` だけが違う 2 実行は `config_sha256` が一致します (決定 **D-20**)。

```bash
# run.json だけでフィンガープリントを再計算できる (reproduction が入力そのもの)
uv run python -c "import json,sys; from harness import fingerprint_digest; \
d=json.load(open(sys.argv[1])); print(fingerprint_digest(d['reproduction']) == d['run_fingerprint'])" \
results/ja_basic/*/run.json
```

## Project Structure

```
.
├── .devcontainer/          # Dev Container 設定
│   ├── Dockerfile          # Python 3.12 + uv ベースイメージ
│   ├── compose.yml         # Docker Compose 設定
│   ├── devcontainer.json   # VS Code Dev Container 設定
│   └── postCreateCommand.sh # コンテナ作成後の初期化スクリプト
├── .github/
│   └── workflows/
│       └── python-ci.yaml  # GitHub Actions CI ワークフロー
├── .vscode/
│   ├── extensions.json      # 推奨拡張機能
│   └── settings.json        # エディタ設定 (フォーマッタ等)
├── .env.example             # 環境変数のテンプレート
├── .gitignore               # Git 除外ファイル
├── .pre-commit-config.yaml  # pre-commit フック設定
├── .claude/
│   ├── decisions.yaml       # 意図的な設計判断 (guard_test 必須)
│   └── schemas/             # reviewer -> fixer の findings スキーマ
├── llmkit/                  # L2: 推論クライアント層 (Phase 1)
│   ├── config.py            # TOML 設定のロードと検証
│   ├── catalog.py           # モデルカタログ (VRAM 見積りの静的テーブル)
│   ├── vram.py              # プロファイル解決・見積り・予算判定
│   ├── client.py            # ChatClient Protocol + ネイティブ/OpenAI 互換の 2 実装
│   ├── manifest.py          # 実行マニフェスト (再現条件の記録)
│   ├── bootstrap.py         # 起動シーケンス
│   ├── cli.py               # doctor / chat サブコマンド
│   └── errors.py            # 対処方法つき例外階層
├── harness/                 # L3: モデル比較ハーネス (Phase 2)
│   ├── suite.py             # スイート TOML の読み込みと実効設定の導出
│   ├── gpu.py               # nvidia-smi による VRAM 実測 (呼ぶのはここだけ)
│   ├── records.py           # 1 実行 1 レコードのスキーマと集計 (中央値 / n/N)
│   ├── runner.py            # 実行計画・run_fingerprint・スイート実行
│   ├── report.py            # report.md / records.jsonl / run.json の書き出し
│   └── cli.py               # python -m harness.cli run [--dry-run]
├── suites/
│   └── ja_basic.toml        # 日本語 8 問 × 3 モデルの比較スイート
├── results/                 # 比較結果 (コミット対象。outputs/ とは扱いが違う)
│   └── ja_basic/<時刻>-<fp12>/ # report.md / records.jsonl / run.json / manifests/
├── configs/
│   ├── default.toml         # ローカル Ollama 用 (ネイティブ /api/chat)
│   ├── ollama_openai_compat.toml # ローカル Ollama 用 (OpenAI 互換経路。A/B 比較)
│   └── external_openai.toml # 外部 OpenAI 互換 API 用の設定
├── docs/
│   ├── localllmrequirements.md   # 要件定義 (v2)
│   ├── phase0-vram-measurements.md # Phase 0 の VRAM 実測記録 (較正の出典)
│   ├── next-pr-candidates.md     # 未対応の改善候補
│   ├── plans/               # planner の仕様書
│   └── adr/                 # architect の設計判断記録
├── scripts/
│   └── start-reranker.sh    # リランカー用 llama-server の起動 (パスは環境変数で上書き可)
├── outputs/runs/            # 実行マニフェスト (gitignore 済み)
├── .python-version          # Python バージョン指定 (3.12)
├── Makefile                 # 検証コマンドの単一の真実 (make ci)
├── main.py                  # エントリポイント
├── tests/                   # pytest テスト
├── pyproject.toml           # プロジェクト・依存関係定義
└── uv.lock                  # 依存関係のロックファイル
```

## File Details

### `pyproject.toml` - プロジェクト定義

uv が使用するプロジェクトメタデータと依存関係の定義ファイルです。

- **requires-python**: `>=3.12`
- **本番依存**: `httpx` (HTTP クライアント), `pydantic` (設定・応答の検証)
- **dev 依存関係**: `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`

型チェックは strict かつ `disallow_any_explicit = true` です。`Any` を明示的に書けません
(`object` / `Protocol` / pydantic dataclass を使う)。この設定のため `pydantic.BaseModel` は
使えません — クラス定義行が `explicit-any` エラーになります (決定 D-08)。

```bash
# 依存関係の同期
uv sync

# パッケージの追加
uv add <package>

# dev 依存関係の追加
uv add --group dev <package>
```

### `.devcontainer/` - Docker 開発環境

Dev Container は VS Code 上でコンテナ内の開発環境を提供します。

| ファイル               | 役割                                                                      |
| ---------------------- | ------------------------------------------------------------------------- |
| `Dockerfile`           | `python:3.12-slim-bookworm` ベースに uv・git・curl 等をインストール       |
| `compose.yml`          | ワークスペースのマウント、`.env` の読み込み、共有メモリ 8GB 設定          |
| `devcontainer.json`    | タイムゾーン (`Asia/Tokyo`)、Google Cloud CLI feature、GPU オプション設定 |
| `postCreateCommand.sh` | コンテナ作成後に git 補完の有効化と `uv sync` を実行                      |

### `.pre-commit-config.yaml` - コミット前の自動チェック

`git commit` 実行時に以下のチェックが自動で走ります:

| フック                   | 説明                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------- |
| **pre-commit-hooks**     | 末尾空白削除、EOF 改行保証、YAML/TOML 構文チェック、秘密鍵検出、大容量ファイル警告 |
| **ruff** (lint + format) | Python / Jupyter のリント (`--fix` 付き) とフォーマット                            |
| **prettier**             | YAML / JSON のフォーマット                                                         |
| **shellcheck**           | シェルスクリプトの静的解析                                                         |
| **mdformat**             | Markdown のフォーマット (GFM、テーブル対応)                                        |
| **codespell**            | スペルミス検出 (`logs/`, `data/`, `*.ipynb` は除外)                                |
| **nbstripout**           | Jupyter Notebook のセル出力を自動削除                                              |

```bash
# pre-commit の初期設定
uv run pre-commit install

# 全ファイルに対して手動実行
uv run pre-commit run --all-files
```

### `.github/workflows/python-ci.yaml` - GitHub Actions CI

Pull Request をトリガーに `uv sync --locked` で依存をインストールした後、`make ci` を実行します。
検証ロジックは Makefile に一元化されており、ローカル・Stop フック・CI がすべて同じコマンドを共有します:

1. **uv lock --check** - lock ファイルの整合性チェック
2. **ruff check** - リント
3. **ruff format --check** - フォーマットチェック
4. **mypy** - 型チェック
5. **pytest** - テスト実行

### `.vscode/` - エディタ設定

- **extensions.json**: Ruff、Python、Jupyter、Docker、Prettier 等の推奨拡張機能
- **settings.json**: Python ファイル保存時に Ruff で自動フォーマット・import 整理、JSON/YAML は Prettier でフォーマット

### `.python-version` - Python バージョン固定

uv や pyenv が参照する Python バージョン指定ファイルです。現在 `3.12` に設定されています。

### `.env.example` - 環境変数テンプレート

`.env` ファイルの雛形です。実際の `.env` は `.gitignore` で除外されています。コピーして使用してください:

```bash
cp .env.example .env
```

## Common Commands

```bash
# 依存関係の同期
uv sync

# テスト実行
uv run pytest

# カバレッジ付きテスト
uv run pytest --cov

# リント
uv run ruff check .

# フォーマット
uv run ruff format .

# pre-commit を全ファイルに実行
uv run pre-commit run --all-files

# CIの実行(ruff formatter mypy pytest の一括実行ができる)
make ci
```

> **`uv run pytest` が `ModuleNotFoundError: No module named 'lark'` で落ちる場合**
>
> シェルに ROS 2 などが `PYTHONPATH` を設定していると、pytest が venv 外のプラグインを
> 自動ロードして失敗します。`make test` / `make ci` は Makefile 側で `PYTHONPATH` を
> 空にするため影響を受けません。素の pytest を使いたい場合は
> `env PYTHONPATH= uv run pytest` としてください。
