# UV Python Template

uv + Docker (Dev Container) を使った Python プロジェクトテンプレート。

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
OpenAI 互換クライアント層です。仕様は `docs/plans/2026-08-22-phase1-inference-client-l2.md`。

### まず `doctor` を実行する

```bash
# 設定の検証 + VRAM 見積り + 実行マニフェスト出力 + ランタイム疎通確認
uv run python -m llmkit.cli doctor --config configs/default.toml

# 一問一答
uv run python -m llmkit.cli chat "日本語で自己紹介して" --config configs/default.toml
```

**Phase 0 (Ollama の導入・`ollama pull`) が未了でも `doctor` は実行できます。**
そのときは原因と対処方法を出して終了コード 1 で終わります。

```
エラー: 推論ランタイムに接続できません (base_url=http://localhost:11434/v1)。
ランタイムが起動していない可能性があります / 対処: `ollama serve` でランタイムを
起動するか、runtime.base_url が正しいか確認してください
```

VRAM 予算を超えるプロファイルを指定した場合は、**HTTP を 1 回も発行せずに**
警告して停止します (要件書 L296 / 決定 D-04)。

### 設定ファイル

| ファイル | 用途 |
|---|---|
| `configs/default.toml` | ローカル Ollama (構成1 = 生成 + 埋め込み + リランカー) |
| `configs/external_openai.toml` | 外部 OpenAI 互換 API への切り替え例 |

- 推論先モデルの切り替えは `[generation] model` の変更**だけ**で済みます (コード変更不要)。
- **api_key の値は設定ファイルに書きません。** 環境変数「名」を `runtime.api_key_env` に書き、
  値は `export LLMKIT_API_KEY=...` で渡します (決定 D-05)。値を直接書くと読み込み時に落ちます。
- VRAM の単位はすべて GiB です (決定 D-03)。

### 実行マニフェスト

起動のたびに、使用した設定・プロファイル・VRAM 内訳・設定ファイルの SHA-256 を
`outputs/runs/{開始時刻}-{run_id}.json` に記録します (`outputs/` は gitignore 済み)。
同じ設定なら `config_sha256` が一致するため、Phase 2 の比較実験の再現条件になります。

### テスト

テストは Ollama 未起動・ネットワーク不通でも全件パスします (`httpx.MockTransport` で完結)。
実ランタイムに接続するテストを書く場合は `@pytest.mark.live` を付けてください。既定でスキップされ、
`uv run pytest --run-live` を指定したときだけ実行されます (決定 D-02)。

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
- **dev 依存関係**: `pytest`, `pytest-cov`, `ruff`

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
