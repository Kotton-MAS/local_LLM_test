# CLAUDE.md — Python モノレポ テンプレート

<!-- ============================================================
  使い方:
  1. TODO コメントをすべて自分のプロジェクトに合わせて書き換える
  2. 該当しないセクションは丸ごと削除する
  3. 200行以下を維持する（長いほどコンテキストを圧迫する）
  4. Claudeがミスをしたら都度このファイルに追記する
  5. チェックリスト的な情報はSkillsに分離する
============================================================ -->

## プロジェクト概要

<!-- TODO: 1〜2行でプロジェクトの目的を書く -->
このプロジェクトは _____ を行うPythonモノレポです。
ISMS（ISO/IEC 27001）準拠の開発プロセスに従います。

## 推奨される進め方

1. Claude Code に /init でプロジェクトを分析させる
2. その結果は参考程度に見る（採用しなくてOK）
3. 今回のテンプレートをプロジェクトルートに配置
4. Claude Code に「このCLAUDE.mdのTODOを埋めて」と指示
5. 出力を自分でレビューし、不要な情報を削除
6. 最終的に自分の目で200行以下に収まっていることを確認

### 指示サンプル
```
このCLAUDE.mdテンプレートを読んで、
現在のプロジェクト構成を調査した上で、
TODOの部分を実際の内容に書き換えて。
ただし構成やルールは変えないで。

```

## プロジェクト構成

```
/
├── .devcontainer/         # Dev Container 設定 (Docker開発環境)
├── .github/
│   └── workflows/
│       └── python-ci.yaml # GitHub Actions CI (make ci を実行)
├── .vscode/               # VS Code 推奨設定・拡張機能
├── .claude/
│   ├── settings.json          # プロジェクト固有の設定のみ（ガード類はキット側）
│   ├── decisions.yaml         # 意図的な設計判断（guard_test 必須。初期状態では未作成）
│   ├── schemas/findings.schema.json  # reviewer→fixer の findings JSON 共通スキーマ
│   └── tmp/                   # /review・/pdca の一時ファイル（gitignore 済み）
│   # フック・ガード・スラッシュコマンド・スキル・agent 群はすべて
│   # user スコープの claude-pdca-kit (~/.claude/) が提供する。
│   # reviewer-uv もキット側にあるため、このリポジトリには置かない
│   # （プロジェクト側にコピーを置くと、キットの改良が上書きで打ち消される）
├── docs/
│   ├── plans/             # planner の仕様書・タスクリスト
│   └── adr/               # architect の設計判断記録 (ADR)
├── .pre-commit-config.yaml # pre-commit フック設定
├── Makefile               # 検証コマンドの単一の真実 (make ci)
├── main.py                # エントリポイント
├── tests/                 # pytest テスト
├── pyproject.toml         # プロジェクト・依存関係定義 (uv)
├── uv.lock                # 依存関係のロックファイル
└── CLAUDE.md              # このファイル
```

<!-- サブパッケージを追加した場合、各ディレクトリにもCLAUDE.mdを配置可能。
     Claude Codeはディレクトリ階層を遡って全CLAUDE.mdを読み込む。 -->

## コマンド一覧

```bash
# 依存インストール
uv sync                              # 依存関係の同期
uv add <package>                     # パッケージ追加
uv add --group dev <package>         # dev依存の追加

# テスト
uv run pytest                        # 全テスト実行
uv run pytest -k "test_name"         # 単一テスト
uv run pytest --cov                  # カバレッジ付き

# リント・フォーマット（Claudeはこれらを自分で実行して確認すること）
uv run ruff check .                  # リント
uv run ruff format .                 # フォーマット
uv run mypy .                        # 型チェック

# pre-commit
uv run pre-commit install            # フックの初期設定
uv run pre-commit run --all-files    # 全ファイルに手動実行

# 実行
uv run python main.py                # エントリポイント実行
```

## アーキテクチャ原則

<!-- TODO: プロジェクト固有のアーキテクチャ原則を記載 -->
- 認証・認可は既存の共通基盤を使用し、独自実装しない

## コード規約

- Python 3.12+、型アノテーション必須
- `Any` 禁止（`object` か `Protocol` を使用）
- 関数は単一責任、50行超は分割を検討

## セキュリティ原則（常時適用）

<!-- 詳細な手順・チェックリストは .claude/skills/isms-security/SKILL.md に記載 -->

**絶対禁止:**
- 認証情報（パスワード、APIキー、トークン）のハードコード
- `.env`, 秘密鍵, `credentials.json` のコミット
- ログへの個人情報・認証情報の出力
- 本番データの開発環境への複製
- 本番環境への直接アクセス・変更
- セキュリティ機能の無効化、監査ログの削除

**必須:**
- シークレットは環境変数またはKey Vaultから取得
- 全ユーザー入力を検証・サニタイズ（SQLi, XSS, コマンドインジェクション対策）
- パラメータ化クエリの使用（生SQLの禁止）
- エラーレスポンスにスタックトレースや内部実装の詳細を含めない
- セキュリティに関わるコードは必ず人間がレビュー

**ログ出力ルール:**
- 含める: タイムスタンプ(UTC), イベント種別, ユーザーID, リソースID, 結果
- 含めない: パスワード, トークン, カード番号, 氏名, 住所, メールアドレス

## テストルール

- 新しいコードには必ずテストを書く
- テストファイル: `test_{モジュール名}.py`
- **実装後は `uv run pytest` を実行し全テストパスを確認してから完了とする**
- カバレッジが下がる変更は理由をコミットメッセージに記載

## Git / PRワークフロー

- ブランチ: `feat/xxx`, `fix/xxx`, `refactor/xxx`
- **作業前に必ず作業ブランチを切る。** キットの自動コミット (`auto-commit.sh`) は
  `main` / `master` では動かないため、main のまま作業すると subagent の成果が
  コミットされずに溜まり続ける
- コミット: Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `ci:`)
- コミット自体はキットの SubagentStop フックが自動で行う（wip コミット。PR 前に squash する）
- コミット前に `git diff` で機密情報の混入がないことを確認。
  ただし自動コミットにより `git diff` は空になり得るので、
  差分は `git diff $(cat .claude/tmp/base-ref)` で見る

**レビュー必須の変更:**
認証・認可、個人情報処理、外部API連携、DBスキーマ変更、インフラ構成

**PR手順:** 実装+テスト → ruff+mypy → pytest → git diff確認 → push → PR作成（`/pr` コマンド使用可）

## CI/CD

- GitHub Actions (`.github/workflows/python-ci.yaml`) がPR時に `make ci` を実行: uv lock --check, ruff check, ruff format --check, mypy, pytest
- シークレットはGitHub Secretsで管理（コードに含めない）
- 本番デプロイは承認済みブランチからのみ

## Claudeへの作業指示

- 複数ファイルにまたがる変更は **Plan Mode** で計画してから実装
- 実装後は テスト + リント + 型チェック で検証
- 既存コードパターンに合わせる。不明点は推測せず調査する
- セキュリティ関連の詳細が必要なときは `isms-security` スキルを参照（user スコープの claude-pdca-kit が提供）
- エージェント成果物の保存先: planner 仕様書 → `docs/plans/`、architect ADR → `docs/adr/`、doc-writer 資料 → `docs/`
- reviewer 系エージェントの findings JSON は `.claude/schemas/findings.schema.json` に準拠させる
  （`evidence` は必須。実測でないものを `measured` と書かない。
  キットの `triage_findings.py` が機械検証し、裏付けの無い実測主張は推論へ降格する）
- ファイル編集後は保存時フォーマットが自動実行される（キットの `format.sh`）
- 「普通はこうするが、今回は理由があって別の選択をする」判断は
  `.claude/decisions.yaml` に guard_test 付きで記録する。
  雛形: `cp ~/.claude/templates/decisions.yaml .claude/decisions.yaml`
  （散文で残すと後続の fixer が一貫性のために潰しても何も落ちない）
- カスタムコマンド: `/security-review`（ISMSレビュー）、`/pr`（PR作成）、
  `/retro`（エージェント運用の振り返り）※ user スコープの claude-pdca-kit が提供
- エージェント運用でうまくいかなかったことは、その場で1行記録する:
  `~/.claude/hooks/record-incident.sh note manual "<課題を1文で>"`

## AI支援開発の制限

- 本番環境への直接操作・認証情報の生成表示・未承認の外部連携は禁止
- AI出力のセキュリティ関連コードは必ず人間がレビュー

## Compaction時の保持ルール

変更ファイル一覧 / テスト結果 / 未完了タスク / ブランチ名 / セキュリティ上の懸念事項

## やってはいけないこと

- `print()` をプロダクションコードに残さない
- テストなしで機能追加しない
- パブリックAPIの型シグネチャを無断変更しない
- `# type: ignore` / `# nosec` / `# noqa` を理由なく使わない
- `.env` やシークレットをハードコードしない
<!-- ============================================================
  ↓ ここから下は、Claudeが間違えるたびに追記するセクション ↓
  例: 「xxxモジュールのインポートパスは packages.core.xxx であること」
============================================================ -->

## 学習メモ（Claudeのミスから追記）

<!-- Claudeに間違いを修正させた後、
     「CLAUDE.mdの学習メモに、今のミスを二度としないようルールを追記して」
     と指示する。このセクションは運用しながら育てる。 -->
