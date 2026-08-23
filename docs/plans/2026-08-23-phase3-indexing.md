# Phase 3 (前半): RAG 取り込み・パース・分割・索引 — 仕様書とタスクリスト

> 前提資料: `docs/localllmrequirements.md`（RAG 詳細設計 ①〜⑤ / Phase 3 受け入れ条件 L309-L319 / 制約 L233-L248）/ `.claude/decisions.yaml`（D-01〜D-24）/ `docs/phase0-vram-measurements.md`（埋め込み実測）/ `docs/plans/` 既存4本
> 位置づけ: **L3 の新規トップレベルパッケージ `rag/`** + **L2 (`llmkit/`) への追加のみ**（既存公開シグネチャ変更なし）。検索・リランキング・回答生成は書かない。
>
> **確定済みの前提（ユーザー承認、変更しない）**: 公開リポジトリのため受け入れ検証は合成 vault で行い実 vault の索引結果・本文はコミットしない / 埋め込みは Ollama `/v1/embeddings` + `hf.co/Targoyle/ruri-v3-310m-GGUF`（768次元、D-14 で確定済み）/ リランカーは本サイクル対象外だが D-15 の `serving_runtime='llama_cpp_server'` は維持

## 0. サイクル分割（確定）

planner が「L タスク4本は分割不足」と判定。**受け入れ条件7項目のうち5項目は埋め込みもストアも無しに検証できる**ため、2サイクルに分割する。

| サイクル | ブランチ | タスク | 満たす受け入れ条件 | 決定 |
|---|---|---|---|---|
| **3a** | `feat/phase3a-parsing` | T1（L2 埋め込みクライアント）・T2（ローダ/パーサ/合成 vault）・T3（チャンカ） | L311 / L312 / L313 / L314 / **L319** | D-25 / D-27 / D-30 / D-31 / D-32 / D-34 |
| **3b** | `feat/phase3b-indexing` | T4（ストア/差分/索引）・T5（CLI・受け入れ・決定・文書） | L309 / **L310** | D-26 / D-28 / D-29 / D-33 |

**3a の危険はすべて T1 に集中する**（`llmkit/client.py` の内部抽出が既存488テストに波及するリスク）。3b は 3a が緑なら HTTP は MockTransport に閉じる。

## 1. ゴール

Obsidian vault のパスを設定で与えると、`.obsidian/` と添付を除外し、frontmatter/wikilink/見出し階層を解決したチャンクを埋め込んで永続化し、**再実行時に変更のないノートを一切再処理せず、vault を1バイトも書き換えない**索引 CLI を作る。

## 2. 現状認識

| パス | 内容 | Phase 3 での使い方 |
|---|---|---|
| `llmkit/client.py` の `_HttpChatClient` | HTTP 送信・接続エラー翻訳・ステータス/本文からの例外翻訳・api_key ヘッダ・`httpx.Client` 所有権を集約 | **埋め込みも同じ翻訳表を通す必要がある。ここを chat 非依存の基底へ抽出する（本サイクル唯一の既存コード改変）** |
| `llmkit/client.py` の `api_style_for` / `endpoint_url_for` | `match` + `assert_never` | 埋め込みエンドポイントの導出も同じ形にそろえる |
| `llmkit/catalog.py` の `ruri-v3-310m` | `served_name="hf.co/Targoyle/ruri-v3-310m-GGUF"` / `role="embedding"` / `serving_runtime="ollama"` | 埋め込みモデルの唯一の出典。**カタログは変更しない** |
| `llmkit/config.py` の `ProfileConfig.embedding` | `configs/default.toml` で `ruri-v3-310m` に配線済み | **埋め込みモデル ID の唯一の出典**（→ D-27） |
| `llmkit/vram.py` | `resolve_profile` / `estimate_resolved_profile` / `check_budget`（純関数、D-01） | 索引前の予算判定にそのまま使う。RAG 側に VRAM ロジックを作らない |
| `harness/` 7モジュール | **L3 の前例**。`llmkit` の公開シンボルのみを使い、`__all__` は和集合、`cli` は非公開 | `rag/` はこの構造を写す |
| `tests/test_harness_layout.py` | (i) L3→llmkit サブモジュール直接 import 0件 (ii) `__all__` 和集合 (iii) llmkit→L3 0件、いずれも **AST** で検査 | `tests/test_rag_layout.py` を同型で新設 |
| `tests/test_layout.py` | `_SUBMODULE_NAMES` と D-08 guard の走査範囲（`llmkit/` + `harness/`） | **`_SUBMODULE_NAMES` に `embeddings` を追加**し、D-08 guard の走査範囲に `rag/` を追加（緩和ではなく拡張） |
| `tests/conftest.py` | `_forbid_real_network` / `RecordingTransport` / `FakeProbe` / `tmp_config` | 埋め込みテストもこれに乗る。**新しい網羅的 fixture を別に作らない** |
| `tests/test_acceptance_phase1.py` / `phase2.py` | `ACCEPTANCE_MAP` が要件書 **L294-L298 / L302-L305 を行番号で**参照 | **要件書の行数を1行も増減させてはならない** |
| `.gitignore` | `outputs/` と **`data/` は ignore 済み**。`results/` は明示的に非 ignore | 索引成果物は `data/` 配下に置けば公開事故が構造的に起きない（→ D-33） |
| `pyproject.toml` | 依存は `httpx` と `pydantic` の**2つだけ**。`pythonpath=["."]`、mypy の exclude は `.venv` のみ | 新規トップレベルパッケージは設定変更なしで strict の対象になる |

### ★ 見落とすと壊れる構造（本仕様の中核 / 3b で実装）

**差分更新は「何を再処理しないか」を決める仕組みであり、前提が変わったときに再処理を止めると索引が静かに壊れる。**

`max_chunk_tokens` を 240→200 に変えて再実行した場合、mtime もハッシュも変わらないノートは「変更なし」と判定されて再チャンクされない。結果として**索引の中に 240 で切ったチャンクと 200 で切ったチャンクが混在**する。埋め込みモデルを差し替えた場合はさらに悪く、**次元も意味空間も異なるベクトルが同じストアに同居**し、コサイン類似度が無意味になる。どちらも例外を出さず、テストも落ちず、検索精度だけが理由不明に劣化する。

これは Phase 2 の D-19（モデル ID が2か所に現れる）と同型の「静かに嘘の成果物を作る」欠陥であり、**Phase 3 全体が防ぐべき唯一最大の欠陥**である（→ D-28、3b）。

### 既存の慣習で守るべきもの

1. **層の境界は AST で機械検証する**（`test_harness_layout.py` / 決定12・43）。`rg` の全文一致は docstring の説明文を違反と誤検出する
2. **スキーマは pydantic dataclass + `TypeAdapter`**。`BaseModel` を継承しない（D-08）。自分で組み立てて書き出すだけの型は素の frozen dataclass（決定23）
3. **欠測は `None`。0 で埋めない**（D-07 / D-13 / D-21）

### 影響範囲（3a）

- 新規: `rag/{__init__,settings,vault,parser,chunker}.py` / `vaults/sample/` / `vaults/sample.toml` / `llmkit/embeddings.py` / `tests/` 6ファイル
- 変更: `llmkit/client.py`（**private 基底の抽出のみ**）/ `llmkit/__init__.py`（再エクスポート）/ `tests/test_layout.py`（走査範囲と `_SUBMODULE_NAMES` の拡張）/ `.claude/decisions.yaml`
- **不変: `llmkit/{config,catalog,vram,manifest,errors,bootstrap,cli}.py` / `harness/` 全体 / `configs/*.toml` / `suites/` / `results/` / `main.py` / `Makefile` / `.github/workflows/` / `pyproject.toml` / `uv.lock` / `docs/localllmrequirements.md`**

## 3. 前提・制約

### ハード制約

- 既存 **488 テストを壊さない。アサーションを緩めない**。既存テストを1行でも修正したくなったら止まって相談（§7 リスク1）
- mypy `strict` + `disallow_any_explicit`。`Any` 明示禁止。`# type: ignore` 0個。`pydantic.BaseModel` 非継承（D-08）
- ruff `T20`。CLI 出力は `sys.stdout.write` / `sys.stderr.write` か logging
- **テストは実 HTTP を1バイトも出さない**（D-02）。埋め込みも `httpx.MockTransport`。実ランタイム接続は `@pytest.mark.live`。**CI（GPU 無し・Ollama 無し）で全件緑**
- **新規依存パッケージを追加しない**（`uv lock --check` が無変更で通ること）
- `llmkit/` の既存**公開**シグネチャを変更しない（追加のみ）。`llmkit/` に GPU 参照・`subprocess` を持ち込まない（D-01 / D-23 維持）
- **`docs/localllmrequirements.md` の行数を1行も増減させない**（3a では内容も変更しない。チェックボックス更新は 3b）
- 実 vault の絶対パス（`/home/<user>/...`）を**コード・テスト・コミットされる設定ファイルに一切書かない**
- **実 vault のノート本文・索引成果物をコミットしない**
- 検索・リランキング・回答生成・評価質問セットを実装しない

### ソフト制約

- ログは INFO で件数のみ、DEBUG で相対パスまで。**ノート本文を1文字もログに出さない**
- 1ノートの処理失敗は記録して次のノートへ進む

### ★ 論点への判定

#### 論点1: 埋め込み呼び出しをどの層に置くか → **(a) `llmkit` に `EmbeddingClient` を足す**

| 案 | 判定 | 根拠 |
|---|---|---|
| **`llmkit/embeddings.py` を新設** | **採用** | 要件書 L239「推論ランタイムへの直接依存をアプリケーション層に持ち込まない。**L2 の抽象を必ず経由する**」。`harness/` は `llmkit` の公開シンボルのみを使う L3 として作られ、`test_harness_layout.py` がそれを AST で固定している。`rag/` に HTTP を持たせると L3 が2種類（ランタイムを叩くものと叩かないもの）に分裂し、既存の境界検査が意味を失う |
| `rag/embed.py` に置く | 却下 | 上記に反する。加えて `_HttpChatClient` の例外翻訳表（`ModelNotFoundError` / `OutOfMemoryError` / `ContextLengthError`）を複製することになり、D-07 とセキュリティ原則（本文を例外に載せない）が2か所に分岐する |

**境界検査は同型のものを新設する。** `tests/test_rag_layout.py` に (i)(ii)(iii) を写し、**さらに1つ強い検査を足す**: `rag/` のどのモジュールにも HTTP メソッド呼び出し（`.post` / `.get` / `.request`）と埋め込みエンドポイントのパス文字列が存在しない（AST + 非 docstring 文字列リテラル）。import 検査だけでは「`httpx` を直接使う」経路を塞げないため（→ D-25）。

#### 論点2: Chroma の導入 → **(b) `VectorStore` 抽象 + 最小の自前実装**（3b で実装）

`chromadb` は **transitive で 79 パッケージ**を引き込む（`uv pip compile` による実測、2026-08-23。プロジェクト外で解決したため lock は無変更）。現在のプロジェクトは dev 依存込みで **35 パッケージ**、直接依存は `httpx` と `pydantic` の **2 個**。内訳には `onnxruntime` / `grpcio` / `kubernetes` / `uvicorn` / `tokenizers` / `numpy` / `opentelemetry` 群（6 パッケージ）が含まれる。対象規模は実 vault で34ノート・257K文字 ≒ 370チャンク前後であり、768次元 × 370件の総当たりコサイン類似度は純 Python でも約30ms、JSONL 永続化で約3.5MB。Chroma が解く問題（大規模・並行・サーバモード）がこの規模では発生しない。

要件書 L192 は「Chroma を初期採用」と同時に「**ストア層も抽象を挟んで差し替え可能にする**」と書いている。**要件の本体は抽象の方**であり、Chroma は充足手段の一つである。よって `VectorStore` Protocol を先に確定し、`JsonlVectorStore` で始める（→ D-26）。

**実測済み（上記）。`uv add --dry-run` はこの uv 版に存在しないフラグのため、`uv pip compile` をプロジェクト外のディレクトリで実行して測定した。再測定する場合も同じ方法を使い、プロジェクトの `uv.lock` に触れないこと。**

#### 論点3: 差分更新の判定方法 → **index ディレクトリ直下の `manifest.json`**（3b で実装）

- 判定は `(st_mtime_ns, st_size)` を**高速経路**に使い、**真偽の決定はファイルバイト列の sha256** が行う（→ D-29）
- マニフェストをストアに持たせない（ストアは差し替え可能でなければならないため）
- マニフェストは `index_fingerprint` を持ち、**不一致なら差分更新を行わず全再構築**（→ D-28）

#### 論点4: チャンク分割とトークン数の数え方 → **文字種別ベースの決定論的近似**

| 案 | 判定 | 根拠 |
|---|---|---|
| `tiktoken` | 却下 | 新規依存。かつ **BPE が cl100k であり ruri-v3（ModernBERT-ja 系）とも Qwen とも別のトークナイザ**なので、正確さも得られない |
| Ollama に数えさせる | 却下 | チャンク境界の決定にネットワーク往復が要り、D-02 と両立しない。`--dry-run` が HTTP 0回で計画を出せなくなる |
| **文字種別ベースの近似** | **採用** | 要件書 L185 が「上限 240 トークン**目安**（設定可能）」と書いており、要求されているのは決定論的で調整可能な上限であって正確なトークン数ではない。純関数なので CI で全経路を検証できる |

`estimate_tokens(text) = ceil(cjk_chars / cjk_chars_per_token + other_chars / ascii_chars_per_token)`（既定 `cjk_chars_per_token=1.0` / `ascii_chars_per_token=4.0`）。係数は設定可能にし、`index_fingerprint` に含める（→ D-32）。**近似であることを関数名・設定キー名・docstring で明示する**。

分割規則:

1. コードフェンス（``` / ~~~）の外側で `^#{1,6}\s+` を見出しとして検出する。**フェンス内の `#` は見出しにしない**
2. 見出し単位でセクションに切る。`heading_path` は **`(ノートタイトル, H1, H2, ...)`**。ノートタイトルは frontmatter の `title`、無ければファイル名の stem
3. セクションが上限を超える場合は空行（段落）境界 → 行境界 → 文字境界の順に分割し、`part_index` を振る。**どの経路でも本文を1文字も落とさない**
4. `embed_text = " > ".join(heading_path) + "\n\n" + body`。**上限は `embed_text` に対して適用する**（接頭辞が上限を静かに超過するのを防ぐ）。`body`（接頭辞なし）は別フィールドで保持する
5. 空文字列・空白のみのチャンクは生成しない（空ノートは0チャンク）

#### 論点5: vault への書き込みをコード上で保証する方法 → **3段構え**

| 層 | 検査 | テスト |
|---|---|---|
| 構造 | vault に触れるモジュールを **`rag/vault.py` 1つに閉じる** | `test_rag_layout.py::test_only_the_vault_module_reads_the_vault` |
| 静的 | `rag/vault.py` に書き込み API 呼び出しが**1つも無い**ことを AST で検査（`open(..., mode)` の w/a/x/+、`write_text` / `write_bytes` / `mkdir` / `touch` / `unlink` / `rename` / `replace` / `chmod` / `shutil.*` / `os.remove`） | `test_rag_vault.py::test_no_vault_module_calls_a_write_api` |
| 動的 | 実行の**前後で vault 配下の全エントリ**の `(相対パス, size, st_mtime_ns, st_mode, sha256)` を採取して完全一致を検査。**エントリの増減も見る**。`st_atime` は比較しない | `test_rag_vault.py::test_indexing_leaves_every_vault_file_byte_identical`（D-30 guard） |
| 設定 | `index_dir` が `vault_dir` 配下に解決される設定を `ConfigError` にする | `test_rag_settings.py::test_index_dir_inside_the_vault_is_rejected` |

補助として、vault ツリーから書き込み権限を落として（`0o555`）読み取りが完走することを見るテストを1本置く。**`os.geteuid() == 0` のときは `pytest.skip`**（root は権限を無視するため恒真になる）。

#### 論点6: 合成 vault の設計 → `vaults/sample/`（12ノート + `.obsidian/` + 添付 + `.trash/`）

**すべて公開可能な内容**（本テンプレートリポジトリの説明・料理・地名など、個人情報・機密・実在の人物を含まない日本語テキスト）。

| # | ファイル | 目的（対応する受け入れ条件・テスト） |
|---|---|---|
| 1 | `notes/project-alpha.md` | H1>H2>H3 の3階層 + 上限超過の長文セクション（L314 / 分割） |
| 2 | `notes/weekly-review.md` | `[[note\|alias]]` / `[[note]]` / `[[note#heading]]` / `[[#heading]]` を各1つ以上（**L312**） |
| 3 | `notes/frontmatter-rich.md` | frontmatter に `title` / `tags`（ブロックリスト）/ `aliases`（インラインリスト）/ 日付スカラ（**L313**） |
| 4 | `notes/frontmatter-broken.md` | 開始 `---` に対して閉じが無い frontmatter（D-31 guard） |
| 5 | `notes/horizontal-rule.md` | 本文冒頭が `---`（水平線）で frontmatter ではない |
| 6 | `notes/code-fence.md` | フェンス内に `# 見出しに見える行` と `[[link\|alias]]`（D-34 guard） |
| 7 | `notes/tags-and-links.md` | インライン `#タグ`、URL 中の `#fragment`、`![[image.png]]` 埋め込み |
| 8 | `notes/empty.md` | 完全に空（0チャンク） |
| 9 | `notes/whitespace-only.md` | 空白と改行のみ（0チャンク） |
| 10 | `notes/no-heading.md` | 見出しが1つも無い（`heading_path` はノートタイトルのみ） |
| 11 | `notes/日本語 ファイル名.md` | 空白と非 ASCII を含むパス（`chunk_id` / JSON 往復） |
| 12 | `notes/diagram.excalidraw.md` | **`.md` だが除外されるべきファイル**（既定 exclude glob） |
| — | `.obsidian/app.json`, `.obsidian/workspace.json` | **L311** |
| — | `attachments/pixel.png`（1×1 px, 約70B）, `attachments/note.pdf`（最小構造）, `attachments/board.canvas` | **L311** |
| — | `.trash/deleted.md` | 除外 |

設定 TOML は `vaults/sample.toml`（`vault_dir = "vaults/sample"` / `index_dir = "data/index/sample"`）。実 vault は `vaults/local.toml`（**`.gitignore` に追加**）に置くか `--vault` で渡す。

## 4. タスク分解（3a = T1〜T3）

**実行順序 T1 → T2 → T3 を厳守。** 各タスク完了時に `make ci` が緑になることを確認してから次へ進む。

### T1: L2 に埋め込みクライアントを足す（HTTP 基底の抽出を含む）— **L**

**`llmkit/client.py`（private のみ改変。公開シンボルは1つも変えない）**

- `_HttpChatClient` から chat 非依存の部分を private 基底 `_HttpEndpointClient` として抽出する: `httpx.Client` の所有権・`close`/`__enter__`/`__exit__`・`_headers`・`_send`・`_raise_for_error_status`・`_raise_translated_error` とその4分岐・`_decode_json`・`_raise_schema_error`
- `OutOfMemoryError` / `ContextLengthError` のメッセージが参照する `config.generation.*` は、基底では `_request_context() -> str` フック経由にし、chat 側の**文言を1文字も変えない**
- `_HttpChatClient` は `_HttpEndpointClient` を継承してボディ組み立てと応答パースだけを持つ

**`llmkit/embeddings.py`（新規）**

```
EmbeddingClient   (Protocol) : embed(texts: Sequence[str]) -> EmbeddingBatch
EmbeddingBatch    (frozen dataclass) : vectors, model, dimensions, latency_s
OpenAIEmbeddingClient (_HttpEndpointClient のサブクラス)
embeddings_url_for(base_url: str) -> str
create_embedding_client(config, *, http_client=None) -> EmbeddingClient
```

- エンドポイントは `base_url.rstrip("/") + "/embeddings"`（D-11 と同じ導出方針で**新しい設定キーを足さない**）。`runtime.kind` では分岐しない（Ollama にネイティブの埋め込み API はあるが、Phase 0 で `/v1/embeddings` の 768 次元応答を実測済みであり、D-10 が OpenAI 互換を退けた理由（`num_ctx` 無視）は埋め込みには存在しない）
- モデルは `config.active_profile().embedding` から解決する。`None` なら `ConfigError`（対処: `[profiles.<名前>] embedding = "ruri-v3-310m"`）。`resolve_model_spec(..., role="embedding")` の結果が `spec.role != "embedding"` なら `ConfigError`
- 応答は pydantic dataclass + `TypeAdapter` で厳格にパース（D-07）。**`len(data) != len(texts)` を `UpstreamError`**、`index` で昇順に整列してから返す、全ベクトルの次元が一致しない場合も `UpstreamError`
- 空文字列を含む入力は `ConfigError`
- `__all__` を宣言し、`llmkit/__init__.py` へ再エクスポート

**`tests/test_layout.py`（拡張のみ）**
- `_SUBMODULE_NAMES` に `"embeddings"` を追加
- **新設**: `test_every_llmkit_submodule_is_covered_by_the_union_check`（`harness` 側の同名検査と同型。今回の追加漏れの再発防止）

**受け入れ基準**
- **既存 488 テストが1件も修正なしで緑**。特に `tests/test_client_errors.py` / `test_client_native.py` / `test_client.py` のメッセージ検査が無修正
- MockTransport で: (a) 3件のテキスト → 3ベクトル・次元768・順序が入力順 (b) `data` が2件しか返らない → `UpstreamError`（本文がメッセージに含まれない） (c) `index` が逆順で返る → 入力順に整列される (d) 404 → `ModelNotFoundError`（`served_name` を含む） (e) 接続不能 → `RuntimeUnavailableError` (f) 200 + 本文 error → 既存の翻訳表に従う
- `embeddings_url_for("http://h:11434/v1") == "http://h:11434/v1/embeddings"`。`kind` を変えても URL が変わらないことを両 `kind` で固定
- `profiles.long_context`（`embedding` 無し）で `create_embedding_client` が `ConfigError` になり、メッセージに `embedding` と対処が入る
- `generation.model` を埋め込みモデルに指定しても `role` 検査で `ConfigError`
- AST で `llmkit/embeddings.py` に `nvidia-smi` / `subprocess` が無い
- `uv lock --check` が無変更

### T2: 合成 vault と読み取り専用ローダ・Obsidian パーサ — **L**

**`vaults/sample/`（新規、§3 論点6 の表のとおり）**

**`rag/settings.py`（新規）** — 入力は**単一 TOML**（D-18 と同じ方針）

| セクション | キー | 既定 | 備考 |
|---|---|---|---|
| `[vault]` | `id`（`[a-z0-9_-]+`）, `dir`, `include_globs`, `exclude_globs` | — / — / `["**/*.md"]` / `[".obsidian/**", ".trash/**", ".git/**", "**/*.excalidraw.md"]` | `dir` は設定ファイルからの相対 or 絶対 |
| `[index]` | `dir` | `data/index/<vault.id>` | **`vault.dir` 配下に解決されたら `ConfigError`** |
| `[chunk]` | `max_tokens`, `cjk_chars_per_token`, `ascii_chars_per_token`, `heading_separator` | 240 / 1.0 / 4.0 / `" > "` | すべて `index_fingerprint` に入る |
| `[embed]` | `batch_size` | 16 | **`index_fingerprint` に入れない**（ベクトルが変わらないため） |

- 例外は `llmkit.ConfigError` を再利用（新しい例外階層を作らない）

**`rag/vault.py`（新規）— vault に触れる唯一のモジュール**

- `VaultFile`（frozen dataclass）: `relpath: str`, `mtime_ns: int`, `size: int`
- `iter_vault_files(settings) -> Iterator[VaultFile]`: `include_globs` の**ホワイトリスト**で拾い、`exclude_globs` で落とす。**シンボリックリンクは辿らずスキップ**。`resolve()` した結果が vault ルート配下でないものはスキップ
- `read_note_bytes(settings, relpath) -> bytes` / `read_note_text(...) -> str`（UTF-8、`errors="strict"`）
- **書き込み API を1つも書かない**（§3 論点5）
- ログ・例外に**絶対パスとノート本文を出さない**

**`rag/parser.py`（新規）**

- `ParsedNote`（frozen dataclass）: `relpath`, `title`, `frontmatter`, `frontmatter_raw`, `tags`, `links`, `embeds`, `body`
- **frontmatter**: 1行目が厳密に `---` で始まり、以降に `---` の閉じがある場合のみ frontmatter とする。閉じが無ければ**全体を本文**として扱う。パースは YAML のサブセット。**解釈できない行があっても例外にせず、`frontmatter_raw` に全文を残す。本文には決して混ぜない**（→ D-31）
- **wikilink**（コードフェンス内を含め一律適用、→ D-34）:

| 記法 | 本文に残る表示テキスト | メタデータ |
|---|---|---|
| `[[note]]` | `note` | `links += ("note",)` |
| `[[note\|alias]]` | `alias` | `links += ("note",)` |
| `[[note#heading]]` | `note > heading` | `links += ("note",)` |
| `[[note#heading\|alias]]` | `alias` | `links += ("note",)` |
| `[[#heading]]` | `heading` | `links += (自ノート,)` |
| `![[image.png]]` | **除去（空文字列）** | `embeds += ("image.png",)` |
| `![[note]]` | `note` | `embeds += ("note",)` |

- **タグ**: frontmatter の `tags` + 本文中の `(?<![\w/#])#[\w一-龥ぁ-んァ-ヶー/-]+`。URL の `#fragment` と見出しの `# ` を除外。**本文からは除去しない**
- 見出し検出はフェンス状態を追跡する。**wikilink 解決はフェンス状態と無関係**

**受け入れ基準**
- 12ノートすべてがパースでき、例外が出ない
- **`[[` と `]]` が `body` に1つも残らない**（全ノートを走査。**L312** の直接検証）
- `frontmatter-rich.md` の `body` に `title:` / `tags:` / `---` が現れず、`frontmatter["tags"] == ("設計", "テンプレート")` のように取得できる（**L313**）
- `frontmatter-broken.md` で例外が出ず、挙動をテストで固定する（どちらを選んだかを §9 に記録）
- `code-fence.md`: フェンス内の `# 行` が見出しとして扱われず、フェンス内の `[[a|b]]` が `b` に解決される（D-34 guard）
- `iter_vault_files` の結果に `.obsidian/` / `attachments/` / `.trash/` / `*.excalidraw.md` が**1件も含まれない**（**L311**）。逆に `notes/*.md` が11件含まれる
- `exclude_globs` に1パターン足すと対象件数が減る（**E26**）
- シンボリックリンク（vault 外を指す）を作っても対象に含まれない
- vault 配下を `0o555` にしても読み取りが完走する（root ではスキップ）
- `rag/vault.py` に書き込み API 呼び出しが0件（AST、D-30 guard の静的側）
- ログ・例外文字列に vault の絶対パスとノート本文が現れない

### T3: 見出しベースのチャンク分割とトークン近似 — **M**

**`rag/chunker.py`（新規）**

- `estimate_tokens(text, *, cjk_chars_per_token, ascii_chars_per_token) -> int`（純関数）
- `Chunk`（frozen dataclass）: `chunk_id`, `relpath`, `ordinal`, `heading_path`, `part_index`, `body`, `embed_text`, `estimated_tokens`, `tags`, `links`
- `chunk_id = f"{relpath}#{ordinal:04d}"`（決定論的・可読・再実行で安定）
- `chunk_note(parsed, settings) -> tuple[Chunk, ...]`（§3 論点4 の規則1〜5）

**受け入れ基準**
- 全チャンクで `estimate_tokens(embed_text) <= max_tokens`（`vaults/sample/` 全件）。**単一の分割不能単位が上限を超える場合の扱い**を1本のテストで固定（文字境界で切る）
- **本文が1文字も失われない**: 1ノートのチャンクの `body` を順に連結すると、空白正規化のうえで元のセクション本文と一致する（**最重要の不変条件**）
- `project-alpha.md` の深い節で `heading_path == ("プロジェクトAlpha", "設計", "データモデル")`（**L314**）、`embed_text` がその接頭辞で始まり、`body` は接頭辞を含まない
- `no-heading.md` の `heading_path` はノートタイトルのみの長さ1
- `empty.md` / `whitespace-only.md` は0チャンク。**どのチャンクの `embed_text` も空でない**
- `max_tokens` を 240→80 に変えるとチャンク総数が**増える**（**E24**）
- `cjk_chars_per_token` を 1.0→2.0 に変えると `estimate_tokens` の値と少なくとも1ノートのチャンク境界が変わる（**E25**）
- `chunk_id` が同一入力の2回実行で完全一致し、vault 全体で一意

## 5. 評価軸（Check フェーズへ）

### 機能観点（3a）
要件書 L311 / L312 / L313 / L314 / L319 の5条件。すべて決定論的に測る。L309 / L310 は 3b。

### 性能観点

| 指標 | 期待値 |
|---|---|
| `uv run pytest` 全体 | **20秒未満**（現行の実測を T1 着手前に記録し、+8秒以内。超えたら報告） |
| 合成 vault のパース + チャンク | 1秒未満 |

### 安全性観点

- `rag/vault.py` に書き込み API 呼び出しが0件（AST）
- `rag/` に HTTP 呼び出しとエンドポイント文字列が0件（AST）
- `llmkit/` に `nvidia-smi` / `subprocess` が0件（既存の D-01 検査が無修正で緑）
- **ログ・例外メッセージにノート本文が1文字も現れず、vault の絶対パスも現れない**（CLAUDE.md ログ出力ルール）
- コミット対象ファイルに実 vault のパス（`/home/`）と個人ノート本文が0件
- `index_dir` が vault 配下に解決される設定が `ConfigError`

### ★ 有効性観点（既存 E1〜E23 を維持したうえで）

| # | 掃引する値 | 変わるべき出力 | テスト | サイクル |
|---|---|---|---|---|
| **E24** | `chunk.max_tokens`（240→80） | チャンク総数 | `test_rag_chunker.py::test_max_tokens_changes_the_chunk_boundaries` | 3a |
| **E25** | `cjk_chars_per_token`（1.0→2.0） | `estimate_tokens` の値 / 少なくとも1ノートのチャンク境界 | `test_rag_chunker.py::test_token_estimate_is_deterministic_and_configurable` | 3a |
| **E26** | `exclude_globs` に1パターン追加 | 索引対象ノート数 | `test_rag_vault.py::test_exclude_globs_change_the_selected_files` | 3a |
| **E27** | `profiles[active].embedding` | リクエストの `model` | `test_embeddings.py::test_embedding_model_comes_from_the_active_profile_only` | 3a |
| **E30** | `vault.dir` を別ディレクトリへ | 索引されるノート集合が変わる（**パスがハードコードされていないことの検証**） | `test_rag_settings.py::test_vault_dir_is_the_only_source_of_the_vault_location` | 3a |
| **E28** | `embed.batch_size`（4→16） | HTTP 回数**のみ**。索引の内容・順序・ベクトルは**完全一致** | `test_rag_indexer.py::test_batch_size_changes_request_count_but_not_the_index` | 3b |
| **E29** | 1ノートの本文を編集 | そのノートのチャンクだけ再埋め込み | `test_rag_indexer.py::test_only_the_edited_note_is_reembedded` | 3b |

### 変異検証（3a で必須）

1. 埋め込み応答の `index` 昇順整列を外す → 整列テストが落ちる
2. wikilink 解決をコードフェンス内で止める → D-34 guard と **L312** が落ちる
3. frontmatter の閉じ `---` 判定を緩める → **L313** が落ちる
4. チャンカの上限適用を `body` に対して行う（`embed_text` ではなく） → 上限テストが落ちる
5. `rag/vault.py` に `Path.write_text` を1行足す → D-30 の静的 guard が落ちる
6. `_HttpEndpointClient` の例外翻訳の1分岐を削る → 既存の `test_client_errors.py` が落ちる

## 6. 意図的な決定（`.claude/decisions.yaml` に D-25 以降で追記。3a は D-25 / D-27 / D-30 / D-31 / D-32 / D-34）

（内容は `.claude/decisions.yaml` を唯一の出典とする。実装者は guard_test が実在してから追記すること）

## 7. 想定リスク（これが起きたら止まって相談）

1. **`llmkit/client.py` の HTTP 基底抽出が既存 488 テストに波及する。**
   `_HttpChatClient` は例外メッセージの文言まで既存テストに固定されている（`test_client_errors.py` が `served_name` / `context_tokens` / `max_context_tokens` を含む文面を検査）。抽出でメッセージ・ログ順・例外送出順が1つでも変わると落ちる。**既存テストを1行でも修正したくなったら、その時点で止めて相談する**（抽出方法が誤っているサイン。Phase 2 の §7 リスク3 と同型で、あのときは分割が正しく既存テストは無修正で通った）。回避案として「埋め込み専用に薄い HTTP 経路を書く」は**採らない**（D-25 の根拠が消える）。

2. **実 vault の初回索引が失敗するか、埋め込み次元が 768 以外で返る**（3b）。
   Phase 0 の実測は短文1件であり、240 トークン相当のチャンク・バッチ16件・34ノート分の連続実行は未検証。`max_context_tokens=8192` は未実測で、長いチャンクが黙って切り詰められる可能性がある。**次元が 768 以外、`data` 件数の不一致、OOM、または 60 秒を超える所要時間が出たら止めて相談する**。

3. **実 vault のノート本文・パスがログ・例外・コミットに漏れる。**
   vault の絶対パスは `/home/<user>/...` を含み、CLAUDE.md のログ出力ルール（氏名を出さない）に抵触する。索引処理は本文を全量メモリに載せるため、例外メッセージに載せる実装を書くと即座に漏れる。**コミット前に `git status` と差分で索引成果物・実 vault 由来の文字列が0件であることを確認し、1件でも見つかったら止めて相談する**（不可逆な公開事故）。

## 8. ファイル構成

```
rag/                        # L3。トップレベル (D-06 / harness と同じフラットレイアウト)
├── __init__.py             # 公開 API の再エクスポート + __all__ (cli は含めない)
├── settings.py             # RagSettings / load_settings   ← 入力は単一 TOML (D-18 と同方針)
├── vault.py                # VaultFile / iter_vault_files / read_note_*  ← vault に触れる唯一のモジュール (D-30)
├── parser.py               # ParsedNote / frontmatter / wikilink / tags / 見出し (D-31 / D-34)
├── chunker.py              # Chunk / chunk_note / estimate_tokens (D-32)
├── store.py                # [3b] VectorStore(Protocol) / JsonlVectorStore (D-26)
├── indexer.py              # [3b] index_fingerprint / plan_index / build_index (D-28 / D-29)
└── cli.py                  # [3b] python -m rag.cli index|status [--dry-run]

llmkit/embeddings.py        # ★ L2 への追加 (D-25)。EmbeddingClient / create_embedding_client
llmkit/client.py            # private 基底 _HttpEndpointClient の抽出のみ (公開シンボル不変)

vaults/sample.toml          # 合成 vault の設定 (コミット)
vaults/sample/              # 合成 vault 本体 (12 ノート + .obsidian/ + attachments/ + .trash/)
vaults/local.toml           # [3b] 実 vault の設定 (.gitignore に追加。コミットしない)
data/index/<vault_id>/      # [3b] 索引成果物 (.gitignore 済み。コミットしない、D-33)
```

**`rag/` を `llmkit/rag/` にしない理由**: 「L3 は `llmkit` の公開シンボルのみを使う」は **L3 が別パッケージであって初めて機械検証できる**（Phase 2 §8 と同じ）。同一パッケージ内では `from llmkit.client import _HttpEndpointClient` が構文上いつでも書けてしまい、境界が散文の主張に戻る。

**`tests/` への波及**

| ファイル | 変更 |
|---|---|
| `tests/test_layout.py` | `_SUBMODULE_NAMES` に `"embeddings"` を**追加** / **新設** `test_every_llmkit_submodule_is_covered_by_the_union_check` |
| `tests/test_harness_layout.py` | **変更なし** |
| `tests/conftest.py` | 埋め込み応答のペイロード定数と、`vaults/sample/` を `tmp_path` へコピーする fixture を追加（既存 fixture は変更しない） |
| 新規 (3a) | `test_embeddings.py` / `test_rag_layout.py` / `test_rag_settings.py` / `test_rag_vault.py` / `test_rag_parser.py` / `test_rag_chunker.py` |
| 新規 (3b) | `test_rag_store.py` / `test_rag_indexer.py` / `test_rag_cli.py` / `test_rag_privacy.py` / `test_acceptance_phase3.py` |

**`pyproject.toml` は変更不要**: `pythonpath=["."]` で `import rag` が解決し、mypy の exclude は `.venv` のみなので `rag/` は自動的に strict の対象。新規依存が無いため `uv.lock` も無変更。

## 9. 実装時に決めたこと（実装者が追記する節）

> T1 以降の実装者は、仕様書に書かれていなかった選択をここに追記すること。次の周の reviewer / fixer が読むのはこの節であり、実装コードのコメントではない。

### T1（`llmkit/client.py` の基底抽出 / `llmkit/embeddings.py`、2026-08-23）

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | `_HttpEndpointClient.__init__(config, spec, *, http_client=None)` とし、**`ModelSpec` を第2位置引数で受け取る**（基底は解決方法を知らない） | chat は `generation.model`、埋め込みは `profiles[active].embedding` と、モデル ID の出典が違う。基底に `AppConfig` だけを渡して解決させると、基底が両方の設定キーを知ることになり D-27（出典を1か所に限る）が基底側で崩れる。`_HttpChatClient.__init__` の公開シグネチャ `(config, *, http_client=None)` は不変 |
| 2 | `ContextLengthError` の分岐は、**判定条件・マーカー表・判定順序を基底に残し、文面だけ `_context_length_message(status)` / `_context_length_remediation()` の2フックに切り出した**。埋め込み側はこの2つだけを上書きする | §4 T1 は「`config.generation.*` を `_request_context()` フック経由にする」と指示しているが、`_request_context()` 1つでは埋め込み側の文面が成立しない。基底の文型は「要求したコンテキスト長 **{要求量}** がモデル X の上限 … を超えています」であり、埋め込み要求には対応する要求量（`generation.context_tokens` に相当する設定値）が存在しない。ここに `embedding=<id>` のような非数量を差し込むと非文になり、`max_context_tokens` を差し込むと「要求 8192 が上限 8192 を超えた」という**実測していない要求量を主張する偽のメッセージ**になる。文面だけをフック化すれば、翻訳表の実装は 1 つのまま（D-25 の根拠が消えない）で、chat 側の文面はバイト単位で不変（`test_client_errors.py` 無修正で緑）。変異検証で 404 分岐を潰すと `test_client_errors.py` と `test_embeddings.py` の**両方**が落ちることを確認済み |
| 3 | `_request_context()` は基底の `@abstractmethod` にし、既定値を与えない。chat は `context_tokens=<値>`、埋め込みは `embedding=<model_id>` を返す | 既定を置くと、将来追加される経路が黙って chat の文言（`context_tokens=…`）を名乗る。D-15 が `serving_runtime` に既定値を与えなかったのと同じ判断 |
| 4 | HTTP 200 + 本文 `error` の処理を基底の `_raise_if_body_reports_error(response)` に切り出し、`OllamaNativeClient` と `OpenAIEmbeddingClient` の双方が `_raise_for_error_status` の中から呼ぶ | 受け入れ基準 (f) を満たすには埋め込みでも同じ処理が要る。3行を複製すると、Ollama が 200 でエラーを返す経路の扱いが 2 か所に分岐する。`OllamaNativeClient` の振る舞いは不変 |
| 5 | 埋め込み応答の検査を**件数 → `index` の並び → 次元の一致 → 次元 0** の 4 段にした（仕様書は件数・整列・次元一致の 3 つを指示） | (i) `index` が重複していると件数が合っていても入力と 1 対 1 に対応せず、整列だけでは取り違えたベクトルを黙って返す。整列後に `(0..n-1)` と一致することを確かめる。(ii) 次元 0 のベクトルは「埋め込めた」ことにしない。どちらも例外もテスト失敗も出さずに索引だけが壊れる型の欠陥（§2 ★ と同型） |
| 6 | 入力が **0 件**の場合も `ConfigError`（空文字列と同じ扱い）。HTTP は発行しない | 仕様書は「空文字列を含む入力 → `ConfigError`」だけを規定している。0 件の要求はランタイム側の挙動が定義されておらず、`EmbeddingBatch.dimensions` に入れる値も無い（0 を入れると D-07 の「欠測を 0 で埋めない」に反する）。3b の `embed.batch_size` 実装は空バッチを作らない側で防ぐ |
| 7 | 応答の `model` を**必須**フィールドにした（`data` と合わせて 2 つだけ必須。`object` / `usage` は無視） | `_ChatCompletion` が `model` を必須にしているのと同じ厳格さ（D-07）。`EmbeddingBatch.model` は設定値ではなくランタイムが名乗った名前を持つ。設定値でフォールバックすると「どのモデルが実際に埋め込んだか」が検証不能になる |
| 8 | `create_embedding_client` に `match` + `assert_never` を置かない（`create_chat_client` と非対称） | 埋め込みは `runtime.kind` で分岐しない（§4 T1）。分岐の無い場所に網羅チェックを書くと「ここは将来分岐する」という誤ったシグナルになる |
| 9 | E27 のテストは**外部 API 設定（`is_local = false`）で掃引する** | カタログの `role="embedding"` は `ruri-v3-310m` の 1 件だけで、ローカル実行では別の値に振れない（未登録 ID は `ConfigError`）。`is_local=false` の passthrough なら `embedding` の値をそのまま `model` として送るため、「`profiles[active].embedding` を変えるとリクエストの `model` が変わる / `generation.model` を変えても変わらない」を実測で示せる |
| 10 | `llmkit/embeddings.py` の `nvidia-smi` / `subprocess` 非混入の AST 検査は、**`tests/test_embeddings.py` にモジュール限定のものを1本置いた**（既存 `test_harness_gpu.py::test_llmkit_never_touches_the_gpu_or_spawns_processes` が `llmkit/*.py` を glob しており重複はする） | 受け入れ基準が本モジュールを名指ししているため、モジュール名で追える形を残した。既存テストは無修正 |
| 11 | 実測（変更前 488 passed / 1.19s → 変更後 528 passed / 1.26s）。**+0.07 秒**で §5 の許容（+8 秒以内）を大きく下回る | — |

**T2 以降への申し送り**: `.claude/decisions.yaml` への D-25 / D-27 の追記は行っていない（guard_test が実在してから T3 完了時にまとめて追記する指示に従った）。D-27 の guard_test として参照するテスト関数名は `tests/test_embeddings.py::test_embedding_model_comes_from_the_active_profile_only`、D-25 側は `tests/test_embeddings.py::test_missing_model_maps_to_model_not_found`（基底の翻訳表を chat と共有していることの検出点）が使える。

### T2（`vaults/sample/` / `rag/{settings,vault,parser}.py`、2026-08-23）

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | `vaults/sample.toml` の値は **`dir = "sample"` / `index.dir = "../data/index/sample"`**（§3 論点6 の表は `vaults/sample` / `data/index/sample` と書いている） | パス解決を「設定ファイルからの相対」に確定した以上、`vaults/sample.toml` に書ける値はこの 2 つになる（カレント基準にすると、同じ設定ファイルが起動場所によって別の vault を指す）。索引先を `../data/...` にしたのは、`.gitignore` 済みの**リポジトリ直下の `data/`** に出すため（D-33 の前提）。`data/index/sample`（設定ファイル相対）だと `vaults/data/` が生まれ、ignore されないまま索引成果物が置かれる |
| 2 | `notes/horizontal-rule.md` の 1 行目を **`----`（4 本線）**にした（表は「本文冒頭が `---`」） | 1 行目が厳密に `---` で以降に閉じ `---` があるものは、Obsidian でも本実装でも frontmatter である。「本文冒頭が `---` で frontmatter ではない」ノートは原理的に作れない（作れば D-31 と矛盾する）。表の意図は「冒頭の水平線を frontmatter と誤認しない」ことなので、**厳密一致の検査**（`----` は不一致）＋本文中の `---` 水平線という構成で満たした |
| 3 | **`frontmatter-broken.md`（閉じ `---` 無し）は「ファイル全体を本文」**とし、`frontmatter_raw` は空文字列にした | §4 T2 の規定（「閉じが無ければ全体を本文」）どおり。もう一方（閉じが無くても frontmatter とみなす）を選ぶと、書きかけのノートや `---` 冒頭の水平線が本文から黙って消える。変異検証3 でこの分岐を潰すと D-31 guard が落ちることを確認済み |
| 4 | パーサは**ファイルを読まない**。入口は `parse_note(relpath, text)` で、vault の読み取りは `rag/vault.py` だけが行う | §3 論点5 の「構造」層の帰結。パーサが読むと、`test_rag_layout.py::test_only_the_vault_module_reads_the_vault` が成立しない。vault 全体を読んでパースする糊は当面テスト側の `parse_whole_vault` に置き、3b で `rag/indexer.py` に移す |
| 5 | D-25 guard は `.post` / `.request` に加えて **`.get()` も一律に落とす**（`Mapping.get` も落ちる）。`rag/` では `in` + 添字で書く | 仕様書 §3 論点1 の指示どおり `.get` を検査対象にすると `dict.get` が誤検出になる。誤検出を許して検査を鈍らせる（受け手を名前で絞る等）より、書き方を 1 つに固定するほうが境界が強い。**ruff の SIM401 が `.get` を勧めてくるため、三項演算子ではなく `if` 文で書く必要がある**（3b への申し送り） |
| 6 | glob の意味づけを `rag/vault.py` に自前実装した（`**` は 0 個以上の階層 / `*` は `/` を跨がない / 末尾 `/**` はそのディレクトリ自身にも一致）。**枝刈りの対象は末尾が `/**` のパターンと `**` そのものに限る**（round-9 fixer で修正、下記参照） | `PurePath.full_match` は Python 3.13 以降、`fnmatch` は `*` が `/` を跨ぐ、`Path.match` は `**` を再帰として扱わない。**include と exclude が同じ実装を通ること**が要件（意味がずれると「除外したつもりのものが索引される」）。当初「結果は枝刈りの有無で変わらない」と書いていたが誤りだった（F-9-002）。`notes/*` や `**/2024` のように末尾が `/**` でないパターンはディレクトリ自身に一致しても配下のファイルには一致しないことがあり、枝刈りに使うとファイル単位では除外対象でないファイルまで消える。末尾 `/**` / `**` に限れば、一致した時点で配下すべてに確実に一致するため等価性が成立する |
| 7 | 走査は `os.walk(root, followlinks=False)` **1 回**で、include / exclude の両方を同じ matcher で判定する（`Path.glob` を include に使わない） | `Path.glob` を include に使うと glob の意味が 2 種類になる（上記 6）。1 回の走査に閉じるとシンボリックリンクの扱いも 1 か所で決まる |
| 8 | `[[#heading]]` の `links` に入れる「自ノート」は**ファイル名の stem**（`title` ではない） | 他の link 先はすべてファイル名であり、参照先の名前空間を 1 つにそろえないと 3b のリンク解決で突き合わせられない |
| 9 | `![[...]]` が「ノートか添付か」は**拡張子**で判定する（suffix が空 or `.md` ならノート扱い＝本文に表示テキストを残す、それ以外は本文から除去） | 仕様書の表は `![[image.png]]` → 除去、`![[note]]` → `note` と結果だけを示していて、両者を分ける規則が書かれていない。拡張子は Obsidian の添付判定と同じ観点で、決定論的に決まる |
| 10 | `links` / `embeds` / `tags` は**出現順で重複除去**する | 同じノートに同じリンクが複数回現れるのは普通で、出現回数を持つ意味が無い。順序を保つのは JSON 往復とテストの決定性のため |
| 11 | frontmatter のスカラは**文字列のまま**持つ（`created: 2026-08-23` → `"2026-08-23"`、`draft: false` → `"false"`） | 型変換を始めると YAML の全機能を自前で持つ方向に引きずられる（PyYAML を入れないという前提が崩れる）。日付・真偽値を使う側が必要になった時点で、使う側で解釈する |
| 12 | frontmatter を切り離したあとの本文は、**改行を含めて 1 文字も加工しない**（`splitlines()` + `join` をやめ、閉じ行の直後からのオフセットで切る） | 最初の実装は末尾の改行を落としていた。T3 の最重要不変条件「本文が 1 文字も失われない」は、その前段であるパーサが原文を保っていて初めて意味を持つ |
| 13 | `read_note_bytes` / `read_note_text` は、絶対パス・`..`・シンボリックリンク経由の参照を `ConfigError` にする | `iter_vault_files` を経由しない呼び出し（3b の差分更新はマニフェストの相対パスから直接読む）でも同じ保証が要る。列挙側だけで守ると、読み取り側に穴が残る |
| 14 | 例外・ログに載せるのは**相対パスと件数だけ**。`UnicodeDecodeError` は位置のみ転記し例外文字列を載せない。`rag/settings.py` の例外は**設定ファイルに書かれたままの文字列**を載せ、解決後の絶対パスを載せない | CLAUDE.md のログ出力ルール。実 vault の絶対パスは利用者名を含み、本文は個人情報そのものになり得る。`test_rag_vault.py::test_logs_never_contain_absolute_paths_or_note_text` が実測で固定する |
| 15 | **`tests/test_layout.py` の D-08 走査範囲に `rag/` を足す（§2 の指示）を行わず**、同型の検査を `tests/test_rag_layout.py::test_rag_schemas_use_pydantic_dataclasses_not_basemodel` として新設した | 本タスクのハード制約「既存 529 テストを 1 行も修正しない」が優先。検査内容は等価（`rag/*.py` の `BaseModel` 継承 0 件）。**3b で統合するか新設側を維持するかを決めること**（現状は 2 ファイルに同型の検査が並ぶ） |
| 16 | `tests/conftest.py` に `SAMPLE_VAULT_DIR` / `SAMPLE_VAULT_CONFIG` / `write_rag_settings()` / `sample_vault_copy` fixture を**追加**した（既存 fixture は 1 行も変更していない） | §8 の「`vaults/sample/` を `tmp_path` へコピーする fixture を追加」に従った。権限変更・シンボリックリンク追加・除外パターン掃引はすべて複製に対して行うため、コミット済みの合成 vault は決して変更されない |
| 17 | `vault.include_globs` が**空配列なら `ConfigError`** にした（仕様書は既定値しか規定していない） | ホワイトリストが空なら索引対象が 0 件になる。「設定は通ったが何も索引されない」は設定ミスとしか解釈できず、静かに空の索引を作るほうが害が大きい |
| 18 | `RagSettings` に `source_path`（読み込んだ TOML のパス）を持たせた | 3b の `index_fingerprint` と CLI のエラーメッセージが「どの設定で索引したか」を示すために要る。解決後の絶対パスを他から再構成させないため |
| 19 | `Heading` / `iter_headings` を **`rag/parser.py` の公開 API** にした | 「見出し検出はフェンス状態を追跡する」はパーサ側の責務として §4 T2 に書かれているが、`ParsedNote` に見出しのフィールドが無い。T3 の chunker が使う入口としてここに置いた。D-34 guard が「フェンス内は見出しにしない」を検査する対象でもある |
| 20 | `project-alpha.md` は **H1 をノートタイトルと同じ「プロジェクトAlpha」**にした | §4 T3 の受け入れ基準が `heading_path == ("プロジェクトAlpha", "設計", "データモデル")` の長さ 3 を求めているのに対し、`heading_path = (ノートタイトル, H1, H2, ...)` で H1>H2>H3 を素直に辿ると長さ 4 になる。H1 がタイトルと一致する構成にしておけば、T3 は「タイトルと同じ H1 を重複させない」規則で受け入れ基準を満たせる。**T3 実装者はこの重複除去を明示的に決めて §9 に記録すること** |
| 21 | 実測: 変更前 529 passed / 1.27s → 変更後 **600 passed / 1.48〜1.50s**（+0.23 秒、§5 の許容 +8 秒以内）。`iter_vault_files` は 11 件、全ノートの `body` に残った `[[` は 0 件 | — |

**T3 への申し送り**: `.claude/decisions.yaml` への追記は行っていない（guard_test が実在してから T3 完了時にまとめる指示に従った）。guard_test として参照する関数名は D-30 静的 = `tests/test_rag_vault.py::test_no_vault_module_calls_a_write_api`、D-30 動的 = `tests/test_rag_vault.py::test_indexing_leaves_every_vault_file_byte_identical`、D-31 = `tests/test_rag_parser.py::test_unparsable_frontmatter_never_leaks_into_the_body`、D-34 = `tests/test_rag_parser.py::test_wikilinks_inside_code_fences_are_resolved_but_headings_are_not`、D-25 = `tests/test_rag_layout.py::test_rag_never_talks_to_the_runtime_directly` で、いずれも実在する。D-30 の動的側は現時点では「vault 全体をパースする処理」の前後比較であり、3b で索引本体に差し替える。

### T3（`rag/chunker.py`、2026-08-23）

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | **`heading_path` は「直前の要素と同じ文字列」だけを畳む**（T2 申し送り1 への判断）。`project-alpha.md` は H1 がノートタイトルと同一なので `("プロジェクトAlpha", "設計", "データモデル")` の長さ 3 になる。離れた位置の同名見出し（`設計 > 概要 > 設計`）は畳まない | 除去する側を採った。(i) §4 T3 の受け入れ基準が長さ 3 を求めており、除去しないと原理的に満たせない。(ii) Obsidian では H1 をタイトルと同じにする書き方が一般的で、`プロジェクトAlpha > プロジェクトAlpha > 設計` は情報が 1 つも増えないのに `embed_text` の接頭辞だけが伸び、**上限が `embed_text` に適用される以上その分だけ本文に使える枠が削られる**（D-32 と直結する実害がある）。(iii) 「タイトルと H1 だけ」ではなく「連続する重複」という一般規則にしたのは、H2 が親 H1 と同名の場合にも同じ実害が出るため。非連続の重複を畳むと `設計 > 概要 > 設計` の最後の 1 つが消えて別の節を指す経路になるので畳まない。`test_the_note_title_is_never_repeated_in_the_heading_path` と `test_a_repeated_heading_further_down_the_path_is_kept` の 2 本で両方向を固定した |
| 2 | `chunk_note(parsed, settings)` の `settings` は **`ChunkSettings`**（`RagSettings` ではない） | チャンカは `vault_dir` / `index_dir` / glob を 1 つも使わない。`RagSettings` を渡すと「チャンカも vault を知っている」という誤ったシグナルになり、§3 論点5 の構造層（vault に触れるのは `rag/vault.py` だけ）と読み手の理解が食い違う。呼び出し側は `settings.chunk` を渡すだけで、E24 / E25 の掃引も `dataclasses.replace(settings.chunk, ...)` で書ける |
| 3 | **見出し行そのものはどのチャンクの `body` にも入れない**（`heading_path` にだけ入る） | 見出しは `embed_text` の接頭辞として必ず載るため、`body` にも残すと同じ文字列を 2 回埋め込むことになる。最重要の不変条件（本文の復元）の期待値も「本文から見出し行を除いたもの」として定義し、テスト側で `rag.iter_headings` から独立に組み立てて突き合わせている |
| 4 | 各チャンクの `body` は前後の空白を落とす（`strip()`）。空白のみになった断片はチャンクにしない | 規則5（空文字列・空白のみのチャンクを作らない）の帰結。段落境界の空行や行末の改行が先頭・末尾に残ると、埋め込みの入力に意味の無い空白が乗る。復元の不変条件は空白正規化のうえで見るため、この加工で本文が失われることはない |
| 5 | **見出し経路だけで `max_tokens` を使い切る場合は `ConfigError`**（`chunk_note` が送出。メッセージは相対パスと段数・トークン数のみで、見出し文字列は載せない） | 仕様書は「単一の分割不能単位が上限を超える場合は文字境界で切る」までしか決めていない。接頭辞が上限を食い切っている状態で本文を 1 文字も落とさずに上限も守ろうとすると、出力は「1 文字ずつのチャンクの山」になる。索引としては壊れているので、静かに作らず設定の誤りとして落とす。見出し文字列はノート本文の一部なので例外にもログにも出さない（CLAUDE.md ログ出力ルール、T2 決定14 と同じ扱い）。1 ノート単位の失敗なので、3b の索引側は「記録して次のノートへ」（§3 ソフト制約）で扱える |
| 6 | それでも 1 文字が上限に収まらない場合（接頭辞の検査を通ったうえで端数が乗る場合）は、**捨てずにそのまま 1 チャンクにする** | 本文を落とすのは無音のデータ欠落で、上限超過は埋め込み側の切り詰めで済む。害の大きさが違う |
| 7 | `Chunk.estimated_tokens` は **`embed_text` の近似値**（`body` ではない） | 上限判定に使った数そのものを持たせる。`body` 基準の数を持つと、記録された値と上限の関係が読み手に分からなくなる。`test_estimated_tokens_is_the_number_used_for_the_limit` が固定 |
| 8 | `tags` / `links` は**ノート単位の値をそのまま全チャンクに載せる**（チャンクごとに切り分けない） | `ParsedNote` はタグ・リンクの出現位置を持たない（T2 決定10 で出現順の重複除去のみ）。位置を持たせるにはパーサの公開形を変える必要があり、T3 のハード制約（`rag/parser.py` を変更しない）に反する。3b の絞り込みは「そのノートに付いたタグ」で行えば足りる |
| 9 | `ordinal` はノート内の通し番号（`chunk_id` の元）、`part_index` は**セクション内**の通し番号。どちらも 0 始まりで、空白のみの断片を捨てたあとに連番を振る | 両方をノート通しにすると `part_index` が「何番目の分割か」を表さなくなる。捨てる前に番号を振ると欠番が出て、再実行時の一致検査（`chunk_id` の安定性）が読みにくくなる |
| 10 | CJK として数える符号位置の表（約物 / ひらがな / カタカナ / 統合漢字と拡張 A・B / 互換漢字 / 全角形と半角カナ / ハングル）を `rag/chunker.py` の定数に置いた | 仕様書は「CJK 文字数」としか書いていない。表そのものが近似の定義であり、変えると同じノートから別のチャンクが出るため、3b では `index_fingerprint` の入力に含める前提を docstring に書いた |
| 11 | **`tests/test_rag_layout.py` の `_SUBMODULE_NAMES` に `"chunker"` を 1 行追加した**（本タスクのハード制約「既存 600 テストを 1 行も修正しない」に対する唯一の例外） | `test_every_rag_submodule_is_covered_by_the_union_check` が `rag/*.py` の集合と `_SUBMODULE_NAMES` の**完全一致**を検査しており、`rag/chunker.py` を置いた時点で（再エクスポートの有無に関わらず）必ず落ちる。テスト自身の docstring が「サブモジュールを足したのに入れ忘れると落ちる」と書いており、**この 1 行は検査を弱めるのではなく `chunker` の `__all__` を和集合検査の対象に加える拡張**である（§4 T1 が `tests/test_layout.py` の `_SUBMODULE_NAMES` に `"embeddings"` を追加したのと同型）。アサーションは 1 つも緩めていない |
| 12 | 実測: 変更前 600 passed / 1.50s → 変更後 **626 passed / 2.20s**（+0.70 秒、§5 の許容 +8 秒以内）。合成 vault は `max_tokens=240` で 24 チャンク・上限超過 0 件・本文不一致 0 ノート、`max_tokens=80` で 32 チャンク（E24 の増加を実測） | — |

**3b への申し送り**: `.claude/decisions.yaml` に **D-25 / D-27 / D-30 / D-31 / D-32 / D-34 を追記済み**（計 30 件で `check_decisions.py` 緑）。3b は D-26 / D-28 / D-29 / D-33 を、guard_test が実在してから追記すること。`rag/chunker.py` は `.get()` を 1 つも使っていない（`headings` の索引も `in` + 添字）。3b で `index_fingerprint` に入れるべきチャンク側の入力は `ChunkSettings` の 4 キーに加えて **`_CJK_RANGES` の内容**（近似の定義そのもの）である。

### fixer (round-9、2026-08-23)

round-9 レビューの HIGH 6 件 (F-9-001 / F-9-002 / F-9-008 / F-9-013 / F-9-014 / F-9-028) を修正した。F-9-016 (`.gitignore`) はメインセッションが対応済みのため対象外。

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | F-9-001: `OpenAIEmbeddingClient._context_length_message` の**上書きを削除**し、基底 `_HttpEndpointClient._context_length_message` (MODEL_CATALOG 分岐を持つ既存の共有実装) にフォールバックさせた。新しいフック (`_context_length_subject` 等) は追加しない | 基底の `_context_length_message` は既に「カタログ登録済みモデルか passthrough か」で文面を分ける実装を持っており (F-2-002)、埋め込み側の上書きがこの分岐を握りつぶしていたのが F-9-001 の原因だった。上書きを消すだけで基底の分岐がそのまま埋め込み経路にも適用される。埋め込み固有の `_request_context()` (`embedding=<model_id>`) は変更していないため、埋め込みの OOM メッセージ (`test_out_of_memory_body_maps_to_out_of_memory_error` が `embedding=ruri-v3-310m` を固定) は無傷。結果としてサブクラスが上書きするフックは `_request_context` / `_context_length_remediation` / `_model_setting_reference` の3つに減った (レビュー観点5 の解消) |
| 2 | F-9-002: `_is_excluded_directory` の枝刈り対象を **末尾が `/**` のパターン、または `**` そのもの**に限定した (`_is_prunable`)。枝刈りをやめてファイル単位判定に一本化する案は採らなかった | 合成 vault (11 ノート) には枝刈りの性能上の理由が無いため一本化も選べたが、`.obsidian/**` 等の実 vault 運用で有効な最適化を無くすのは過剰な後退だと判断した。末尾 `/**` パターンは「一致した時点で配下すべてに確実に一致する」性質を持つため、この形に限れば枝刈りとファイル単位判定は数学的に等価になり、`notes/*` のような部分一致パターンで配下が丸ごと消える事故が起きない |
| 3 | F-9-008: D-30 / D-27 / D-32 の `rule` に、**guard_test が引いていない他の層・他の条件を別途固定しているテスト名を明記**した (guard_test フィールド自体はスキーマ上 1 件しか持てないため) | `check_decisions.py` の `guard_test` は単一の pytest node id しか検証できない。4 層 (D-30) や複数条件 (D-27 / D-32) のうち 1 つしか引けないなら、残りをどのテストが守っているかを rule に明記しないと、Stop hook (guard_test だけを実走) が守っていない層の劣化を検出できない (D-15 と同型の抜け) |
| 4 | F-9-013: `_resolve_wikilinks` の重複判定を list の `in` (線形探索) から `dict` の keys (O(1)) に変えた。公開 API (`ParsedNote.links` / `.embeds` の型・順序) は不変 | `_dedupe()` と同じ「dict で O(1) 判定」の方針に揃えた。実測: ユニークリンク数 8000 で 0.195s → 0.0064s (約30倍)。差分フォジングテスト (300 試行、旧実装との出力完全一致) で振る舞いが変わっていないことも確認した |
| 5 | F-9-014: 文字境界フォールバック (`_pack_characters`) を、`embed_text` 全体を毎回 `estimate_tokens` する方式から、**バッファの CJK 文字数・総文字数を差分更新する**方式に変えた。見出し経路由来の接頭辞の文字数は 1 回だけ数える | 改行の無い長い1行 (外れ値ノート) を字単位分割する経路が、バッファリセットのたびに 0 から数百文字を再走査していたのが原因。実測: 160,000 文字で 31.86s → 0.49s (約65倍)。300 試行の差分フォジング (ランダム CJK/ASCII 混在、境界値含む) で旧実装 (`_pack` + 文字列ベース `fits()`) と出力が完全一致することを確認した |
| 6 | 変異検証はすべて「1箇所を壊す→対応するテストが落ちる→復元する→`git status` クリーンを確認」の順で実施した。F-9-013 / F-9-014 は**性能のみの修正**であり、正しさを検査する既存テスト (`test_repeated_links_are_deduplicated_in_order` 等) は退行させても落ちない (呼び出し側の `_dedupe()` が最終的に正しさを保証するため)。この2件は実測時間の比較でのみ検出できる | 誠実に報告するため。「対応するテストが落ちる」という要求を機械的に満たすために不要な assertion を追加する (スコープ外の作り込み) より、性能改善は数値実測で示すほうが適切だと判断した |
| 7 | 実測: 変更前 636 passed / 約2.2s → 変更後 **641 passed / 約1.8s**（新規テスト5件を追加。うち F-9-001 用 1 件、F-9-002 用 3 件 (parametrize)、F-9-028 用 1 件） | F-9-013 / F-9-014 は性能のみの修正のためテスト件数は増えていない |

**申し送り**: F-9-011 (`rag/settings.py` の `source_path` 未正規化) と F-9-018 (`index_dir` が追跡対象を指せる) は今回のスコープ外 (MEDIUM 相当) だが、3b の `index_fingerprint` 実装に直結するため `docs/next-pr-candidates.md` に「次サイクルで必ず対応」と明記した。

### fixer (round-10、2026-08-23)

round-10 レビュー (reviewer-architecture) の MEDIUM 3 件を修正した。いずれもメインセッションが再現確認済みで、`fixer-input.json` は round-10 に存在しなかったため、プロンプト本文に直接記載された要点を出典とした。

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | `rag/chunker.py`: D-32 の近似式を `_tokens_from_counts(cjk_count, total_count, *, cjk_chars_per_token, ascii_chars_per_token)` に切り出した。`estimate_tokens` は `_count_chars` + `_tokens_from_counts` の合成、文字境界フォールバック (`_pack_characters` 内 `tokens_for`) は `_tokens_from_counts` の部分適用にした。F-9-014 の性能特性 (バッファの CJK/総文字数を差分更新し、文字列を毎回数え直さない) は変更していない | `tokens_for` が独自に `ceil(...)` 式を再実装しており、近似式が 2 か所に分岐していた (round-10 レビュー)。式を 1 関数に集約すれば、性能のための「文字列ではなくカウントで持つ」設計は保ったまま、式そのものは単一実装になる。差分更新の途中経過はカウント (int) であって文字列ではないため、`_tokens_from_counts` はカウントを受け取るシグネチャにした (テキストを受け取る `estimate_tokens` には合わせられない) |
| 2 | `rag/vault.py`: `_scan_vault` の走査中に、**末尾が `/**` でない (= 枝刈りに使われない) exclude パターンごとに「ディレクトリ一致の有無」と「ファイル一致件数」を集計**し、走査後に `_warn_about_exclude_globs_matching_no_file` で「ディレクトリには一致したがファイル単位では 0 件」のパターンだけを WARNING にした。末尾 `/**` のパターンは監視対象から外した | 末尾 `/**` のパターンは `_is_prunable` の等価性保証によりファイル単位判定と数学的に一致するため、枝刈りされたディレクトリの中身を歩かずには確認しようが無く、確認する必要も無い。逆に「ディレクトリにも一致しなかった」パターン (既定の `.git/**` など、その vault に単に存在しないパス) は設定ミスとは限らないため警告対象から除いた。これが無いと、既定値 (`DEFAULT_EXCLUDE_GLOBS` は `.git/**` を含むが合成 vault に `.git/` は無い) でも警告が出てしまい、「合成 vault の既定値では警告が出ない」という受け入れ条件を満たせなかった (実測で確認: 当初案は `.obsidian/**` / `.trash/**` の 2 件が偽陽性で警告された) |
| 3 | `rag/settings.py`: `DEFAULT_EXCLUDE_GLOBS` の直上コメントに「ディレクトリ全体を外すには `dir/**` と書く (書き忘れは `rag.vault` が WARNING ログで検出する)」を追記した | 修正方針の指示どおり。`exclude_globs` フィールド自体には個別の docstring が無いため、既定値の定数コメント (フィールドの説明として最も近い場所) に足した |
| 4 | `llmkit/client.py`: `_HttpEndpointClient._context_length_remediation` を `@abstractmethod` にし、既存の文面 (`generation.context_tokens を減らすか…`) をそのまま `_HttpChatClient._context_length_remediation` へ移した。`OpenAIEmbeddingClient` は round-9 (F-9-001) の時点で既に独自の `_context_length_remediation` を持っており、変更不要だった | 基底に既定値が残っていると、次にこの基底へ載る経路 (リランカー) が上書きを忘れても `ABC` も `mypy` も何も言わない。F-9-001 と同じ型の欠陥が対処メッセージ側で再発する経路を、実装を待たずに機械的に塞いだ |
| 5 | D-32 の `rule` に「近似式は `_tokens_from_counts` を唯一の実装とし、`estimate_tokens` と文字境界フォールバックはどちらもその合成/部分適用でなければならない (式を独立に複製しない)」を追記し、`tests/test_rag_chunker.py::test_character_boundary_fallback_shares_the_token_formula` を guard_test 以外の固定テストとして rule 本文に明記した | F-9-008 と同型の対応 (guard_test は 1 件しか持てないため、rule 本文に他の層を固定するテスト名を書く)。この参照は `tests/test_decisions_guards.py` が実在を検証する |
| 6 | 新規テストは 3 本 (`test_rag_chunker.py` 1 本、`test_rag_vault.py` 2 本 [parametrize 込みで 3 件]、`test_client_errors.py` 1 本)。`test_context_length_remediation_is_abstract_on_the_shared_base` は `_HttpEndpointClient` を継承して `_context_length_remediation` だけを未実装のまま残すサブクラス (リランカーを模した架空クラス) を定義し、`TypeError` で `pytest.raises` することで abstractness を直接検証する | `__abstractmethods__` の集合を見るだけでは「実際にインスタンス化を防いでいる」ことまでは示せないため、インスタンス化そのものを試すテストにした |
| 7 | 変異検証はすべて「1 回の Bash 呼び出しの中でファイルをバックアップ→ `python3` で該当箇所だけ元の実装に戻す→対象テストを実行→バックアップから復元→`git diff --stat` で汚れが無いことを確認」の順で実施し、間に他の作業を挟まなかった (前回 round-9 fixer が報告した自動コミット混入を避けるため)。3 件とも狙ったテストが red → green で復元できることを確認した | ハード制約「変異検証は 1 回の tool 呼び出しの中で完結させる」に対する具体的な手順 |
| 8 | 実測: 変更前 686 passed → 変更後 **687 passed**（新規 5 テストのうち 1 件は `tests/test_decisions_guards.py` の既存パラメタライズドテストが D-32 rule への新規参照を自動的に拾って増えた分）。`make ci` 緑、チャット例外文面の md5 は `b1181779cb1fe452780e5d542e5d5944` で不変、`python3 ~/.claude/scripts/check_decisions.py .claude/decisions.yaml` 緑 (30 件) | — |

**申し送り**: round-10 でメインセッションから渡された MEDIUM 3 件はすべて修正した (round-10 は他に findings が無い)。round-9 由来で未対応のまま残っている MEDIUM/INFO (`rag/parser.py::ParsedNote.frontmatter` の可変 dict、`llmkit/embeddings.py` の本文 error 翻訳の非対称など) は `docs/next-pr-candidates.md`「round-9 (Phase 3a) の MEDIUM / INFO」節に既に記載済みであり、本 round では変更していない。
