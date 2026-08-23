# Phase 3 後半 (3b: ベクトルストア・差分更新・索引本体・CLI) — 仕様書とタスクリスト

> 前提資料: `docs/plans/2026-08-23-phase3-indexing.md` (§0 サイクル分割 / §2 ★ / §3 論点2・3 / §9 T1・T2・T3・fixer round-9・round-10) / `.claude/decisions.yaml` 全 30 件 / `docs/next-pr-candidates.md` round-9 節 / 要件書 L309-L319
> 位置づけ: **3a の続き**。`rag/` に `store.py` / `indexer.py` / `cli.py` を足し、`llmkit/embeddings.py` に公開関数を 1 つ追加する。3a で確定した 6 決定 (D-25 / D-27 / D-30 / D-31 / D-32 / D-34) と 687 テストは 1 つも動かさない。
> ブランチ: `feat/phase3b-indexing`

> **★ 決定番号の対応表**: 3a の仕様書が素案として書いた D-26 / D-28 / D-29 / D-33 は `decisions.yaml` に**一度も存在しない欠番**である (実測確認済み)。3b はこれらを使わず D-35 以降を使う (D-14 の前例に従い既存 id の意味を動かさない)。
> 旧 D-26 → **D-39** / 旧 D-28 → **D-35** / 旧 D-29 → **D-36** / 旧 D-33 → **D-40**

> **ユーザー確定事項** (planner の Q1/Q2/Q3 に推奨案で回答):
> Q1 = (a) `--settings <toml>` のみ。vault パスの指定は TOML の `[vault] dir` が担う
> Q2 = (a) 実機実行の数値は本仕様書 §9 に記録する
> Q3 = (a) `Chunk.embed_text` の property 化は次サイクルに送る (永続化から外すだけに留める)

> **メインセッションが着手前に潰した 2 件** (仕様は「直っている」前提):
> 1. `RagSettings.source_path` を `expanduser().resolve()` で正規化済み (F-9-004)
> 2. `load_settings` が gitignore されない場所を指す `index_dir` を `ConfigError` にする (F-9-018)

---

## 1. ゴール

設定 TOML で与えた vault を埋め込んで `index_dir` に永続化し、**再実行時に変更のないノートを 1 件も再処理せず、索引成果物がバイト単位で一致する**索引 CLI を作る (要件書 L309 / L310)。

---

## 2. 現状認識

### 2.1 3b が乗る土台 (すべて実装済み・変更しない)

| パス | 3b での使い方 |
|---|---|
| `rag/settings.py` `RagSettings` | `vault_id` / `vault_dir` / `index_dir` / `chunk` / `embed` / `source_path`。3b の入力はこれだけ |
| `rag/settings.py` `ChunkSettings` | 4 キーすべて `index_fingerprint` に入る (docstring に宣言済み。**まだ配線されていない**) |
| `rag/settings.py` `EmbedSettings.batch_size` | **fingerprint に入れない**と docstring が宣言済み (E28 が固定する) |
| `rag/vault.py` `iter_vault_files` | 索引対象の唯一の出典。相対パス昇順で決定論的 |
| `rag/vault.py` `read_note_bytes` / `read_note_text` | **vault を読む唯一の経路** (D-30)。sha256 も `read_note_bytes` の戻り値から取る (indexer が `open` してはいけない) |
| `rag/chunker.py` `Chunk` | `body` と `embed_text` を両方持つ (→ §4 論点3) |
| `rag/chunker.py` `_tokens_from_counts` / `_count_chars` | 近似式の唯一の実装 (D-32)。CJK 範囲表は同モジュールの定数 |
| `llmkit/embeddings.py` `_resolve_embedding_spec` | **private**。fingerprint は `ModelSpec` を要るので公開版が必要 (→ T5) |
| `llmkit/embeddings.py` `embed` | 0 件入力・空白のみは `ConfigError`、件数/index/次元不整合は `UpstreamError` |
| `harness/runner.py` | **L3 の前例**。`fingerprint_inputs` (入力そのものを残す) → `fingerprint_digest` (正規化 JSON の sha256) → `run_fingerprint`。`_VOLATILE_MANIFEST_KEYS` で「毎回変わる 2 項目」だけを落とす |
| `harness/report.py` | `mkdir(parents=True, exist_ok=True)` → `write_text` → `OSError` を `ConfigError` に翻訳。索引の書き出しもこの形にそろえる |
| `harness/cli.py` | `--dry-run` は計画のみ・HTTP 0 回・exit 0。失敗はケース単位で継続し、全滅時のみ exit 1 |
| `tests/conftest.py` | `SAMPLE_VAULT_DIR` / `write_rag_settings()` / `sample_vault_copy`。**既存 fixture は 1 行も変更しない** |
| `tests/test_rag_privacy.py` | `.gitignore` ホワイトリスト・利用者名検出・symlink 0 件を機械化済み。3b は**索引成果物**の検査をここに足す |
| `tests/test_rag_layout.py` | `_SUBMODULE_NAMES` に「3b で `store` / `indexer` / `cli` が加わる。`cli` は入れない」とコメント済み |
| `tests/test_decisions_guards.py` | decisions.yaml 本文に書いた `tests/...::test_...` の実在を機械検証する。**rule に書いたテスト名は嘘をつけない** |

### 2.2 ★ 3b が防ぐべき唯一最大の欠陥 (3a §2 ★ の再掲・実装はここ)

`max_tokens` を 240→200 にして再実行すると、mtime もハッシュも変わらないノートは「変更なし」と判定されて再チャンクされない。**240 で切ったチャンクと 200 で切ったチャンクが同じ索引に混在する。** 埋め込みモデルを差し替えた場合はさらに悪く、**次元も意味空間も違うベクトルが同居**してコサイン類似度が無意味になる。どちらも例外を出さず、テストも落ちず、検索精度だけが理由不明に劣化する (D-19 と同型)。

→ `index_fingerprint` の設計 (§4 論点1 / D-35) が本サイクルの中核であり、**「何を入れるか」より「入れ忘れると静かに壊れる」ことが問題**である。

### 2.3 既存の慣習で守るべきもの

1. **層の境界は AST で機械検証する**。`rag/` に HTTP と `.get()` を書かない (D-25。`.get` は `dict.get` ごと落ちる。`in` + 添字で書く。ruff SIM401 が `.get` を勧めるので三項演算子ではなく `if` 文にする)
2. **重複した真実を作らない** (D-19 / D-27)。モデル ID・次元・区切り文字を 2 か所に持たない
3. **欠測は `None`。0 で埋めない** (D-07 / D-13 / D-21)

### 2.4 影響範囲

- 新規: `rag/{store,indexer,cli}.py` / `tests/{test_rag_store,test_rag_indexer,test_rag_cli,test_acceptance_phase3}.py`
- 変更: `llmkit/embeddings.py` (**公開関数の追加のみ**) / `llmkit/__init__.py` / `rag/__init__.py` / `tests/test_rag_layout.py` (`_SUBMODULE_NAMES` に 2 行) / `tests/conftest.py` (**追加のみ**) / `tests/test_rag_privacy.py` (追加のみ) / `.claude/decisions.yaml` / `docs/localllmrequirements.md` (**チェックボックスのみ**) / `docs/next-pr-candidates.md`
- **不変: `rag/{settings,vault,parser,chunker}.py` / `llmkit/` の他 7 モジュール / `harness/` 全体 / `configs/` / `suites/` / `results/` / `vaults/sample/` / `main.py` / `Makefile` / `.github/` / `pyproject.toml` / `uv.lock`**

---

## 3. 前提・制約

### ハード制約

- 既存 **692 テストを壊さない。アサーションを緩めない。** 唯一の例外は `tests/test_rag_layout.py::_SUBMODULE_NAMES` への `"store"` / `"indexer"` の追加 (§9 T3 決定11 と同型の**拡張**。`test_every_rag_submodule_is_covered_by_the_union_check` が完全一致を要求するため、追加しないと必ず落ちる)。これ以外で既存テストを 1 行でも直したくなったら**止まって相談** (§8 リスク3)
- mypy `strict` + `disallow_any_explicit`。`Any` 明示禁止。`# type: ignore` 0 個。`pydantic.BaseModel` 非継承 (D-08)
- ruff `T20`。CLI 出力は `sys.stdout.write` / `sys.stderr.write` か logging
- **テストは実 HTTP を 1 バイトも出さない** (D-02)。**CI (GPU 無し・Ollama 無し) で全件緑**
- **新規依存を追加しない** (`uv lock --check` 無変更)
- **`docs/localllmrequirements.md` の行数を 1 行も増減させない** (`ACCEPTANCE_MAP` が L294-L298 / L302-L305 を行番号参照)
- 実 vault の絶対パス・ノート本文・タイトルを、コード・テスト・コミット対象の設定ファイル・**索引成果物以外の出力 (ログ / stdout / 例外)** に一切書かない
- **実 vault の索引成果物をコミットしない**
- 検索・リランキング・回答生成・評価質問セットを実装しない (L315-L318 は次サイクル)

### ソフト制約

- ログは INFO で件数のみ、DEBUG で相対パスまで。**本文を 1 文字も出さない**
- 1 ノートの処理失敗は記録して次のノートへ (例外種別による分岐は D-38 で確定)
- **正規化 JSON の sha256 は `harness/runner.py` と同じ規則** (`sort_keys=True, ensure_ascii=False`) を `rag/indexer.py` に置く。層構造上 `rag` は `harness` を import できず、L2 に上げるのは `llmkit` の責務 (推論ランタイムの抽象) ではないため、3 行の重複を許容する。ただし**両者が同じ dict に同じ digest を返すことをテストで固定**する。D-32 が禁じた「独自の近似式の複製」とは性質が違う (こちらは stdlib 呼び出し 1 行)
- `ParsedNote` / `Chunk` を差分判定・キャッシュキーに使わない (→ §4 論点4)

---

## 4. ★ 論点への判定

### 論点1: `index_fingerprint` の入力に `source_path` を入れるか → **入れない**

| 案 | 判定 | 根拠 |
|---|---|---|
| `source_path` (絶対パス) を入れる | **却下** | (i) **設定ファイルの置き場所は生成されるベクトルに 1 ビットも影響しない。** fingerprint の定義は「前提が変わったら再処理する」であり、影響しない値を入れると偽の全再構築を生む (ii) ディレクトリ移動・シンボリックリンク経由・別ホストでの実行が全再構築になる (iii) **実 vault の絶対パス (利用者名を含む) が成果物に書き出される唯一の経路になる。** manifest は `data/` 配下で gitignore されるが、`status` サブコマンドと INFO ログで画面に出る |
| `vault_dir` を入れる | **却下** | 同上 (絶対パス)。「別 vault なのに同じ索引を使う」事故は `(relpath, sha256)` の集合差分で全ノートが新規扱いになるため自動的に正しく処理される |
| `sha256(str(source_path))` を入れる | **却下** | パス由来の値であることは変わらず、防げる事故は集合差分で既に防げている。「一方向だから安全」は、利用者名が既知なら総当たりで戻せるため成立しない |
| **設定の内容のうちベクトルに影響する値だけを入れる** | **採用** | 下記 |

**確定形** (`rag/indexer.py`):

```
fingerprint_inputs = {
  "schema_version":         <索引成果物の形式バージョン (int)>,
  "vault_id":               settings.vault_id,
  "chunk":                  {max_tokens, cjk_chars_per_token, ascii_chars_per_token, heading_separator},
  "chunk_algorithm_sha256": <chunker の CJK 範囲表の正規化表現の sha256>,
  "embedding":              {"model_id": spec.model_id, "served_name": spec.served_name},
}
index_fingerprint = sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False))
```

- **入れない**: `source_path` / `vault_dir` / `index_dir` / `runtime.base_url` / `embed.batch_size` / `include_globs` / `exclude_globs` / mtime / 時刻 / run_id
- `include_globs` / `exclude_globs` を入れない理由: glob を変えると**索引対象の集合**が変わるが、これは集合差分 (追加 → 新規埋め込み、消失 → チャンク削除) で正しく処理される。fingerprint に入れると 1 パターン足しただけで**残り全ノートの再埋め込み**が走る (→ E33 が固定)
- `chunk_algorithm_sha256` は手書きの `VERSION = 1` 定数にしない。手で上げ忘れるのは D-19 と同型の欠陥であり、**表そのものから導出**すれば忘れられない
- **fingerprint 不一致 → 差分更新を行わず全再構築** (既存 `chunks.jsonl` は破棄して作り直す)
- fingerprint 値だけでなく `fingerprint_inputs` そのものを manifest に残す (`harness/runner.py` と同じ方針。「何が変わったから変わったのか」を追える)
- 宣言 (設定) と実際 (ランタイムの応答) は別物なので、**ランタイムが名乗ったモデル名と次元を manifest に別枠で持ち、既存 manifest と食い違ったら書き出す前に中断する** (D-35 の第2条項)

### 論点2: `VectorStore` Protocol に検索メソッドを今定義するか → **定義しない**

3a の「使わない抽象を先に固めない」を維持する。

- 検索の形は次サイクルの要件 (L315-L318) が決める。`search(vector, k)` を今決めると、タグ絞り込み・スコア閾値・MMR を足す段階で **Protocol を変えること**になり、「公開シグネチャを断りなく変更しない」に自分で違反する
- 代わりに **`iter_records()` (全走査) を定義する。** 370 チャンクの総当たりコサイン類似度は純 Python で約 30ms (3a 実測) なので、次サイクルの検索は store の**外** (`rag/search.py`) に書ける。Protocol を変えずに検索を足せる形が確保される

**確定形**:

```
ChunkRecord (frozen dataclass): chunk_id / relpath / ordinal / part_index / heading_path / body
                                / estimated_tokens / tags / links / vector
VectorStore (Protocol):
    note_chunk_counts() -> Mapping[str, int]        # 整合検査と plan に使う
    replace_note(relpath, records) -> None          # ノート単位の置換 (差分更新の粒度と一致)
    delete_note(relpath) -> None
    iter_records() -> Iterator[ChunkRecord]
    commit() -> None                                # 永続化の確定 (原子的)
```

- 実装は **`JsonlVectorStore`** (単一 `chunks.jsonl` を生成時に全読み込み、`commit()` で一時ファイル → `os.replace`) と **`InMemoryVectorStore`** (`commit()` は no-op) の 2 つ
- **同じ適合テストを 2 実装で parametrize して回す**
- 単一ファイル方式を採る理由: ノート単位のファイル分割は部分書き込み状態を作り、原子性の保証が難しい。6.36MB (実測) の書き直しは数十 ms で、**「再処理しない」は再埋め込みをしないことであってファイルを書かないことではない**
- 差し替えの発動条件を D-39 に明記する (チャンク数 5 万超 / 200MB 超 / 並行アクセスが要る)

### 論点3: `Chunk` が `body` と `embed_text` を両方持つ件 → **永続化しない。型は今サイクルでは変えない**

| 案 | 判定 | 根拠 |
|---|---|---|
| **JSONL に `embed_text` を書かず、`heading_path` + `body` から再構成する** | **採用** | reviewer の懸念の実害部分 (「片方だけ差し替わった状態が**索引に残る**」) が構造的に消える。本文の 2 重保存も消え、成果物が約 4 割小さくなる。再構成は `render_embed_text(heading_path, body, separator)` の**唯一の実装**を通し、`chunk_note` もそれを使う (D-32 の `_tokens_from_counts` と同じパターン) |
| `Chunk.embed_text` を `@property` 化する | **見送り (Q3 = a)** | 完全な解だが `Chunk` に `heading_separator` フィールドを足す必要があり、`rag.__all__` の公開型変更になる。L309 / L310 に 1 ビットも寄与しないのに 692 テストの緑を賭ける。実測で既存テストは `Chunk(...)` を直接構築しておらず属性アクセスのみなので**次サイクルで安全に実施できる** → `docs/next-pr-candidates.md` に移す |

### 論点4: `ParsedNote.frontmatter` が可変 dict (F-9-003) → **3b では潰さない。使わない設計にする**

- 3b の差分判定は**ファイルのバイト列の sha256 だけ**で決まる (D-36)。`ParsedNote` の同一性・ハッシュ可能性は 1 か所も要らない
- パース結果をキャッシュしない (再処理対象と決まったノートだけを、その場でパース → チャンク → 埋め込みする)
- 型を変えるのは公開 API 変更 + 既存テスト波及で、受け入れ条件に寄与しない
- **guard_test を書ける形にならないため、決定にせずソフト制約に落とす**。`docs/next-pr-candidates.md` の記載は維持

### 論点5: バッチの途中で失敗したときのマニフェスト → **ノート単位のトランザクション。部分成功を記録しない**

- バッチ (`embed.batch_size` 件) は**ノート境界を跨いでよい**が、**manifest にノートを載せるのは、そのノートの全チャンクが埋め込まれ `store.replace_note()` を通った後だけ**
- 部分的に成功したチャンクは破棄する。理由: 半分だけ載せると次回は sha256 が一致するので「変更なし」と判定され、**欠けたまま永久に治らない** (§2.2 と同型の静かな破壊)
- **例外種別で挙動を分ける**:

| 例外 | 挙動 | 根拠 |
|---|---|---|
| `RuntimeUnavailableError` / `OutOfMemoryError` | **即座に中断** (それまでに完了したノートは commit 済み) | 次のノートでも必ず再発する。34 回同じエラーを出すのは利用者への嫌がらせで、部分索引を作る時間も無駄 |
| `ContextLengthError` / `UpstreamError` | そのノートをスキップして継続、失敗件数に計上 | 特定ノート固有の問題。1 件で索引全体を止めない |
| `ConfigError` (0 件・空白のみ) | **起こさない**。空チャンクはチャンカが作らない | — |

- **書き出し順序は `store.commit()` → `manifest` の順に固定する。** 逆にすると manifest だけ進んだ状態でクラッシュしたとき、次回「変更なし」と判定されて**チャンクが欠けたまま固定**される。この順序ならクラッシュ時は manifest が古い = 再処理されるだけ (安全側に倒れる)
- 全ノートが失敗した場合は exit 1

### 論点6: 実機実行で何を数値報告すれば L309 / L310 を満たしたと言えるか

実 vault (34 ノート / 257K 文字) の設定は `vaults/local.toml` (gitignore 済み) に置く。**成果物はコミットしない。報告してよいのは数値と fingerprint だけで、ノートのパス・タイトル・本文・vault の絶対パスは 1 文字も出さない。**

| # | 実行 | 報告する数値 | 満たす条件 |
|---|---|---|---|
| 1 | `index --dry-run` | 対象ノート数 / 新規・変更・削除・変更なしの内訳 / `index_fingerprint` / **HTTP 0 回** | — |
| 2 | `index` (初回) | ノート数 N / チャンク数 C / リクエスト数 R=⌈C/batch⌉ / 所要 t₁ 秒 / **次元 768** / 失敗ノート 0 / `chunks.jsonl` バイト数 | **L309** |
| 3 | `index` (無変更で再実行) | **再処理ノート 0 / 再埋め込みチャンク 0 / リクエスト 0** / 所要 t₂ と t₂/t₁ / `manifest.json` と `chunks.jsonl` の sha256 が #2 と**一致** | **L310** |
| 4 | 1 ノートの末尾に 1 文字足して再実行 | 再処理ノート 1 / 再埋め込みチャンク k / リクエスト ⌈k/batch⌉ / **他ノートの `chunk_id` とベクトルが #2 とバイト一致** | **L310** (差分の粒度) |
| 5 | `max_tokens` を変えて再実行 | `index_fingerprint` が変化 / 全 C′ チャンクを再埋め込み | §2.2 の欠陥が塞がれていること |
| 6 | #2〜#5 の前後 | vault 配下全エントリの `(relpath, size, st_mtime_ns, st_mode, sha256)` ダイジェストが完全一致 | **L319** の実 vault 版 |

> **「予定リクエスト数」を報告項目から落とした** (T7 決定4 への回答)。正確に出すには CLI が `parse_note` → `chunk_note` を回すことになり、`build_index` の前半を CLI に二重実装する (D-27 の趣旨に反する)。`plan_index` に `pending_chunk_counts` を持たせるのが正しい置き場所だが、`--dry-run` の目的 (HTTP 0 回で「何を再処理するか」を出す) は対象ノート数と内訳で足りている。リクエスト数は #2 の実測で報告する。`IndexPlan` の拡張は `docs/next-pr-candidates.md` に送る。

- **止める条件**: 次元が 768 以外 / `data` 件数不一致 / OOM / #2 が 60 秒超
- 報告先は本仕様書 §9。実行コマンド例に絶対パスを書かない

---

## 5. タスク分解 (T4〜T8)

**実行順序 T4 → T5 → T6 → T7 → T8 を厳守。** 各タスク完了時に `make ci` の緑を確認してから次へ進む。

**タスクサイズの判定**: L は T6 (差分更新の実行本体・失敗方針・原子性) と T8 (実機実行を含む) の **2 本**。T4 / T5 / T7 はいずれも既存パターン (`harness/report.py` / `harness/runner.py` / `harness/cli.py`) の写しで新規設計判断を含まないため M。**分割不足の警告は出ない。**

### T4: `rag/store.py` — VectorStore 抽象と 2 実装 — **M**

- `ChunkRecord` (frozen dataclass、§4 論点2 の形) / `VectorStore` Protocol / `JsonlVectorStore` / `InMemoryVectorStore` / `render_embed_text(heading_path, body, separator)` (**唯一の実装**。`rag/chunker.py` に置くか `store.py` に置くかは実装者判断だが、`chunk_note` もこれを通すこと)
- JSONL 1 行 = `json.dumps(record, sort_keys=True, ensure_ascii=False)`。**`embed_text` / `model` / `dimensions` をレコードに入れない** (モデル名と次元は manifest に 1 か所だけ)
- 行順は `(relpath, ordinal)` 昇順。`commit()` は一時ファイル → `os.replace` (原子的)。`OSError` は `ConfigError` に翻訳
- 読み込みは pydantic dataclass + `TypeAdapter` で厳格に (D-07)。壊れた行・次元不一致は `ConfigError`
- `.get()` を 1 つも書かない (D-25 guard)

**受け入れ基準**

- 適合テストが **2 実装で parametrize され、両方で緑** (`ids=` を付ける)。契約: 空 store の `iter_records()` は 0 件 / `replace_note` の 2 回目が置換になる (追記でない) / `delete_note` 後は 0 件 / `note_chunk_counts()` が `iter_records()` の集計と一致 / 存在しないノートの `delete_note` は例外にならない
- `JsonlVectorStore`: 書いて読み直すと `ChunkRecord` が**完全一致**
- **同じ内容を 2 回 commit すると `chunks.jsonl` がバイト一致**
- JSONL のどの行にも `"embed_text"` キーが無く、`render_embed_text` で再構成した文字列が `chunk_note` の出力と一致する (**D-41 guard**)
- `commit()` を `OSError` で失敗させると `ConfigError` になり、**既存ファイルが壊れていない** (原子性)
- `rag/store.py` に HTTP 呼び出し・`.get()` が 0 件 (既存 D-25 guard が自動で拾う)
- `tests/test_rag_layout.py::_SUBMODULE_NAMES` に `"store"` を追加 (唯一許可された既存テスト編集)

### T5: `rag/indexer.py` 前半 — fingerprint とマニフェストと計画 — **M**

- **`llmkit/embeddings.py` に `resolve_embedding_spec(config) -> ModelSpec` を公開追加** (現 `_resolve_embedding_spec` の rename + `__all__` + `llmkit/__init__.py` 再エクスポート)。**既存公開シグネチャは 1 つも変えない。** `rag` は `llmkit` の公開シンボルしか使えないため、これが唯一の経路 (`config.active_profile().embedding` を `rag` が直接読む案は、role 検査と `ConfigError` 文言が 2 か所に分岐するので採らない ← D-27 の趣旨)
- `INDEX_SCHEMA_VERSION` / `MANIFEST_FILENAME` / `CHUNKS_FILENAME` / `fingerprint_inputs(settings, spec)` / `fingerprint_digest(inputs)` / `index_fingerprint(settings, spec)`
- `IndexManifest` (`schema_version` / `index_fingerprint` / `fingerprint_inputs` / `embedding{reported_model, dimensions}` / `notes[{relpath, sha256, chunk_count}]` (relpath 昇順) / `totals{notes, chunks}`) + `load_manifest` / `write_manifest`。**時刻・run_id・絶対パスを 1 つも持たない**
- `IndexPlan` / `plan_index(settings, config, *, store, manifest)` — **HTTP 0 回・書き込み 0 バイト**。`iter_vault_files` → `read_note_bytes` → sha256 → manifest と突き合わせて `unchanged` / `changed` / `new` / `deleted` に分類。fingerprint 不一致なら全件 `changed`。**manifest の `chunk_count` と `store.note_chunk_counts()` が食い違うノートも `changed` に入れる** (自己修復)

**受け入れ基準**

- 合成 vault で `plan_index` が **HTTP 0 回**、`index_dir` に 1 バイトも書かずに 11 ノートを分類する
- `chunk` の 4 キーをそれぞれ 1 つ変えると `index_fingerprint` が変わる (4 パラメタライズ、**E31**)
- CJK 範囲表に 1 範囲足すと `chunk_algorithm_sha256` と `index_fingerprint` が変わる (**E32**)
- `profiles[active].embedding` を変えると `index_fingerprint` が変わる (**E34**)
- **`source_path` / `vault_dir` / `index_dir` / `base_url` / `embed.batch_size` / `include_globs` / `exclude_globs` を変えても `index_fingerprint` が変わらない** (7 パラメタライズ、**D-35 guard**)
- `fingerprint_digest` が `harness.runner.fingerprint_digest` と同じ dict に同じ値を返す
- 同一入力で `manifest.json` を 2 回書くと**バイト一致**
- manifest の `chunk_count` を手で 1 減らすと、そのノートだけ `changed` になる (自己修復)
- `llmkit` 側: `test_layout.py` の和集合検査が無修正で緑
- `tests/test_rag_layout.py::_SUBMODULE_NAMES` に `"indexer"` を追加

### T6: `rag/indexer.py` 後半 — `build_index` (差分更新の実行) — **L**

- `IndexResult` (`indexed_notes` / `skipped_notes` / `failed_notes` / `deleted_notes` / `embedded_chunks` / `request_count` / `elapsed_s` / `fingerprint` / `dimensions`) と `build_index(settings, config, *, embedding_client, store, ...)`
- `plan_index` の `changed`+`new` だけをパース → チャンク → `embed.batch_size` 件ずつ埋め込み → **ノートの全チャンクが揃った時点で `store.replace_note`**
- `deleted` は `store.delete_note` + manifest から除去
- 完了後に **`store.commit()` → `write_manifest()` の順**で確定
- ランタイムが名乗ったモデル名・次元が既存 manifest と食い違ったら、**1 バイトも書かずに `ConfigError`** で中断
- 失敗の扱いは §4 論点5 の表のとおり
- ログ: INFO は件数のみ。**本文・絶対パスを出さない**

**受け入れ基準** (すべて `httpx.MockTransport` + 決定論的フェイク埋め込み)

- 合成 vault (11 ノート / 24 チャンク) で初回 24 チャンクが埋め込まれ、`chunks.jsonl` と `manifest.json` が書かれる (**L309**)
- **無変更で再実行 → 再埋め込み 0 / リクエスト 0 / `chunks.jsonl` と `manifest.json` がバイト一致** (**L310**、**D-37 guard**)
- ファイルを `touch` (mtime だけ変更) しても再処理 0 件 (**D-36 guard**)
- 1 ノートを編集 → **そのノートのチャンクだけ再埋め込み**、他ノートのベクトルはバイト一致 (**E29**)
- `batch_size` を 4→16 に変えるとリクエスト数だけ変わり、`chunks.jsonl` はバイト一致 (**E28**)
- `exclude_globs` に 1 パターン足す → そのノートのチャンクが消え、**他ノートは再埋め込みされず fingerprint も不変** (**E33**)
- `max_tokens` を変える → fingerprint 不一致で全 24 チャンク再埋め込み (**E31** の実行側)
- 1 ノートで `UpstreamError` → そのノートは manifest に**載らず**、他 10 ノートは索引され、次回の再実行でそのノートだけ再試行される (**D-38 guard**)
- `RuntimeUnavailableError` → 1 ノート目で中断し、リクエスト数がノート数より少ない
- `store.commit()` を失敗させると **manifest が更新されない** (順序の証拠)
- ランタイムが別のモデル名/次元を名乗ると `index_dir` に 1 バイトも書かずに中断
- **索引実行の前後で vault 配下の全エントリが完全一致** (D-30 動的 guard を「vault 全体のパース」から**索引本体**に差し替える)
- ログ・例外・stdout にノート本文と絶対パスが 0 件

### T7: `rag/cli.py` — `index` / `status` — **M**

```
python -m rag.cli index  --settings vaults/sample.toml [--config configs/default.toml] [--dry-run] [--rebuild]
python -m rag.cli status --settings vaults/sample.toml [--config configs/default.toml]
```

- `--settings` が **vault パスを与える唯一の入口** (Q1 = a)。`--vault <dir>` は作らない (`vault_id` と `index_dir` が決まらず、出典が 2 か所になる ← D-27 の趣旨)
- `--dry-run`: 計画と `index_fingerprint` と出力先の予定だけを出す。**HTTP 0 回・書き込み 0 バイト・exit 0**
- `--rebuild`: fingerprint が一致していても全再構築する (逃げ道)
- `status`: manifest を読んで fingerprint 一致・ノート数・チャンク数・次元を出す。未構築なら「未構築」と出して exit 0
- 出力は**件数と fingerprint のみ**。相対パスも出さない (実機実行の報告をそのまま貼れる状態にする)
- 終了コード: 成功 0 / `LlmkitError` 捕捉 1 / 全ノート失敗 1
- `main(argv, *, http_client=None, embedding_client=None, stdout=None, stderr=None)` の注入点は `harness/cli.py` と同型
- `rag/__init__.py` の `__all__` に `cli` を含めない

**受け入れ基準**

- `--dry-run` で `RecordingTransport.call_count == 0` かつ `index_dir` が**作られない**
- `index` → `status` の順で回すと `status` が「fingerprint 一致 / 再処理 0 件」を出す
- `index_dir` を変えると別の索引が作られ、元の索引が無傷 (**E35**)
- `--settings` に別 vault を指す設定を渡すと索引されるノート集合が変わる (**E36**、L309 の直接検証)
- `ConfigError` (存在しない設定ファイル / vault 配下の `index_dir` / gitignore されない `index_dir`) が exit 1 + 対処つきメッセージ
- stdout / stderr に絶対パス・本文・`[[` が 0 件
- `tests/test_rag_layout.py::_SUBMODULE_NAMES` に `"cli"` を**入れない**

### T8: 受け入れ検証・決定の追記・実機実行・文書 — **L**

- `tests/test_acceptance_phase3.py` を新設。`ACCEPTANCE_MAP` の **7 行 7 関数** (L309-L314 / L319。L315-L318 は次サイクルなので含めない。含めない理由を module docstring に書く)。`assert_is_acceptance_line` は `test_acceptance_phase2.py` を写す
- `docs/localllmrequirements.md` の L309-L314 / L319 を `- [ ]` → `- [x]` に置換 (**行数不変**)
- `.claude/decisions.yaml` に **D-35〜D-41** を追記 (**guard_test が実在してから**)。`check_decisions.py` が **37 件**で緑
- `tests/test_rag_privacy.py` に索引成果物の検査を追加 (D-40 guard)
- 実機実行 (§4 論点6 の #1〜#6) を行い、**数値だけ**を本仕様書 §9 に記録
- `docs/next-pr-candidates.md` に 3b の申し送りを追記

**受け入れ基準**

- `test_acceptance_phase3.py` の 7 テストが緑で、要件書の行数が変わっていない
- `data/index/**` と `vaults/local.toml` が `git check-ignore` で無視される (**D-40 guard**)
- `test_no_tracked_file_exposes_the_real_user_identity` が緑 (実機実行の報告を書いた後に必ず再実行)
- `check_decisions.py` が **37 件**で緑。`test_decisions_guards.py` が全参照の実在を確認
- 実機 #3 で **再処理 0 / リクエスト 0 / 成果物 sha256 一致**を実測し、数値を §9 に記録
- 実機 #6 で vault ダイジェストが前後一致
- `make ci` 緑 / `uv lock --check` 無変更

---

## 6. 評価軸 (Check フェーズへ)

### 機能観点

要件書 **L309 / L310** を `tests/test_acceptance_phase3.py` が行番号参照つきで機械検証する。加えて 3a で満たした L311 / L312 / L313 / L314 / L319 も同ファイルに集約し、チェックボックスを更新する。**実機での数値は §4 論点6 の表**。

### 性能観点

| 指標 | 期待値 |
|---|---|
| `uv run pytest` 全体 | **T4 着手前の実測 (692 passed / 約 1.9s) から +8 秒以内** |
| 合成 vault の索引 (MockTransport) | 1 秒未満 |
| 実 vault の初回索引 | **60 秒以内** (超えたら止まって相談) |
| 実 vault の無変更再実行 | 初回の **10% 未満** |
| `chunks.jsonl` | 実測値を報告 (見積り約 6.4MB) |

### 安全性観点

- 索引実行の**前後で vault 配下の全エントリが完全一致**
- `rag/` に HTTP 呼び出し・エンドポイント文字列・`.get()` が 0 件 (D-25、新規 3 モジュールも自動で対象)
- ログ・例外・stdout に**ノート本文が 1 文字も現れず、絶対パスも現れない**
- `index_dir` が vault 配下 / gitignore されない場所を指す設定が `ConfigError`
- 追跡対象ファイルに実 vault の痕跡が 0 件
- `store.commit()` は原子的 (失敗しても既存索引が壊れない)

### テスト観点

- 新規: `test_rag_store.py` (適合テスト × 2 実装) / `test_rag_indexer.py` / `test_rag_cli.py` / `test_acceptance_phase3.py`
- 拡張のみ: `conftest.py` (**決定論的フェイク埋め込み**: 入力テキストの sha256 からベクトルを導出。「同じテキストなら同じベクトル」が成り立たないと E28 と D-37 のバイト一致検査が書けない。既定は軽量な低次元、768 次元を通すテストを 1 本置く) / `test_rag_privacy.py` / `test_rag_layout.py` (2 行)
- 既存 692 テストは**上記 2 行以外 1 行も変更しない**
- 新規 `parametrize` にはすべて `ids=` を付ける

### ★ 有効性観点 (既存 E1〜E30 を維持したうえで)

| # | 掃引する値 | 変わるべき出力 | テスト |
|---|---|---|---|
| **E28** | `embed.batch_size` (4→16) | HTTP 回数**のみ**。索引はバイト一致 | `test_rag_indexer.py::test_batch_size_changes_request_count_but_not_the_index` |
| **E29** | 1 ノートの本文を編集 | そのノートのチャンクだけ再埋め込み | `test_rag_indexer.py::test_only_the_edited_note_is_reembedded` |
| **E31** | `chunk` の 4 キー各々 | `index_fingerprint` → 全再構築 | `test_rag_indexer.py::test_changing_chunk_settings_invalidates_the_whole_index` |
| **E32** | chunker の CJK 範囲表 | `chunk_algorithm_sha256` → `index_fingerprint` | `test_rag_indexer.py::test_the_token_algorithm_is_part_of_the_fingerprint` |
| **E33** | `exclude_globs` に 1 パターン | そのノートのチャンクだけ消える。**fingerprint 不変・他ノート再埋め込み 0** | `test_rag_indexer.py::test_excluding_a_note_deletes_its_chunks_without_reembedding_the_rest` |
| **E34** | `profiles[active].embedding` | `index_fingerprint` → 全再構築 (§2.2 の中核) | `test_rag_indexer.py::test_changing_the_embedding_model_invalidates_the_index` |
| **E35** | `index.dir` | 索引の出力先。元の索引は無傷 | `test_rag_cli.py::test_index_dir_is_the_only_output_location` |
| **E36** | `--settings` の vault | 索引されるノート集合 (**L309** の直接検証) | `test_rag_cli.py::test_the_indexed_note_set_follows_the_given_config` |

### 変異検証 (3b で必須。1 回の tool 呼び出しで完結させる)

1. `index_fingerprint` の入力から `chunk` を落とす → E31 と D-35 guard が落ちる
2. `index_fingerprint` の入力に `source_path` を足す → D-35 guard (不変側) が落ちる
3. 差分判定を「mtime 一致なら変更なし」にする → D-36 guard が落ちる
4. 書き出し順序を manifest → store に入れ替える → D-37 の順序テストが落ちる
5. 埋め込み失敗時に部分成功チャンクを manifest に載せる → D-38 guard が落ちる
6. `chunks.jsonl` の行順ソートを外す → バイト一致テスト (D-37) が落ちる
7. JSONL レコードに `embed_text` を足す → D-41 guard が落ちる

---

## 7. 意図的な決定 (`.claude/decisions.yaml` に **D-35 以降**で追記)

内容は planner の出力どおり (D-35 / D-36 / D-37 / D-38 / D-39 / D-40 / D-41)。**guard_test が実在してから T8 でまとめて追記する。** 追記後に `check_decisions.py` (37 件) と `tests/test_decisions_guards.py` を必ず走らせる。

| id | 主旨 | guard_test |
|---|---|---|
| D-35 | fingerprint の入力を限定する。ランタイム申告の食い違いは書く前に中断 | `tests/test_rag_indexer.py::test_the_fingerprint_covers_every_input_that_changes_a_vector` |
| D-36 | 差分判定はバイト列の sha256 だけ。mtime を判定にもマニフェストにも使わない | `tests/test_rag_indexer.py::test_touching_a_note_without_changing_bytes_reindexes_nothing` |
| D-37 | 確定は store.commit() → write_manifest() の順。成果物はバイト一致 | `tests/test_rag_indexer.py::test_an_unchanged_rerun_leaves_the_index_byte_identical` |
| D-38 | ノート単位のトランザクション。部分成功を記録しない。例外種別で中断/継続 | `tests/test_rag_indexer.py::test_a_failed_note_is_never_recorded_as_indexed` |
| D-39 | chromadb を入れず Protocol + 2 実装。検索メソッドは定義しない | `tests/test_rag_store.py::test_every_store_implementation_passes_the_same_contract` |
| D-40 | 索引成果物と実 vault 設定をコミットしない。index.dir を設定層で検査 | `tests/test_rag_privacy.py::test_the_index_directory_is_never_tracked` |
| D-41 | 永続化レコードに embed_text を持たせない。モデル名と次元は manifest に 1 か所 | `tests/test_rag_store.py::test_the_persisted_record_never_stores_the_body_twice` |

---

## 8. 想定リスク (これが起きたら止まって人間に相談)

1. **実 vault の初回索引が失敗する / 埋め込みが 768 次元以外を返す / 60 秒を超える。**
   Phase 0 の実測は短文 1 件であり、240 トークン相当のチャンク・バッチ 16 件・34 ノートの連続実行は未検証。`max_context_tokens=8192` も未実測で、長いチャンクが黙って切り詰められる可能性がある (3a §7 リスク2 の再掲、**まだ解消していない**)。**次元が 768 以外・`data` 件数の不一致・OOM・60 秒超のいずれかが出たら止めて相談する。**

2. **実 vault のノート本文・絶対パスが成果物・ログ・コミットに漏れる。**
   3b は初めて**本文を全量ディスクに書く**サイクルであり、漏洩面が 3a より一段広い。しかもキットの SubagentStop 自動コミットは `git add -A` を行い、テストを回す Stop フックより**先**に走る。D-40 の設定層検査と `test_rag_privacy.py` で機械化するが、**実機実行の直後に `git status` と `git check-ignore` を確認し、1 件でも索引成果物・実 vault 由来の文字列が追跡側に現れたら止めて相談する** (不可逆な公開事故)。

3. **`Chunk` / `ParsedNote` / `VaultFile` の公開形を変えたくなる。**
   論点3・論点4 で「今サイクルでは変えない」と判断したが、実装中に「property 化しないと綺麗に書けない」場面が来る可能性がある。公開型の変更は `rag.__all__` の変更であり、既存 692 テストの緑を賭けることになる。**既存テストを (許可された `_SUBMODULE_NAMES` の 2 行以外で) 1 行でも直したくなったら、その時点で止めて相談する。**

---

## 9. 実装時に決めたこと / 実機実測 (実装者が追記する節)

> T4 以降の実装者は、仕様書に書かれていなかった選択をここに追記すること。次の周の reviewer / fixer が読むのはこの節であり、実装コードのコメントではない。
> 実機実測 (§4 論点6 の #1〜#6) の数値もここに記録する。**ノートのパス・タイトル・本文・vault の絶対パスは 1 文字も書かないこと。**

### T4（`rag/store.py` / `rag/chunker.py` の `render_embed_text` 共有化、2026-08-23）

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | **`render_embed_text(heading_path, body, separator)` は `rag/chunker.py` に置いた**（`rag/store.py` ではない）。`chunk_note` は既存の `_embed_text(heading_path, body, settings)` を「`settings.heading_separator` を取り出して `render_embed_text` に渡すだけ」の薄いラッパに変え、全経路がこの 1 関数を通る | §5 T4 は置き場所を実装者判断としている。`store.py` に置くと `chunker.py` → `store.py` の import が生まれ、**純粋モジュール（ファイルにもネットワークにも触れない）が I/O を持つモジュールに依存する**。逆向き（store → chunker）なら層の向きが自然で、区切り文字定数 `_PREFIX_SEPARATOR` と、上限判定 `_fits_predicate`（`embed_text` に対して上限を適用する D-32 の要）も同じモジュールに残る。実測: 合成 vault の**全 24 チャンク**で `render_embed_text(chunk.heading_path, chunk.body, settings.chunk.heading_separator) == chunk.embed_text`（不一致 0 件） |
| 2 | **`tests/test_rag_layout.py` の `_FILESYSTEM_READ_ALLOWANCES` に `"store.py": frozenset({"read_text"})` を追加した**（`_SUBMODULE_NAMES` への `"store"` 追加に加えて、同ファイルの 2 か所目の変更）。★**仕様外の変更であり、次の周で承認/再設計の判断が要る** | §3 ハード制約は `test_rag_layout.py` の変更を `_SUBMODULE_NAMES` だけに限っているが、**この制約のままでは T4 は原理的に緑にならない**。D-30 の構造層 guard `test_only_the_vault_module_reads_the_vault` は `rag/vault.py` 以外での `read_text` / `open` / `read_bytes` 等を名前ベースで一律に落とすため、`chunks.jsonl` を読む `store.py`（および T5 の `manifest.json`、T7 の CLI）が必ず違反になる。回避策として「名前表に載っていない読み取り API（`io.FileIO` 等）を使う」「モジュールレベルで `Path.read_text` を別名に束ねる」が技術的には可能だが、いずれも**検査を弱めたことを記録に残さずに迂回する**行為であり、以後どのモジュールでも同じ抜け方ができるようになる。既に `settings.py` に対して同じ形の例外（「設定ファイル自身を読む。vault 相対パスを受け取る関数を 1 つも持たない」）が用意されており、`store.py` はその条件を満たす（読むのは `index_dir` 配下の索引ファイルだけ）ため、**設計された拡張点に 1 行足し、理由をコメントに残す**形を採った。`open` / `read_bytes` / `glob` / `iterdir` 等は `store.py` でも依然 1 つも許していない |
| 3 | `ChunkRecord` は**素の frozen dataclass ではなく pydantic frozen dataclass**（`extra="forbid"`）にし、書き出す型と読み込む型を 1 つにした | §4 論点2 は「frozen dataclass」、§5 T4 は「読み込みは pydantic dataclass + TypeAdapter」と書いており、素直に読むと**書き出し用と読み込み用の 2 つの型**になる。2 つに分けると、書き出し側にフィールドを足しても読み込み側が黙って無視する状態が作れる（`llmkit/config.py` の `_RawSettings` は入力専用なので分ける理由があるが、索引レコードは同じ型で往復する）。`extra="forbid"` の副次効果として、`embed_text` を書き足した索引は**読み込み時にも落ちる**ため D-41 が書き・読みの両方向から守られる（`test_an_unknown_key_in_the_index_is_rejected`） |
| 4 | **`replace_note(relpath, [])`（空のチャンク列）は `delete_note` と同じ扱い**にした（0 件のノートを表に残さない） | 残すと `note_chunk_counts()` に `{relpath: 0}` が現れるが、`iter_records()` の集計には 0 件のノートが現れないため、受け入れ基準の「`note_chunk_counts()` が `iter_records()` の集計と一致」が原理的に成立しなくなる。空ノート（0 チャンク）は合成 vault に実在する（`chunk_note` は空白のみのノートに 0 チャンクを返す）ので、T6 が踏む経路である。**T5/T6 への申し送り**: manifest の `chunk_count` と突き合わせるときは `relpath in counts` で分岐し、無ければ 0 として扱うこと（`.get()` は D-25 で使えない） |
| 5 | `iter_records()` の順序を **`(relpath, ordinal)` 昇順に固定**した（仕様は JSONL の**行順**しか規定していない） | 2 つの順序（メモリ上と行順）を持つと、「行順は正しいがメモリ上は投入順」という状態が生まれ、T6 の「他ノートのベクトルがバイト一致」検査を `iter_records()` で書けなくなる。並べ替えは書き出し直前の 1 か所（`_ordered`）だけで行い、`replace_note` の呼び出し順や dict の挿入順が成果物のバイト列に漏れないようにした |
| 6 | ベクトルの**次元検査は読み込み時のみ**行い、`replace_note` では行わない | 次元をレコードに書かない（D-41）以上、ファイル単位の整合を見られるのは読み込み時だけである。書き込み時にも「既存レコードと同じ次元であること」を課すと、**モデル差し替え時の全再構築（fingerprint 不一致 → 既存 `chunks.jsonl` を破棄して作り直す、§4 論点1）で T6 が採る手順を先回りして縛る**ことになる。読み込み時に落とせば「次元が混ざった索引」は次回の起動で必ず検出される |
| 7 | `replace_note` に**別ノートのチャンクを渡すと `ConfigError`**（黙って受け入れない） | 表の鍵（relpath）と中身が食い違うと、そのノートを次に差し替えたときに他ノートのチャンクが道連れで消える。例外種別は `llmkit` の階層内に留める方針（新しい例外を作らない）に従い `ConfigError` を使った |
| 8 | 索引の読み書きで送出する `ConfigError` のメッセージには**ファイル名（`path.name`）と行番号だけ**を載せ、解決済みの絶対パスも行の中身も載せない。pydantic の `ValidationError` は `loc` と `msg` だけを整形し、`input`（＝ノート本文）は使わない | 索引の中身はノート本文そのものであり、例外メッセージに載せると CLAUDE.md のログ出力ルールに反する経路が索引側にできる（`rag/settings.py` の「設定ファイルに書かれたままの文字列だけを載せる」と同じ扱い）。`test_a_broken_line_never_leaks_the_note_body` が本文と `tmp_path` の両方が現れないことを固定する |
| 9 | 空行（末尾改行を含む）は**レコードとして数えない**。JSON として壊れている行も `ValidationError` 経路（`json_invalid`）で `ConfigError` になるため、`ValueError` の分岐は置かない | JSONL の末尾は改行で終わるため、空行を弾かないと毎回 1 件の偽レコードが生まれる。分岐を 2 本置くと片方が到達不能な死んだコードになる |
| 10 | `JsonlVectorStore` に読み取り専用の `path` プロパティを足した（`__all__` には出さない） | T7 の `status` / `--dry-run` が「出力先の予定」を報告するのに要る。書き換え口は与えない |
| 11 | `CHUNKS_FILENAME` を `rag/store.py` に置かず、`JsonlVectorStore` は**フルパスを受け取るだけ**にした | §5 T5 が `CHUNKS_FILENAME` を `rag/indexer.py` の定数と定めている。store 側にも置くと出典が 2 か所になる（D-27 の趣旨）。T4 のテストは自前で `"chunks.jsonl"` を定義している |

**変異検証（各 1 回の tool 呼び出しで backup → 変異 → 実行 → 復元まで完結）**

| 変異 | 落ちたテスト |
|---|---|
| JSONL レコードに `embed_text` を足す | `test_the_persisted_record_never_stores_the_body_twice`（**D-41 guard**）/ `test_a_jsonl_store_round_trips_every_record_exactly` / `test_committing_the_same_content_twice_is_byte_identical` / `test_mixed_vector_dimensions_are_rejected` の 4 件 |
| `chunks.jsonl` の行順ソートを外す | `test_committing_the_same_content_twice_is_byte_identical` / `test_every_store_implementation_passes_the_same_contract[jsonl]` / `[memory]` の 3 件 |
| `replace_note` を追記にする | `test_every_store_implementation_passes_the_same_contract[jsonl]` と `[memory]` の **2 実装とも**（D-39 guard） |

**T5 への申し送り**: `.claude/decisions.yaml` は指示どおり編集していない（T8 でまとめて追記する）。guard_test として参照する関数名は D-39 = `tests/test_rag_store.py::test_every_store_implementation_passes_the_same_contract`、D-41 = `tests/test_rag_store.py::test_the_persisted_record_never_stores_the_body_twice` で、いずれも実在する。**D-30 の rule 本文は「vault への I/O は rag/vault.py だけ」と書いているが、構造層 guard の実体は「vault 以外の読み取りも `_FILESYSTEM_READ_ALLOWANCES` に登録した 2 モジュール以外は 0 件」になった**ので、T8 で D-30 を追記・修正する際はこの例外を rule 本文に明記すること（散文と guard が食い違ったままにしない）。実測: `make ci` 緑（708 passed / 1.92s、着手前 692 passed / 約 1.9s から +16 件・+0.02 秒）。`rag/store.py` の `.get()` は 0 件、HTTP メソッド呼び出しとネットワーク系 import も 0 件（AST 実測）。

### T5（`rag/indexer.py` 前半 = fingerprint / マニフェスト / 計画、`llmkit.resolve_embedding_spec` の公開昇格、2026-08-23）

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | **`fingerprint_inputs` / `fingerprint_digest` を `rag.indexer.__all__` に入れない**（関数名は §5 T5 のとおり `harness` と同名のまま。`from rag.indexer import ...` で使う） | `tests/test_rag_layout.py::test_rag_and_harness_never_import_each_other` が `set(rag.__all__).isdisjoint(harness.__all__)` を要求し、`harness.__all__` には既に `fingerprint_digest` / `fingerprint_inputs` が入っている（実測）。`rag.__all__` はサブモジュールの `__all__` の**和集合と一致**することも別テストが要求するので、指示どおりの関数名を保ったまま緑にできる形はこれだけである。改名（`index_fingerprint_inputs` 等）は仕様が指定した名前を変えることになり採らなかった。名前をそろえること自体は意図的（同じ規則であることの表明）で、`test_the_digest_uses_the_same_canonical_json_rules_as_the_harness` が両者の一致を固定する |
| 2 | `chunk_algorithm_sha256` は **`rag.chunker._CJK_RANGES` をモジュール属性として実行時に読み**、`sorted()` してから `json.dumps` した文字列の sha256 にした | (i) `from rag.chunker import _CJK_RANGES` で束ねると**束ねた時点の値に固定**され、表を差し替えても伝わらない（E32 が monkeypatch で検出できなくなる）ため `from rag import chunker` でモジュールごと持つ (ii) private 定数を読むのは、`rag/chunker.py` が変更禁止で公開の出口が無いため。表を複製すれば D-32 が禁じた「近似の定義が 2 か所」になる (iii) **昇順に並べ替えてから畳む**のは、`_is_cjk` が `any` で判定するので**並び順が挙動に影響しない**ため。並び順まで拾うと、意味の変わらない並べ替えが全再構築を引き起こす。実測: 1 範囲追加 → `f2865f4a` → `4a9b9c67`（変化）／並べ替えのみ → `f2865f4a`（不変） |
| 3 | `manifest_path` / `chunks_path` / `load_manifest` / `write_manifest` は **`Path` ではなく `RagSettings` を受け取る** | ファイル名の出典を `rag/indexer.py` の 2 定数だけに閉じる（T4 決定11 の続き）。`Path` を受ける形にすると T7 の CLI が `index_dir / "manifest.json"` を組み立てることになり、出典が 2 か所になる（D-27 の趣旨）。`chunks_path` は T6 が `JsonlVectorStore` に渡すために公開する |
| 4 | `IndexManifest.fingerprint_inputs` の型を **`dict[str, object]`**（`Any` ではない）にした。pydantic が `object` を任意値として検証できることを実測して採用 | fingerprint の入力は「何が変わったのかを追うための記録」であって、索引側が解釈する構造ではない。ここを型で固定すると、fingerprint に項目を足すたびに manifest のスキーマ変更（= 全再構築）が要る。`harness` の `run.json` が入力をそのまま載せているのと同じ扱い |
| 5 | `load_manifest` は **(a) `schema_version` が実装と違う (b) 記録された `index_fingerprint` が `fingerprint_inputs` から再計算した値と食い違う** の 2 つを `ConfigError` にする（対処は `--rebuild`） | 同じ真実を 2 つ載せる以上、食い違った manifest は「どちらが本当か」を決められない。そのまま差分判定に使うと「前提が変わったのに変わっていないと名乗る索引」を信じることになる（§2.2 と同型）。知らない形式バージョンを黙って上書きするのも不可逆な破棄になる |
| 6 | `totals` は **書き出す側でだけ** `notes[]` との一致を検査し（`write_manifest` が `ConfigError`）、**読み込み側では検査しない** | `totals` は `notes[]` から導出できる重複（D-19 の対象）だが、読み込み側で拒否すると「manifest とストアが食い違った索引を**再処理で直す**」という T5 の自己修復経路そのものが到達不能になる（拒否して止まるのは、直せるものを直さない）。生成経路を 1 本に絞れば「集計だけがずれた索引」は作れない |
| 7 | fingerprint 不一致のときは **全件 `changed`**（`new` を空にする）。マニフェストが**無い**ときは全件 `new`。加えて `IndexPlan.full_rebuild` フラグを持たせた | §5 T5 の「fingerprint 不一致なら全件 changed」に従った。不一致のときは `new` と `changed` の区別に情報が無く（どちらも全量再埋め込み）、T6 が依存する不変条件は「`unchanged` が空であること」だけになる。一方 manifest が無い初回は「再利用できる索引が存在しない」だけなので `new` のままにした（`--dry-run` の表示が「変更 11 件」になると嘘になる）。「全再構築か」を CLI が件数から推測せずに済むよう、判定結果そのものを `full_rebuild` として持たせた |
| 8 | `IndexPlan.note_digests` は `types.MappingProxyType` で包んで返す | frozen dataclass に素の `dict` を持たせると呼び出し側から書き換えられる（F-9-003 と同型）。T6 はこの表をそのまま次の manifest の材料にするため、計画と成果物の間で内容が変わる経路を作らない。テストで「`dict` そのものではない」ことを固定した |
| 9 | 「ストアに無いノートは 0 チャンク」の分岐を `_stored_chunk_count` という**関数に切り出した** | D-25 で `.get()` が使えず、`in` + 添字を三項演算子で書くと ruff **SIM401** が `.get` を勧め、`if`/`else` ブロックで書くと **SIM108** が三項演算子を勧める（実測で SIM108 が発火した）。早期 return の関数にすると両方と衝突しない。`# noqa` を 1 つも足さずに済む形はこれだけだった |
| 10 | `Sha256Hex` の `Field(pattern=...)` は `\A...\Z` ではなく `^...$` で書いた | pydantic v2 の `pattern` は **Rust の正規表現**で評価され、`\A` / `\Z` は `unrecognized escape sequence` で**クラス定義時に落ちる**（実測）。`rag/settings.py` の `_VAULT_ID_PATTERN` は Python 側で `re.compile` しているため同じ書き方が使えない |
| 11 | **`tests/test_rag_layout.py` の `_FILESYSTEM_READ_ALLOWANCES` に `"indexer.py": frozenset({"read_text"})` を追加**した（`_SUBMODULE_NAMES` への `"indexer"` 追加に加えて、同ファイルの 2 か所目の変更）。T4 決定2 と同型の 3 件目 | `load_manifest` が `manifest.json` を読むため、T4 と同じ理由で必要になる（D-30 の構造層 guard は名前ベースで `read_text` を一律に落とす）。許可は `read_text` だけで、`open` / `read_bytes` / `glob` / `iterdir` は 1 つも許していない。迂回策（`io.FileIO`、`Path.read_text` を変数に束ねる等）は採らなかった。**`test_only_the_vault_and_settings_modules_know_where_the_vault_is` は緑**（下記12） |
| 12 | `rag/indexer.py` に文字列 `vault_dir` を **docstring も含めて 1 度も書かない**（`vault のルート` と書く） | 上記の不変条件テストは AST ではなく**行のテキスト走査**で `vault_dir` を探すため、説明文に書いただけで落ちる（実際に 2 か所で落ちた）。検査を弱める（走査を AST 化する）のではなく、実装側が「vault の場所を知らない」ことを字義どおり満たす形にした。結果として `indexer.py` は vault 配下のパスを組み立てられず、`read_text` の許可があっても vault には届かない |
| 13 | E34（埋め込みモデルの掃引）のテストは **2 つの設定をどちらも `is_local = false` にして** `embedding` の 1 行だけを変えた | `llmkit/catalog.py` に role=embedding のモデルは `ruri-v3-310m` の**1 つしか無い**ため、ローカル経路では「別の埋め込みモデル」を作れない。外部 API 経路（`resolve_model_spec` の passthrough）を使えばカタログを増やさずに掃引できる。カタログにダミーのモデルを足す案は D-01（静的テーブルが唯一の出典）を汚すので採らない |
| 14 | `tests/conftest.py` を**変更していない**。fingerprint の掃引用の `RagSettings` はテスト側で直接組み立てる | 指示どおり既存テストの編集を `test_rag_layout.py` の 2 か所に限った。`write_rag_settings` は `[chunk]` / `[embed]` を書けないが、fingerprint はファイルを 1 バイトも読まないので、素の `RagSettings` を組めば掃引できる（「掃引にディレクトリを用意する必要が無いこと」自体が D-35 の主張の一部）。T6 で決定論的フェイク埋め込みを conftest に足す際も、この 2 ファイルは触らずに済むはず |

**変異検証（各 1 回の tool 呼び出しで backup → 変異 → 実行 → 復元まで完結）**

| 変異 | 落ちたテスト |
|---|---|
| fingerprint の入力から `chunk` を落とす | `test_the_fingerprint_covers_every_input_that_changes_a_vector` の **7 パラメタすべて**（D-35 guard、`chunk.max_tokens` で検出）/ `test_changing_chunk_settings_invalidates_the_whole_index` の **4 パラメタすべて**（E31）の計 11 件 |
| fingerprint の入力に `source_path` を足す | `test_the_fingerprint_covers_every_input_that_changes_a_vector[source_path]`（**D-35 guard の不変側**）/ `test_the_manifest_records_no_time_no_run_id_and_no_absolute_path`（成果物に絶対パスが載ることを検出）の 2 件 |
| `chunk_algorithm_sha256` を手書き定数にする | `test_the_token_algorithm_is_part_of_the_fingerprint`（**E32**）/ `test_the_fingerprint_covers_every_input_that_changes_a_vector` の 7 パラメタすべての計 8 件 |

**fingerprint の掃引（実測、ハッシュ先頭 8 文字）**: baseline `f2865f4a`。
**変わらない 7 項目** — `source_path` / vault のルート / `index_dir` / `runtime.base_url` / `embed.batch_size` / `include_globs` / `exclude_globs` はいずれも `f2865f4a`（変化 0 件）。
**変わる項目** — `max_tokens` 240→200 `e9792f78` / `cjk_chars_per_token` 1.0→1.5 `f93a9316` / `ascii_chars_per_token` 4.0→3.0 `c70130b0` / `heading_separator` `" > "`→`" / "` `e27a93de` / `embedding.model_id` `71115834` / `embedding.served_name` `9d45f3cb` / `vault_id` `5dd8aafd` / CJK 範囲表に 1 範囲追加 `4a9b9c67`（並べ替えのみは `f2865f4a` のまま）。

**`plan_index` の実測（合成 vault 11 ノート / 24 チャンク、`httpx.MockTransport` の埋め込みクライアントを生成した状態で計画）**

| 実行 | notes | new | changed | unchanged | deleted | rebuild |
|---|---|---|---|---|---|---|
| 初回（manifest 無し） | 11 | 11 | 0 | 0 | 0 | true |
| 無変更の再計画 | 11 | 0 | 0 | 11 | 0 | false |
| 1 編集 + 1 削除 + 1 追加 | 11 | 1 | 1 | 9 | 1 | false |
| fingerprint 不一致 | 11 | 0 | 11 | 0 | 1 | true |

**HTTP 呼び出し 0 回 / `index_dir` は作られず 0 バイト**（実測）。0 チャンクのノート（`empty.md` / `whitespace-only.md`）はストアの表に現れず（11 ノート中 9 ノートのみ）、それでも `unchanged` に入る（T4 決定4 の申し送りどおり `relpath in counts` で分岐した結果）。

**T6 への申し送り**:
1. `.claude/decisions.yaml` は指示どおり編集していない（T8 でまとめて追記）。guard_test として参照する関数名は D-35 = `tests/test_rag_indexer.py::test_the_fingerprint_covers_every_input_that_changes_a_vector` で、実在する（E31 = `test_changing_chunk_settings_invalidates_the_whole_index` / E32 = `test_the_token_algorithm_is_part_of_the_fingerprint` / E34 = `test_changing_the_embedding_model_invalidates_the_index` も同様）。
2. **D-35 の第 2 条項（ランタイムが名乗ったモデル名・次元が既存 manifest と食い違ったら 1 バイトも書かずに中断）は T5 では未実装**。`IndexManifest.embedding`（`ManifestEmbedding`）という置き場所だけを用意した。T6 が `EmbeddingBatch.model` / `.dimensions` と突き合わせること。
3. `write_manifest` は `totals` の整合を検査するので、T6 は `notes[]` から導出した値を渡すこと（手で数えない）。
4. `plan.note_digests` をそのまま次の manifest の `sha256` に使える（vault を 2 回読まない）。
5. マニフェストの書き出しは `harness/report.py` と同じ「`mkdir` → `write_text`」で、**一時ファイル + `os.replace` にしていない**（原子性を持つのは `store.commit()` 側）。D-37 の順序（`store.commit()` → `write_manifest()`）を守る限り、manifest 書き出し中のクラッシュは「manifest が古い = 再処理される」安全側に倒れる。

**実測**: `make ci` 緑（**742 passed / 2.21〜2.23s**、T5 着手前 709 passed / 約 1.96s から +33 件・+0.3 秒以内）。`rag/indexer.py` の `.get()` 呼び出しは 0 件（残る 3 件はすべて説明文中の文字列）、HTTP メソッド呼び出しとネットワーク系 import も 0 件、`vault_dir` の出現 0 件。`llmkit` 側の `tests/test_layout.py` の和集合検査は**無修正で緑**（`resolve_embedding_spec` を `llmkit/embeddings.py` の `__all__` と `llmkit/__init__.py` の両方へ足したため自動的に一致する）。

### T6（`rag/indexer.py` 後半 = `build_index`、2026-08-23）

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | **`build_index` に `rebuild` フラグを作らず、`manifest=None` を渡す経路を全再構築とした**（T7 の `--rebuild` は `load_manifest` を呼ばずに `None` を渡す） | §5 T6 は引数を `(settings, config, *, embedding_client, store, ...)` としか決めていない。フラグを足すと「全再構築か」の判定が `plan_index`（fingerprint 不一致）と `build_index`（フラグ）の 2 か所に分かれる（D-27 の趣旨）。`manifest=None` は `plan_index` が既に「再利用できる索引が無い」と解釈する既存の入力で、意味が 1 つしかない |
| 2 | **致命的な障害（`RuntimeUnavailableError` / `OutOfMemoryError`）は `store.commit()` → `write_manifest()` で確定してから再送出する**。`IndexResult` は返さない | 仕様は「即座に中断（それまでに完了したノートは commit 済み）」とだけ書いており、戻り値か例外かを決めていない。結果を返すと、ランタイムが落ちて 3/11 ノートしか索引できなかった実行が「成功」として exit 0 を名乗れる（T7 の終了コードは「全ノート失敗で 1」なので部分成功は 0 になる）。§5 T7 が「`LlmkitError` 捕捉 → exit 1」を既に用意しているので、受け皿は存在する |
| 3 | **`ModelNotFoundError` を致命的グループに加えた**（§4 論点5 の表は 4 種類しか挙げていない） | 「次のノートでも必ず再発する」という表の区分基準にそのまま当てはまる。表に無いからと未捕捉のまま通すと、確定を行わずに送出され**完了済みのノートまで捨てる**（他の致命的障害と挙動が食い違う） |
| 4 | ノートの**読み取り・パース・チャンク分割で出た `ConfigError`**（見出し経路だけで `max_tokens` を使い切る等）は、そのノート固有の失敗として計上し継続する | 埋め込み以前の失敗を仕様は扱っていない。この `ConfigError` は見出しの深さに依存するのでノート固有であり、1 件で索引全体を止める理由が無い。要求を 1 回も出さないので `request_count` にも影響しない |
| 5 | **ノート境界を跨いだバッチが `ContextLengthError` / `UpstreamError` で落ちたら、そのバッチをノート単位に切り直して送り直す**（`_isolate`） | 仕様は「バッチはノート境界を跨いでよい」と「失敗したノートをスキップする」を同時に要求しているが、**切り分け方法を決めていない**。バッチ単位で失敗を記録すると、長すぎるチャンクを 1 つ含むノートと同居しただけで健全なノートが索引から静かに落ちる（受け入れ基準「他 10 ノートは索引され」が原理的に満たせない）。切り直しの追加要求は最大でもバッチ内のノート数（実測: 合成 vault で +2 件） |
| 6 | 失敗が確定したノートの**残りチャンクは送らず、後続のチャンクを詰め直す** | どうせ破棄するチャンクに要求を払う理由が無い。実測: 9 チャンクのノートが 2 件目のバッチで失敗した場合、残り 2 バッチ分（6 件）を送らずに済む |
| 7 | **失敗したノート・未処理のノートは、前回のマニフェストの記録（古い `sha256` と `chunk_count`）をそのまま残す**。全再構築のときだけ落とす | 失敗したノートのチャンクはストアに古いまま残っている（`replace_note` を呼んでいないため）。記録を消すとマニフェストとストアが食い違い、`plan_index` の自己修復に頼るだけの状態になる。古い記録を残せば「古い版が索引されている」という**真**の主張になり、`sha256` の違いで次回必ず再試行される。全再構築ではストアを空にしているので、残すと「チャンクの無いノートを索引済みと名乗る」ことになるため落とす。実装は「前回のマニフェスト − 消えたノート」を土台にし、今回書き終えたノートだけを上書きする 1 本の経路にした（新規で失敗したノートは 1 度も載らない = D-38） |
| 8 | `IndexResult` は**件数だけ**を持ち、相対パスの一覧を持たない。`dimensions` は欠測時 `None`（0 で埋めない、D-07） | §5 T7 が「出力は件数と fingerprint のみ。相対パスも出さない」と決めており、型に一覧があると CLI がうっかり出せる。どのノートが失敗したかは WARNING ログとマニフェストの差分で分かる |
| 9 | **埋め込みが 1 度も成立せず、既存マニフェストの申告も無い場合はマニフェストを書かない**（`IndexResult.dimensions` も `None`） | `ManifestEmbedding` は `reported_model` と `dimensions` を必須にしており、ランタイムが 1 度も答えていない状態で埋めると**捏造**になる（D-07 の「欠測を 0 で埋めない」と同型）。索引すべきチャンクが 0 件（空ノートだけの vault）でも要求は 0 回なので、この経路は実在する |
| 10 | `embedded_chunks` は「**索引に書き込まれた**チャンク数」と定義した（埋め込みには成功したがノートが完成せず破棄したチャンクは数えない） | 「再埋め込み 0」を数える指標が、破棄された作業を数えると受け入れ基準（L310）の検証に使えなくなる。要求回数のほうは `request_count` が失敗した要求も含めて数える（実際に送った回数という別の量） |
| 11 | 埋め込みへ送るテキストは `chunk.embed_text` をそのまま使う（`render_embed_text` で組み直さない） | 上限判定（D-32）を通した値そのものを送るのが最も素直で、レコードから再構成した文字列と一致することは T4 の D-41 guard が既に固定している。T6 側でも `test_every_stored_vector_matches_the_text_that_was_embedded` が「保存済みレコードから再構成した文字列の埋め込み == 保存されたベクトル」を全チャンクで確かめる |
| 12 | ランタイム申告の一致検査は、**全再構築でないときだけ**既存マニフェストを基準にし、それ以外は**1 回目の応答**を基準にする | 全再構築では全チャンクを埋め直すので、モデルが変わっていても混ざらない（むしろ E34 の正常系）。一方で 1 回目の応答を基準に据えることで、**実行の途中でランタイムがモデルを差し替えた**場合も同じ 1 つの検査で落ちる |
| 13 | 失敗の WARNING には `relpath` と例外**クラス名**を載せる（例外メッセージは載せない）。INFO は件数のみ | ソフト制約は「INFO は件数、DEBUG で相対パスまで」だが、識別子の無い WARNING は運用で役に立たない。`rag/vault.py` が既に WARNING で `relpath` を出している前例に合わせた。例外メッセージを載せないのは `harness/runner.py` と同じ（`type(exc).__name__` だけ）。実測: 索引 1 回分の全ログ 63 行に本文 0 件 / vault の絶対パス 0 件 / `index_dir` の絶対パス 0 件 |
| 14 | 0 チャンクのノートは**要求を 1 回も出さずに確定**し、マニフェストに `chunk_count=0` で載せる（ストアの表には現れない = T4 決定4） | 載せないと毎回「新規」として計画され続ける。要求は 0 回のままなので件数の一致では気づけない静かな欠陥になる |
| 15 | テストの埋め込みは**すべて `httpx.MockTransport` 経由**にし、`EmbeddingClient` のフェイク実装（クラス）を作らなかった。障害注入は応答（ステータス + 本文）と `httpx.ConnectError` で行う | `llmkit` の例外翻訳表を通らないフェイクを置くと、`rag` が受け取る例外の形をテストが**自分で決めてしまう**。応答から作れば、翻訳表を変えた瞬間に T6 のテストが追随する。conftest に足したのは `fake_embedding_vector`（入力テキストの sha256 から導出）/ `fake_embedding_payload` / `fake_embedding_handler` / `fake_embedding_transport` の 4 つで、既存 fixture は 1 行も変更していない |
| 16 | **`tests/test_rag_vault.py` の D-30 動的 guard を「vault 全体のパース」から `build_index` の実行に差し替えた**（既存テストの**強化**。指示で明示的に許可された唯一の変更）。`parse_whole_vault` ヘルパは権限・シンボリックリンクのテストが引き続き使うので残した | 3a はまだ索引本体が無く代用していた。パースだけでは「書き出しの段階で vault に触れる」実装を 1 つも検出できない。差し替えたテストは索引が**実際に成果物を書いた**こと（`indexed_notes == 11` / `embedded_chunks == 24` / 2 ファイルの存在）も同時に主張するので、「何もしなかったから一致した」では通らない |
| 17 | `tests/test_rag_indexer.py` の module docstring を更新した（T5 が書いた「差分判定の検証は T6 で追加する」が事実に合わなくなるため） | 散文と実体を食い違わせない。アサーションは 1 つも緩めていない |

**変異検証（各 1 回の tool 呼び出しで backup → 変異 → 実行 → 復元まで完結。復元後に `git status` が空であることと、`git log -p` に変異の痕跡が 0 件であることを確認済み）**

| 変異 | 落ちたテスト |
|---|---|
| 差分判定を mtime にする | **15 件**。`test_touching_a_note_without_changing_bytes_reindexes_nothing`（**D-36 guard**）/ `test_only_the_edited_note_is_reembedded`（E29）/ `test_an_unchanged_vault_plans_no_work_at_all` ほか T5 の計画テスト群 |
| 書き出し順序を manifest → store に入れ替える | **1 件**。`test_the_manifest_is_written_only_after_the_store_is_committed`（D-37 の順序） |
| 失敗時に部分成功チャンクを索引とマニフェストに載せる | **2 件**。`test_a_failed_note_is_never_recorded_as_indexed[upstream]` / `[context_length]`（**D-38 guard**。他は 1 件も落ちない = この guard だけが見ている性質） |
| `chunks.jsonl` の行順ソートを外す | **5 件**。`test_a_failed_note_is_never_recorded_as_indexed[upstream]` / `[context_length]`（後から書き足したノートの行が末尾に固まる）/ T4 の `test_committing_the_same_content_twice_is_byte_identical` / `test_every_store_implementation_passes_the_same_contract[jsonl]` / `[memory]` |
| `RuntimeUnavailableError` でも継続する | **1 件**。`test_a_fatal_runtime_failure_stops_the_index_immediately[unavailable]` |

> 補足: 「無変更の再実行がバイト一致」だけでは行順ソートの欠落を検出できない（`replace_note` が dict のキー位置を保つため、初回の投入順がたまたま昇順になる）。そのため `chunks.jsonl` の**ファイル上の行順**を直接読む `stored_order()` を足し、初回索引と「失敗したノートを後から書き足した」再実行の 2 か所で `(relpath, ordinal)` 昇順を主張している。

**合成 vault の実測（11 ノート / 24 チャンク、`httpx.MockTransport` + 決定論的フェイク埋め込み、既定 `batch_size=16`）**

| 実行 | indexed | 再埋め込み | 要求 | skipped | 所要 | 成果物 |
|---|---|---|---|---|---|---|
| 初回 | 11 | 24 | **2**（= ⌈24/16⌉。ノート数 11 ではない = バッチがノート境界を跨いでいる） | 0 | 7.7 ms | `chunks.jsonl` 12,708 B / `manifest.json` 2,518 B |
| 無変更の再実行 | 0 | **0** | **0** | 11 | 2.4 ms | 2 ファイルとも **sha256 一致** |
| 1 ノート編集 | 1 | **3**（そのノートのチャンク数と一致） | 1 | 10 | — | 他 10 ノートのレコードが**バイト一致**、`index_fingerprint` 不変 |

その他の実測: 次元 8 / 768 のどちらでも成立（768 は `test_a_768_dimensional_runtime_round_trips_through_the_index`）。`batch_size` 4 → 6 要求 / 16 → 2 要求で `chunks.jsonl` と `manifest.json` はバイト一致（E28）。`exclude_globs` に 1 パターン追加 → 再埋め込み 0 / 要求 0 / fingerprint 不変（E33）。`max_tokens` 240 → 120 で全ノート再構築（E31 実行側）。`batch_size=4` で 9 チャンクのノートの**最後の 1 チャンクだけ**を失敗させると、要求 9 回・索引 10 ノート / 15 チャンク・失敗 1 ノートになり、先行して埋め込めた 7 チャンクは**破棄**される（D-38）。

**`make ci` 緑（762 passed / 2.54s、T6 着手前 742 passed / 2.21〜2.26s から +20 件・+0.3 秒）。** `uv lock --check` 無変更。`rag/indexer.py` の `.get()` 呼び出し 0 件 / `vault_dir` の属性参照 0 件 / HTTP メソッド呼び出し 0 件（既存の AST guard が緑）。

**T7 への申し送り**:
1. `.claude/decisions.yaml` は指示どおり編集していない（T8 でまとめて追記）。T6 が固定する guard_test はすべて実在する: D-36 = `tests/test_rag_indexer.py::test_touching_a_note_without_changing_bytes_reindexes_nothing` / D-37 = `::test_an_unchanged_rerun_leaves_the_index_byte_identical` / D-38 = `::test_a_failed_note_is_never_recorded_as_indexed`（parametrize 2 件）/ D-30 動的 = `tests/test_rag_vault.py::test_indexing_leaves_every_vault_file_byte_identical`（**索引本体に差し替え済み**）。
2. **`--rebuild` は `build_index` に渡すフラグではない**。`manifest=None`（= `load_manifest` を呼ばない）で全再構築になる（上記決定1）。
3. `index` サブコマンドの終了コードは `IndexResult.indexed_notes == 0 and failed_notes > 0` で「全ノート失敗」を判定できる。ランタイム障害は `LlmkitError` として送出されるので、`harness/cli.py` と同じ捕捉で exit 1 になる（成果物は送出前に確定済み）。
4. `--dry-run` は `plan_index` だけを呼ぶこと（`build_index` は必ず `store.commit()` を呼ぶので、計画だけのつもりで呼ぶと `index_dir` が作られる）。
5. `status` に出せる値は `IndexManifest`（`totals` / `embedding.dimensions` / `index_fingerprint`）と `plan_index` の件数。`IndexResult` は相対パスを持たないので、そのまま画面へ出して差し支えない。
6. 索引の実行を注入込みで書く形は `tests/test_rag_indexer.py::index_once` にある（`JsonlVectorStore(chunks_path(settings))` + `load_manifest(settings)` + `create_embedding_client(config, http_client=...)`）。CLI もこの 3 つを組み立てるだけで済む。

### T7（`rag/cli.py` = `index` / `status`、2026-08-23）

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | **`tests/test_rag_layout.py::test_every_rag_submodule_is_covered_by_the_union_check` の右辺を `set(_SUBMODULE_NAMES)` → `{*_SUBMODULE_NAMES, "cli"}` に変えた**（`_SUBMODULE_NAMES` 自体は無変更。指示で許されていない既存テストの編集であり、**次の周で承認/再設計の判断が要る**） | このテストは `rag/*.py` の実ファイル集合と `_SUBMODULE_NAMES` の**完全一致**を要求する。仕様は「`cli` を `_SUBMODULE_NAMES` に入れない」と明示しているので、`rag/cli.py` を作った瞬間にこのテストは**原理的に落ちる**（実測: 762 件中この 1 件だけが落ちた）。回避策として `rag/cli/__init__.py` のパッケージ化（`glob("*.py")` に引っかからない）が技術的には可能だが、検査を迂回するためだけにレイアウトを歪める行為であり採らない。採った形は `tests/test_harness_layout.py` の同名テスト（`discovered == {*_SUBMODULE_NAMES, "cli"}`、L122）の**逐語の写し**で、完全一致の要求は保たれる（サブモジュールを足して再エクスポートを忘れれば依然落ちる = 検査の強さは 1 ビットも下がっていない）。`_SUBMODULE_NAMES` を含む側に緩める形（`discovered >= ...`）は採らなかった |
| 2 | **`main()` の注入点を `http_client` ではなく `embedding_client` 1 つにした**（§5 T7 が書いた `main(argv, *, http_client=..., embedding_client=..., stdout=..., stderr=...)` から `http_client` を落とした）。★**仕様と明示的に食い違う唯一の点** | `http_client` を受けるには型 `httpx.Client \| None` を書く必要があり、そのためには `rag/cli.py` が `httpx` を import することになる。すると **D-25 guard（`tests/test_rag_layout.py::test_rag_never_talks_to_the_runtime_directly`）が落ちる**（同テストは `rag/*.py` 全件について `httpx` / `requests` / `urllib` / `http` / `socket` の import を一律に禁じており、`cli.py` も対象）。緑にするには guard に `cli.py` の例外を足すしかなく、それは**アーキテクチャ境界の検査を弱める**編集になる。一方 `embedding_client` は `llmkit` の公開 Protocol であり、テストは `llmkit.create_embedding_client(config, http_client=MockTransport のクライアント)` を渡すだけで同じ注入ができる（`tests/test_rag_indexer.py::index_once` と同型）。`harness/cli.py` が `http_client` を持つのは harness に D-25 が無いからで、`rag` に同じ形を持ち込む必然性は無い。**「CLI の私的な注入点の形」より「L3 が HTTP を知らない」ほうが守る価値が高い**と判断した。`test_rag_cli.py::test_the_cli_never_imports_an_http_library` が名指しでも固定する |
| 3 | **`--settings` を `required=True` にした**（既定値を持たせない）。`--config` は `configs/default.toml` を既定にする | §5 T7 の書式が `--settings vaults/sample.toml`（角括弧なし）/ `[--config ...]`（角括弧あり）で必須と任意を書き分けている。既定値を `vaults/sample.toml` にすると、`--settings` を打ち忘れた実行が**エラーにならず出荷済みの合成 vault を索引する**。実 vault を索引したつもりの実行が黙って別の索引を更新するのは、`harness` の `--suite` 既定（結果が `results/` に別 id で残るので気づける）とは危険度が違う |
| 4 | **`--dry-run` の出力に「予定リクエスト数」を入れなかった**（§4 論点6 の表 #1 は挙げている） | 正確な予定リクエスト数は「再処理対象ノートのチャンク数の合計 ÷ `batch_size`」で、チャンク数を知るには CLI が `read_note_text` → `parse_note` → `chunk_note` を回す必要がある。それは `build_index` の前半を L4 に**もう 1 本実装する**ことであり（D-27 の趣旨に反する）、vault を 2 度読むことにもなる。§5 T7 の受け入れ基準はこの項目を要求していない。**T8 への申し送り**: 実測 #2〜#5 の要求回数は `IndexResult.request_count`（画面に「リクエスト : N 回」として出る）から取れる。#1 の「予定リクエスト数」も報告に要るなら、`plan_index` に `pending_chunk_counts` を持たせるのが正しい置き場所で、それは仕様変更なので planner に戻す |
| 5 | **「出力先の予定」はファイル名だけを出す**（`index.dir 直下の manifest.json / chunks.jsonl`）。ディレクトリは相対パスも出さない | §5 T7 は「計画と `index_fingerprint` と**出力先の予定**を出す」と「出力は件数と fingerprint のみ。**相対パスも出さない**」を同時に要求している。両立する形は「ファイル名だけ」しかない。`rag/indexer.py` の例外メッセージが `path.name` だけを載せているのと同じ扱いで、出典（`index.dir`）は設定ファイルにあると案内する |
| 6 | `IndexResult.dimensions` が `None` のときは **`—`**（`harness/report.py` の `MISSING_CELL` と同じ字）を出す。`未構築` とは書き分ける | 1 度も埋め込めなかった実行（全ノート失敗）で `0` を出すと「0 次元のベクトルを作った」と読める嘘になる（D-07）。当初 `未構築` を流用していたが、実機実行の出力を見て「索引は書きかけで存在するのに未構築と名乗る」矛盾に気づいたので分けた |
| 7 | `status` は `load_manifest` に加えて **`plan_index` も呼ぶ**（HTTP 0 回・書き込み 0 バイト） | 受け入れ基準が「`index` → `status` で **fingerprint 一致 / 再処理 0 件**を出す」と要求しており、「再処理 0 件」はマニフェスト単体からは出せない（vault の現在のバイト列と突き合わせて初めて決まる）。`plan_index` は 1 バイトも書かないので `status` の副作用ゼロは保たれる（`test_status_never_writes_anything` が成果物のバイト一致で固定） |
| 8 | `status` は fingerprint 不一致でも **exit 0** | 「前提が変わっているので次回は全再構築になる」は正しい状態の報告であって失敗ではない。失敗にすると `status` が CI の判定に使われたときに「索引を作り直すまで赤」になり、`index` を回す前に必ず赤という無意味な状態が生まれる |
| 9 | **`--rebuild` は `chunks.jsonl` が壊れている場合の逃げ道にはならない**（既知の限界。今回は塞いでいない） | `--rebuild` は `load_manifest` を呼ばないので壊れた `manifest.json` からは復帰できるが、`JsonlVectorStore(chunks_path(settings))` の生成時読み込みは通るため、壊れた `chunks.jsonl` は `ConfigError` になる。塞ぐには「空のストアから始める」入口が `rag/store.py` に要り、それは変更禁止のモジュールへの API 追加になる。利用者は索引ディレクトリを消せば回復できる。**T8 / 次サイクルへの申し送り**: `JsonlVectorStore(path, *, load=True)` か `--rebuild` 時の破棄を `rag/store.py` 側に足すのが筋 |
| 10 | ノート単位の失敗があっても **1 件でも索引できていれば exit 0**（`indexed_notes == 0 and failed_notes > 0` のときだけ exit 1） | §5 T7 の「全ノート失敗 1」をそのまま実装した（T6 申し送り3 と同じ判定）。合成 vault は 0 チャンクのノートを 2 件含み、それらは要求を 1 回も出さずに確定する（T6 決定14）ため、**合成 vault では「全ノート失敗」を作れない**。全件失敗の検証には本文を持つ 2 ノートだけの別 vault を使った |
| 11 | `.get()` を 1 つも書かず、`argparse.Namespace` からの取り出しは `_get_path` / `_get_flag` のヘルパに閉じた | D-25 guard は `.get()` 呼び出しを一律に落とす（`dict.get` ごと）。`Namespace` は属性アクセスなので衝突しないが、`vars(args).get(...)` と書きたくなる場所を最初から潰しておく。`namespace.command` だけは ruff **B009**（`getattr` に定数を渡すな）に触れたので直接の属性参照にした |
| 12 | `tests/conftest.py` を**変更していない**。合成 vault の複製は `shutil.copytree` をテスト側で直接呼ぶ（`sample_vault_copy` fixture を使わない） | 1 テストの中で **2 つの vault**（合成 vault + E36 用の別 vault）と **2 つの出力先**（E35）を作る必要があり、fixture の固定パス（`tmp_path/vault`）では足りない。既存 fixture を拡張すると他の 700 件超に影響する |

**変異検証（各 1 回の tool 呼び出しで backup → 変異 → 実行 → 復元まで完結。復元後に `git log -p -- rag/cli.py` へ変異の痕跡が 0 件であることを確認済み）**

| 変異 | 落ちたテスト |
|---|---|
| `--dry-run` の経路に `store.commit()` を足す（計画だけのつもりが書き込む） | **1 件**。`test_dry_run_issues_no_http_and_creates_no_index_directory`（`index_dir` が作られたことを検出） |
| `--rebuild` を無視する（常に `load_manifest` を呼ぶ） | **2 件**。`test_rebuild_reindexes_everything_even_when_the_fingerprint_matches`（索引 11 件 → 0 件）/ `test_rebuild_with_dry_run_plans_a_full_rebuild_without_any_http`（全再構築 はい → いいえ）。モード表示だけを見る `test_rebuild_reports_the_full_rebuild_mode` は落ちない（= 挙動を見ているのはこの 2 件だけ、という切り分けの証拠） |

**合成 vault の実測（11 ノート / 24 チャンク、`httpx.MockTransport` + 決定論的フェイク埋め込み、次元 8、`batch_size=16`）**

| 実行 | exit | HTTP | 索引したノート | 埋め込みチャンク | 再処理しない | `index_dir` |
|---|---|---|---|---|---|---|
| `index --dry-run` | 0 | **0 回** | — | — | — | **作られない** |
| `index`（初回） | 0 | 2 回 | 11 | 24 | 0 | 作られる |
| `status` | 0 | 0 回 | — | — | 再処理 **0 件** / fingerprint 一致 **はい** | 変化なし |
| `index`（2 回目） | 0 | **0 回** | **0** | **0** | 11 | 成果物バイト一致 |
| `index --rebuild` | 0 | 2 回 | 11 | 24 | 0 | 成果物バイト一致 |

`index_fingerprint` は 5 回とも `f2865f4a…`（T5 が記録した baseline と一致）。

**終了コードの実測**: 成功 = 0 / `ConfigError`（`index.dir` が vault 配下）= 1 / `ConfigError`（設定ファイルが無い）= 1 / 全ノート失敗（本文を持つ 2 ノートの vault を 5xx で全滅）= 1。

**出力の走査（実測）**: 上表 5 回分の stdout + stderr **計 58 行**に対し、絶対パス **0 件** / 合成 vault の本文行（10 文字以上、照合 53 行）**0 件** / `[[` **0 件**。

**`make ci` 緑（796 passed / 2.95s、T7 着手前 762 passed / 2.55s から +34 件・+0.4 秒）。** `uv lock --check` 無変更。`rag/cli.py` の `.get()` 呼び出し 0 件 / `vault_dir` の属性参照 0 件 / HTTP メソッド呼び出し・ネットワーク系 import 0 件（既存の AST guard が緑）。`_FILESYSTEM_READ_ALLOWANCES` に **`cli.py` を追加していない**（設定ファイルを読むのは `rag/settings.py` の `load_settings` と `llmkit.load_config` で、`rag/cli.py` 自身は `read_text` / `open` を 1 つも呼ばないため）。

**T8 への申し送り**:
1. `.claude/decisions.yaml` は指示どおり編集していない。T7 が固定する評価軸の関数名は実在する: E35 = `tests/test_rag_cli.py::test_index_dir_is_the_only_output_location` / E36 = `tests/test_rag_cli.py::test_the_indexed_note_set_follows_the_given_config`。
2. **上記決定1・決定2 は仕様の明示的な制約（既存テストを 1 行も直さない / `http_client` の注入点）と食い違う。** どちらも「代替案は検査を弱めるか迂回する」ため現在の形を採ったが、T8 の前に承認を取ること。
3. 実機実行（§4 論点6）のコマンドは `python -m rag.cli index --settings <実 vault の設定> --dry-run` → `--rebuild` 無しの `index` → `status` の順。**画面出力はそのまま §9 に貼ってよい**（上記の走査で絶対パス 0 件を実測済み）。ただし `logging` の WARNING はノートの `relpath` を出す（T6 決定13）ので、**ログは貼らない**こと。
4. `--dry-run` は「予定リクエスト数」を出さない（決定4）。§4 論点6 #1 の報告項目からこの 1 項目を落とすか、`plan_index` の拡張を planner に戻すかを T8 で決めること。

---

### T8（受け入れ検証・決定の追記・文書、2026-08-23。**実機実行はメインセッションが行うため未実施**）

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | **`tests/test_acceptance_phase3.py` の 7 テストはすべて「索引成果物」に対して主張する**（`chunks.jsonl` を `JsonlVectorStore` で読み直したレコードとマニフェスト）。3a で満たした L311-L314 も中間表現（`ParsedNote` / `Chunk`）ではなく索引の中身で測り直した | §5 T8 は「3a で満たした L311-L314 / L319 も同ファイルに集約する」としか書いておらず、既存テストの写しでも要件を満たせる。しかし受け入れ条件が問うているのは「**インデックス**に含まれていない」「チャンク本文が `alias` に解決されている」であって、中間表現の性質ではない。列挙 (`iter_vault_files`) が正しくても、索引側が別経路でファイルを拾えば条件は破れる。索引を 1 回回せば 7 条件すべてを同じ経路で測れるため、テストの独立性を落とさずに検査を強められる。実測: 7 テストで 0.12 秒 |
| 2 | **`ACCEPTANCE_MAP` と関数定義の 1 対 1 対応を機械検証するテスト（Phase 1 の `test_l298_automated_tests_exist_for_every_condition` 相当）を置かなかった** | Phase 3 の受け入れ条件に「上記に対する自動テストが存在しパスする」に当たる行が無い。置くと 8 個目のテスト関数になり、指示の「7 行 7 関数」が崩れる（対応表に無い関数が 1 つ増える）。写し元に指定された `test_acceptance_phase2.py` も同じ理由でこの検査を持っていない。**次サイクルで L315-L318 を足すときに、Phase 3 側にも対応検査を置くかを判断すること** |
| 3 | `assert_is_acceptance_line` は指示どおり `test_acceptance_phase2.py` の写しで、**参照行が `- [` で始まることだけ**を見る（行の内容は見ない） | 行番号のずれは検出しないが、要件書は L315-L318 を触らない限り行数が変わらず（本 PR も `wc -l` = 355 を維持）、次サイクルのチェックボックス更新でも行数は変わらない。内容の照合（キーワード一致）を足す案は、要件書の文言を微修正しただけでテストが落ちる形になり、決定47（チェック状態ではなくチェックボックスを見る）と同じ理由で採らない |
| 4 | **D-40 guard は `git check-ignore` の判定に加えて「`data/` 配下に追跡済みファイルが 0 件」も見る**。検査するパスは実在しないもの（`data/index/sample/chunks.jsonl` 等）を名指しで並べる | `.gitignore` の記述を読む案は再包含規則で打ち消され得るので採らない（`rag/settings.py` の設定層検査と同じ判断）。`check-ignore` はパスが実在しなくても判定できるため、索引をまだ作っていない状態でも検査が空回りしない。追跡済み 0 件を併せて見るのは、`.gitignore` が正しくても**既に追跡済みのファイルには無視規則が効かない**ため |
| 5 | **D-30 の rule 本文を、実際に守られている内容へ書き直した**（T4 決定2 / T5 申し送り1 への対応）。旧: 「vault への I/O は `rag/vault.py` だけが行い…」／新: 「vault の場所を知るモジュールを `rag/vault.py` と `rag/settings.py` の 2 つに閉じる。vault の読み取りは `rag/vault.py` だけが行い、他モジュールのファイル読み取りは `_FILESYSTEM_READ_ALLOWANCES` に登録した `settings.py`（設定ファイル自身）/ `store.py`（`chunks.jsonl`）/ `indexer.py`（`manifest.json`）の `read_text` に限る。`open` / `read_bytes` / `glob` / `iterdir` は許可モジュールでも 1 つも書かない…」。`index.dir` の条項に「リポジトリ内の `data/` 以外を指す設定も `ConfigError`」を追記し、構造層に `tests/test_rag_layout.py::test_only_the_vault_and_settings_modules_know_where_the_vault_is`、設定層に `tests/test_rag_settings.py::test_index_dir_inside_the_repository_must_be_ignored_by_git` を列挙した | 散文と guard が食い違ったままの決定は、壊しても何も落ちないのと同じで、決定として弱い（`tests/test_decisions_guards.py` の存在理由そのもの）。列挙した 2 件は同テストが実在を検証するので、改名・削除すれば落ちる。**guard_test（動的層）は変えていない** |
| 6 | `.claude/decisions.yaml` の追記は既存 30 件と同じ「1 行 1 フィールドの二重引用符スカラ」で書き、`rule` に**採らなかった案**を含めた（D-35 の `sha256(str(source_path))`、D-39 の chromadb、D-41 のレコードごとのモデル名など） | `check_decisions.py` の最小パーサは複数行スカラを扱えない（PyYAML 非依存で動く必要がある）。「なぜその案を採らなかったか」を残すのは、後続の fixer が一貫性のために潰す動機を先回りして塞ぐため（D-34 の rationale と同じ趣旨） |
| 7 | **実機実行（§4 論点6 の #1〜#6）は行っていない。** 本節の実測欄はメインセッションが埋める | 指示により実機実行はメインセッションの担当。T8 の実装者は実 vault に一切アクセスしていない（`vaults/local.toml` を作っておらず、`data/` 配下にも 1 バイトも書いていない）。合成 vault だけで検証できる範囲（受け入れテスト・決定の追記・追跡側の検査）を完了させた |

**検証の実測**

| 検証 | 結果 |
|---|---|
| `make ci` | **緑（813 passed / 3.02s）**。着手前 796 passed / 2.88s から +17 件（受け入れ 7 / 索引成果物の追跡検査 1 / `test_decisions_guards.py` の parametrize +9） |
| `wc -l docs/localllmrequirements.md` | **355**（変更は 7 行の `- [ ]` → `- [x]` のみ。7 insertions / 7 deletions） |
| `tests/test_acceptance_phase1.py` / `test_acceptance_phase2.py` | **無修正で 9 passed**（L294-L298 / L302-L305 の行番号参照は無傷） |
| `check_decisions.py` | **37 件で緑** |
| `tests/test_decisions_guards.py` | **47 passed**（決定本文に書いたテスト参照がすべて実在） |
| D-35〜D-41 の guard_test 7 本 | 個別実行ですべて緑（7 / 1 / 1 / 2 / 2 / 1 / 1 件） |
| `git diff --stat` | `rag/` / `llmkit/` / `harness/` のコード変更 **0 件**（テストと文書のみ） |

**メインセッションへの申し送り**:
1. 実機実行（§4 論点6 の #1〜#6）の数値を本節に追記すること。**実行の直後に `git status` と `git check-ignore` を確認**し、`tests/test_rag_privacy.py` を再実行すること（`test_the_index_directory_is_never_tracked` と `test_no_tracked_file_exposes_the_real_user_identity` の 2 本が要）。
2. 上記決定7 のとおり、L309 / L310 のチェックボックスは**合成 vault での機械検証**を根拠に `- [x]` にしてある。実機で #2 / #3 が満たせなかった場合は、チェックを戻すか要件書側に条件を追記するかの判断が要る。
3. 次サイクルへの申し送り（`Chunk.embed_text` の property 化 / `ParsedNote.frontmatter` の不変化 / `IndexPlan.pending_chunk_counts` / `rag/indexer.py` 1,123 行の分割 / 壊れた `chunks.jsonl` からの復帰 / `VaultFile.mtime_ns` の未使用と欠番 D-29 への言及 / T4 決定2・T7 決定1・決定2 の承認）は `docs/next-pr-candidates.md` の「Phase 3b の申し送り」節にまとめた。

---

## 10. ファイル構成

```
rag/
├── __init__.py             # __all__ に store / indexer の公開シンボルを追加 (cli は含めない)
├── settings.py             # [不変] RagSettings / load_settings
├── vault.py                # [不変] vault に触れる唯一のモジュール (D-30)
├── parser.py               # [不変] ParsedNote / iter_headings (D-31 / D-34)
├── chunker.py              # [不変 or render_embed_text の追加のみ] Chunk / chunk_note (D-32)
├── store.py                # ★ T4  ChunkRecord / VectorStore / JsonlVectorStore / InMemoryVectorStore (D-39 / D-41)
├── indexer.py              # ★ T5,T6  index_fingerprint / IndexManifest / plan_index / build_index (D-35〜D-38)
└── cli.py                  # ★ T7  python -m rag.cli index|status [--dry-run] [--rebuild]

llmkit/embeddings.py        # resolve_embedding_spec を公開追加 (既存公開シグネチャは不変)
llmkit/__init__.py          # 再エクスポート

data/index/<vault_id>/      # 索引成果物 (gitignore 済み。コミットしない、D-40)
├── manifest.json           #   schema_version / index_fingerprint / fingerprint_inputs /
│                           #   embedding{reported_model, dimensions} / notes[{relpath, sha256, chunk_count}] / totals
└── chunks.jsonl            #   ChunkRecord 1 件 1 行、(relpath, ordinal) 昇順、embed_text を持たない

vaults/sample.toml          # [不変] 合成 vault の設定 (コミット)
vaults/local.toml           # 実 vault の設定 (gitignore 済み。コミットしない)

tests/
├── conftest.py             # [追加のみ] 決定論的フェイク埋め込み / 索引用ヘルパ
├── test_rag_store.py       # ★ 適合テスト (2 実装 parametrize)
├── test_rag_indexer.py     # ★ fingerprint / 差分更新 / 失敗方針 / E28-E34
├── test_rag_cli.py         # ★ index / status / --dry-run / E35 / E36
├── test_rag_privacy.py     # [追加のみ] 索引成果物が追跡されないこと (D-40 guard)
├── test_rag_layout.py      # [2 行のみ追加] _SUBMODULE_NAMES に "store" / "indexer"
└── test_acceptance_phase3.py  # ★ L309-L314 / L319 の 7 条件 = 7 関数

docs/
├── localllmrequirements.md # [チェックボックスのみ] L309-L314 / L319 を - [x] に (行数不変)
├── plans/2026-08-23-phase3b-indexing.md  # ★ 本仕様書
└── next-pr-candidates.md   # [追記] 3b の申し送り

.claude/decisions.yaml      # [追記] D-35 〜 D-41 (計 37 件)
```

**`pyproject.toml` は変更不要**: `pythonpath=["."]` で `rag` が解決し、mypy の exclude は `.venv` のみなので新規 3 モジュールは自動的に strict の対象。新規依存が無いため `uv.lock` も無変更。

---

## 11. 実機実測 (メインセッション実施、2026-08-23)

> 実 vault の設定は `vaults/local.toml` (gitignore 済み) に置き、実行後に索引成果物ごと削除した。
> **ノートのパス・タイトル・本文・vault の絶対パスは 1 文字も記録していない。**
> 環境: RTX 4070 Ti SUPER / Ollama 0.32.15 / 埋め込み `hf.co/Targoyle/ruri-v3-310m-GGUF`

### 実行結果

| # | 実行 | 結果 | 判定 |
|---|---|---|---|
| 1 | `index --dry-run` | 対象 33 ノート (新規 33 / 変更 0 / 変更なし 0 / 削除 0) / **HTTP 0 回** / **`index_dir` 未作成** / fingerprint `04cbb62a…` | — |
| 2 | `index` (初回) | 33 ノート / **1,212 チャンク** / 76 リクエスト / **768 次元** / 失敗 0 件 / **16.9 秒** / `chunks.jsonl` 13,762,919 B (13.1 MiB) | **L309** |
| 3 | `index` (無変更) | 索引 0 / 再処理しない 33 / **埋め込み 0 / リクエスト 0** / **0.25 秒 (初回の 1.5%)** / 成果物 sha256 が #2 と**完全一致** | **L310** |
| 4 | 1 ノートに 1 文字追記 | 索引 1 / 再処理しない 32 / **埋め込み 1 チャンク / 1 リクエスト**。**ファイルを元に戻して再実行すると成果物が #2 とバイト一致** | **L310** (粒度) |
| 5 | `max_tokens` 240→200 | fingerprint `04cbb62a…` → `ed2eecd9…` / **全 33 ノート・1,399 チャンクを再構築** / 10.2 秒 | §2.2 の欠陥が塞がれている |
| 6 | 索引実行の前後 | vault 配下 53 エントリの `(size, mtime_ns, mode, sha256)` が**完全一致** | **L319** |

### 期待値との比較

| 指標 | 期待 | 実測 |
|---|---|---|
| 初回索引 | 60 秒以内 | **16.9 秒** |
| 無変更の再実行 | 初回の 10% 未満 | **1.5%** (0.25 秒) |
| 次元 | 768 | **768** |
| `chunks.jsonl` | 約 6.4 MB (見積り) | **13.1 MiB** |

見積りの約 2 倍になったのは、チャンク数の見積り (約 370) が実測 1,212 と 3 倍以上ずれたため。1 チャンクあたりのバイト数 (13,762,919 / 1,212 ≒ 11.4 KB) は 768 次元 × 約 14 B/要素の見積りと整合する。**ノート数から総チャンク数を線形に外挿した見積りが甘かった** (実 vault のノートは合成 vault より 1 件あたりが長い)。

### 注意した点

- **#6 の最初の測定でダイジェストが不一致になった。** 原因は #4 で編集したノートを `cp` で戻したことによる `mtime` の変化で、索引処理ではない。索引実行だけを挟んで測り直したところ完全一致した。**測定手順が汚染源になり得るので、切り分けてから結論を出す必要がある。**
- 実行後に `git status` / `git check-ignore` / `tests/test_rag_privacy.py` (13 件) を確認し、索引成果物・実 vault 設定・実 vault の痕跡が追跡側に **0 件**であることを確認したうえで、成果物と `vaults/local.toml` を削除した。

### 残った疑問 (次サイクル)

- 実 vault の 33 ノートのうち **2 ノートがチャンク 0 件**だった (索引に現れたのは 31 ノート)。合成 vault の空ノート 2 件と同じ挙動だが、実 vault では意図した空ノートとは限らない。**「取り込んだのにチャンクが 0 件」を警告で知らせるか**を判断する価値がある (現在は無言)。
