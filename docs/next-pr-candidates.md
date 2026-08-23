# 次の PR 候補 (Phase 1 PDCA サイクルの MEDIUM / INFO)

> 出典: `.claude/tmp/findings/round-{1,2,3}/triage.json`
> Phase 1 のサイクルでは BLOCKER/HIGH のみを修正した。以下は未対応。

合計 43 件 (MEDIUM 29 / INFO 14)

| severity | round | id | 観点 | 箇所 | 内容 |
|---|---|---|---|---|---|
| MEDIUM | 1 | F-1-003 | reviewer-architecture | `llmkit/bootstrap.py:103` | bootstrap() は http_client 未注入時に OpenAICompatibleClient 内で httpx.Client を新規生成するが、その所有権が誰にも渡らない。BootstrapResult.client の静的型は ChatClient Protocol で close()/__exit_… |
| ~~MEDIUM~~ | 1 | ~~F-1-004~~ | reviewer-architecture | `llmkit/bootstrap.py:61` | bootstrap() は入口が Path 固定 (config_path) で、内部で load_config → 具象 OpenAICompatibleClient 生成まで一気通貫にしている。差し替え点として外へ出ているのは httpx.Client のみ。このため (a) Phase 2 の比較ハーネスが『同一…  **対応済み (本 PR。Phase 2)**: `llmkit/bootstrap.py` に `bootstrap_from_config(config, config_path, *, ...)` を追加し、`bootstrap(config_path, ...)` はそれへ委譲する薄いラッパにした (既存シグネチャは無変更)。差し替え点が `httpx.Client` だけだった状態は解消し、比較ハーネス (`harness/`) はモデルごとに `AppConfig` を組み替えて起動できる。帰結として `manifest.config_sha256` はベース設定ファイルの同一性しか表さなくなるため、実効値の同一性は `run_fingerprint` が担う (**D-20**) |
| MEDIUM | 1 | F-1-005 | reviewer-architecture | `llmkit/client.py:178` | OpenAICompatibleClient が AppConfig 全体を受け取り、OOM メッセージの組み立てで self._config.vram.active_profile (client.py:343) を読んでいる。トランスポート層が VRAM プロファイルという別関心事を知っており、責務が混ざっている。… |
| MEDIUM | 1 | F-1-006 | reviewer-architecture | `llmkit/client.py:96` | ChatResult が持つ時間指標は latency_s (HTTP 送信直前〜受信直後の総時間) のみで、プロンプト処理時間を表現する場所が無い。要件書 Phase 2 受け入れ条件 (L303)『出力にモデルごとの生成速度・プロンプト処理速度・設定コンテキスト長が含まれる』のうち『プロンプト処理速度』は、Open… |
| MEDIUM | 1 | F-1-009 | reviewer-docs | `README.md:106` | README に新設した「## llmkit (Phase 1)」節では llmkit/, configs/, docs/plans/ の使い方を説明しているが、その下の「## Project Structure」のディレクトリツリー (104-128行目) はテンプレート初期状態のままで、これら新規ディレクトリが1つ… |
| MEDIUM | 1 | F-1-010 | reviewer-docs | `docs/localllmrequirements.md:337` | 「未確定事項」表の Q7 (ハイブリッド運用の切り替えを実装に含めるか) は Phase 1 で「入れる」に確定し configs/external_openai.toml を実装しているにもかかわらず、要件書側の Q7 行は未解決のまま残っている。同ファイルに既にある「v2 で解決済み」節 (Q1/Q2 の取消線パタ… |
| MEDIUM | 1 | F-1-012 | reviewer-performance | `llmkit/bootstrap.py:82` | bootstrap() は load_config() 内で設定ファイルを path.read_text() (llmkit/config.py:157) で読み、その後 build_manifest() 経由の compute_config_sha256() (llmkit/manifest.py:207) が同じフ… |
| MEDIUM | 1 | F-1-013 | reviewer-performance | `llmkit/bootstrap.py:89` | bootstrap() は estimate_resolved_profile() で VRAM 見積りを計算済み (line 84) だが、直後の check_budget(profile, config) (vram.py:176) は戻り値の estimate を受け取らず内部で estimate_resolve… |
| MEDIUM | 1 | F-1-014 | reviewer-performance | `llmkit/cli.py:122` | _run_doctor() / _run_chat() は bootstrap() が生成した BootstrapResult.client (内部で自前の httpx.Client を new する経路) を使った後、result.client.close() を一度も呼んでいない。OpenAICompatibleC… |
| MEDIUM | 1 | F-1-017 | reviewer-security | `llmkit/client.py:276` | `_headers()` は api_key が非空なら scheme を問わず `Authorization: Bearer <key>` を付与する。`_validate_base_url` は `http` を無条件に許可しており、ホストがループバックかどうかも判定していないため、リモートホスト宛の平文 HTTP… |
| MEDIUM | 1 | F-1-018 | reviewer-security | `llmkit/config.py:191` | `api_key_env` に任意の環境変数名を指定でき、その値が `base_url` (これも任意ホスト指定可) 宛の Authorization ヘッダに載る。TOML 2 行 (`base_url` と `api_key_env`) の変更だけで、開発者マシンや CI 上の任意の環境変数 (`AWS_SECRE… |
| MEDIUM | 1 | F-1-019 | reviewer-security | `llmkit/config.py:157` | `_read_toml` は `OSError` と `tomllib.TOMLDecodeError` のみを ConfigError に翻訳しているが、`path.read_text(encoding="utf-8")` は非 UTF-8 バイト列に対して `UnicodeDecodeError` (ValueEr… |
| MEDIUM | 1 | F-1-022 | reviewer-style | `llmkit/client.py:161` | `_format_validation_error` が `llmkit/config.py:146` とほぼ同一実装 (docstring 1文言のみ差異) で重複定義されている。ValidationError を「キー名: 理由」に整形するロジックが2箇所に存在し、片方を直した際にもう片方が追随しない事故のリスクが… |
| MEDIUM | 1 | F-1-023 | reviewer-style | `tests/test_client.py:31` | `REPO_ROOT` / `DEFAULT_CONFIG` が `tests/conftest.py:27-28` に既に定義されているにもかかわらず、11個のテストファイル全てで再定義されている。しかも同じファイル内で `SUCCESS_PAYLOAD` は `from conftest import SUCCES… |
| MEDIUM | 1 | F-1-024 | reviewer-style | `llmkit/vram.py:186` | `VramBudgetExceededError` のメッセージが `estimate.summary()` を丸ごと内包しており、`total_gib` と `budget_gib` が1メッセージ内に二重に出力される。実測したところ同じ数値が2回現れ、CLI エラー出力としては冗長で読みにくい。 |
| MEDIUM | 1 | F-1-025 | reviewer-style | `llmkit/client.py:314` | `_raise_for_error_status` が69行で CLAUDE.md の「50行超は分割を検討」を超過している。ネストは浅く可読性自体は保たれているが、4種の判定 (model_not_found / out_of_memory / context_length / generic) が1関数に同居し長い… |
| MEDIUM | 1 | F-1-026 | reviewer-style | `llmkit/manifest.py:215` | `build_manifest` が62行で CLAUDE.md の50行ガイドラインを超過している。各 Manifest* データクラスは既に to_dict() を個別に持つパターンがあるのに、build_manifest 側では対応するセクション別ヘルパーに分けず全フィールドをインライン構築している。 |
| MEDIUM | 1 | F-1-028 | reviewer-style | `llmkit/config.py:82` | 同一概念 (カタログの model_id) を指す名前がレイヤーごとに割れている: catalog.ModelSpec.model_id / config.ProfileConfig.generation / config.GenerationParams.model。GenerationParams.model が … |
| MEDIUM | 1 | F-1-032 | reviewer-test | `llmkit/manifest.py:208` | compute_config_sha256() の except OSError (208-210行目) と write_manifest() の except OSError (290-292行目、書き込み失敗を ConfigError に翻訳する分岐) がいずれも未実行。tests/test_manifest.py… |
| MEDIUM | 1 | F-1-033 | reviewer-test | `llmkit/config.py:185` | _reject_inline_secrets() の TOML ルート直下に api_key を書いた場合の拒否 (185行目) が未実行。tests の test_inline_api_key_is_rejected は [runtime] 配下のみをテストしており、D-05 のガード対象のうちルート直下ケースの拒否… |
| MEDIUM | 2 | F-2-002 | reviewer-architecture | `llmkit/client.py:366` | F-1-001 の修正で導入した placeholder _PASSTHROUGH_MAX_CONTEXT_TOKENS = 1_048_576 が、下流で「モデルの真の上限」として無条件に消費されている。結果、passthrough 経路で上流が context 長超過の 400 を返すと『要求したコンテキスト長 c… |
| MEDIUM | 2 | F-2-003 | reviewer-architecture | `llmkit/client.py:339` | RuntimeConfig.kind (Literal['ollama','openai_compatible']) が挙動に一切分岐していない。実測で参照箇所は manifest.py:145/266 とテスト2箇所だけで、client / bootstrap / vram のどこも見ていない。結果、kind='op… |
| MEDIUM | 2 | F-2-004 | reviewer-architecture | `llmkit/bootstrap.py:86` | passthrough が weights_gib=0.0 / kv_gib_per_1k_tokens=0.0 を VRAM 見積り経路にそのまま流すため、外部 API + 未登録モデルの構成で INFO ログとマニフェストに『想定 VRAM 使用量 0.80 GiB (重み 0.00 + KV 0.00 + オーバ… |
| MEDIUM | 2 | F-2-005 | reviewer-architecture | `tests/test_layout.py:117` | F-1-002 の修正自体は妥当で内部専用シンボルの漏れは発生していないが、境界の宣言方法が『手書きの公開一覧』から『派生』に変わった結果ガードの向きが2点で反転している。(1) 等号アサートにより、サブモジュールが __all__ に足したシンボルは必ずパッケージ公開になる = 誤って公開 API が広がるケースを原… |
| MEDIUM | 2 | F-2-007 | reviewer-security | `llmkit/config.py:68` | _validate_base_url は userinfo (user:pass@) のみを拒否し、URL の path / query / fragment に置かれた認証情報は素通りする。fixer の「秘密が書けてしまう設定項目は base_url だけで、userinfo を塞げばクラスごと閉じた」という主張は… |
| MEDIUM | 2 | F-2-009 | reviewer-style | `llmkit/catalog.py:138` | resolve_model_spec の docstring は get_model_spec との違い(is_local/role で分岐、passthrough を合成する)を丁寧に説明しているが、逆方向 (get_model_spec 側からの案内) が無い。catalog.py 内では get_model_sp… |
| MEDIUM | 2 | F-2-011 | reviewer-test | `tests/test_client.py:246` | F-1-031で追加された test_client_with_no_injected_http_client_creates_and_owns_one / test_client_with_no_injected_http_client_uses_configured_timeout が client._http / … |
| MEDIUM | 3 | F-3-001 | reviewer-architecture | `llmkit/client.py:368` | 「この ModelSpec は passthrough の合成物か」という catalog の内部事情を client.py が self._spec.model_id in MODEL_CATALOG で再導出している。passthrough である事実は catalog.py 側に既に別形で符号化されており (so… |
| MEDIUM | 3 | F-3-003 | reviewer-style | `llmkit/client.py:318` | _raise_for_error_status が round 3 の F-2-002 修正 (ContextLengthError のメッセージをカタログ登録有無で2分岐にした変更) により 71行 → 85行 に伸び、関数長の目安 80行を新たに超えた。CLAUDE.md の『50行超は分割を検討』の目安も踏まえる… |
| INFO | 1 | F-1-007 | reviewer-architecture | `llmkit/config.py:113` | AppConfig が『TOML ファイルのスキーマ』と『実行時に解決された値 (api_key: SecretStr)』を 1 型に混ぜている。api_key は AppConfig のフィールドなので extra='forbid' では塞げず、_reject_inline_secrets() という手書きの特例ガー… |
| INFO | 1 | F-1-008 | reviewer-architecture | `llmkit/bootstrap.py:84` | bootstrap() が estimate_resolved_profile() で見積りを計算した直後に check_budget() を呼び、check_budget 内でも同じ見積りが再計算されている。check_budget が VramEstimate を返す設計は『判定と内訳取得を 2 回計算に分けない』… |
| INFO | 1 | F-1-011 | reviewer-docs | `llmkit/vram.py:151` | estimate_profile() は docstring が一行のみで Args/Raises が無い。同じモジュール内の resolve_profile / check_budget は Args・Raises を明記しており、estimate_profile は内部で resolve_profile を呼ぶため… |
| INFO | 1 | F-1-015 | reviewer-performance | `llmkit/config.py:65` | RuntimeConfig.timeout_s の既定値が 120.0 秒。要件書の非機能要件 (対話用途で生成 30 t/s 以上、RAG クエリ応答 2 秒以内 (仮)) と比べるとかなり緩い上限であり、ランタイム未応答時の失敗検知が遅くなる。 |
| INFO | 1 | F-1-020 | reviewer-security | `llmkit/manifest.py:194` | `manifest_filename` は `run_id` を検証せずにファイル名へ連結し、`write_manifest` (manifest.py:286) が `directory / manifest.filename()` で書き込み先を組み立てる。`run_id` に `../` を含めると出力ディレクト… |
| INFO | 1 | F-1-021 | reviewer-security | `llmkit/manifest.py:289` | `destination.write_text(...)` はパーミッションを umask 任せ (一般的な環境で 0644) にしている。マニフェストには base_url・ホスト platform 文字列・設定ハッシュが入り、上記 HIGH が未修正の状態では資格情報も入りうる。共用ホストでは同一マシンの他ユーザー… |
| INFO | 1 | F-1-027 | reviewer-style | `llmkit/bootstrap.py:61` | `bootstrap` が58行で CLAUDE.md の50行ガイドラインをわずかに超過している。ただしモジュール冒頭の docstring に「仕様書 §4 T4 の順序をそのまま1関数に集約する」という設計意図が明記されており、意図的な選択に見える。 |
| INFO | 1 | F-1-034 | reviewer-test | `llmkit/vram.py:52` | ResolvedProfile.generation プロパティの ConfigError 分岐が未テスト。ただし ProfileConfig.generation は必須のため現行スキーマ下では実質到達不能な防御的コードであり優先度は低い。 |
| INFO | 1 | F-1-035 | reviewer-test | `llmkit/errors.py:36` | LlmkitError.__str__ の remediation が空文字列のケースの分岐が未テスト。全ての例外送出箇所が remediation を必ず渡しているため実害は小さい。 |
| INFO | 1 | F-1-036 | reviewer-uv | `Makefile:6` | `export PYTHONPATH :=` は環境変数を空文字列に設定するのであって unset ではない。今回のシェル (ROS 2 由来の PYTHONPATH 汚染) では実測で問題なく動作しており、CI (PYTHONPATH 未設定) でも空文字列は実質無害だが、将来 PYTHONPATH の「未設定である… |
| INFO | 2 | F-2-006 | reviewer-architecture | `llmkit/manifest.py:97` | check_budget は is_local=false で判定をスキップするが、ManifestVram.within_budget はスキップされた run でも total_gib <= budget_gib の計算結果として true が記録される。『判定して通った』と『判定していない』が同じ true にな… |
| INFO | 2 | F-2-008 | reviewer-security | `llmkit/config.py:75` | RuntimeConfig は __all__ で公開 API になっているため、L3 が RuntimeConfig(base_url=...) を直接構築する経路が正規に存在する。この経路で userinfo 付き URL を渡すと生の pydantic ValidationError が送出され、その str()… |
| INFO | 3 | F-3-002 | reviewer-security | `llmkit/config.py:98` | urlparse が分離する 6 成分のうち query / fragment / userinfo は拒否されるが params (RFC 3986 の path parameter、/v1;key=SECRET) は素通りする。fixer は「path 中の秘密は機械判別できないため注意喚起のみ」としたが、para… |
| INFO | 3 | F-3-004 | reviewer-test | `tests/test_config.py:103` | fixer の判断『gt=0 系フィールド (context_tokens, max_output_tokens, budget_gib, timeout_s) は既存の「0を拒否する」テストで gt/ge 境界が固定済み』について、read-only 制約により Field(gt=0) → Field(ge=0) へ… |

## Phase 0 実測で判明した課題 (2026-08-22 追記)

出典: `docs/phase0-vram-measurements.md`

| 優先度 | 課題 | 決定した扱い |
|---|---|---|
| ~~HIGH~~ | 埋め込み `ruri-v3-310m` とリランカー `ruri-reranker-large` は GGUF を持たず `ollama pull` できない。Phase 0 受け入れ条件「埋め込み・リランカーの取得と単発実行」を満たせず、構成1 (14B + 埋め込み + リランカー) の同居実測もできていなかった | **対応済み (本 PR)**。選択肢 (A)+(C) の**ハイブリッド**を採った (2026-08-23 実機検証)。埋め込みは有志の GGUF 変換版 `hf.co/Targoyle/ruri-v3-310m-GGUF` を `ollama pull` して Ollama で動かし (768 次元 / VRAM 0.57 GiB)、リランカーは **Ollama にリランキング API が無い**ため別ランタイム llama.cpp `llama-server` に載せた (`Ruri Reranker` は GGUF 非提供のため `BGE Reranker v2-m3` Q6_K を採用。VRAM 0.28 GiB / 50 ペア 140 ms)。(B) の `bge-m3` への変更は不要だった。構成1 の同居実測も完了 (12.53 GiB 累計 / 増分 11.97 GiB、余力 3.46 GiB)。**「Phase 3 着手時に方式を決める」とした D-14 は撤回した** (→ D-14 は「取得不能・(仮) を残さない」旨に書き換え。載せ先ランタイムは **D-15** の `ModelSpec.serving_runtime`、別プロセス分の合算は **D-16**、起動スクリプトは **D-17**) |
| ~~MEDIUM~~ | 要件書の VRAM 表と実測が乖離している (`gpt-oss-20b` は 128k で CPU オフロード、GPU 内上限は 65,536〜98,303 の間)。要件書 L69 の「20B は 128k まで VRAM 内完結」は本機では成立しない | **対応済み (本 PR)**。要件書の VRAM 表・構成表・採用モデル表を実測値に訂正した (`docs/localllmrequirements.md` L64 / L74-L76 / L81 / L93) |
| ~~MEDIUM~~ | 要件書の生成速度と実測が乖離 (14B: 84.3 → 61.5 t/s、20B: 57.5 → 134.3 t/s) | **対応済み (本 PR)**。要件書の採用モデル表を実測値に置き換え、記載の無かった 8B (104.6 t/s) も追記した |
| ~~INFO~~ | `gpt-oss-20b` の `max_context_tokens` はカタログ上 131072 だが、本機で GPU 内に収まるのは 65,536 まで。プロファイル定義で上限を分けるべきか | **対応済み (本 PR) / 分けるのは不要と結論**。GPU 常駐上限は同居する埋め込み・リランカーの有無に依存するため、モデル単位の定数では表せない。プロファイル単位の `vram.budget_gib` (既定 14.0) で表現する (**D-12**)。`max_context_tokens` はモデル自身の上限のまま 131072 を維持する |

> **本 PR で併せて解消した既存 finding** (上の一覧は未更新のまま残す。再スキャン時に重複しないための注記):
> `F-1-006` (ChatResult に prompt/eval 分離の受け皿が無い) は `ChatTimings` の追加で解消 (**D-13**)。
> `F-2-003` (`runtime.kind` が挙動に分岐していない) は `create_chat_client` の追加で解消 (**D-10**)。

## round-9 (Phase 3a) の MEDIUM / INFO (2026-08-23 追記)

出典: `.claude/tmp/findings/round-9/triage.json`。HIGH 6 件 (F-9-001 / F-9-002 / F-9-008 / F-9-013 / F-9-014 / F-9-028) は本 PR で対応済み。F-9-016 (`.gitignore`) はメインセッションが対応済み。以下 24 件 (MEDIUM 15 / INFO 9) は未対応。

**次サイクル (3b) で必ず対応すべき2件**:

| severity | id | 観点 | 箇所 | 内容 |
|---|---|---|---|---|
| MEDIUM | F-9-004 | reviewer-architecture | `rag/settings.py:293` | `load_settings` は `vault_dir` / `index_dir` を `expanduser().resolve()` で正規化するが `source_path` だけは呼び出し側が渡した `Path` をそのまま保持する。相対パスと絶対パスで同じ設定ファイルを渡すと `RagSettings` の等価性が偽になる。§9 T2 決定18 で `source_path` は 3b の `index_fingerprint` に入れると明記済みのため、このまま載せると**起動ディレクトリによって fingerprint が変わり全再構築になる** (D-28)。**3b 着手前に必ず `source_path=path.expanduser().resolve()` へ揃えること**。ただし fingerprint に入れる際は絶対パスそのもの (実 vault のパスは利用者名を含む) ではなく設定内容のハッシュにすることも併せて検討する |
| MEDIUM | F-9-018 | reviewer-security | `rag/settings.py:233` | `_validate_index_dir` は `index.dir` が `vault_dir` 配下に解決されることだけを禁止しており、**リポジトリ内の追跡対象ディレクトリを指すことは禁止していない**。索引成果物 (`Chunk.body` / `embed_text`) はノート本文を全量保持するため、実 vault を索引しつつ `index.dir` を `data/` 以外のリポジトリ内パスに向けると個人ノート本文が **PUBLIC リポジトリへ入り得る**。3a には索引の書き出しコードがまだ無いため実害は未発生だが、**3b で索引を書き出す前に、`index_dir` が `.gitignore` で無視される場所であることを要求する検査を追加すること** (D-30 に条件を1つ足し guard_test を追加する) |

(上記2件について、当初の依頼文中の ID 表記「F-9-011」は triage.json 上の実際の内容とは一致しなかった。F-9-011 は実際には `harness/runner.py` のコメント陳腐化を指す別件であり、下表に別掲した。内容が一致する正しい ID は F-9-004 であるため、そちらで記載した)

**その他 (優先度は通常の MEDIUM/INFO)**:

| severity | id | 観点 | 箇所 | 内容 |
|---|---|---|---|---|
| MEDIUM | F-9-003 | reviewer-architecture | `rag/parser.py:109` | `ParsedNote` は frozen dataclass だが `frontmatter` フィールドの型注釈は `Mapping` でも実体は可変な生 `dict` のままで、呼び出し側から書き換えられ frozen の保証が境界で破れる (`hash()` も `TypeError`) |
| MEDIUM | F-9-005 | reviewer-architecture | `llmkit/embeddings.py:284` | 「HTTP 200 + 本文 error を翻訳表へ流すか」がサブクラスごとの上書きで決まっており、`OpenAIEmbeddingClient` は既存の `OpenAICompatibleClient` と逆の選択をしている (同条件で chat は `UpstreamError`、埋め込みは `ModelNotFoundError`) |
| MEDIUM | F-9-009 | reviewer-docs | `.claude/decisions.yaml:143` | D-27 の rule 3条項のうち guard_test が検証するのは1条項目のみ (本 PR の F-9-008 対応で rule 側に他条項の担当テストを追記済みだが、根本の「guard_test は単一 node id」というスキーマ制約自体は残る) |
| MEDIUM | F-9-010 | reviewer-docs | `.claude/decisions.yaml:158` | D-32 の rule の「上限は embed_text に適用する」条項を guard_test が検証していない (同上、本 PR で rule 側に追記済み) |
| MEDIUM | F-9-011 | reviewer-docs | `harness/runner.py:629` | `_execute` 内のコメントが httpx.Client の所有権規則を「`llmkit._HttpChatClient` と同じ規則」と説明しているが、Phase 3a T1 でこの規則は chat 非依存の基底 `_HttpEndpointClient` に移った (埋め込みも継承する)。コメントを基底の名前に更新する |
| MEDIUM | F-9-017 | reviewer-security | `rag/vault.py:88` | `_compile_glob` が中間の `**` を `(?:[^/]+/)*` に展開するため、パターンに `**` が複数現れると破滅的バックトラッキングが起きる (実測: `**` の個数 n=11 で 3.59 秒)。既定値は `**` 1個のみなので現状は無害 |
| MEDIUM | F-9-019 | reviewer-security | `rag/vault.py:206` | `_resolve_note_path` は絶対パス・vault 外脱出・シンボリックリンクは弾くが通常ファイルであること (`S_ISREG`) を確認していない。`_accept_file` にはある判定が `read_note_bytes` / `read_note_text` の経路に無い |
| MEDIUM | F-9-023 | reviewer-style | `tests/test_rag_layout.py:78` | `imported_module_names` が `tests/test_harness_layout.py:43-52` と完全に同一定義のまま複製されている (Phase 2 の `FakeProbe` 重複と同型の再発) |
| MEDIUM | F-9-024 | reviewer-style | `tests/test_rag_layout.py:305` | D-08 の BaseModel 非継承検査が `tests/test_layout.py` と `tests/test_rag_layout.py` に事実上同一ロジックで並存 (§9 T2 決定15 で「3b で統合するか決めること」と未解決のまま申し送り済み) |
| MEDIUM | F-9-025 | reviewer-style | `tests/test_rag_chunker.py:168` | `rag/` 側の新規 parametrize 4箇所が `ids=` を付けていない (同じ diff 内の `test_embeddings.py` は5箇所すべて付けており規約適用が割れている) |
| MEDIUM | F-9-026 | reviewer-style | `rag/settings.py:97` | `EmbedSettings.batch_size` の既定値16に出所の記載が無い (`ChunkSettings` は数値の出典を Attributes docstring で追跡可能にしている) |
| MEDIUM | F-9-029 | reviewer-test | `rag/vault.py:60` | `_compile_glob` の `?` ワイルドカード変換、および `**` が最初かつ末尾の特殊系が未到達。中間 `**` の2階層以上ネスト動作を直接検証するテストも無い |
| MEDIUM | F-9-030 | reviewer-test | `tests/test_rag_vault.py:205` | D-30 動的 guard の健全性確認テストが本文変更とエントリ増加しか見ておらず、`st_mode` だけの変更を検出できることを検証していない |
| INFO | F-9-006 | reviewer-architecture | `rag/parser.py:56` | `_LINK_HEADING_SEPARATOR = " > "` と `ChunkSettings.heading_separator` の既定値が独立した2か所に置かれている。設定を変えても wikilink 展開の区切りは追随しない |
| INFO | F-9-007 | reviewer-architecture | `rag/chunker.py:130` | `Chunk` が `body` と `embed_text` を両方保持しており、3b の JSONL 永続化で本文が2重に書かれ得る。他2件 (D-08 走査の2ファイル並存、`vaults/sample.toml` 関連) も同 finding に同梱 |
| INFO | F-9-012 | reviewer-docs | `docs/plans/2026-08-23-phase3-indexing.md:386` | §9 T1 の申し送りで挙げた D-25 の guard_test 候補と、実際に `.claude/decisions.yaml` に採用された guard_test が異なる |
| INFO | F-9-015 | reviewer-performance | `docs/plans/2026-08-23-phase3-indexing.md:93` | D-26 の JSONL 永続化サイズ見積り (約3.5MB) が実測 (約6.36MB) の半分程度に過小評価されている |
| INFO | F-9-020 | reviewer-security | `rag/vault.py:232` | `read_note_bytes` / `read_note_text` は vault 内であることは検査するが `exclude_globs` を適用しない (`.obsidian/app.json` 等を relpath 指定で読み出せる) |
| INFO | F-9-021 | reviewer-security | `rag/settings.py:192` | 設定ファイル自身のパスが例外メッセージに絶対パスのまま出ることがある (`vault.dir` は `configured` 文字列のみを載せる方針が徹底されているのと非対称) |
| INFO | F-9-022 | reviewer-security | `rag/vault.py:222` | `_resolve_note_path` の検査後に `read_note_bytes` が別途開き直すため TOCTOU の窓がある (ローカル単一利用者ツールのため実害は小さい) |
| INFO | F-9-027 | reviewer-style | `rag/settings.py:48` | `NonEmptyStr` / `PositiveInt` / `PositiveFloat` の型エイリアスが `llmkit/config.py` / `harness/suite.py` に続き3重複製 |
| INFO | F-9-031 | reviewer-test | `rag/parser.py:173` | frontmatter パース中の「空行または `#` コメント行をスキップ」分岐が未到達 (テストデータに該当ケース無し) |
| INFO | F-9-032 | reviewer-uv | `docs/plans/2026-08-23-phase3-indexing.md` | 3b で Chroma 導入 (D-26) を検討する際の依存数見積りが read-only 権限のため transitive 総数まで測定できていない |

## Phase 3b (索引・差分更新・CLI) の申し送り (2026-08-23 追記)

出典: `docs/plans/2026-08-23-phase3b-indexing.md` §9 (T4〜T7 の実装者が残した判断 54 件と申し送り) と T8 の実測。
本 PR (3b) で対応済みのものは行末に **対応済み** と書く。以下は**次サイクル以降**の候補。

### 1. 仕様と食い違う実装 / 承認が要る変更 (次の周で必ず判断する)

| 優先 | 箇所 | 内容 |
|---|---|---|
| HIGH | `tests/test_rag_layout.py:85` (`_FILESYSTEM_READ_ALLOWANCES`) | §9 T4 決定2。仕様は `test_rag_layout.py` の変更を `_SUBMODULE_NAMES` だけに許していたが、D-30 の構造層 guard が `read_text` を名前ベースで一律に落とすため、索引ファイルを読む `store.py` / `indexer.py` は許可を足さないと**原理的に緑にならない**。許可リストは読むモジュールが増えるたびに伸びる。**T8 で D-30 の rule 本文を実際の許可内容へ書き直した (対応済み)** が、「許可リストを伸ばし続ける設計でよいか」は未判断。代案は `rag/store.py` / `rag/indexer.py` の読み書きを 1 モジュール (例 `rag/artifacts.py`) に閉じて許可を 1 件に戻すこと |
| HIGH | `tests/test_rag_layout.py::test_every_rag_submodule_is_covered_by_the_union_check` | §9 T7 決定1。`rag/cli.py` を作った時点でこのテストは原理的に落ちるため、右辺を `set(_SUBMODULE_NAMES)` → `{*_SUBMODULE_NAMES, "cli"}` に変えた (`_SUBMODULE_NAMES` 自体は無変更)。**指示で許されていない既存テストの編集**であり、承認または再設計 (「公開 API を持たない入口モジュール」の表を別に持つ等) が要る |
| HIGH | `rag/cli.py` `main()` の注入点 | §9 T7 決定2。仕様は `main(argv, *, http_client=None, embedding_client=None, ...)` だったが、`http_client: httpx.Client \| None` と書くと `rag/cli.py` が `httpx` を import することになり **D-25 guard が落ちる**ため `embedding_client` 1 つにした。**仕様と明示的に食い違う唯一の点**。恒久策は「`rag` に `httpx` の型だけを持ち込める例外を D-25 に作る」か「注入点を `embedding_client` に一本化すると仕様側を直す」かの二択 |

### 2. 型・API の整理 (3b では意図的に見送った)

| 優先 | 箇所 | 内容 |
|---|---|---|
| MEDIUM | `rag/chunker.py` `Chunk.embed_text` | 仕様 §4 論点3 / ユーザー確定事項 Q3 = (a)。3b では**永続化から外す**に留めた (D-41)。完全な解は `@property` 化だが `Chunk` に `heading_separator` フィールドが要り、`rag.__all__` の公開型変更になる。既存テストは `Chunk(...)` を直接構築しておらず属性アクセスのみなので**次サイクルで安全に実施できる** |
| MEDIUM | `rag/parser.py` `ParsedNote.frontmatter` (F-9-003 の再掲) | 仕様 §4 論点4。型注釈は `Mapping` だが実体は可変な生 `dict` で、frozen の保証が境界で破れる (`hash()` も `TypeError`)。3b は差分判定をバイト列の sha256 だけで行う (D-36) ため `ParsedNote` の同一性を 1 か所も使っておらず、**guard_test を書ける形にならない**ので決定にせず見送った。次サイクルで検索結果のキャッシュキーに使うなら、その時点で不変化と guard_test をセットで入れる |
| MEDIUM | `rag/indexer.py` `IndexPlan` | 仕様 §4 論点6 の脚注 / §9 T7 決定4。`--dry-run` が「予定リクエスト数」を出せない。正確に出すには CLI が `parse_note` → `chunk_note` を回すことになり `build_index` の前半を二重実装する (D-27 の趣旨に反する)。正しい置き場所は `plan_index` が返す `IndexPlan` に `pending_chunk_counts` (再処理対象ノートごとのチャンク数) を持たせること。**リクエスト数 = ⌈合計 / batch_size⌉ を CLI が計算できるようになる** |
| INFO | `rag/vault.py:50` `VaultFile.mtime_ns` / `size` | D-36 (差分判定はバイト列の sha256 だけ) を採ったため、この 2 フィールドは `rag/` の中で**1 度も読まれていない** (実測: 生成箇所以外の参照 0 件)。しかも docstring が「差分更新の高速経路に使う (3b / D-29)」と書いており、**D-29 は `decisions.yaml` に一度も存在しない欠番**を指している。フィールドを消すか、docstring を「列挙の副産物であって索引は読まない」に直すか (D-36 の rationale と揃える) |

### 3. 構造・運用 (実測から出たもの)

| 優先 | 箇所 | 内容 |
|---|---|---|
| MEDIUM | `rag/indexer.py` (1,123 行) | fingerprint / マニフェスト / 計画 (`plan_index`) / 実行 (`build_index`) の **4 責務が 1 モジュールに同居**している (`rag/` 全体 3,518 行の 32%)。分割の自然な線は「再現条件とマニフェスト (読み書き)」と「計画と実行」。ただし `rag/__init__.py` の再エクスポートと `_SUBMODULE_NAMES` / `_FILESYSTEM_READ_ALLOWANCES` に波及するため、上記 1. の判断と同時に行うのが安い |
| MEDIUM | `rag/cli.py` `--rebuild` | §9 T7 決定9。`--rebuild` は `load_manifest` を呼ばないので**壊れた `manifest.json` からは復帰できるが、壊れた `chunks.jsonl` からは復帰できない** (`JsonlVectorStore` の生成時読み込みが `ConfigError` になる)。塞ぐには「空のストアから始める」入口が `rag/store.py` に要る。現状の逃げ道は「索引ディレクトリを手で消す」で、CLI のメッセージにその案内が無い |
| INFO | `rag/indexer.py` の正規化 JSON sha256 | 仕様 §3 ソフト制約で `harness/runner.py` との 3 行重複を許容した (層構造上 `rag` は `harness` を import できない)。両者が同じ dict に同じ digest を返すことはテストで固定済み。共通化するなら置き場所は `llmkit` になるが、「推論ランタイムの抽象」という `llmkit` の責務からは外れる |
