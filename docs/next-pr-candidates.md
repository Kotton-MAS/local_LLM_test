# 次の PR 候補 (Phase 1 PDCA サイクルの MEDIUM / INFO)

> 出典: `.claude/tmp/findings/round-{1,2,3}/triage.json`
> Phase 1 のサイクルでは BLOCKER/HIGH のみを修正した。以下は未対応。

合計 43 件 (MEDIUM 29 / INFO 14)

| severity | round | id | 観点 | 箇所 | 内容 |
|---|---|---|---|---|---|
| MEDIUM | 1 | F-1-003 | reviewer-architecture | `llmkit/bootstrap.py:103` | bootstrap() は http_client 未注入時に OpenAICompatibleClient 内で httpx.Client を新規生成するが、その所有権が誰にも渡らない。BootstrapResult.client の静的型は ChatClient Protocol で close()/__exit_… |
| MEDIUM | 1 | F-1-004 | reviewer-architecture | `llmkit/bootstrap.py:61` | bootstrap() は入口が Path 固定 (config_path) で、内部で load_config → 具象 OpenAICompatibleClient 生成まで一気通貫にしている。差し替え点として外へ出ているのは httpx.Client のみ。このため (a) Phase 2 の比較ハーネスが『同一… |
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
