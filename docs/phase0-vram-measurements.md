# Phase 0: VRAM 実測記録

> 要件書 `docs/localllmrequirements.md` Phase 0 受け入れ条件「各モデルの実測 VRAM 使用量を記録した表が存在する」に対応。

## 測定環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (16376 MiB) |
| ドライバ / CUDA | 595.84 / 13.2 |
| OS | Ubuntu 24.04.4 LTS |
| Ollama | 0.32.15 (systemd 管理) |
| 測定日 | 2026-08-22 |

## 測定方法

1. 対象以外のモデルを `ollama stop` で全てアンロードし 4 秒待つ
2. `nvidia-smi --query-gpu=memory.used` で基準値を取得 (アイドル時 611 MiB)
3. **ネイティブ `/api/chat`** に `options.num_ctx` を指定して 128 トークン生成
   (OpenAI 互換エンドポイントは `num_ctx` を無視するため。後述)
4. ロード状態で再度 `nvidia-smi` を取得し、差分を VRAM 使用量とする
5. `ollama ps` で実 CONTEXT と GPU/CPU 配置を確認

**増分はモデル重み + KV キャッシュ + ランタイムオーバーヘッドの合計**です。個別の内訳は直接測っていません。

## 実測結果

| モデル | num_ctx | VRAM 増分 (GiB) | 生成 t/s | 配置 |
|---|---|---|---|---|
| `qwen3:14b-q4_K_M` | 4,096 | 9.24 | 61.6 | 100% GPU |
| `qwen3:14b-q4_K_M` | 16,384 | 11.13 | 61.5 | 100% GPU |
| `qwen3:14b-q4_K_M` | 32,768 | 13.65 | 61.8 | 100% GPU |
| `qwen3:8b-q4_K_M` | 8,192 | 6.12 | 105.0 | 100% GPU |
| `qwen3:8b-q4_K_M` | 16,384 | 7.26 | 104.6 | 100% GPU |
| `qwen3:8b-q4_K_M` | 32,768 | 9.54 | 103.9 | 100% GPU |
| `gpt-oss:20b` | 32,768 | 12.90 | 134.3 | 100% GPU |
| `gpt-oss:20b` | 65,536 | 13.68 | 134.0 | 100% GPU |
| `gpt-oss:20b` | 98,304 | 13.91 | 105.5 | 14%/86% CPU/GPU ⚠ |
| `gpt-oss:20b` | 131,072 | 13.90 | 87.1 | 20%/80% CPU/GPU ⚠ |

## 線形フィットによる較正値

GPU 内に収まった点のみを使い `合計 = (重み + オーバーヘッド) + KV係数 × (num_ctx / 1024)` で回帰しました。

オーバーヘッドを現行カタログと同じ **0.8 GiB** と仮定して重みを分離しています (実測は合計値のみのため、この分離は仮定に依存します)。

| モデル | 実測 KV (GiB/1k) | 現行カタログ | 実測 重み (GiB) | 現行カタログ |
|---|---|---|---|---|
| `qwen3-14b` | **0.1575** | 0.1 | **7.81** | 9.0 |
| `qwen3-8b` | **0.1425** | 0.08 | **4.18** | 4.5 |
| `gpt-oss-20b` | **0.0244** | 0.004 | **11.32** | 11.5 |

**カタログは重みをやや過大に、KV 係数を大きく過小に見積もっています。**
特に `gpt-oss-20b` の KV 係数は実測の約 1/6 でした。

## 要件書との相違

### 1. `gpt-oss-20b` は 128k で VRAM 内に収まらない

要件書は「20B (MXFP4) は 128k コンテキストまで VRAM 内で完結」としていますが、実測では
**98,304 トークンで既に CPU オフロードが発生**しました。GPU 内に収まる上限は 65,536〜98,303 の間です。

| num_ctx | 配置 | 生成 t/s |
|---|---|---|
| 65,536 | 100% GPU | 134.0 |
| 98,304 | 14%/86% CPU/GPU | 105.5 |
| 131,072 | 20%/80% CPU/GPU | 87.1 |

オフロード時も 87〜105 t/s 出ており要件の 30 t/s は満たしますが (MoE で活性パラメータが少ないため)、
「128k が VRAM 内完結」という前提でプロファイルを組むと破綻します。

### 2. 生成速度は要件書の記載と異なる

| モデル | 要件書 | 実測 | 要件 (30 t/s) |
|---|---|---|---|
| `qwen3-14b` | 84.3 t/s | **61.5 t/s** | 満たす |
| `gpt-oss-20b` | 57.5 t/s | **134.3 t/s** (32k) | 満たす |
| `qwen3-8b` | 記載なし | **104.6 t/s** | 満たす |

3モデルとも対話用途の要件 30 t/s を満たします。

### 3. OpenAI 互換エンドポイントは `options.num_ctx` を無視する

**仕様書 `docs/plans/2026-08-22-phase1-inference-client-l2.md` §7 リスク1 が現実化しました。**

| 送信先 | 送った num_ctx | `ollama ps` の CONTEXT |
|---|---|---|
| `/api/chat` (ネイティブ) | 16384 | **16384** |
| `/v1/chat/completions` | 16384 | 4096 |
| `/v1/chat/completions` | 32768 | 4096 |

測定時点の `llmkit` は OpenAI 互換経路のみを使っていたため (元仕様書の決定 Q-B)、コンテキスト長の
設定が実機に反映されず、VRAM 見積り・マニフェスト記録と実態が乖離していました。
サーバ側の既定を変える `OLLAMA_CONTEXT_LENGTH` 環境変数は存在しますが、プロファイルごとの切り替えはできません。

→ **解決済み (2026-08-23)。** `docs/plans/2026-08-22-phase1-calibration-and-native-chat.md` §4 T3/T4 で
`runtime.kind` によるクライアント実装の切り替えを実装し、`kind = "ollama"` はネイティブ `/api/chat` を
使うようにしました (**D-10**)。エンドポイントは `runtime.base_url` から導出します (**D-11**)。
実機 (Ollama 0.32.15) で `configs/default.toml` の `doctor` を実行し、送信先が `/api/chat` になり
`ollama ps` の CONTEXT が設定どおり 16384 になることを確認済みです。OpenAI 互換経路を残したまま
A/B 比較するための設定として `configs/ollama_openai_compat.toml` を追加しました
(こちらでは CONTEXT が 4096 のままになる = 上表の再現)。

## 埋め込み・リランカー (2026-08-23 実測、当初「取得不能」としていた項目)

当初 `hf.co/cl-nagoya/ruri-v3-310m` と `hf.co/cl-nagoya/ruri-reranker-large` は GGUF 非提供のため
`ollama pull` できず、Phase 0 の受け入れ条件2項目を満たせていなかった。**いずれも解決した。**

### 埋め込み: 有志の GGUF 変換版で取得できた

`ollama pull` できなかったのは Hugging Face の**元リポジトリ** (sentence-transformers 形式) だけで、
GGUF 変換版なら Ollama で扱える。

```bash
ollama pull hf.co/Targoyle/ruri-v3-310m-GGUF   # 336 MB
```

| 項目 | 結果 |
|---|---|
| エンドポイント | `/v1/embeddings` (Ollama) |
| 次元数 | 768 |
| 日本語の意味的分離 | 関連あり 0.9194 / 無関係 0.7820 (差 **+0.1375**) |
| VRAM 実測 | **0.57 GiB** |

### リランカー: Ollama では不可能。llama.cpp で解決

**Ollama にはリランキング API が無い。**

| エンドポイント | 結果 |
|---|---|
| `POST /api/rerank` | HTTP 404 |
| `POST /v1/rerank` | HTTP 404 |

最新の v0.33.0-rc2 でも未対応で、GitHub issue (`Reranking models` / `Add reranking support` /
`Add reranking in new engine`) はいずれも open のまま。GGUF 自体は存在するが、
**スコアを返すエンドポイントが無いため Ollama に載せても使えない。**

そこで **llama.cpp の `llama-server`** を別ポートで併走させた。要件書が第一候補としていた
`Ruri Reranker` は GGUF が存在しないため、要件書が代替として挙げている
**`BGE Reranker v2-m3`** を採用した。

- この環境には CUDA toolkit が無いが、**Vulkan が RTX 4070 Ti SUPER を認識**したため
  プリビルドバイナリ (33 MB) で GPU が使える。**ソースビルド不要**
- llama.cpp: `b10586` の `llama-b10586-bin-ubuntu-vulkan-x64.tar.gz`
- モデル: `gpustack/bge-reranker-v2-m3-GGUF` の `bge-reranker-v2-m3-Q6_K.gguf` (478 MB)

| 項目 | 結果 |
|---|---|
| エンドポイント | `/v1/rerank` (llama-server, port 8081) |
| 日本語のスコア分離 | 関連 **+0.80 / +0.63** vs 無関係 **-11.02 / -11.03** |
| **50ペアの所要時間** | **140 ms** (3回の中央値。要件書の目安 500〜800ms を下回る) |
| VRAM 実測 | **0.28 GiB** |

## 構成1 の同居実測 (2026-08-23)

生成 + 埋め込み + リランカーを同時に常駐させた状態を測定した。

| 段階 | VRAM 累計 | 増分 |
|---|---|---|
| アイドル (デスクトップ等) | 0.56 GiB | — |
| + リランカー (llama-server 常駐) | 0.84 GiB | **0.28** |
| + 生成 `qwen3:14b-q4_K_M` @16384 | 11.96 GiB | **11.12** |
| + 埋め込み `ruri-v3-310m` | **12.53 GiB** | **0.57** |

- **要件書の構成1 目安「約 12〜13GB」に収まった。** 余力 3.46 GiB (15.99 − 12.53)
- 生成・埋め込みとも `ollama ps` で **100% GPU**
- `OLLAMA_MAX_LOADED_MODELS` の設定変更は**不要**だった (既定で2モデル同居)
- リランカーは別プロセス (llama-server) が確保するため、`ollama ps` には現れない

## 受け入れ条件の充足状況

**Phase 0 は全項目クリア (2026-08-23)。**

| 条件 | 状態 |
|---|---|
| `nvidia-smi` が GPU を認識し VRAM 16GB が報告される | 達成 (16376 MiB) |
| Ollama が起動し OpenAI 互換エンドポイントが応答する | 達成 |
| 生成・埋め込み・リランカーの全てが取得済みで単発実行が成功 | **達成** (埋め込みは GGUF 変換版、リランカーは llama-server 経由) |
| 各モデルの実測 VRAM 使用量を記録した表が存在する | 達成 (本ドキュメント) |
| 構成1 が実測で 16GB 以内に収まることを確認 | **達成** (12.53 GiB、余力 3.46 GiB) |

## 環境の再現手順

### Ollama (生成 + 埋め込み)

インストールは公式スクリプト (`https://ollama.com/install.sh`) を取得して内容を確認してから実行する。
その後、以下を取得する。

```bash
ollama pull qwen3:14b-q4_K_M                   # 9.3 GB
ollama pull gpt-oss:20b                        # 13.8 GB
ollama pull qwen3:8b-q4_K_M                    # 5.2 GB
ollama pull hf.co/Targoyle/ruri-v3-310m-GGUF   # 336 MB (埋め込み)
```

### llama.cpp (リランカー)

CUDA toolkit は不要。Vulkan が NVIDIA GPU を認識していれば動く (`vulkaninfo --summary` で確認)。

```bash
# バイナリ (ソースビルド不要)
mkdir -p ~/.local/opt/llama.cpp
curl -sLo /tmp/llama-vulkan.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b10586/llama-b10586-bin-ubuntu-vulkan-x64.tar.gz
tar xzf /tmp/llama-vulkan.tar.gz -C ~/.local/opt/llama.cpp --strip-components=1

# モデル
mkdir -p ~/.local/share/llama-models
curl -sLo ~/.local/share/llama-models/bge-reranker-v2-m3-Q6_K.gguf \
  https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q6_K.gguf
```

起動はリポジトリの `scripts/start-reranker.sh` を使う (パスは環境変数で上書き可能)。
