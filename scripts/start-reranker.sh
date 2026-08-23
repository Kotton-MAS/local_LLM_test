#!/usr/bin/env bash
#
# リランカー (llama.cpp llama-server) を起動する。
#
# なぜ Ollama ではないのか:
#   Ollama にはリランキング API が無い (v0.33.0-rc2 時点で /api/rerank・
#   /v1/rerank とも 404、GitHub issue は open のまま)。スコアを返す
#   エンドポイントが無いため、リランカーの GGUF を載せても使えない。
#   そのため llama.cpp の llama-server を別ポートで併走させる (決定 D-17)。
#
# 検証済みの構成 (docs/phase0-vram-measurements.md):
#   llama.cpp : b10586 / llama-b10586-bin-ubuntu-vulkan-x64.tar.gz
#               CUDA toolkit は不要。Vulkan が NVIDIA GPU を認識すれば動く
#               (`vulkaninfo --summary` で確認できる)
#   モデル    : gpustack/bge-reranker-v2-m3-GGUF の bge-reranker-v2-m3-Q6_K.gguf
#   実測      : VRAM 0.28 GiB / 50 ペアの再ランキング 140 ms
#
# ⚠ RERANKER_CTX_SIZE を既定から上げると、カタログ (llmkit/catalog.py) の
#   weights_gib = 0.28 が過小評価になる。0.28 は --ctx-size 2048 かつ
#   --n-gpu-layers 99 の 1 点でしか実測していない。上げる場合は VRAM を
#   測り直してカタログを更新すること。
#
# ⚠ llama-server には認証機構が無い。RERANKER_HOST を 127.0.0.1 (既定) 以外
#   (例 0.0.0.0) にすると、LAN 上の誰でも無認証で /v1/rerank を叩けてしまう。
#   リモートから使いたい場合は RERANKER_HOST を変えず、SSH ポートフォワード
#   (例: ssh -L 8081:127.0.0.1:8081 <このホスト>) を使うこと。
#
# 使い方:
#   scripts/start-reranker.sh                 # 起動する (フォアグラウンド)
#   scripts/start-reranker.sh --dry-run       # 解決したコマンド行を出して終了する (存在チェックなし)
#   scripts/start-reranker.sh --print-command # --dry-run の別名 (同じ動作)
#
# 上記以外の引数 (--help・打ち間違い等) は使い方を表示して exit 2 する。
# 黙って無視して既定値で本番起動しない (F-6-002)。
#
# パスはすべて環境変数で上書きできる。リポジトリ外の絶対パスは既定値としてのみ持つ。

set -euo pipefail

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${HOME}/.local/opt/llama.cpp/llama-server}"
RERANKER_MODEL="${RERANKER_MODEL:-${HOME}/.local/share/llama-models/bge-reranker-v2-m3-Q6_K.gguf}"
RERANKER_HOST="${RERANKER_HOST:-127.0.0.1}"
RERANKER_PORT="${RERANKER_PORT:-8081}"
RERANKER_NGL="${RERANKER_NGL:-99}"
RERANKER_CTX_SIZE="${RERANKER_CTX_SIZE:-2048}"

command_line() {
	printf '%s --model %s --reranking --host %s --port %s --n-gpu-layers %s --ctx-size %s\n' \
		"${LLAMA_SERVER_BIN}" "${RERANKER_MODEL}" "${RERANKER_HOST}" \
		"${RERANKER_PORT}" "${RERANKER_NGL}" "${RERANKER_CTX_SIZE}"
}

usage() {
	cat >&2 <<-'EOF'
		使い方:
		  scripts/start-reranker.sh                 # 起動する (フォアグラウンド)
		  scripts/start-reranker.sh --dry-run       # 解決したコマンド行を出して終了する (存在チェックなし)
		  scripts/start-reranker.sh --print-command # --dry-run の別名 (同じ動作)
	EOF
}

# --dry-run / --print-command は解決結果を出すだけで終わる。存在チェックを
# 行わないのは、llama-server が入っていない CI でも環境変数の配線を検証
# できるようにするため。それ以外の引数 (--help・打ち間違い等) は黙って無視
# して本番起動せず、使い方を表示して exit 2 する (F-6-002)。
case "${1:-}" in
"") ;;
--dry-run | --print-command)
	command_line
	exit 0
	;;
*)
	echo "エラー: 不明な引数です: ${1}" >&2
	usage
	exit 2
	;;
esac

if [ ! -x "${LLAMA_SERVER_BIN}" ]; then
	cat >&2 <<-EOF
		エラー: llama-server が見つかりません: ${LLAMA_SERVER_BIN}

		対処:
		  1. https://github.com/ggml-org/llama.cpp/releases から
		     llama-<build>-bin-ubuntu-vulkan-x64.tar.gz を取得する
		     (CUDA toolkit は不要。Vulkan で NVIDIA GPU が使える)
		  2. mkdir -p ~/.local/opt/llama.cpp && tar xzf <取得したファイル> \\
		       -C ~/.local/opt/llama.cpp --strip-components=1
		  3. 別の場所に置く場合は LLAMA_SERVER_BIN で上書きする
	EOF
	exit 1
fi

if [ ! -f "${RERANKER_MODEL}" ]; then
	cat >&2 <<-EOF
		エラー: リランカーのモデルが見つかりません: ${RERANKER_MODEL}

		対処:
		  1. mkdir -p ~/.local/share/llama-models
		  2. curl -sLo ~/.local/share/llama-models/bge-reranker-v2-m3-Q6_K.gguf \\
		       https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q6_K.gguf
		  3. 別の場所に置く場合は RERANKER_MODEL で上書きする
	EOF
	exit 1
fi

# llama.cpp の共有ライブラリはバイナリと同じディレクトリに置かれる。
LLAMA_DIR="$(dirname "${LLAMA_SERVER_BIN}")"
export LD_LIBRARY_PATH="${LLAMA_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "起動します: $(command_line)" >&2
exec "${LLAMA_SERVER_BIN}" \
	--model "${RERANKER_MODEL}" \
	--reranking \
	--host "${RERANKER_HOST}" \
	--port "${RERANKER_PORT}" \
	--n-gpu-layers "${RERANKER_NGL}" \
	--ctx-size "${RERANKER_CTX_SIZE}"
