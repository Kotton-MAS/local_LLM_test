"""テスト全体で共有する設定・fixture。

このファイルが担保するのは 2 点:

1. **live マーカーの既定スキップ** — 実ランタイムに接続するテストは
   ``@pytest.mark.live`` を付け、``--run-live`` を渡したときだけ実行する (D-02)。
2. **実ネットワークの遮断** — live 以外のテストではソケット接続を機械的に禁止する。
   ``httpx.MockTransport`` はソケットを使わないため影響を受けない。

加えて、各テストモジュールに散っていた「``LLMKIT_*`` 環境変数の除去」
「成功応答ペイロード」「リクエスト捕捉つき MockTransport」をここに集約する。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import httpx
import pytest

from harness.gpu import GpuMemory

pytest_plugins = ["pytester"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"
EXTERNAL_CONFIG = REPO_ROOT / "configs" / "external_openai.toml"

LIVE_MARKER = "live"
RUN_LIVE_OPTION = "--run-live"

#: 正常な OpenAI 互換 chat completion 応答。全テストが共有する。
SUCCESS_PAYLOAD: dict[str, object] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "qwen3:14b-q4_K_M",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "テスト応答"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
}

#: 正常な Ollama ネイティブ /api/chat 応答 (stream=false)。SUCCESS_PAYLOAD と
#: 同じ内容を、ネイティブ形状で表したもの。
NATIVE_SUCCESS_PAYLOAD: dict[str, object] = {
    "model": "qwen3:14b-q4_K_M",
    "created_at": "2026-08-22T00:00:00.000000000Z",
    "message": {"role": "assistant", "content": "テスト応答"},
    "done": True,
    "done_reason": "stop",
    "total_duration": 1_600_000_000,
    "prompt_eval_count": 11,
    "prompt_eval_duration": 550_000_000,
    "eval_count": 7,
    "eval_duration": 1_000_000_000,
}

#: ネイティブ経路の判定に使うパス。runtime.kind = "ollama" の送信先。
NATIVE_CHAT_PATH = "/api/chat"

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------
# live マーカー (D-02)
# --------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        RUN_LIVE_OPTION,
        action="store_true",
        default=False,
        help="実ランタイムに接続する live マーカー付きテストも実行する",
    )


def pytest_configure(config: pytest.Config) -> None:
    # pyproject.toml にも登録しているが、rootdir が変わる入れ子実行でも
    # マーカーが未知にならないようにここでも宣言する。
    config.addinivalue_line(
        "markers",
        f"{LIVE_MARKER}: 実ランタイムに接続するテスト (既定スキップ)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """``--run-live`` が無い限り live マーカー付きテストをスキップする。"""
    if config.getoption(RUN_LIVE_OPTION):
        return
    skip_live = pytest.mark.skip(
        reason=f"live テストは既定でスキップします ({RUN_LIVE_OPTION} で実行)"
    )
    for item in items:
        if item.get_closest_marker(LIVE_MARKER) is not None:
            item.add_marker(skip_live)


# --------------------------------------------------------------------------
# 実行環境の隔離
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_llmkit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ローカルの LLMKIT_* 環境変数でテスト結果が変わらないようにする。"""
    for name in list(os.environ):
        if name.startswith("LLMKIT_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _forbid_real_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """live 以外のテストからの実ネットワーク接続を禁止する (D-02)。

    ``httpx.MockTransport`` はソケットを開かないため、正しく書かれたテストは
    この fixture の影響を受けない。実接続が混入した瞬間に落ちる。
    """
    if request.node.get_closest_marker(LIVE_MARKER) is not None:
        return

    def _blocked(*args: object, **kwargs: object) -> object:
        message = (
            "テストが実ネットワークへ接続しようとしました。"
            "httpx.MockTransport を使うか @pytest.mark.live を付けてください (D-02)"
        )
        raise AssertionError(message)

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


# --------------------------------------------------------------------------
# 共有 fixture
# --------------------------------------------------------------------------


class RecordingTransport:
    """発行されたリクエストをすべて記録する ``httpx.MockTransport`` のラッパ。

    ``requests`` が空であること自体が「HTTP を 1 回も出していない」ことの証拠に
    なるため、受け入れ条件3 の検証にもそのまま使える。
    """

    def __init__(
        self,
        handler: Handler | None = None,
        *,
        requests: list[httpx.Request] | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = requests if requests is not None else []
        self._handler = handler if handler is not None else _default_handler
        self.transport = httpx.MockTransport(self._record)

    def _record(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    def client(self) -> httpx.Client:
        """このトランスポートを使う ``httpx.Client`` を作る。"""
        return httpx.Client(transport=self.transport)

    @property
    def call_count(self) -> int:
        return len(self.requests)


def _default_handler(request: httpx.Request) -> httpx.Response:
    """送信先パスに合わせた正常応答を返す。

    ネイティブ ``/api/chat`` と OpenAI 互換 ``/chat/completions`` では応答形状が
    違うため、URL で振り分ける (ここで分岐しないと、ネイティブ経路のテストが
    OpenAI 形状の応答を受け取って UpstreamError になる)。
    """
    if request.url.path.endswith(NATIVE_CHAT_PATH):
        return httpx.Response(200, json=NATIVE_SUCCESS_PAYLOAD)
    return httpx.Response(200, json=SUCCESS_PAYLOAD)


@pytest.fixture
def mock_transport() -> RecordingTransport:
    """成功応答を返し、リクエストを捕捉する MockTransport。"""
    return RecordingTransport()


@pytest.fixture
def mock_http_client(mock_transport: RecordingTransport) -> Iterator[httpx.Client]:
    """``mock_transport`` を使う httpx.Client (使用後に閉じる)。"""
    with mock_transport.client() as client:
        yield client


class FakeProbe:
    """注入する :class:`~harness.gpu.VramProbe`。実プロセスを起動しない。

    ``harness/`` のテストに共有する (F-8-006: 4 ファイルにバイト単位で完全に
    同一定義されていたものをここへ集約)。呼ばれるたびに ``readings`` を順に
    返すため、「全モデルが同じアイドル基準を共有する」ような掃引テストが
    単一値フェイクで検出漏れを起こさない (2026-08-23 実機実行で発見した不具合
    の再発防止と同じ設計)。
    """

    def __init__(self, readings: Sequence[GpuMemory | None] = ()) -> None:
        self._readings = list(readings)
        self.call_count = 0

    def read(self) -> GpuMemory | None:
        self.call_count += 1
        if not self._readings:
            return None
        index = min(self.call_count - 1, len(self._readings) - 1)
        return self._readings[index]


class ConfigWriter(Protocol):
    """``tmp_config`` fixture が返すファクトリの型。"""

    def __call__(
        self,
        edits: Mapping[str, str] | None = None,
        *,
        name: str = "config.toml",
    ) -> Path: ...


def write_config_variant(
    directory: Path,
    edits: Mapping[str, str] | None = None,
    *,
    name: str = "config.toml",
) -> Path:
    """``configs/default.toml`` を一部置換して ``directory`` に書き出す。

    ``edits`` は「置換前の文字列 -> 置換後の文字列」。置換対象が見つからなければ
    その場で落とす (設定ファイルの書式が変わったのにテストだけ通る事故を防ぐ)。
    ``bootstrap`` は Path しか受け取らないため、設定ファイル経由でしか振れない値を
    テストから掃引するのに使う。
    """
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    for before, after in (edits or {}).items():
        replaced = text.replace(before, after)
        assert replaced != text, f"置換対象が見つかりません: {before}"
        text = replaced
    destination = directory / name
    destination.write_text(text, encoding="utf-8")
    return destination


@pytest.fixture
def tmp_config(tmp_path: Path) -> ConfigWriter:
    """``write_config_variant`` を ``tmp_path`` に束ねたファクトリ。"""

    def write(
        edits: Mapping[str, str] | None = None, *, name: str = "config.toml"
    ) -> Path:
        return write_config_variant(tmp_path, edits, name=name)

    return write


# --------------------------------------------------------------------------
# 合成 vault (rag / Phase 3)
# --------------------------------------------------------------------------

#: コミット済みの合成 vault。読み取りだけを行うテストはこれを直接使う。
SAMPLE_VAULT_DIR = REPO_ROOT / "vaults" / "sample"
SAMPLE_VAULT_CONFIG = REPO_ROOT / "vaults" / "sample.toml"


def write_rag_settings(
    directory: Path,
    *,
    vault_dir: str = "vault",
    index_dir: str = "index",
    vault_id: str = "sample",
    include_globs: Sequence[str] | None = None,
    exclude_globs: Sequence[str] | None = None,
    name: str = "rag.toml",
) -> Path:
    """索引設定 TOML を ``directory`` に書き出してそのパスを返す。

    パスは設定ファイルからの相対として解決されるため、リポジトリの外 (tmp_path)
    でも実 vault の絶対パスを 1 つも書かずに掃引できる (E30)。
    """
    lines = ["[vault]", f'id = "{vault_id}"', f'dir = "{vault_dir}"']
    if include_globs is not None:
        rendered = ", ".join(f'"{pattern}"' for pattern in include_globs)
        lines.append(f"include_globs = [{rendered}]")
    if exclude_globs is not None:
        rendered = ", ".join(f'"{pattern}"' for pattern in exclude_globs)
        lines.append(f"exclude_globs = [{rendered}]")
    lines += ["", "[index]", f'dir = "{index_dir}"', ""]
    destination = directory / name
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


@pytest.fixture
def sample_vault_copy(tmp_path: Path) -> Path:
    """``vaults/sample/`` を ``tmp_path/vault`` に複製してそのパスを返す。

    複製に対して書き込み権限の変更・除外パターンの掃引・シンボリックリンクの
    追加を行うため、コミット済みの合成 vault 自体は決して変更されない。
    """
    destination = tmp_path / "vault"
    shutil.copytree(SAMPLE_VAULT_DIR, destination)
    return destination


# --------------------------------------------------------------------------
# 決定論的なフェイク埋め込み (rag / Phase 3b)
# --------------------------------------------------------------------------

#: フェイクのランタイムが名乗るモデル名 (応答の ``model``)。設定に書いた
#: モデル ID とは別物で、索引はこちらをマニフェストに記録する (D-35)。
FAKE_EMBEDDING_MODEL = "fake-embedding"

#: 既定の次元。検査したいのは「同じテキストなら同じベクトル」であって次元
#: そのものではないので軽量な低次元にする (768 次元は専用のテストが通す)。
FAKE_EMBEDDING_DIMENSIONS = 8

#: 埋め込み応答を差し替えるフック。入力テキストを受け取り、``None`` を返すと
#: 通常の応答、``httpx.Response`` を返すとその応答になる。例外を送出すれば
#: 接続断 (``httpx.ConnectError``) も再現できる。
EmbeddingIntercept = Callable[[tuple[str, ...]], httpx.Response | None]


def fake_embedding_vector(
    text: str, *, dimensions: int = FAKE_EMBEDDING_DIMENSIONS
) -> tuple[float, ...]:
    """入力テキストの sha256 から決定論的にベクトルを導出する。

    「同じテキストなら必ず同じベクトル」でないと、索引成果物のバイト一致
    (D-37) も「編集していないノートのベクトルが変わっていない」(E29) も
    検査できない。乱数やカウンタで作ると、再実行のたびに索引が変わるので
    差分更新が正しくてもテストが落ちる (逆に壊れていても気づけない)。

    ハッシュを ``dimensions`` 分だけ伸ばすためにカウンタを連結して繰り返す
    (768 次元でも同じ規則で作れる)。値は ``[-1, 1)`` に収める。
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    stream = bytearray()
    counter = 0
    while len(stream) < dimensions:
        stream += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    return tuple((stream[index] - 128) / 128.0 for index in range(dimensions))


def fake_embedding_payload(
    texts: Sequence[str],
    *,
    dimensions: int = FAKE_EMBEDDING_DIMENSIONS,
    model: str = FAKE_EMBEDDING_MODEL,
) -> dict[str, object]:
    """OpenAI 互換 ``/embeddings`` の正常応答 (入力と同じ順序)。"""
    return {
        "object": "list",
        "model": model,
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": list(fake_embedding_vector(text, dimensions=dimensions)),
            }
            for index, text in enumerate(texts)
        ],
    }


def fake_embedding_handler(
    *,
    dimensions: int = FAKE_EMBEDDING_DIMENSIONS,
    model: str = FAKE_EMBEDDING_MODEL,
    intercept: EmbeddingIntercept | None = None,
) -> Handler:
    """``/embeddings`` に決定論的な応答を返すハンドラ。

    実 HTTP は 1 バイトも出さないが、``llmkit`` の
    :class:`~llmkit.OpenAIEmbeddingClient` を素通りするので、例外の翻訳表
    (次元不整合・件数不整合・エラーステータス) も本番と同じ経路を通る。
    """

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        texts = tuple(str(text) for text in body["input"])
        if intercept is not None:
            replacement = intercept(texts)
            if replacement is not None:
                return replacement
        return httpx.Response(
            200, json=fake_embedding_payload(texts, dimensions=dimensions, model=model)
        )

    return handle


def fake_embedding_transport(
    *,
    dimensions: int = FAKE_EMBEDDING_DIMENSIONS,
    model: str = FAKE_EMBEDDING_MODEL,
    intercept: EmbeddingIntercept | None = None,
) -> RecordingTransport:
    """:func:`fake_embedding_handler` を積んだ :class:`RecordingTransport`。

    ``call_count`` が索引の要求回数そのものになる (E28 の掃引に使う)。
    """
    return RecordingTransport(
        fake_embedding_handler(dimensions=dimensions, model=model, intercept=intercept)
    )
