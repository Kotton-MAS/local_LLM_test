"""GPU メモリの実測プローブ。**外部コマンドを起動するのはこのモジュールだけ**。

D-01 (VRAM 見積りは静的テーブルのみを出典とし、GPU に触れない) は L2 に対する
制約であり、L2 の純粋性を機械検証できるようにするため、実測は L3 のこのファイルに
隔離する。``llmkit/`` 側には ``subprocess`` も GPU 参照も 1 つも持ち込まない。

方針:

- 記録するのは **GPU 名・used MiB・total MiB の 3 つだけ**。実行中プロセスの一覧
  (``--query-compute-apps``) は取得しない。ユーザー名・コマンドライン・他ユーザーの
  ジョブ名が比較結果に混入するため (CLAUDE.md のログ出力ルール)。
- コマンド列と実行関数はコンストラクタで注入できる。テストは偽の実行関数を渡し、
  **実プロセスを 1 つも起動しない**。
- 失敗 (未インストール / 非 0 終了 / パース失敗 / タイムアウト) は例外を上に投げず
  ``None`` を返す。VRAM 実測は比較の付加情報であり、``nvidia-smi`` が無い環境
  (CI) で比較実行そのものが落ちてよい理由にならない。
- 生出力はログにも記録にも出さない。パース失敗時に出力全文をログへ流すと、
  ドライバのエラーメッセージ経由でホスト名等が混入し得る。
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "DEFAULT_NVIDIA_SMI_COMMAND",
    "DEFAULT_TIMEOUT_S",
    "CommandResult",
    "CommandRunner",
    "GpuMemory",
    "NvidiaSmiProbe",
    "VramProbe",
    "run_command",
]

logger = logging.getLogger(__name__)

#: 名前と使用量・総量だけを CSV で問い合わせる。プロセス一覧は取らない。
DEFAULT_NVIDIA_SMI_COMMAND: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=name,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)

DEFAULT_TIMEOUT_S = 5.0

#: 複数 GPU が刺さっている環境では先頭 (index 0) の 1 台だけを読む。
#: 本プロジェクトの対象は単一 GPU 機であり、合算すると「どのカードに載ったか」が
#: 読めなくなるため。
_TARGET_GPU_INDEX = 0

_CSV_FIELD_COUNT = 3


@dataclass(frozen=True, slots=True)
class GpuMemory:
    """``nvidia-smi`` が返した GPU メモリの実測値。

    Attributes:
        name: GPU 名 (例 ``NVIDIA GeForce RTX 5070 Ti``)。
        used_mib: デバイス全体の使用量 (MiB)。プロセス単位ではない。
        total_mib: デバイスの総容量 (MiB)。
    """

    name: str
    used_mib: int
    total_mib: int


class VramProbe(Protocol):
    """VRAM 実測値の取得口。取得できない環境では ``None`` を返す。"""

    def read(self) -> GpuMemory | None:
        """現在の GPU メモリ使用量を読む。失敗時は例外を投げず ``None``。"""
        ...


class CommandResult(Protocol):
    """``subprocess.CompletedProcess[str]`` のうち本モジュールが使う部分。

    テストが偽の結果を渡せるよう、具象クラスではなく構造で受ける。
    """

    @property
    def returncode(self) -> int: ...

    @property
    def stdout(self) -> str: ...


class CommandRunner(Protocol):
    """コマンドを 1 回実行して結果を返す関数。"""

    def __call__(
        self, command: Sequence[str], *, timeout_s: float
    ) -> CommandResult: ...


def run_command(command: Sequence[str], *, timeout_s: float) -> CommandResult:
    """既定の実行関数。``subprocess.run`` をタイムアウト付きで呼ぶ。

    ``shell=False`` (リスト渡し) 固定であり、シェルを介さないためコマンド
    インジェクションの経路を作らない。
    """
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


class NvidiaSmiProbe:
    """``nvidia-smi`` を 1 回起動して GPU メモリ使用量を読む :class:`VramProbe`。

    Args:
        command: 実行するコマンド列。既定は
            :data:`DEFAULT_NVIDIA_SMI_COMMAND`。
        runner: コマンド実行関数。既定は :func:`run_command`。テストは実プロセスを
            起動しない偽の関数を渡す。
        timeout_s: 1 回の実行に許す秒数 (既定 5 秒)。
    """

    def __init__(
        self,
        *,
        command: Sequence[str] = DEFAULT_NVIDIA_SMI_COMMAND,
        runner: CommandRunner = run_command,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._command = tuple(command)
        self._runner = runner
        self._timeout_s = timeout_s
        #: WARNING は 1 実行につき 1 回だけ出す。モデルとプロンプトの組ごとに
        #: 読むため、毎回警告するとログが実測不能の告知で埋まる。
        self._warned = False

    def read(self) -> GpuMemory | None:
        """GPU メモリ使用量を読む。取得できなければ ``None`` を返す。

        例外は外に出さない。呼び出し側 (ランナー) は実測列を欠測として扱う。
        """
        try:
            completed = self._runner(self._command, timeout_s=self._timeout_s)
        except (OSError, subprocess.SubprocessError) as exc:
            # FileNotFoundError (未インストール) / TimeoutExpired / PermissionError。
            # 例外の型名だけを残す。メッセージにはパスや環境が入り得る。
            self._give_up(f"{type(exc).__name__} が発生しました")
            return None
        if completed.returncode != 0:
            self._give_up(f"終了コード {completed.returncode} で失敗しました")
            return None
        return self._parse(completed.stdout)

    def _parse(self, stdout: str) -> GpuMemory | None:
        """``name, used, total`` の CSV 1 行目を :class:`GpuMemory` にする。

        生出力はログにも戻り値にも出さない (ドライバのメッセージ経由で環境情報が
        混入するのを避けるため)。
        """
        lines = [line for line in stdout.splitlines() if line.strip()]
        if len(lines) <= _TARGET_GPU_INDEX:
            self._give_up("出力に GPU の行がありませんでした")
            return None
        fields = [field.strip() for field in lines[_TARGET_GPU_INDEX].split(",")]
        if len(fields) != _CSV_FIELD_COUNT or not fields[0]:
            self._give_up("出力の列数が想定と異なります")
            return None
        try:
            used_mib = int(fields[1])
            total_mib = int(fields[2])
        except ValueError:
            self._give_up("メモリ量を整数として解釈できませんでした")
            return None
        return GpuMemory(name=fields[0], used_mib=used_mib, total_mib=total_mib)

    def _give_up(self, reason: str) -> None:
        """失敗理由を記録する (WARNING は 1 実行につき 1 回だけ)。"""
        logger.debug("%s の実行に失敗しました: %s", self._command[0], reason)
        if not self._warned:
            self._warned = True
            logger.warning(
                "GPU メモリの実測値を取得できません (以降は記録を省略します): %s",
                reason,
            )
