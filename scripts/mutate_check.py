#!/usr/bin/env python3
"""変異検証をリポジトリの複製上で行う。

## なぜ必要か

変異検証 (実装を壊してテストが落ちることを確かめる) を作業ツリーで行うと、
キットの SubagentStop 自動コミットが「壊れた中間状態」を拾う。

「1 回の tool 呼び出しの中で 壊す → 実行 → 復元 を完結させる」という
プロンプト層の緩和策は **5 回失敗した**。並列でエージェントを動かすと、
別のエージェントの SubagentStop が自分の変異中に発火するため、自分の
呼び出しをいくら短くしても防げない。層が違う。

CLAUDE.md は「再発している事象への対策をプロンプト層に置かない。2 回目
以降は hook / スキーマ / テストへ層を下げる」と定めている。複製の上で
壊せば、追跡ツリーに触れることが**物理的に不可能**になる。

シェルではなく Python なのは、後片付けを ``tempfile.TemporaryDirectory``
に任せられるため。シェルで書くと再帰削除にパス展開を組み合わせることに
なり、ハード制約 (およびキットのガード) に触れる。

## 使い方

    scripts/mutate_check.py <変異の Python 式> -- <pytest の引数...>

第 1 引数は複製のルートで実行されるコード。``Path`` が使える。

    scripts/mutate_check.py \\
      'p=Path("rag/indexer.py"); p.write_text(p.read_text().replace("A","B"))' \\
      -- tests/test_rag_indexer.py -q -k some_guard

複製には追跡ファイルと未コミットの変更が入る (``.venv`` と ``data/`` は
持ち込まず、仮想環境は元のものを参照する)。作業ツリーは 1 バイトも
変更されない。

終了コードは複製上で走らせた pytest のものをそのまま返す。「変異させたら
落ちる」ことを確かめる用途なので、**落ちる (非 0) のが期待どおり**である
点に注意する。
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def _populate(work_dir: Path, repo_root: Path) -> None:
    """追跡ファイル + 未コミットの変更を複製に展開する。

    ``git archive`` を使うのは、``.venv`` や ``data/`` (索引成果物) を
    複製に持ち込まないため。前者は数百 MB、後者は実 vault を索引した
    場合にノート本文を含む。
    """
    archive = work_dir / "_snapshot.tar"
    with archive.open("wb") as handle:
        subprocess.run(
            ["git", "archive", "HEAD"], cwd=repo_root, stdout=handle, check=True
        )
    with tarfile.open(archive) as tar:
        tar.extractall(work_dir, filter="data")
    archive.unlink()

    # 変異検証は「いまの実装」に対して行うので、未コミットの変更も反映する。
    diff = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if diff.strip():
        subprocess.run(
            ["patch", "-p1", "-s"], cwd=work_dir, input=diff, text=True, check=True
        )


def main(argv: list[str]) -> int:
    if "--" not in argv or argv.index("--") == 0:
        sys.stderr.write(
            "使い方: scripts/mutate_check.py <変異の Python 式> -- <pytest の引数...>\n"
        )
        return 2
    separator = argv.index("--")
    mutation = "\n".join(argv[:separator])
    pytest_args = argv[separator + 1 :]
    if not pytest_args:
        sys.stderr.write("pytest の引数がありません\n")
        return 2

    repo_root = _repo_root()
    with tempfile.TemporaryDirectory(prefix="mutate-check-") as raw:
        work_dir = Path(raw)
        _populate(work_dir, repo_root)
        subprocess.run(
            [sys.executable, "-c", f"from pathlib import Path\n{mutation}"],
            cwd=work_dir,
            check=True,
        )
        sys.stderr.write(f"--- 複製に変異を適用しました: {work_dir} ---\n")
        # 仮想環境は複製せず元のものを使う (複製に uv sync すると数十秒かかる)。
        result = subprocess.run(
            [str(repo_root / ".venv" / "bin" / "python"), "-m", "pytest", *pytest_args],
            cwd=work_dir,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "", "HOME": str(Path.home())},
            check=False,
        )
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
