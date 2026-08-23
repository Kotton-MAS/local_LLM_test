"""RAG 索引の設定。入力は **単一の TOML ファイル** に閉じる (D-18 と同方針)。

- 形式は TOML (stdlib ``tomllib``)。YAML 依存を持ち込まない。
- スキーマは **pydantic dataclass + TypeAdapter** で定義し、``pydantic.BaseModel``
  を継承しない (D-08)。
- 検証エラーは新しい例外階層を作らず :class:`llmkit.ConfigError` に翻訳する。
  メッセージには該当キー名と対処を入れる。

パスの解決規則は 1 つだけ: **相対パスは設定ファイルのあるディレクトリからの相対**
として解決する。カレントディレクトリからの相対にすると、同じ設定ファイルが
起動場所によって別の vault を指すことになる。

``index.dir`` が ``vault.dir`` の配下に解決される設定は :class:`ConfigError` に
する (§3 論点5 の「設定」層)。索引成果物を vault の中に書くと、読み取り専用で
あるべき vault へ書き込む唯一の経路が設定ファイルから開いてしまう。

例外メッセージには **設定ファイルに書かれたままの文字列**を載せ、解決後の絶対
パスは載せない (CLAUDE.md のログ出力ルール: 実 vault のパスを出さない)。
"""

from __future__ import annotations

import dataclasses
import logging
import re
import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from pydantic.dataclasses import dataclass

from llmkit import ConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_EXCLUDE_GLOBS",
    "DEFAULT_INCLUDE_GLOBS",
    "ChunkSettings",
    "EmbedSettings",
    "RagSettings",
    "load_settings",
]

_FORBID_EXTRA = ConfigDict(extra="forbid")

NonEmptyStr = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]

#: vault ID に許す文字。``data/index/<id>/`` のディレクトリ名になるため、
#: パス区切りと ``..`` を構文レベルで排除する (``suite.id`` と同じ扱い)。
_VAULT_ID_PATTERN = re.compile(r"\A[a-z0-9_-]+\Z")

#: 索引対象のホワイトリスト。ここに一致しないファイルは最初から見ない。
DEFAULT_INCLUDE_GLOBS: tuple[str, ...] = ("**/*.md",)

#: ホワイトリストに一致したうえで落とすパターン。判定は常に**ファイル単位**
#: (相対パスとの照合) で行う。``.gitignore`` と違い ``notes`` のような末尾に
#: ワイルドカードの無いパターンはディレクトリ自身にしか一致せず、配下の
#: ファイルは除外されない。ディレクトリ全体を外すには ``dir/**`` と書く
#: (書き忘れは ``rag.vault`` が WARNING ログで検出する)。
DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = (
    ".obsidian/**",
    ".trash/**",
    ".git/**",
    "**/*.excalidraw.md",
)


@dataclass(frozen=True, config=_FORBID_EXTRA)
class ChunkSettings:
    """``[chunk]`` セクション。**すべて index_fingerprint に入る** (3b / D-28)。

    値を変えると同じノートから別のチャンクが生まれるため、差分更新の前提が
    変わったことを索引側が検出できなければならない。

    Attributes:
        max_tokens: 1 チャンクの推定トークン数の上限 (目安、要件書 L185)。
        cjk_chars_per_token: 近似トークン数の CJK 係数 (D-32)。
        ascii_chars_per_token: 近似トークン数の非 CJK 係数 (D-32)。
        heading_separator: ``heading_path`` を ``embed_text`` に展開する区切り。
    """

    max_tokens: PositiveInt = 240
    cjk_chars_per_token: PositiveFloat = 1.0
    ascii_chars_per_token: PositiveFloat = 4.0
    heading_separator: NonEmptyStr = " > "


@dataclass(frozen=True, config=_FORBID_EXTRA)
class EmbedSettings:
    """``[embed]`` セクション。

    ``batch_size`` は **index_fingerprint に入れない**。まとめ方を変えても
    個々のベクトルは変わらないため、これを前提に含めると意味のない全再構築を
    誘発する (E28 が固定する性質)。
    """

    batch_size: PositiveInt = 16


@dataclass(frozen=True, config=_FORBID_EXTRA)
class _VaultSection:
    """``[vault]`` セクションの生値 (パス未解決)。"""

    id: NonEmptyStr
    dir: NonEmptyStr
    include_globs: tuple[NonEmptyStr, ...] = DEFAULT_INCLUDE_GLOBS
    exclude_globs: tuple[NonEmptyStr, ...] = DEFAULT_EXCLUDE_GLOBS

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if _VAULT_ID_PATTERN.match(value) is None:
            msg = (
                "vault.id は [a-z0-9_-]+ のみ使えます "
                "(索引ディレクトリ名になるため、パス区切りや '..' は使えません)"
            )
            raise ValueError(msg)
        return value

    @field_validator("include_globs")
    @classmethod
    def _validate_include_globs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            msg = "vault.include_globs を空にはできません (索引対象が 0 件になります)"
            raise ValueError(msg)
        return value


@dataclass(frozen=True, config=_FORBID_EXTRA)
class _IndexSection:
    """``[index]`` セクションの生値 (パス未解決)。"""

    dir: NonEmptyStr | None = None


@dataclass(frozen=True, config=_FORBID_EXTRA)
class _RawSettings:
    """設定ファイル 1 つ分の生の内容。"""

    vault: _VaultSection
    index: _IndexSection = Field(default_factory=_IndexSection)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    embed: EmbedSettings = Field(default_factory=EmbedSettings)


_RAW_SETTINGS_ADAPTER: TypeAdapter[_RawSettings] = TypeAdapter(_RawSettings)


@dataclasses.dataclass(frozen=True)
class RagSettings:
    """解決済みの索引設定 (パスは絶対、glob は正規化済み)。

    自分で組み立てて渡すだけの型なので素の frozen dataclass にする (決定23)。

    Attributes:
        vault_id: vault の識別子。索引ディレクトリ名にもなる。
        vault_dir: 解決済みの vault ルート (絶対パス)。
        index_dir: 解決済みの索引出力先 (絶対パス)。**vault_dir の外**。
        include_globs: 索引対象のホワイトリスト。
        exclude_globs: ホワイトリストから落とすパターン。
        chunk: チャンク分割の設定。
        embed: 埋め込み呼び出しの設定。
        source_path: この設定を読み込んだ TOML のパス。
    """

    vault_id: str
    vault_dir: Path
    index_dir: Path
    include_globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    chunk: ChunkSettings
    embed: EmbedSettings
    source_path: Path


def _format_validation_error(exc: ValidationError) -> str:
    """pydantic の ValidationError を「キー名: 理由」の一覧に整形する。

    ``llmkit.config`` / ``harness.suite`` と同じ書式にそろえる。
    """
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(ルート)"
        lines.append(f"{location}: {error['msg']}")
    return "; ".join(lines)


def _read_toml(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"索引設定ファイルを読めません: {path}"
        raise ConfigError(
            msg, remediation=f"パスが正しいか確認してください ({exc.strerror})"
        ) from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = f"索引設定ファイルの TOML 構文が不正です: {path} ({exc})"
        raise ConfigError(msg, remediation="TOML の構文を修正してください") from exc


def _resolve_against(base_dir: Path, value: str) -> Path:
    """設定ファイルのあるディレクトリを基準に相対パスを解決する。"""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _validate_vault_dir(vault_dir: Path, configured: str) -> None:
    """vault ルートが実在するディレクトリであることを確かめる。

    メッセージには設定ファイルに書かれた文字列だけを載せる (解決後の絶対パスを
    載せない)。
    """
    if not vault_dir.exists():
        msg = f"vault.dir が存在しません: {configured}"
        raise ConfigError(
            msg,
            remediation=(
                "設定ファイルからの相対パス、または絶対パスで "
                "vault.dir を指定してください"
            ),
        )
    if not vault_dir.is_dir():
        msg = f"vault.dir がディレクトリではありません: {configured}"
        raise ConfigError(
            msg, remediation="vault のルートディレクトリを指定してください"
        )


def _validate_index_dir(index_dir: Path, vault_dir: Path, configured: str) -> None:
    """索引の出力先が vault の中に解決されていないことを確かめる (§3 論点5)。

    ``resolve()` 済みの絶対パスどうしで比較する。``../`` やシンボリックリンクを
    挟んだ書き方で vault の中を指す設定も、ここで落ちる。
    """
    if index_dir == vault_dir or vault_dir in index_dir.parents:
        msg = f"index.dir が vault.dir の配下に解決されました: {configured}"
        raise ConfigError(
            msg,
            remediation=(
                "索引の出力先を vault の外 (例: data/index/<vault.id>) に "
                "変更してください。vault は読み取り専用として扱います"
            ),
        )


def load_settings(path: Path) -> RagSettings:
    """索引設定 TOML を読み込んで検証済みの :class:`RagSettings` を返す。

    Args:
        path: 設定ファイルのパス。相対パスの基準にもなる。

    Raises:
        ConfigError: 読み込み・構文・スキーマ・パス解決のいずれかに失敗した場合。
    """
    raw_document = _read_toml(path)
    try:
        raw = _RAW_SETTINGS_ADAPTER.validate_python(raw_document)
    except ValidationError as exc:
        msg = f"索引設定エラー ({path}): {_format_validation_error(exc)}"
        raise ConfigError(
            msg, remediation="該当キーの値・型・綴りを確認してください"
        ) from exc

    base_dir = path.expanduser().resolve().parent
    vault_dir = _resolve_against(base_dir, raw.vault.dir)
    _validate_vault_dir(vault_dir, raw.vault.dir)

    configured_index_dir = (
        raw.index.dir if raw.index.dir is not None else f"data/index/{raw.vault.id}"
    )
    index_dir = _resolve_against(base_dir, configured_index_dir)
    _validate_index_dir(index_dir, vault_dir, configured_index_dir)

    logger.debug(
        "索引設定を読み込みました: path=%s vault_id=%s include=%d exclude=%d",
        path,
        raw.vault.id,
        len(raw.vault.include_globs),
        len(raw.vault.exclude_globs),
    )
    return RagSettings(
        vault_id=raw.vault.id,
        vault_dir=vault_dir,
        index_dir=index_dir,
        include_globs=tuple(raw.vault.include_globs),
        exclude_globs=tuple(raw.vault.exclude_globs),
        chunk=raw.chunk,
        embed=raw.embed,
        source_path=path,
    )
