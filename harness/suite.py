"""比較スイート (入力) の読み込みと、ベース設定への適用。

入力は **単一の TOML ファイル** に閉じる (D-18)。``[[models]]`` とプロンプト集を
1 ファイルに収めることで、「その比較実行にどの入力を与えたか」を
:func:`suite_sha256` の 1 個のハッシュで表せる。YAML を採らないのは新規依存を
足さないため (llmkit と同じく stdlib の ``tomllib`` で読む)。出力側の生ログだけは
1 行 1 レコードの JSONL にする。

スキーマは llmkit と同じく **pydantic dataclass + TypeAdapter** で定義する
(``pydantic.BaseModel`` を継承しない、D-08)。検証エラーは新しい例外階層を作らず
:class:`llmkit.ConfigError` に翻訳し、メッセージに該当キー名を含める。

★ このモジュールの中核は :func:`apply_case` である。モデル ID は設定の 2 か所
(``generation.model`` と ``profiles[active].generation``) に現れ、前者はリクエストの
``model`` フィールドに、後者は VRAM 見積りと実行マニフェストの ``profile.models[]``
に届く。``configs/default.toml`` は両者が一致しているため、**片方だけ差し替えても
何もエラーにならず**「20B に投げているのに 14B の VRAM 見積りを記録した比較結果」が
静かに生成される。:func:`apply_case` は常に両方を同時に差し替える (D-19)。
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import re
import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from pydantic.dataclasses import dataclass

from llmkit import AppConfig, ConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "ComparisonSuite",
    "ModelCase",
    "PromptSpec",
    "SuiteMeta",
    "apply_case",
    "load_suite",
    "suite_sha256",
]

_FORBID_EXTRA = ConfigDict(extra="forbid")

NonEmptyStr = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
Temperature = Annotated[float, Field(ge=0.0, le=2.0)]
TopP = Annotated[float, Field(gt=0.0, le=1.0)]

#: スイート ID に許す文字。``id`` は ``results/<id>/`` のディレクトリ名になるため、
#: パス区切り (``/`` ``\``) と ``..`` を構文レベルで排除する。
_SUITE_ID_PATTERN = re.compile(r"\A[a-z0-9_-]+\Z")


@dataclass(frozen=True, config=_FORBID_EXTRA)
class SuiteMeta:
    """``[suite]`` セクション。スイート自体の同一性と実行条件。

    Attributes:
        id: ``results/<id>/`` のディレクトリ名になる識別子。
        description: 人間向けの説明 (出力ヘッダに写す)。
        warmup_runs: 各モデルにつき計測前に捨てずに実行する回数 (D-22)。
            0 を許すのは「コールド実行そのものを測る」ためで、負値は拒否する。
    """

    id: NonEmptyStr
    description: str = ""
    warmup_runs: Annotated[int, Field(ge=0)] = 1

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if _SUITE_ID_PATTERN.match(value) is None:
            msg = (
                "スイート ID は [a-z0-9_-]+ のみ使えます "
                "(出力ディレクトリ名になるため、パス区切りや '..' は使えません)"
            )
            raise ValueError(msg)
        return value


@dataclass(frozen=True, config=_FORBID_EXTRA)
class ModelCase:
    """``[[models]]`` の 1 件。比較対象 1 モデル分の実行条件。

    ``model_id`` 以外はすべて任意で、省略した項目はベース :class:`AppConfig` の値を
    そのまま使う。``profile`` を省略した場合は ``vram.active_profile`` が対象になる。
    """

    model_id: NonEmptyStr
    profile: NonEmptyStr | None = None
    context_tokens: PositiveInt | None = None
    temperature: Temperature | None = None
    top_p: TopP | None = None
    max_output_tokens: PositiveInt | None = None
    seed: int | None = None


@dataclass(frozen=True, config=_FORBID_EXTRA)
class PromptSpec:
    """``[[prompts]]`` の 1 件。

    ``id`` はスイート内で一意でなければならない (出力の突き合わせキーになるため)。
    """

    id: NonEmptyStr
    text: NonEmptyStr
    system: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, config=_FORBID_EXTRA)
class ComparisonSuite:
    """比較スイート 1 本 (= TOML ファイル 1 つ) の内容。"""

    suite: SuiteMeta
    models: Annotated[tuple[ModelCase, ...], Field(min_length=1)]
    prompts: Annotated[tuple[PromptSpec, ...], Field(min_length=1)]

    @field_validator("prompts")
    @classmethod
    def _validate_unique_prompt_ids(
        cls, value: tuple[PromptSpec, ...]
    ) -> tuple[PromptSpec, ...]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for prompt in value:
            if prompt.id in seen:
                duplicates.append(prompt.id)
            seen.add(prompt.id)
        if duplicates:
            msg = (
                "prompts[].id がスイート内で重複しています: "
                f"{', '.join(sorted(set(duplicates)))}"
            )
            raise ValueError(msg)
        return value

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(prompt.id for prompt in self.prompts)


_SUITE_ADAPTER: TypeAdapter[ComparisonSuite] = TypeAdapter(ComparisonSuite)


def _or_base[T](override: T | None, current: T) -> T:
    """スイート側の上書き値 (``None`` なら未指定) とベース値のどちらを使うか決める。

    ``or`` ではなく ``is None`` で判定する。``seed = 0`` や ``temperature = 0.0``
    は正当な上書き値であり、真偽値で判定すると黙ってベース値に戻る。
    """
    return current if override is None else override


def _format_validation_error(exc: ValidationError) -> str:
    """pydantic の ValidationError を「キー名: 理由」の一覧に整形する。

    ``llmkit.config`` と同じ書式にそろえる (利用者が読む先が 2 種類にならない)。
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
        msg = f"スイートファイルを読めません: {path}"
        raise ConfigError(
            msg, remediation=f"パスが正しいか確認してください ({exc.strerror})"
        ) from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = f"スイートファイルの TOML 構文が不正です: {path} ({exc})"
        raise ConfigError(msg, remediation="TOML の構文を修正してください") from exc


def load_suite(path: Path) -> ComparisonSuite:
    """比較スイート TOML を読み込んで検証済みの :class:`ComparisonSuite` を返す。

    Raises:
        ConfigError: 読み込み・構文・スキーマのいずれかに失敗した場合。
    """
    raw = _read_toml(path)
    try:
        suite = _SUITE_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        msg = f"スイート定義エラー ({path}): {_format_validation_error(exc)}"
        raise ConfigError(
            msg, remediation="該当キーの値・型・綴りを確認してください"
        ) from exc
    logger.debug(
        "スイートを読み込みました: path=%s id=%s models=%d prompts=%d",
        path,
        suite.suite.id,
        len(suite.models),
        len(suite.prompts),
    )
    return suite


def suite_sha256(path: Path) -> str:
    """スイートファイルの内容ハッシュ (入力の同一性)。

    ``llmkit.compute_config_sha256`` と同じ役割をスイート側に対して果たす。

    Raises:
        ConfigError: スイートファイルを読めない場合。
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        msg = f"スイートファイルを読めないためハッシュを計算できません: {path}"
        raise ConfigError(
            msg, remediation=f"パスが正しいか確認してください ({exc.strerror})"
        ) from exc


def apply_case(base: AppConfig, case: ModelCase) -> AppConfig:
    """ベース設定に :class:`ModelCase` を適用した実効 :class:`AppConfig` を返す。

    ★ **モデル ID は必ず 2 か所を同時に差し替える** (D-19):

    1. ``generation.model`` — ``resolve_model_spec`` を経てリクエストの ``model``
       フィールドになる。
    2. ``profiles[<対象プロファイル>].generation`` — ``resolve_profile`` を経て
       VRAM 見積りの内訳と実行マニフェストの ``profile.models[]`` になる。

    加えて ``vram.active_profile`` を対象プロファイルにそろえる。1 だけを
    差し替える経路はこのモジュールに存在しない。片方だけを差し替えると、
    リクエストは 20B に飛ぶのに VRAM 見積りは 14B のまま、という**何のエラーも
    出さずに嘘の比較結果を書き出す**状態になる。

    ``model_id`` 以外の項目は ``None`` (スイート側で省略) ならベースの値を残す。

    Args:
        base: ベース設定ファイルから読み込んだ :class:`AppConfig`。
        case: 適用する ``[[models]]`` 1 件。

    Raises:
        ConfigError: ``case.profile`` がベース設定に定義されていない場合。
    """
    profile_name = (
        case.profile if case.profile is not None else base.vram.active_profile
    )
    profile = base.profiles.get(profile_name)
    if profile is None:
        known = ", ".join(sorted(base.profiles)) or "(なし)"
        msg = (
            f"models[].profile '{profile_name}' に対応する [profiles.*] が"
            f"ベース設定にありません。定義済み: {known}"
        )
        raise ConfigError(
            msg,
            remediation=(
                "スイートの models[].profile を修正するか、ベース設定に "
                "[profiles.<名前>] を追加してください"
            ),
        )

    generation = dataclasses.replace(
        base.generation,
        # (1) リクエストの model フィールドへ届く側
        model=case.model_id,
        context_tokens=_or_base(case.context_tokens, base.generation.context_tokens),
        temperature=_or_base(case.temperature, base.generation.temperature),
        top_p=_or_base(case.top_p, base.generation.top_p),
        max_output_tokens=_or_base(
            case.max_output_tokens, base.generation.max_output_tokens
        ),
        seed=_or_base(case.seed, base.generation.seed),
    )
    profiles = dict(base.profiles)
    # (2) VRAM 見積り・実行マニフェストへ届く側。(1) と同じ case.model_id を使う。
    profiles[profile_name] = dataclasses.replace(profile, generation=case.model_id)
    vram = dataclasses.replace(base.vram, active_profile=profile_name)
    return dataclasses.replace(
        base, generation=generation, vram=vram, profiles=profiles
    )
