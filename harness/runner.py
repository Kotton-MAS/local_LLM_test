"""比較実行の計画 (``plan_run``)・実行 (``run_suite``)・再現条件 (``run_fingerprint``)。

★ このモジュールの中核は **``run_fingerprint`` が ``config_sha256`` の代わりに
ならないこと**である (D-20)。

ハーネスは設定を実行時に in-memory で上書きする (:func:`harness.suite.apply_case`)
ため、``manifest.config_sha256`` は**ベース設定ファイルの同一性しか表さない**。
モデルの違う 2 実行が同じ ``config_sha256`` を持つ。したがって「同じ入力で
同じ結果条件になるか」の判定は、実効値そのものから作る
:func:`run_fingerprint` が担う。

フィンガープリントに入れるもの / 入れないもの:

* 入れる … スイートのハッシュ / ベース設定のハッシュ / レコードスキーマ版 /
  全ケースの実行マニフェスト (実効値がそのまま写っている) / 全プロンプトの
  id と本文ハッシュ / ``warmup_runs``
* **入れない** … ``run_id`` と ``started_at_utc`` (実行のたびに変わるため、
  含めると「2 回実行で一致」が恒真ではなく**恒偽**に落ちる)、応答テキスト、
  レイテンシ、速度、VRAM 実測値、実行時刻

実行順序は **model-major** (1 モデルにつき全プロンプトを連続実行) で、
ウォームアップの結果も ``phase="warmup"`` として残す (D-22)。ケース単位の失敗は
記録して次のモデルへ進む。VRAM 予算超過は起動前に停止するため **HTTP を 1 バイトも
発行しない** (D-04)。
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from harness.gpu import GpuMemory, VramProbe
from harness.records import (
    RECORD_SCHEMA_VERSION,
    RecordError,
    ResponseFacts,
    RunPhase,
    RunRecord,
    RunSummary,
    SpeedMetrics,
    VramReading,
    summarize_records,
)
from harness.suite import ComparisonSuite, ModelCase, PromptSpec, apply_case
from harness.suite import suite_sha256 as compute_suite_sha256
from llmkit import (
    AppConfig,
    ChatClient,
    ChatMessage,
    LlmkitError,
    ModelSpec,
    ResolvedProfile,
    RunManifest,
    VramBudgetExceededError,
    VramEstimate,
    bootstrap_from_config,
    build_manifest,
    check_budget,
    compute_config_sha256,
    estimate_resolved_profile,
    resolve_model_spec,
    resolve_profile,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MANIFESTS_DIRNAME",
    "CasePlan",
    "Clock",
    "RunPlan",
    "RunResult",
    "SuiteFilters",
    "fingerprint_digest",
    "fingerprint_inputs",
    "plan_run",
    "run_fingerprint",
    "run_suite",
]

#: 実行マニフェストを書き出す ``results/<...>/`` 配下のサブディレクトリ名。
MANIFESTS_DIRNAME = "manifests"

#: 現在時刻を返す関数。テストが実行時刻を固定できるよう注入可能にする。
Clock = Callable[[], datetime]

#: フィンガープリントの入力からマニフェストの何を落とすか (D-20)。
_VOLATILE_MANIFEST_KEYS = ("run_id", "started_at_utc")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _isoformat_utc(moment: datetime) -> str:
    """マニフェストの ``started_at_utc`` と同じ書式にそろえる。"""
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# 計画 (HTTP を 1 バイトも出さない)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CasePlan:
    """比較対象 1 モデル分の実効条件。``--dry-run`` はこれを出すだけで足りる。

    Attributes:
        manifest: 計画段階の実行マニフェスト。フィンガープリントの入力。
            ``run_id`` / ``started_at_utc`` は毎回変わる (だからこそ
            :func:`run_fingerprint` はその 2 つを除外する、D-20)。実行時の
            マニフェストは ``bootstrap_from_config`` が改めて組み立てる。
        spec: ``generation.model`` を解決したカタログ項目。リクエストに送る
            ``served_name`` の出典。
        starts_without_budget_abort: 起動が VRAM 予算で停止しないか。
            :func:`llmkit.check_budget` を実際に呼んで
            :class:`llmkit.VramBudgetExceededError` を捕まえたかどうかから
            導出する (規則の再実装をしない)。:attr:`VramEstimate.within_budget`
            とは名前も意味も別物であり、``is_local=false`` (外部 API) では
            見積りが予算を超えていても起動は止まらないため両者は一致しない
            (F-8-003)。
    """

    case_index: int
    case: ModelCase
    config: AppConfig
    profile: ResolvedProfile
    spec: ModelSpec
    estimate: VramEstimate
    manifest: RunManifest
    starts_without_budget_abort: bool

    @property
    def model_id(self) -> str:
        return self.case.model_id

    @property
    def profile_name(self) -> str:
        return self.profile.name


@dataclass(frozen=True, slots=True)
class SuiteFilters:
    """CLI の ``--models`` / ``--limit`` がスイートに適用されたかどうかの記録。

    ``plan_run`` に渡す ``suite`` はすでに絞り込み後の in-memory 値であり、
    :func:`harness.suite.suite_sha256` はベースファイルのハッシュのままなので、
    絞り込みが適用された事実そのものは ``suite_sha256`` からは読み取れない
    (F-8-002)。この値を ``run.json`` / ``report.md`` の再現条件に明示することで、
    成果物だけから実行コマンドを復元できるようにする。``run_fingerprint`` 自体は
    絞り込み後の実効値 (プロンプト集合・ケース一覧) から計算されるため既に
    フル実行とは別値になるが (D-20 は健在)、どちらのフィルタで絞り込んだのかは
    フィンガープリントの桁からは読めない。
    """

    models: str | None = None
    limit: int | None = None

    @property
    def is_empty(self) -> bool:
        """``--models`` / ``--limit`` のどちらも指定されていないか。"""
        return self.models is None and self.limit is None

    def describe(self) -> str:
        """人間向けの 1 行表現。指定が無ければその旨を明示する (0 件表示にしない)。"""
        if self.is_empty:
            return "(絞り込みなし: 全モデル・全プロンプト)"
        parts = []
        if self.models is not None:
            parts.append(f"--models {self.models}")
        if self.limit is not None:
            parts.append(f"--limit {self.limit}")
        return " ".join(parts)

    def to_json(self) -> dict[str, object]:
        return {"models": self.models, "limit": self.limit}


#: 絞り込みを指定しなかった実行の既定値。
_NO_FILTERS = SuiteFilters()


@dataclass(frozen=True, slots=True)
class RunPlan:
    """比較実行 1 回分の計画。HTTP を出さずにここまで確定する。"""

    suite: ComparisonSuite
    suite_path: Path
    base_config: AppConfig
    config_path: Path
    suite_sha256: str
    config_sha256: str
    cases: tuple[CasePlan, ...]
    fingerprint: str
    #: ``--models`` / ``--limit`` の指定内容 (F-8-002)。CLI を経由しない
    #: 呼び出しでは既定で「絞り込みなし」を表す。
    filters: SuiteFilters

    @property
    def suite_id(self) -> str:
        return self.suite.suite.id

    @property
    def warmup_runs(self) -> int:
        return self.suite.suite.warmup_runs

    @property
    def request_count(self) -> int:
        """予算内のケースが発行する HTTP リクエストの総数 (予定)。"""
        per_case = self.warmup_runs + len(self.suite.prompts)
        return sum(per_case for case in self.cases if case.starts_without_budget_abort)


def _plan_case(
    case_index: int, case: ModelCase, base_config: AppConfig, config_path: Path
) -> CasePlan:
    config = apply_case(base_config, case)
    profile = resolve_profile(config)
    estimate = estimate_resolved_profile(profile, config)
    try:
        check_budget(profile, config)
        starts_without_budget_abort = True
    except VramBudgetExceededError:
        # llmkit.check_budget が実際に投げた例外を捕まえるだけで、判定規則
        # 自体は再実装しない (L2 の check_budget が唯一の出典、F-8-003)。
        starts_without_budget_abort = False
    return CasePlan(
        case_index=case_index,
        case=case,
        config=config,
        profile=profile,
        spec=resolve_model_spec(
            config.generation.model, is_local=config.runtime.is_local
        ),
        estimate=estimate,
        manifest=build_manifest(config, profile, estimate, config_path),
        starts_without_budget_abort=starts_without_budget_abort,
    )


def plan_run(
    suite: ComparisonSuite,
    suite_path: Path,
    base_config: AppConfig,
    config_path: Path,
    *,
    filters: SuiteFilters | None = None,
) -> RunPlan:
    """スイートとベース設定から実行計画を組み立てる (``--dry-run`` の実体)。

    HTTP を 1 バイトも発行せず、プロセスも起動しない。全ケースの実効
    :class:`~llmkit.AppConfig`・VRAM 見積り・予算判定・
    :func:`run_fingerprint` をここで確定する。

    Args:
        suite: 読み込み済みスイート (``--models`` / ``--limit`` を適用済みなら
            絞り込み後の値)。
        suite_path: ``suite`` の出所となったファイル。ハッシュの計算に使う。
        base_config: ベース設定 (``[[models]]`` で上書きされる前の値)。
        config_path: ``base_config`` の出所となったファイル。
        filters: ``suite`` に適用した ``--models`` / ``--limit`` の値。省略時は
            「絞り込みなし」として記録する (F-8-002)。

    Raises:
        ConfigError: プロファイル・カタログの解決に失敗した場合。
    """
    cases = tuple(
        _plan_case(index, case, base_config, config_path)
        for index, case in enumerate(suite.models)
    )
    suite_hash = compute_suite_sha256(suite_path)
    config_hash = compute_config_sha256(config_path)
    fingerprint = run_fingerprint(
        suite,
        [case.manifest for case in cases],
        suite_sha256=suite_hash,
        config_sha256=config_hash,
    )
    logger.debug(
        "実行計画を作成しました: suite=%s models=%d prompts=%d fingerprint=%s",
        suite.suite.id,
        len(cases),
        len(suite.prompts),
        fingerprint,
    )
    return RunPlan(
        suite=suite,
        suite_path=suite_path,
        base_config=base_config,
        config_path=config_path,
        suite_sha256=suite_hash,
        config_sha256=config_hash,
        cases=cases,
        fingerprint=fingerprint,
        filters=filters if filters is not None else _NO_FILTERS,
    )


# --------------------------------------------------------------------------
# 再現条件 (D-20)
# --------------------------------------------------------------------------


def _manifest_fingerprint_dict(manifest: RunManifest) -> dict[str, object]:
    """マニフェストから実行のたびに変わる 2 項目を落とす (D-20)。

    ``run_id`` は毎回 uuid、``started_at_utc`` は毎回現在時刻であり、含めると
    「同じ入力の 2 回実行でフィンガープリントが一致する」が原理的に成立しなく
    なる。逆に、生成パラメータ・プロファイル構成・VRAM 見積り・接続先はすべて
    残す (これらが変われば再現条件が変わったということ)。
    """
    payload = manifest.to_dict()
    for key in _VOLATILE_MANIFEST_KEYS:
        del payload[key]
    return payload


def fingerprint_inputs(
    suite: ComparisonSuite,
    manifests: Sequence[RunManifest],
    *,
    suite_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    """フィンガープリントの入力を素の辞書で返す (``run.json`` に載せる形)。

    ハッシュ値だけを記録すると「何が変わったから変わったのか」が追えない。
    入力そのものを残し、:func:`fingerprint_digest` で再計算できるようにする。
    """
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "suite_sha256": suite_sha256,
        "config_sha256": config_sha256,
        "warmup_runs": suite.suite.warmup_runs,
        "prompts": [
            {
                "id": prompt.id,
                "text_sha256": _sha256_text(prompt.text),
                "system_sha256": _sha256_text(prompt.system or ""),
            }
            for prompt in suite.prompts
        ],
        "cases": [_manifest_fingerprint_dict(manifest) for manifest in manifests],
    }


def fingerprint_digest(inputs: Mapping[str, object]) -> str:
    """正規化 JSON (``sort_keys=True``, ``ensure_ascii=False``) の sha256。"""
    canonical = json.dumps(dict(inputs), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_fingerprint(
    suite: ComparisonSuite,
    manifests: Sequence[RunManifest],
    *,
    suite_sha256: str,
    config_sha256: str,
) -> str:
    """実効値から再現条件のフィンガープリントを計算する (D-20)。

    ``config_sha256`` では代替できない。ハーネスは設定を in-memory で上書き
    するため、``config_sha256`` はベース設定ファイルの同一性しか表さず、
    **モデルの違う 2 実行が同じ値を持つ**。
    """
    return fingerprint_digest(
        fingerprint_inputs(
            suite,
            manifests,
            suite_sha256=suite_sha256,
            config_sha256=config_sha256,
        )
    )


# --------------------------------------------------------------------------
# 実行
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunResult:
    """比較実行 1 回分の結果。``report.py`` はこれだけを見れば書き出せる。"""

    plan: RunPlan
    run_id: str
    started_at_utc: str
    records: tuple[RunRecord, ...]
    manifests: tuple[RunManifest, ...]
    manifest_paths: tuple[Path, ...]
    summary: RunSummary
    #: 実行開始直後に 1 回だけ測ったアイドル時の VRAM 使用量 (MiB)。
    #: プローブが使えない環境では ``None`` (D-23)。全モデルの
    #: ``vram_increment_gib`` はこの値を共通の基準として使う。
    vram_idle_mib: int | None
    #: ``run_suite`` に渡した出力先。``report.write_run_outputs`` /
    #: ``report.build_run_json`` が値の出所として使う唯一の場所であり、
    #: 呼び出し側が別の Path を独自に持ち回る必要をなくす (F-8-001)。
    results_dir: Path

    @property
    def fingerprint(self) -> str:
        """再現条件。計画時に確定しており、応答内容には依存しない (D-20)。"""
        return self.plan.fingerprint


@dataclass(frozen=True, slots=True)
class _Attempt:
    """1 回の生成の試行結果 (成功・失敗のどちらか)。"""

    response: ResponseFacts
    speed: SpeedMetrics
    error: RecordError | None


class _SuiteRunner:
    """1 実行分の状態 (連番・レコード・マニフェスト) を持つ内部ランナー。

    ``run_suite`` を 1 関数で書くと連番とレコード列を引数で持ち回ることになり、
    ウォームアップと計測の順序が読み取りにくくなるため、状態をここに閉じる。
    """

    def __init__(
        self,
        plan: RunPlan,
        *,
        run_id: str,
        probe: VramProbe,
        idle: GpuMemory | None,
        http_client: httpx.Client,
        results_dir: Path,
    ) -> None:
        self._plan = plan
        self._run_id = run_id
        self._probe = probe
        # アイドル基準は実行全体で 1 回だけ測り、全モデルで共有する
        # (仕様書 §4 T3)。モデルごとに測り直すと、2 モデル目以降の
        # ``vram_increment_gib`` が「前のモデルがロード済みの状態からの差」
        # になってしまい、実測値が意味を成さなくなる。
        self._idle = idle
        self._http_client = http_client
        self._results_dir = results_dir
        self._records: list[RunRecord] = []
        self._manifests: list[RunManifest] = []
        self._manifest_paths: list[Path] = []

    @property
    def records(self) -> tuple[RunRecord, ...]:
        return tuple(self._records)

    @property
    def manifests(self) -> tuple[RunManifest, ...]:
        return tuple(self._manifests)

    @property
    def manifest_paths(self) -> tuple[Path, ...]:
        return tuple(self._manifest_paths)

    def run(self) -> None:
        """model-major (1 モデルにつき全プロンプトを連続実行) で回す (D-22)。"""
        for case_plan in self._plan.cases:
            self._run_case(case_plan)

    def _run_case(self, case_plan: CasePlan) -> None:
        try:
            boot = bootstrap_from_config(
                case_plan.config,
                self._plan.config_path,
                profile_name=case_plan.profile_name,
                http_client=self._http_client,
                output_dir=self._results_dir / MANIFESTS_DIRNAME,
            )
        except LlmkitError as exc:
            # VramBudgetExceededError はここで止まる = HTTP を 1 バイトも
            # 発行しない (D-04)。他の起動失敗も同じ扱いで次のモデルへ進む。
            self._skip_case(case_plan, exc)
            return

        self._manifests.append(boot.manifest)
        if boot.manifest_path is not None:
            self._manifest_paths.append(boot.manifest_path)

        warmups = [
            (prompt, self._generate(boot.client, prompt))
            for prompt in self._warmup_prompts()
        ]
        # VRAM はモデルがロードされ切った時点 (ウォームアップ後) に 1 回読む。
        # ウォームアップ前は計測対象のモデルがまだ載っていないことがある。
        # アイドル基準 (self._idle) はここでは測り直さず、実行開始時に 1 回
        # 測ったものを全モデルで共有する。
        vram = VramReading.from_probe(idle=self._idle, current=self._probe.read())

        for prompt, attempt in warmups:
            self._append(case_plan, "warmup", prompt, attempt, vram)
        for prompt in self._plan.suite.prompts:
            attempt = self._generate(boot.client, prompt)
            self._append(case_plan, "measure", prompt, attempt, vram)

    def _warmup_prompts(self) -> Iterator[PromptSpec]:
        """ウォームアップに使うプロンプト。件数を超える分は先頭から巡回する。"""
        prompts = self._plan.suite.prompts
        for index in range(self._plan.warmup_runs):
            yield prompts[index % len(prompts)]

    def _generate(self, client: ChatClient, prompt: PromptSpec) -> _Attempt:
        messages: list[ChatMessage] = []
        if prompt.system:
            messages.append(ChatMessage(role="system", content=prompt.system))
        messages.append(ChatMessage(role="user", content=prompt.text))
        try:
            result = client.chat(messages)
        except LlmkitError as exc:
            logger.warning(
                "ケースを記録して次へ進みます: prompt_id=%s error=%s",
                prompt.id,
                type(exc).__name__,
            )
            return _Attempt(
                response=ResponseFacts.missing(),
                speed=SpeedMetrics.missing(),
                error=RecordError.from_exception(exc),
            )
        return _Attempt(
            response=ResponseFacts.from_chat_result(result),
            speed=SpeedMetrics.from_chat_result(result),
            error=None,
        )

    def _skip_case(self, case_plan: CasePlan, exc: LlmkitError) -> None:
        """起動できなかったモデルを、プロンプト数分の失敗レコードとして残す。

        件数をそろえるのは、比較表の ``n/N`` の N (試行数) をモデル間で
        比較可能に保つため。空欄にすると「速かったから少ない」のか
        「そもそも走っていない」のかが表から読めなくなる。
        """
        logger.warning(
            "モデルを起動できなかったため全プロンプトをスキップします: "
            "model_id=%s error=%s",
            case_plan.model_id,
            type(exc).__name__,
        )
        attempt = _Attempt(
            response=ResponseFacts.missing(),
            speed=SpeedMetrics.missing(),
            error=RecordError.from_exception(exc),
        )
        for prompt in self._plan.suite.prompts:
            self._append(case_plan, "measure", prompt, attempt, VramReading.missing())

    def _append(
        self,
        case_plan: CasePlan,
        phase: RunPhase,
        prompt: PromptSpec,
        attempt: _Attempt,
        vram: VramReading,
    ) -> None:
        generation = case_plan.config.generation
        speed = attempt.speed
        response = attempt.response
        self._records.append(
            RunRecord(
                schema_version=RECORD_SCHEMA_VERSION,
                run_id=self._run_id,
                run_fingerprint=self._plan.fingerprint,
                sequence_index=len(self._records),
                case_index=case_plan.case_index,
                model_id=case_plan.model_id,
                served_name=case_plan.spec.served_name,
                quantization=case_plan.spec.quantization,
                serving_runtime=case_plan.spec.serving_runtime,
                profile_name=case_plan.profile_name,
                context_tokens=generation.context_tokens,
                temperature=generation.temperature,
                top_p=generation.top_p,
                max_output_tokens=generation.max_output_tokens,
                seed=generation.seed,
                phase=phase,
                prompt_id=prompt.id,
                prompt_sha256=_sha256_text(prompt.text),
                response_text=response.response_text,
                finish_reason=response.finish_reason,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                latency_s=speed.latency_s,
                eval_tokens_per_second=speed.eval_tokens_per_second,
                prompt_tokens_per_second=speed.prompt_tokens_per_second,
                wallclock_tokens_per_second=speed.wallclock_tokens_per_second,
                generation_tokens_per_second=speed.generation_tokens_per_second,
                generation_tokens_per_second_source=(
                    speed.generation_tokens_per_second_source
                ),
                vram_estimate_gib=case_plan.estimate.total_gib,
                vram_budget_gib=case_plan.estimate.budget_gib,
                vram_used_mib=vram.used_mib,
                vram_total_mib=vram.total_mib,
                vram_increment_gib=vram.increment_gib,
                error=attempt.error,
            )
        )


def _execute(
    plan: RunPlan,
    *,
    run_id: str,
    probe: VramProbe,
    idle: GpuMemory | None,
    http_client: httpx.Client | None,
    results_dir: Path,
) -> _SuiteRunner:
    """``httpx.Client`` の所有権だけを扱う薄いラッパ。

    注入されたクライアントは閉じない (呼び出し側が所有している)。自前で
    生成したものだけを閉じる。``llmkit._HttpChatClient`` と同じ規則。
    """
    if http_client is not None:
        runner = _SuiteRunner(
            plan,
            run_id=run_id,
            probe=probe,
            idle=idle,
            http_client=http_client,
            results_dir=results_dir,
        )
        runner.run()
        return runner
    with httpx.Client(timeout=plan.base_config.runtime.timeout_s) as owned:
        runner = _SuiteRunner(
            plan,
            run_id=run_id,
            probe=probe,
            idle=idle,
            http_client=owned,
            results_dir=results_dir,
        )
        runner.run()
    return runner


def run_suite(
    plan: RunPlan,
    *,
    probe: VramProbe,
    http_client: httpx.Client | None = None,
    results_dir: Path,
    clock: Clock | None = None,
    run_id: str | None = None,
) -> RunResult:
    """計画を実行してレコードと集計を返す。

    実行順序は model-major (1 モデルにつき全プロンプトを連続実行)。各モデルは
    ``bootstrap_from_config`` → ウォームアップ → VRAM プローブ → 全プロンプト
    の計測、の順に進む。ウォームアップの結果は ``phase="warmup"`` として残し、
    集計からのみ除外する (D-22)。

    ケース単位の失敗 (:class:`llmkit.LlmkitError`) は記録して次のモデルへ進む。
    VRAM 予算超過は起動時点で停止するため **HTTP は 1 バイトも発行されない**
    (D-04)。

    Args:
        plan: :func:`plan_run` が作った実行計画。
        probe: VRAM 実測プローブ。取得できない環境では ``None`` を返す (D-23)。
        http_client: 注入する ``httpx.Client``。テストは ``MockTransport`` を
            渡す。省略時はここで生成し、終了時に閉じる (注入されたものは
            閉じない = ``llmkit`` 側と同じ所有権の規則)。
        results_dir: 出力先ディレクトリ。実行マニフェストは
            ``<results_dir>/manifests/`` に書き出す。
        clock: 現在時刻を返す関数 (テスト用)。省略時は UTC の現在時刻。
        run_id: この実行の識別子。省略時は uuid から 12 桁を生成する。
    """
    now = clock if clock is not None else _utc_now
    identifier = run_id if run_id is not None else uuid.uuid4().hex[:12]
    started_at = _isoformat_utc(now())
    logger.info(
        "比較実行を開始します: suite=%s models=%d prompts=%d fingerprint=%s",
        plan.suite_id,
        len(plan.cases),
        len(plan.suite.prompts),
        plan.fingerprint,
    )

    # アイドル基準は実行全体で 1 回だけ測る (仕様書 §4 T3)。モデルごとに
    # 測り直すと、2 モデル目以降の増分が「前のモデルがロード済みの状態
    # からの差」になり、実測値が意味を成さなくなる。
    idle = probe.read()

    runner = _execute(
        plan,
        run_id=identifier,
        probe=probe,
        idle=idle,
        http_client=http_client,
        results_dir=results_dir,
    )

    records = runner.records
    summary = summarize_records(records)
    logger.info(
        "比較実行が終了しました: records=%d (warmup=%d measure=%d error=%d)",
        len(records),
        summary.warmup_record_count,
        summary.measure_record_count,
        summary.error_record_count,
    )
    return RunResult(
        plan=plan,
        run_id=identifier,
        started_at_utc=started_at,
        records=records,
        manifests=runner.manifests,
        manifest_paths=runner.manifest_paths,
        summary=summary,
        vram_idle_mib=None if idle is None else idle.used_mib,
        results_dir=results_dir,
    )
