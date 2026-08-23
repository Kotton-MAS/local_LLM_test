"""比較実行の 1 レコードと、その集計 (中央値 / n/N)。

**1 モデルに 1 プロンプトを 1 回投げた生成 = 1 レコード**であり、
ウォームアップも含めてすべて記録する。集計から除外するのは
``phase == "measure"`` 以外と失敗レコードだけで、捨てはしない (D-22)。
捨てるとコールド実行とウォーム実行の差そのものが観測できなくなる。

★ このモジュールの中核は **欠測を ``None`` のまま運ぶこと**である (D-21)。
:attr:`llmkit.ChatResult.tokens_per_second` は「計測不能」を 0.0 で表す仕様
なので、**そのまま記録してはならない**。0 を書くと「実測 0 トークン/秒」と
「未計測」が区別できなくなり、中央値がゼロ方向へ静かに歪む。速度は 3 値
(eval / prompt / 壁時計) を別カラムで保持し、代表値には出典ラベル
(``"eval"`` / ``"wallclock"``) を必ず添える。

スキーマ定義は llmkit の :mod:`llmkit.manifest` と同じく **素の frozen
dataclass + ``to_dict()``** にする。ここは外部入力を検証する層ではなく
「自分で組み立てて書き出す」層であり、``pydantic.BaseModel`` は使わない (D-08)。
"""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from harness.gpu import GpuMemory
from llmkit import ChatResult

__all__ = [
    "MIXED_SPEED_SOURCE",
    "RECORD_KEYS",
    "RECORD_SCHEMA_VERSION",
    "MetricSummary",
    "ModelSummary",
    "RecordError",
    "ResponseFacts",
    "RunPhase",
    "RunRecord",
    "RunSummary",
    "SpeedMetrics",
    "SpeedSource",
    "VramReading",
    "summarize_records",
    "wallclock_tokens_per_second",
]

#: レコードスキーマの版。フィールドを足し引きしたら上げる。
#: ``run_fingerprint`` の入力にも含める (スキーマが変われば再現条件も変わる)。
RECORD_SCHEMA_VERSION = "1"

#: 実行区分。``warmup`` も記録するが集計には入れない (D-22)。
RunPhase = Literal["warmup", "measure"]

#: 生成速度の代表値がどの計測に由来するか (D-21)。
#: ``eval`` は Ollama 内部の生成時間のみ、``wallclock`` は HTTP 往復全体。
SpeedSource = Literal["eval", "wallclock"]

#: 1 モデルの中で出典が混在した場合に集計が示すラベル。
#: 中央値を出す元の値の分母が途中で変わったことを、表の上で隠さないため。
MIXED_SPEED_SOURCE = "mixed"

_MIB_PER_GIB = 1024.0


@dataclass(frozen=True, slots=True)
class RecordError:
    """ケース単位の失敗。例外の型名と説明文だけを残す。

    スタックトレースは残さない (CLAUDE.md: エラーレスポンスに内部実装の詳細を
    含めない)。``message`` は llmkit 例外の ``str()`` であり、api_key の値や
    応答本文全文を含まないことが llmkit 側で保証されている。
    """

    type: str
    message: str

    @classmethod
    def from_exception(cls, exc: Exception) -> RecordError:
        return cls(type=type(exc).__name__, message=str(exc))

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "message": self.message}


@dataclass(frozen=True, slots=True)
class ResponseFacts:
    """応答そのものから読める値。失敗したケースはすべて ``None``。"""

    response_text: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    @classmethod
    def from_chat_result(cls, result: ChatResult) -> ResponseFacts:
        return cls(
            response_text=result.text,
            finish_reason=result.finish_reason,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        )

    @classmethod
    def missing(cls) -> ResponseFacts:
        """応答が得られなかった (失敗・スキップ) 場合の全欠測。"""
        return cls(
            response_text=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )


def wallclock_tokens_per_second(result: ChatResult) -> float | None:
    """壁時計ベースの生成速度。計測不能なら ``None`` (0.0 ではない)。

    :attr:`llmkit.ChatResult.tokens_per_second` は ``latency_s <= 0`` や
    ``completion_tokens == 0`` のときに 0.0 を返す。その 0.0 は「この値からは
    分からない」を表す欠測のシグナルであって実測値ではないため、比較記録に
    そのまま書かない (D-21)。ここで同じ条件を ``None`` に翻訳する。
    """
    if result.latency_s > 0.0 and result.usage.completion_tokens > 0:
        return result.usage.completion_tokens / result.latency_s
    return None


@dataclass(frozen=True, slots=True)
class SpeedMetrics:
    """速度指標 3 値 + 代表値 + 出典 (D-21)。欠測はすべて ``None``。

    Attributes:
        latency_s: HTTP 往復全体の秒数。
        eval_tokens_per_second: Ollama 内部の生成時間のみを分母にした値。
        prompt_tokens_per_second: プロンプト処理速度。
        wallclock_tokens_per_second: 壁時計ベース (モデルロード・キュー待ち込み)。
        generation_tokens_per_second: 代表値
            (:attr:`llmkit.ChatResult.measured_tokens_per_second`)。
        generation_tokens_per_second_source: 代表値がどれに由来するか。
    """

    latency_s: float | None
    eval_tokens_per_second: float | None
    prompt_tokens_per_second: float | None
    wallclock_tokens_per_second: float | None
    generation_tokens_per_second: float | None
    generation_tokens_per_second_source: SpeedSource | None

    @classmethod
    def from_chat_result(cls, result: ChatResult) -> SpeedMetrics:
        """1 応答から速度指標を取り出す。

        代表値は llmkit の :attr:`ChatResult.measured_tokens_per_second`
        (eval 優先、無ければ壁時計、どちらも無ければ ``None``) をそのまま使う。
        出典ラベルはその優先順位を写したもので、値とラベルが食い違わないよう
        同じ条件式で決める。
        """
        timings = result.timings
        eval_tps = timings.eval_tokens_per_second if timings is not None else None
        prompt_tps = timings.prompt_tokens_per_second if timings is not None else None
        wallclock = wallclock_tokens_per_second(result)
        source: SpeedSource | None
        if eval_tps is not None:
            source = "eval"
        elif wallclock is not None:
            source = "wallclock"
        else:
            source = None
        return cls(
            latency_s=result.latency_s,
            eval_tokens_per_second=eval_tps,
            prompt_tokens_per_second=prompt_tps,
            wallclock_tokens_per_second=wallclock,
            generation_tokens_per_second=result.measured_tokens_per_second,
            generation_tokens_per_second_source=source,
        )

    @classmethod
    def missing(cls) -> SpeedMetrics:
        """応答が得られなかった場合の全欠測。"""
        return cls(
            latency_s=None,
            eval_tokens_per_second=None,
            prompt_tokens_per_second=None,
            wallclock_tokens_per_second=None,
            generation_tokens_per_second=None,
            generation_tokens_per_second_source=None,
        )


@dataclass(frozen=True, slots=True)
class VramReading:
    """VRAM の実測値。プローブが使えない環境ではすべて ``None`` (D-23)。

    ``increment_gib`` は「そのモデルを載せたことによる増分」であり、
    ``idle`` (実行全体で 1 回だけ測ったアイドル基準。``harness.runner`` が
    全モデルで共有する) と計測直後の差で求める。負値でも丸めない。負値は
    「アイドル基準より使用量が減った」実態を表しており、0 に潰すと消えてしまう。
    """

    used_mib: int | None
    total_mib: int | None
    increment_gib: float | None

    @classmethod
    def from_probe(
        cls, *, idle: GpuMemory | None, current: GpuMemory | None
    ) -> VramReading:
        if current is None:
            return cls.missing()
        increment = (
            None if idle is None else (current.used_mib - idle.used_mib) / _MIB_PER_GIB
        )
        return cls(
            used_mib=current.used_mib,
            total_mib=current.total_mib,
            increment_gib=increment,
        )

    @classmethod
    def missing(cls) -> VramReading:
        return cls(used_mib=None, total_mib=None, increment_gib=None)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """1 モデルに 1 プロンプトを 1 回投げた生成の記録 (JSONL の 1 行に対応)。

    フィールドは仕様書 §4 T3 の表と 1 対 1 で、同一性 / 条件 / 区分 / 応答 /
    速度 / VRAM / 失敗 の順に並べる。:data:`RECORD_KEYS` がこの並びの真実で
    あり、``records.jsonl`` の各行のキー集合と一致する。
    """

    # 同一性
    schema_version: str
    run_id: str
    run_fingerprint: str
    sequence_index: int
    case_index: int
    # 条件
    model_id: str
    served_name: str
    quantization: str
    serving_runtime: str
    profile_name: str
    context_tokens: int
    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int
    # 区分
    phase: RunPhase
    prompt_id: str
    prompt_sha256: str
    # 応答
    response_text: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    # 速度
    latency_s: float | None
    eval_tokens_per_second: float | None
    prompt_tokens_per_second: float | None
    wallclock_tokens_per_second: float | None
    generation_tokens_per_second: float | None
    generation_tokens_per_second_source: SpeedSource | None
    # VRAM
    vram_estimate_gib: float
    vram_budget_gib: float
    vram_used_mib: int | None
    vram_total_mib: int | None
    vram_increment_gib: float | None
    # 失敗
    error: RecordError | None

    def to_dict(self) -> dict[str, object]:
        """JSONL 1 行分の素の辞書。キー集合は :data:`RECORD_KEYS` と一致する。"""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_fingerprint": self.run_fingerprint,
            "sequence_index": self.sequence_index,
            "case_index": self.case_index,
            "model_id": self.model_id,
            "served_name": self.served_name,
            "quantization": self.quantization,
            "serving_runtime": self.serving_runtime,
            "profile_name": self.profile_name,
            "context_tokens": self.context_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "seed": self.seed,
            "phase": self.phase,
            "prompt_id": self.prompt_id,
            "prompt_sha256": self.prompt_sha256,
            "response_text": self.response_text,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": self.latency_s,
            "eval_tokens_per_second": self.eval_tokens_per_second,
            "prompt_tokens_per_second": self.prompt_tokens_per_second,
            "wallclock_tokens_per_second": self.wallclock_tokens_per_second,
            "generation_tokens_per_second": self.generation_tokens_per_second,
            "generation_tokens_per_second_source": (
                self.generation_tokens_per_second_source
            ),
            "vram_estimate_gib": self.vram_estimate_gib,
            "vram_budget_gib": self.vram_budget_gib,
            "vram_used_mib": self.vram_used_mib,
            "vram_total_mib": self.vram_total_mib,
            "vram_increment_gib": self.vram_increment_gib,
            "error": None if self.error is None else self.error.to_dict(),
        }

    @property
    def is_aggregatable(self) -> bool:
        """集計対象か。計測フェーズかつ成功したレコードだけを数える (D-22)。"""
        return self.phase == "measure" and self.error is None


#: ``records.jsonl`` の 1 行が持つキーの並び (= :class:`RunRecord` の宣言順)。
RECORD_KEYS: tuple[str, ...] = tuple(
    field.name for field in dataclasses.fields(RunRecord)
)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """1 指標の集計。**``None`` は分母にも分子にも混ぜない** (D-21)。

    Attributes:
        median: 非欠測値の中央値。1 件も無ければ ``None`` (0.0 ではない)。
        n: 中央値に寄与した件数。
        total: 集計対象になったレコード数 (成功した計測レコードの数)。
    """

    median: float | None
    n: int
    total: int

    @property
    def coverage(self) -> str:
        """レポートの「測定数 n/N」に出す文字列。"""
        return f"{self.n}/{self.total}"


def _summarize_metric(values: Sequence[float | None], total: int) -> MetricSummary:
    present = [value for value in values if value is not None]
    median = statistics.median(present) if present else None
    return MetricSummary(median=median, n=len(present), total=total)


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """比較表 1 行分 (= スイートの ``[[models]]`` 1 件) の集計。

    Attributes:
        attempted: 計測フェーズのレコード数 (失敗・スキップを含む N)。
        succeeded: そのうち成功した数。
    """

    case_index: int
    model_id: str
    served_name: str
    quantization: str
    context_tokens: int
    generation_tokens_per_second: MetricSummary
    generation_tokens_per_second_source: str | None
    prompt_tokens_per_second: MetricSummary
    wallclock_tokens_per_second: MetricSummary
    vram_estimate_gib: float
    vram_used_mib: int | None
    vram_increment_gib: float | None
    attempted: int
    succeeded: int


@dataclass(frozen=True, slots=True)
class RunSummary:
    """比較実行 1 回分の集計。"""

    models: tuple[ModelSummary, ...]
    warmup_record_count: int
    measure_record_count: int
    error_record_count: int


def _speed_source(records: Sequence[RunRecord]) -> str | None:
    """代表値に寄与したレコードの出典ラベル。混在は隠さず ``"mixed"``。"""
    sources = {
        record.generation_tokens_per_second_source
        for record in records
        if record.generation_tokens_per_second is not None
    }
    sources.discard(None)
    if not sources:
        return None
    if len(sources) == 1:
        return next(iter(sources))
    return MIXED_SPEED_SOURCE


def _first_not_none[T](values: Sequence[T | None]) -> T | None:
    for value in values:
        if value is not None:
            return value
    return None


def _summarize_case(case_records: Sequence[RunRecord]) -> ModelSummary:
    """1 ケース分のレコード (warmup を含む) を比較表 1 行へ畳む。"""
    measured = [record for record in case_records if record.phase == "measure"]
    aggregated = [record for record in measured if record.is_aggregatable]
    total = len(aggregated)
    head = case_records[0]
    return ModelSummary(
        case_index=head.case_index,
        model_id=head.model_id,
        served_name=head.served_name,
        quantization=head.quantization,
        context_tokens=head.context_tokens,
        generation_tokens_per_second=_summarize_metric(
            [record.generation_tokens_per_second for record in aggregated], total
        ),
        generation_tokens_per_second_source=_speed_source(aggregated),
        prompt_tokens_per_second=_summarize_metric(
            [record.prompt_tokens_per_second for record in aggregated], total
        ),
        wallclock_tokens_per_second=_summarize_metric(
            [record.wallclock_tokens_per_second for record in aggregated], total
        ),
        vram_estimate_gib=head.vram_estimate_gib,
        vram_used_mib=_first_not_none([record.vram_used_mib for record in measured]),
        vram_increment_gib=_first_not_none(
            [record.vram_increment_gib for record in measured]
        ),
        attempted=len(measured),
        succeeded=total,
    )


def summarize_records(records: Sequence[RunRecord]) -> RunSummary:
    """レコード列を比較表の行へ集計する。

    集計に入れるのは ``phase == "measure"`` かつ ``error is None`` のレコード
    だけで、ウォームアップは**除外するが捨てない** (D-22)。統計は中央値を使う
    (平均はコールド実行の外れ値に引きずられる)。欠測は分母にも分子にも
    混ぜず、件数を ``n/N`` で併記する (D-21)。

    Args:
        records: 1 実行分のレコード (warmup 含む)。並び順は保持する。

    Returns:
        ケース (``[[models]]`` の宣言順) ごとの集計。
    """
    grouped: dict[int, list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.case_index, []).append(record)
    return RunSummary(
        models=tuple(
            _summarize_case(case_records) for _, case_records in sorted(grouped.items())
        ),
        warmup_record_count=sum(1 for record in records if record.phase == "warmup"),
        measure_record_count=sum(1 for record in records if record.phase == "measure"),
        error_record_count=sum(1 for record in records if record.error is not None),
    )
