"""Evaluate U/S/L/F/T/R without changing physical constraints during tuning.

Inputs are observations from a constraint checker, a rollout engine and a robot
verifier. This module never certifies geometry or robot feasibility itself.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

FEATURES = ("U", "S", "L", "F", "T", "R")
SIGNS = {"U": 1, "S": 1, "L": 1, "F": 1, "T": -1, "R": -1}


def _finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _unit(name: str, value: float) -> None:
    _finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class ScoreWeights:
    # Implementation proposal, not an official judging rubric or meeting agreement.
    U: float = 0.20
    S: float = 0.30
    L: float = 0.15
    F: float = 0.20
    T: float = 0.10
    R: float = 0.05

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _unit(name, value)
        if not math.isclose(sum(asdict(self).values()), 1.0, abs_tol=1e-9):
            raise ValueError("six score weights must sum to 1")


@dataclass(frozen=True)
class ScoreConfig:
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    support_threshold: float = 0.75
    support_cog_min_mm: float = 10.0
    support_cog_target_mm: float = 30.0  # desired EXTRA margin above the hard minimum
    time_reference_s: float = 8.0
    tail_alpha: float = 0.20
    risk_tail_mix: float = 0.40

    def __post_init__(self) -> None:
        _unit("support_threshold", self.support_threshold)
        if self.support_threshold >= 1:
            raise ValueError("support_threshold must be below 1 for residual-margin scoring")
        for name in ("support_cog_min_mm", "support_cog_target_mm", "time_reference_s"):
            _finite(name, getattr(self, name))
        if self.support_cog_min_mm < 0 or self.support_cog_target_mm <= 0 or self.time_reference_s <= 0:
            raise ValueError("normalization distances/times must be positive (minimum margin may be zero)")
        _unit("tail_alpha", self.tail_alpha)
        _unit("risk_tail_mix", self.risk_tail_mix)
        if self.tail_alpha == 0:
            raise ValueError("tail_alpha must be positive")

    @classmethod
    def load(cls, path: str | Path) -> "ScoreConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["weights"] = ScoreWeights(**data["weights"])
        return cls(**data)


@dataclass(frozen=True)
class ImmediateMetrics:
    compactness: float  # placed volume after candidate / (pallet footprint * used height)
    support_ratio: float
    support_cog_margin_mm: float  # smallest signed distance to support-polygon edges
    min_load_slack_ratio: float  # min remaining top-load capacity / capacity after propagation
    pallet_cog_offset_mm: float
    pallet_cog_limit_mm: float  # the SAME hard limit for every candidate of this decision
    cycle_time_s: float  # full cycle, including shelf movements when the action requires them

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite(name, value)
        for name in ("compactness", "support_ratio", "min_load_slack_ratio"):
            _unit(name, getattr(self, name))
        if self.pallet_cog_offset_mm < 0 or self.pallet_cog_limit_mm <= 0 or self.cycle_time_s < 0:
            raise ValueError("invalid COG limit/offset or cycle time")


@dataclass(frozen=True)
class RobotVerification:
    status: str = "UNKNOWN"  # UNKNOWN, PASS, FAIL
    cycle_time_s: float | None = None  # path timing + gripper timing, required for PASS
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("UNKNOWN", "PASS", "FAIL"):
            raise ValueError("robot status must be UNKNOWN/PASS/FAIL")
        if self.cycle_time_s is not None:
            _finite("verified cycle_time_s", self.cycle_time_s)
            if self.cycle_time_s < 0:
                raise ValueError("cycle time cannot be negative")
        if self.status == "PASS" and self.cycle_time_s is None:
            raise ValueError("PASS needs a path-based full-cycle time")


@dataclass(frozen=True)
class CandidateInput:
    candidate_id: str
    metrics: ImmediateMetrics
    physics_passed: bool = False
    physics_reason: str = ""
    robot: RobotVerification = field(default_factory=RobotVerification)

    def __post_init__(self) -> None:
        if not self.candidate_id or not isinstance(self.candidate_id, str):
            raise ValueError("candidate_id must be a nonempty string")
        if not isinstance(self.physics_passed, bool):
            raise ValueError("physics_passed must be an explicit boolean")


@dataclass(frozen=True)
class RolloutOutcome:
    scenario_id: str
    offered_future_volume: float  # excludes the current candidate and already placed volume
    placed_future_volume: float
    blocked: bool  # at least one placement failure in this finite horizon

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty")
        for name in ("offered_future_volume", "placed_future_volume"):
            _finite(name, getattr(self, name))
        if not 0 <= self.placed_future_volume <= self.offered_future_volume:
            raise ValueError("placed future volume must be between zero and offered future volume")
        if not isinstance(self.blocked, bool):
            raise ValueError("blocked must be a boolean")
        if self.offered_future_volume == 0 and self.blocked:
            raise ValueError("an empty future cannot have a blocked box")


@dataclass(frozen=True)
class FutureSummary:
    mean: float
    cvar: float
    block_rate: float
    risk: float
    scenario_count: int


@dataclass(frozen=True)
class CandidateScore:
    candidate: CandidateInput
    features: dict[str, float]
    contributions: dict[str, float]
    total: float
    future: FutureSummary | None = None
    rejection_reason: str = ""

    @property
    def eligible(self) -> bool:
        return self.candidate.physics_passed and not self.rejection_reason and self.candidate.robot.status != "FAIL"

    @property
    def executable(self) -> bool:
        return self.eligible and self.candidate.robot.status == "PASS"

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(eligible=self.eligible, executable=self.executable)
        return result


@dataclass(frozen=True)
class Shortlist:
    selected: tuple[CandidateScore, ...]
    deferred: tuple[CandidateScore, ...]  # retain other candidates for fallback
    ranking_source: str
    fallback_reason: str = ""


def _score(candidate: CandidateInput, features: dict[str, float], cfg: ScoreConfig,
           future: FutureSummary | None = None, reason: str = "") -> CandidateScore:
    contributions = {k: SIGNS[k] * getattr(cfg.weights, k) * features[k] for k in FEATURES}
    return CandidateScore(candidate, features, contributions, sum(contributions.values()), future, reason)


def evaluate_immediate(candidate: CandidateInput, cfg: ScoreConfig = ScoreConfig()) -> CandidateScore:
    m = candidate.metrics
    reason = candidate.physics_reason if not candidate.physics_passed else ""
    if not candidate.physics_passed:
        reason = reason or "physics_not_verified"
    # Defense against inconsistent upstream status, in addition to its full constraint checker.
    if m.support_ratio < cfg.support_threshold - 1e-9:
        reason = "support_below_threshold"
    elif m.support_cog_margin_mm < cfg.support_cog_min_mm - 1e-9:
        reason = "support_cog_below_minimum"
    elif m.pallet_cog_offset_mm > m.pallet_cog_limit_mm + 1e-9:
        reason = "pallet_cog_outside_limit"
    support_slack = _clip((m.support_ratio - cfg.support_threshold) / (1 - cfg.support_threshold))
    cog_slack = _clip((m.support_cog_margin_mm - cfg.support_cog_min_mm) / cfg.support_cog_target_mm)
    t = candidate.robot.cycle_time_s if candidate.robot.status == "PASS" else m.cycle_time_s
    f = {
        "U": m.compactness,
        "S": 0.6 * support_slack + 0.4 * cog_slack,
        "L": 0.6 * m.min_load_slack_ratio + 0.4 * (1 - _clip(m.pallet_cog_offset_mm / m.pallet_cog_limit_mm)),
        "F": 0.0, "T": t / (t + cfg.time_reference_s), "R": 0.0,
    }
    return _score(candidate, f, cfg, reason=reason)


def _ordered(scores: Sequence[CandidateScore]) -> list[CandidateScore]:
    # Deterministic tie break; the candidate id must be stable across logging and replays.
    return sorted(scores, key=lambda s: (-s.total, s.candidate.candidate_id))


def _unique(scores: Sequence[CandidateScore]) -> None:
    ids = [s.candidate.candidate_id for s in scores]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate ids")


def preselect(scores: Sequence[CandidateScore], k: int,
              predicted_future: Mapping[str, float] | None = None,
              heuristic_reserve: int = 2, cfg: ScoreConfig = ScoreConfig()) -> Shortlist:
    """AI Future Value/Ranking -> top candidates; heuristic fallback is always available.

    Future predictions must be unpenalized expected F in [0,1], not V_future that
    already subtracts a blocking penalty. They are used only for screening.
    """
    _unique(scores)
    if isinstance(k, bool) or k < 1 or heuristic_reserve < 0:
        raise ValueError("k must be positive and reserve must be nonnegative")
    eligible = _ordered([s for s in scores if s.eligible])
    fallback = ""
    predicted = predicted_future
    if predicted is not None:
        try:
            for s in eligible:
                _unit("predicted_future", predicted[s.candidate.candidate_id])
        except (KeyError, ValueError, TypeError):
            predicted = None
            fallback = "missing_or_invalid_prediction"
    if predicted is None:
        return Shortlist(tuple(eligible[:k]), tuple(eligible[k:]), "fixed_score", fallback)
    by_model = sorted(eligible, key=lambda s: (-(s.total + cfg.weights.F * predicted[s.candidate.candidate_id]), s.candidate.candidate_id))
    reserve = min(heuristic_reserve, max(0, k - 1), len(eligible))
    chosen = list(by_model[: max(0, k - reserve)])
    chosen_ids = {s.candidate.candidate_id for s in chosen}
    for s in eligible[:reserve] + by_model + eligible:
        if len(chosen) >= min(k, len(eligible)):
            break
        if s.candidate.candidate_id not in chosen_ids:
            chosen.append(s)
            chosen_ids.add(s.candidate.candidate_id)
    return Shortlist(tuple(chosen), tuple(s for s in eligible if s.candidate.candidate_id not in chosen_ids), "ai_with_heuristic_reserve")


def lower_tail_mean(values: Sequence[float], alpha: float) -> float:
    """Empirical lower-tail CVaR with fractional mass at the quantile boundary."""
    _unit("alpha", alpha)
    if not values or alpha <= 0:
        raise ValueError("CVaR needs values and a positive tail fraction")
    for value in values:
        _unit("CVaR value", value)
    ordered = sorted(values)
    mass = alpha * len(ordered)
    whole = int(math.floor(mass))
    fraction = mass - whole
    total = sum(ordered[:whole])
    if fraction > 1e-12 and whole < len(ordered):
        total += fraction * ordered[whole]
    return total / mass


def summarize_future(outcomes: Sequence[RolloutOutcome], cfg: ScoreConfig) -> FutureSummary:
    if not outcomes:
        raise ValueError("future evaluation needs at least one shared scenario")
    ids = [o.scenario_id for o in outcomes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ids")
    fill = [o.placed_future_volume / o.offered_future_volume if o.offered_future_volume else 0.0 for o in outcomes]
    mean = sum(fill) / len(fill)
    cvar = lower_tail_mean(fill, cfg.tail_alpha)
    blocked = sum(o.blocked for o in outcomes) / len(outcomes)
    risk = (1 - cfg.risk_tail_mix) * blocked + cfg.risk_tail_mix * max(0.0, mean - cvar)
    return FutureSummary(mean, cvar, blocked, _clip(risk), len(outcomes))


def rank_final(scores: Sequence[CandidateScore], rollouts: Mapping[str, Sequence[RolloutOutcome]],
               cfg: ScoreConfig = ScoreConfig()) -> list[CandidateScore]:
    """Detailed rollout re-ranking; compare exactly the same scenarios/horizons."""
    _unique(scores)
    valid = [s for s in scores if s.eligible]
    signatures = []
    for s in valid:
        outcomes = rollouts[s.candidate.candidate_id]
        summary = summarize_future(outcomes, cfg)  # validates duplicates and emptiness
        signatures.append({o.scenario_id: o.offered_future_volume for o in outcomes})
    if signatures and any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("candidates need identical scenario ids AND offered volumes (common horizon)")
    result = []
    for s in valid:
        future = summarize_future(rollouts[s.candidate.candidate_id], cfg)
        f = dict(s.features, F=future.mean, R=future.risk)
        result.append(_score(s.candidate, f, cfg, future))
    return _ordered(result)


def rerank_after_robot(scores: Sequence[CandidateScore], verifications: Mapping[str, RobotVerification],
                       cfg: ScoreConfig = ScoreConfig()) -> list[CandidateScore]:
    """Update path times and robot feasibility, then recompute the common score."""
    _unique(scores)
    result = []
    for old in scores:
        candidate = replace(old.candidate, robot=verifications.get(old.candidate.candidate_id, old.candidate.robot))
        immediate = evaluate_immediate(candidate, cfg)
        f = dict(old.features, T=immediate.features["T"])
        result.append(_score(candidate, f, cfg, old.future, immediate.rejection_reason))
    return _ordered(result)


def select_executable(scores: Sequence[CandidateScore]) -> CandidateScore | None:
    """Return the highest verified score, or None (HOLD); never execute UNKNOWN."""
    return next((s for s in _ordered(scores) if s.executable), None)


def write_decision_log(path: str | Path, scores: Sequence[CandidateScore], metadata: Mapping | None = None) -> None:
    """JSONL stores features and score parts for supervised labels and tuning."""
    record = {"metadata": dict(metadata or {}), "candidates": [s.to_dict() for s in scores]}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
