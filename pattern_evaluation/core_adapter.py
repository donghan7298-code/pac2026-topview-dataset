"""Optional adapter for yang8988/pac-mission1's ROS-independent geometry core.

Install/add src/palletizing_core from the reference repo to PYTHONPATH first.
No upstream files are copied into this deliverable. No torch/PPO import occurs.
The upstream core currently has no physical shelf state or robot path verifier.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

from .evaluator import (
    CandidateInput, CandidateScore, ImmediateMetrics, RobotVerification,
    RolloutOutcome, ScoreConfig, evaluate_immediate, preselect, rank_final,
    rerank_after_robot, select_executable,
)


def _polygon_margin(points: Sequence[tuple[float, float]], center: tuple[float, float]) -> float:
    from palletizing_core.constraints import convex_hull

    hull = convex_hull(points)
    if len(hull) < 3:
        raise ValueError("supported candidate needs a nondegenerate support polygon")
    distances = []
    for a, b in zip(hull, hull[1:] + hull[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        distances.append((dx * (center[1] - a[1]) - dy * (center[0] - a[0])) / math.hypot(dx, dy))
    return min(distances)


def estimate_full_cycle(state, cand) -> float:
    """Rough time from above pickup to above the next pickup, all phases included.

    grasp/release use fixed_time. Linear approach, lift, lower, retreat use
    approach_clearance; round-trip transport and wrist rotation are added.
    Replace this estimate with MoveIt trajectory timing before execution.
    """
    rob, pal = state.cfg.robot, state.cfg.pallet
    if rob.lin_speed <= 0 or rob.fixed_time < 0 or rob.rot_time < 0 or rob.approach_clearance < 0:
        raise ValueError("invalid robot timing configuration")
    cx, cy, _ = cand.center(pal.grid)
    px, py, pz = rob.pick
    destination_z = cand.z + cand.h
    transport = math.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (destination_z - pz) ** 2)
    rotation = 2 * rob.rot_time if cand.o else 0.0
    return rob.fixed_time + (2 * transport + 4 * rob.approach_clearance) / rob.lin_speed + rotation


def candidate_input(state, cand) -> CandidateInput:
    """Only call on an output of the core's full Tier-1 checker."""
    from palletizing_core.constraints import cog_radius

    pal = state.cfg.pallet
    cx, cy, _ = cand.center(pal.grid)
    if cand.z <= 0:
        margin = min(cand.w, cand.d) / 2
    else:
        points = []
        x0, y0 = cand.i * pal.grid, cand.j * pal.grid
        for index, _ in cand.supporters:
            pb = state.placed[index]
            xa, xb = max(x0, pb.x), min(x0 + cand.w, pb.x + pb.w)
            ya, yb = max(y0, pb.y), min(y0 + cand.d, pb.y + pb.d)
            if xb > xa and yb > ya:
                points.extend(((xa, ya), (xb, ya), (xa, yb), (xb, yb)))
        margin = _polygon_margin(points, (cx, cy))
    slack = []
    for index, pb in enumerate(state.placed):
        after = pb.load + cand.load_delta.get(index, 0.0)
        if pb.box.max_load <= 0:
            if after > 1e-8:
                raise ValueError("overloaded zero-capacity box passed upstream checker")
            slack.append(0.0)
        else:
            ratio = (pb.box.max_load - after) / pb.box.max_load
            if ratio < -1e-8:
                raise ValueError("overloaded box passed upstream checker")
            slack.append(max(0.0, min(1.0, ratio)))
    slack.append(1.0 if cand.box.max_load > 0 else 0.0)
    used_volume = sum(pb.box.volume for pb in state.placed) + cand.box.volume
    used_height = max(float(state.H.max()), cand.z + cand.h)
    compactness = used_volume / (pal.length * pal.width * used_height)
    if compactness > 1 + 1e-8:
        raise ValueError("invalid compactness: overlapping geometry or inconsistent dimensions")
    metrics = ImmediateMetrics(
        compactness=max(0.0, min(1.0, compactness)),
        support_ratio=float(cand.support_ratio), support_cog_margin_mm=float(margin),
        min_load_slack_ratio=min(slack),
        pallet_cog_offset_mm=math.hypot(cand.cog_after[0] - pal.length / 2, cand.cog_after[1] - pal.width / 2),
        pallet_cog_limit_mm=float(cog_radius(state, 1)),
        cycle_time_s=estimate_full_cycle(state, cand),
    )
    return CandidateInput(f"{cand.box.box_id}:{cand.i}:{cand.j}:{cand.o}", metrics, physics_passed=True)


def observed_scenarios(state, queue: Sequence[str], seed: int, count: int, depth: int) -> list[list[str]]:
    """Keep observed order, then shuffle ONLY the remaining unknown inventory.

    Contract: the current box has already been consumed from state.inventory;
    observed future boxes have not been consumed. Shelf inventory is NOT modeled.
    """
    if count < 1 or depth < 1:
        raise ValueError("scenario count and depth must be positive")
    remaining = dict(state.inventory)
    for tid in queue:
        if remaining.get(tid, 0) <= 0:
            raise ValueError("observed queue exceeds remaining inventory")
        remaining[tid] -= 1
    pool = [tid for tid, n in sorted(remaining.items()) for _ in range(n)]
    rng = random.Random(f"{seed}:{state.processed}")
    scenarios = []
    for index in range(count):
        unknown = list(pool)
        rng.shuffle(unknown)
        if index % 4 == 0:  # one unfavourable small-first sequence per four scenarios
            unknown.sort(key=lambda tid: state.types[tid].volume)
        scenarios.append((list(queue) + unknown)[:depth])
    return scenarios


@dataclass
class PatternPlan:
    ranked: list[CandidateScore]
    candidates: dict
    deferred: list[CandidateScore]
    rollouts: dict[str, list[RolloutOutcome]]
    generated_count: int
    physics_count: int
    ranking_source: str
    fallback_reason: str
    elapsed_s: float

    @property
    def executable(self) -> CandidateScore | None:
        return select_executable(self.ranked)


class CorePatternPlanner:
    """AHEAD score -> (optional AI screen) -> shared rollouts -> common final score.

    `ranker` is a batch callable(inputs) -> {candidate_id: predicted F in [0,1]}.
    A missing/failed ranker falls back to fixed scoring. No model is trained here.
    `robot_verifier`, if supplied, must verify IK/collision/gripper/payload and
    return a RobotVerification with full path timing. Without it no execution
    candidate is returned, even when a geometry-only simulation has a ranking.
    """

    def __init__(self, config: ScoreConfig = ScoreConfig(), shortlist: int = 4,
                 scenario_count: int = 4, depth: int = 4, seed: int = 0,
                 rollout_limit: int = 8, ranker: Callable | None = None,
                 robot_verifier: Callable | None = None) -> None:
        if min(shortlist, scenario_count, depth, rollout_limit) < 1:
            raise ValueError("planner counts and horizon must be positive")
        self.config = config
        self.shortlist, self.scenario_count, self.depth = shortlist, scenario_count, depth
        self.seed, self.rollout_limit = seed, rollout_limit
        self.ranker, self.robot_verifier = ranker, robot_verifier

    def _config(self, state) -> ScoreConfig:
        # Match hard thresholds from the checker; tuning never changes them.
        cp = state.cfg.constraints
        return replace(self.config, support_threshold=cp.min_support_ratio, support_cog_min_mm=cp.cog_margin)

    def _feasible(self, state, box, limit=None):
        from palletizing_core.candidates import generate
        from palletizing_core.constraints import evaluate_many

        if box.damaged or box.mass > state.cfg.robot.payload:
            return [], 0
        actions = generate(state, box)
        feasible, _ = evaluate_many(state, box, actions, limit=limit)
        return feasible, len(actions)

    def _rollout(self, state, candidate, seq, scenario_id: str) -> RolloutOutcome:
        from palletizing_core.model import Box

        simulated = state.copy()
        simulated.commit(candidate)
        offered = placed = 0.0
        blocked = False
        cp = state.cfg.constraints
        for step, tid in enumerate(seq):
            b = Box.from_type(f"rollout:{step}", simulated.types[tid], cp.default_load_factor,
                              None, cp.area_load_capacity)
            simulated.consume(tid)
            offered += b.volume
            feasible, _ = self._feasible(simulated, b, self.rollout_limit)
            if not feasible:
                simulated.reject(b)
                blocked = True
                continue
            scored = [(evaluate_immediate(candidate_input(simulated, c), self._config(simulated)), c) for c in feasible]
            scored = [(s, c) for s, c in scored if s.eligible]
            if not scored:
                simulated.reject(b)
                blocked = True
                continue
            _, best = min(scored, key=lambda item: (-item[0].total, item[0].candidate.candidate_id))
            simulated.commit(best)
            placed += b.volume
        return RolloutOutcome(scenario_id, offered, placed, blocked)

    def plan(self, state, box, queue: Sequence[str] = ()) -> PatternPlan:
        """Read-only; caller consumes the current box before calling, commits after."""
        t0 = time.perf_counter()
        cfg = self._config(state)
        feasible, generated = self._feasible(state, box)
        inputs = [candidate_input(state, c) for c in feasible]
        candidates = {i.candidate_id: c for i, c in zip(inputs, feasible)}
        immediate = [evaluate_immediate(i, cfg) for i in inputs]
        predicted = None
        fallback = ""
        if self.ranker is not None:
            try:
                predicted = self.ranker(tuple(i for i, s in zip(inputs, immediate) if s.eligible))
                if predicted is None or not isinstance(predicted, Mapping):
                    fallback = "invalid_ranker_output"
                    predicted = None
            except Exception:
                fallback = "ranker_failed"
        shortlist = preselect(immediate, self.shortlist, predicted, cfg=cfg)
        ranked = list(shortlist.selected)
        outcomes = {}
        if ranked:
            scenarios = observed_scenarios(state, queue, self.seed, self.scenario_count, self.depth)
            if scenarios and any(scenarios):
                # Complete the same scenario batch for EVERY candidate. If the application
                # needs a deadline, decrease these counts BEFORE starting a batch.
                for s in ranked:
                    cid = s.candidate.candidate_id
                    outcomes[cid] = [self._rollout(state, candidates[cid], seq, f"scenario:{index}")
                                     for index, seq in enumerate(scenarios)]
                ranked = rank_final(ranked, outcomes, cfg)
            else:
                ranked.sort(key=lambda s: (-s.total, s.candidate.candidate_id))
        if self.robot_verifier is not None:
            verified = {}
            for s in ranked:
                verified[s.candidate.candidate_id] = self.robot_verifier(state, candidates[s.candidate.candidate_id])
            ranked = rerank_after_robot(ranked, verified, cfg)
        return PatternPlan(ranked, candidates, list(shortlist.deferred), outcomes, generated,
                           len(feasible), shortlist.ranking_source,
                           fallback or shortlist.fallback_reason, time.perf_counter() - t0)
