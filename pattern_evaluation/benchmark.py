"""Closed-loop geometry smoke benchmark; requires the optional reference core.

Usage:
  python -m pattern_evaluation.benchmark --core-root ../pac-mission1/src/palletizing_core

This is not a robot/shelf benchmark, neural training, or official PAC scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

from .evaluator import ScoreConfig
from .tuning import PROFILES, choose_on_validation

FAMILIES = ("random", "small_first", "large_first", "clustered")


def _case(seed: int, family: str):
    from palletizing_core.config import Config, PalletSpec
    from palletizing_core.model import Box, BoxType
    rng = random.Random(seed)
    cfg = Config(pallet=PalletSpec(length=600, width=400, max_height=600, max_mass=300, grid=20))
    types = [BoxType("a", 200, 200, 120, 5, max_load=250),
             BoxType("b", 300, 200, 180, 10, max_load=600),
             BoxType("c", 200, 100, 100, 3, max_load=120),
             BoxType("d", 300, 200, 240, 15, max_load=800)]
    counts = {t.type_id: rng.randint(3, 6) for t in types}
    sequence = [Box.from_type(f"{t.type_id}:{index}", t) for t in types for index in range(counts[t.type_id])]
    rng.shuffle(sequence)
    if family == "small_first":
        sequence.sort(key=lambda b: b.volume)
    elif family == "large_first":
        sequence.sort(key=lambda b: -b.volume)
    elif family == "clustered":
        order = list(counts)
        rng.shuffle(order)
        sequence.sort(key=lambda b: order.index(b.type_id))
    return cfg, types, counts, sequence


def run_case(seed: int, family: str, profile: str, shortlist: int, scenario_count: int, depth: int,
             log_path: Path | None = None) -> dict:
    from palletizing_core.constraints import cog_radius
    from palletizing_core.model import PalletState
    from palletizing_core.planner import GreedyPlanner, PLACE
    from palletizing_core.simulate import validate
    from .core_adapter import CorePatternPlanner, estimate_full_cycle
    from .evaluator import write_decision_log
    cfg, types, counts, sequence = _case(seed, family)
    state = PalletState.new(cfg, types, counts)
    scorer = CorePatternPlanner(replace(ScoreConfig(), weights=PROFILES.get(profile, PROFILES["default"])),
                               shortlist=shortlist, scenario_count=scenario_count, depth=depth, seed=seed)
    baseline = GreedyPlanner("irap") if profile == "upstream_greedy" else None
    elapsed = []
    cycles = []
    supports = []
    chosen_ids = []
    for index, box in enumerate(sequence):
        state.consume(box.type_id)
        queue = [b.type_id for b in sequence[index + 1:index + 3]]  # two observed upstream boxes
        if baseline:
            start = time.perf_counter()
            decision = baseline.decide(state, box, queue)
            elapsed.append(time.perf_counter() - start)
            best = decision.best if decision.action == PLACE else None
        else:
            plan = scorer.plan(state, box, queue)
            elapsed.append(plan.elapsed_s)
            # GEOMETRY simulation only: a physical robot would need plan.executable.
            best = plan.candidates[plan.ranked[0].candidate.candidate_id] if plan.ranked else None
            if log_path and plan.ranked:
                write_decision_log(log_path, plan.ranked, {
                    "profile": profile, "seed": seed, "family": family, "step": index,
                    "queue": queue, "weights": asdict(scorer.config.weights),
                    "geometry_only": True,
                    "rollouts": {cid: [asdict(o) for o in outcomes] for cid, outcomes in plan.rollouts.items()},
                })
        if best is None:
            state.reject(box)
            chosen_ids.append("reject")
        else:
            cycles.append(estimate_full_cycle(state, best))
            supports.append(best.support_ratio)
            chosen_ids.append(f"{best.i}:{best.j}:{best.o}")
            state.commit(best)
    violations = validate(state)
    if state.mass > cfg.pallet.max_mass + 1e-6:
        violations.append("pallet mass exceeded")
    cog = state.cog_xy()
    if cog is not None and math.hypot(cog[0] - cfg.pallet.length/2, cog[1] - cfg.pallet.width/2) > cog_radius(state) + 1e-6:
        violations.append("terminal pallet COG outside configured limit")
    sorted_time = sorted(elapsed)
    return {
        "seed": seed, "family": family, "profile": profile,
        "final_utilization": state.utilization(), "placed": len(state.placed),
        "rejected": len(state.rejected), "hard_violations": len(violations), "violation_details": violations,
        "min_support_ratio": min(supports) if supports else None,
        "cycle_s_sum_for_placed": sum(cycles), "cycle_s_per_placed": mean(cycles) if cycles else 0.0,
        "decision_ms_mean": mean(elapsed) * 1000,
        "decision_ms_p95": sorted_time[max(0, math.ceil(0.95 * len(sorted_time)) - 1)] * 1000,
        "chosen_candidate_ids": chosen_ids,
    }


def _summary(episodes: list[dict]) -> dict:
    return {"mean_final_utilization": mean(ep["final_utilization"] for ep in episodes),
            "mean_placed": mean(ep["placed"] for ep in episodes),
            "mean_rejected": mean(ep["rejected"] for ep in episodes),
            "hard_violations": sum(ep["hard_violations"] for ep in episodes),
            "mean_cycle_s_per_placed": mean(ep["cycle_s_per_placed"] for ep in episodes),
            "mean_decision_ms": mean(ep["decision_ms_mean"] for ep in episodes),
            "max_episode_p95_ms": max(ep["decision_ms_p95"] for ep in episodes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/geometry_smoke.json"))
    parser.add_argument("--log", type=Path)
    parser.add_argument("--cases-per-family", type=int, default=1)
    parser.add_argument("--shortlist", type=int, default=4)
    parser.add_argument("--scenarios", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    if args.cases_per_family < 1:
        parser.error("cases-per-family must be positive")
    sys.path.insert(0, str(args.core_root.resolve()))
    splits = {"validation": 1000, "test": 2000}
    all_results = {}
    profiles = list(PROFILES) + ["upstream_greedy"]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("", encoding="utf-8")
    for split, base_seed in splits.items():
        all_results[split] = {}
        for profile in profiles:
            episodes = []
            for family_index, family in enumerate(FAMILIES):
                for index in range(args.cases_per_family):
                    seed = base_seed + family_index * 100 + index
                    result = run_case(seed, family, profile, args.shortlist, args.scenarios, args.depth,
                                      args.log if split == "validation" and profile == "default" else None)
                    episodes.append(result)
            all_results[split][profile] = episodes
            summary = _summary(episodes)
            print(f"{split:10} {profile:16} fill={summary['mean_final_utilization']:.4f} "
                  f"violations={summary['hard_violations']} mean_ms={summary['mean_decision_ms']:.1f}", flush=True)
    selection = choose_on_validation({p: eps for p, eps in all_results["validation"].items() if p in PROFILES})
    result = {
        "benchmark_kind": "small_geometry_smoke_no_shelf_no_IK_no_trained_AI",
        "reference_repo": "https://github.com/yang8988/pac-mission1",
        "reference_commit": "2312192e05964e573187922c11932d19e6181952",
        "conditions": {"shortlist": args.shortlist, "scenario_count": args.scenarios, "depth": args.depth,
                       "observed_queue": 2, "pallet_mm": [600, 400, 600], "grid_mm": 20,
                       "robot_time_is_estimate": True, "NG_and_shelf_time_not_measured": True,
                       "hard_real_time_deadline": False, "neural_training": False},
        "weights": {p: asdict(w) for p, w in PROFILES.items()},
        "selection_from_validation_only": selection,
        "summaries": {split: {p: _summary(eps) for p, eps in values.items()} for split, values in all_results.items()},
        "episodes": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Validation-only recommendation:", selection, flush=True)


if __name__ == "__main__":
    main()
