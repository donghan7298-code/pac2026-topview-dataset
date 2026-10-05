"""Small, explainable weight search using CLOSED-LOOP validation episodes.

The callback must actually repack each case with the proposed weights. Re-scoring
a fixed log alone cannot measure changed placements/future states.
"""

from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Callable, Iterable, Mapping

from .evaluator import ScoreWeights

PROFILES = {
    "default": ScoreWeights(),
    "space": ScoreWeights(U=0.30, S=0.25, L=0.15, F=0.20, T=0.05, R=0.05),
    "time": ScoreWeights(U=0.20, S=0.25, L=0.15, F=0.20, T=0.15, R=0.05),
    "future": ScoreWeights(U=0.15, S=0.25, L=0.15, F=0.30, T=0.10, R=0.05),
}


def choose_on_validation(results: Mapping[str, list[dict]], fill_tolerance: float = 0.002) -> dict:
    """Exclude physical violations; maximize final fill, then reduce cycle time.

    The tolerance is an explicit engineering choice, not a statistical significance
    threshold. Do not use candidate scores as the tuning objective (circular).
    """
    if fill_tolerance < 0:
        raise ValueError("fill tolerance must be nonnegative")
    feasible = {name: episodes for name, episodes in results.items()
                if episodes and all(ep["hard_violations"] == 0 for ep in episodes)}
    if not feasible:
        return {"selected_profile": None, "reason": "no_profile_passed_validation"}
    avg_fill = {name: mean(ep["final_utilization"] for ep in episodes) for name, episodes in feasible.items()}
    best = max(avg_fill.values())
    near = [name for name in feasible if avg_fill[name] >= best - fill_tolerance]
    selected = min(near, key=lambda name: (mean(ep["cycle_s_per_placed"] for ep in feasible[name]), name))
    return {"selected_profile": selected, "fill_tolerance": fill_tolerance,
            "validation_mean_fill": avg_fill[selected],
            "reason": "zero_violations_then_fill_then_placement_cycle_time"}


def search_weights(cases: Iterable, runner: Callable, profiles: Mapping[str, ScoreWeights] = PROFILES) -> dict:
    cases = list(cases)
    if not cases:
        raise ValueError("validation cases must not be empty")
    by_profile = {}
    for name, weights in profiles.items():
        # All profiles see the same cases/seeds and the same fixed hard constraints.
        by_profile[name] = [runner(case, weights) for case in cases]
    return {"selection": choose_on_validation(by_profile),
            "weights": {name: asdict(weights) for name, weights in profiles.items()},
            "validation": by_profile}
