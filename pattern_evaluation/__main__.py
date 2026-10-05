"""Dependency-free synthetic score demo: python -m pattern_evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .evaluator import (
    CandidateInput, ImmediateMetrics, RobotVerification, RolloutOutcome, ScoreConfig,
    evaluate_immediate, preselect, rank_final, rerank_after_robot, select_executable,
)


def demo(config: ScoreConfig) -> dict:
    specs = {
        "A_compact": ImmediateMetrics(0.82, 0.88, 25, 0.45, 180, 400, 7.0),
        "B_supported": ImmediateMetrics(0.75, 1.0, 60, 0.80, 90, 400, 8.0),
        "C_fast": ImmediateMetrics(0.73, 0.96, 50, 0.68, 130, 400, 4.0),
        "D_unsafe": ImmediateMetrics(0.95, 0.60, 5, 0.10, 200, 400, 3.0),
    }
    initial = [evaluate_immediate(CandidateInput(cid, m, physics_passed=cid != "D_unsafe"), config)
               for cid, m in specs.items()]
    # These are hand-written example outputs, not a trained AI model.
    predicted = {"A_compact": 0.94, "B_supported": 0.80, "C_fast": 0.72}
    shortlist = preselect(initial, 3, predicted, heuristic_reserve=1, cfg=config)
    ratios = {"A_compact": (0.95, 0.82, 0.15, 0.88),
              "B_supported": (0.82, 0.80, 0.79, 0.86),
              "C_fast": (0.83, 0.70, 0.63, 0.72)}
    outcomes = {cid: [RolloutOutcome(f"s:{i}", 1000.0, ratio * 1000, ratio < 0.5)
                      for i, ratio in enumerate(values)] for cid, values in ratios.items()}
    ranked = rank_final(shortlist.selected, outcomes, config)
    before_verification = select_executable(ranked)
    # Mock robot checks for demonstration only. They are NOT IK/collision tests.
    verifications = {"A_compact": RobotVerification("PASS", 7.2),
                     "B_supported": RobotVerification("FAIL", reason="mock_path_collision"),
                     "C_fast": RobotVerification("PASS", 4.4)}
    verified = rerank_after_robot(ranked, verifications, config)
    chosen = select_executable(verified)
    return {
        "demo_kind": "synthetic_metrics_and_mock_robot_checks_no_trained_ai",
        "ranking_source": shortlist.ranking_source,
        "weights": vars(config.weights),
        "ranked_before_robot": [s.to_dict() for s in ranked],
        "selected_before_robot": before_verification,
        "ranked_after_mock_robot": [s.to_dict() for s in verified],
        "selected_after_mock_robot": chosen.candidate.candidate_id if chosen else None,
        "unsafe_filtered": not any(s.candidate.candidate_id == "D_unsafe" for s in shortlist.selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = demo(ScoreConfig.load(args.config) if args.config else ScoreConfig())
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print("합성 입력 예제: 학습된 AI/실제 로봇 검증 결과가 아닙니다.")
    for s in result["ranked_before_robot"]:
        f = s["features"]
        print(f"{s['candidate']['candidate_id']:15} score={s['total']:.4f} "
              + " ".join(f"{key}={value:.3f}" for key, value in f.items()))
    print("로봇 검증 전 실행 후보:", result["selected_before_robot"])
    print("모의 로봇 검증 후 선택:", result["selected_after_mock_robot"])


if __name__ == "__main__":
    main()
