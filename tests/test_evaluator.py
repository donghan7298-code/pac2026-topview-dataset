import math
import unittest
from dataclasses import replace

from pattern_evaluation.evaluator import (
    CandidateInput, ImmediateMetrics, RobotVerification, RolloutOutcome, ScoreConfig,
    ScoreWeights, evaluate_immediate, lower_tail_mean, preselect, rank_final,
    rerank_after_robot, select_executable, summarize_future,
)


def candidate(cid="a", **changes):
    metrics = ImmediateMetrics(0.8, 1.0, 60.0, 0.8, 50.0, 400.0, 6.0)
    return CandidateInput(cid, replace(metrics, **changes), physics_passed=True)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = ScoreConfig()

    def test_space_cannot_compensate_for_unsafe_support(self):
        safe = evaluate_immediate(candidate("safe", compactness=0.5), self.config)
        unsafe = evaluate_immediate(candidate("unsafe", compactness=1.0, support_ratio=0.6), self.config)
        self.assertEqual([s.candidate.candidate_id for s in preselect([unsafe, safe], 2).selected], ["safe"])

    def test_upstream_physics_must_be_explicitly_verified(self):
        s = evaluate_immediate(replace(candidate(), physics_passed=False))
        self.assertFalse(s.eligible)
        self.assertFalse(preselect([s], 1).selected)

    def test_cog_outside_hard_limit_is_excluded(self):
        s = evaluate_immediate(candidate(pallet_cog_offset_mm=401.0))
        self.assertFalse(s.eligible)

    def test_support_cog_minimum_is_not_a_soft_penalty(self):
        s = evaluate_immediate(candidate(support_cog_margin_mm=9.0))
        self.assertFalse(s.eligible)

    def test_unknown_robot_cannot_execute_even_highest_score(self):
        s = evaluate_immediate(candidate())
        self.assertIsNone(select_executable([s]))

    def test_real_path_time_can_change_the_winner(self):
        a = evaluate_immediate(candidate("a", cycle_time_s=1.0))
        b = evaluate_immediate(candidate("b", cycle_time_s=5.0))
        self.assertGreater(a.total, b.total)
        ranked = rerank_after_robot([a, b], {"a": RobotVerification("PASS", 80.0), "b": RobotVerification("PASS", 4.0)})
        self.assertEqual(select_executable(ranked).candidate.candidate_id, "b")

    def test_failed_first_candidate_uses_next_verified_candidate(self):
        scores = [evaluate_immediate(candidate("a", compactness=0.95)), evaluate_immediate(candidate("b"))]
        ranked = rerank_after_robot(scores, {"a": RobotVerification("FAIL", reason="collision"), "b": RobotVerification("PASS", 6.0)})
        self.assertEqual(select_executable(ranked).candidate.candidate_id, "b")

    def test_all_robot_failures_return_hold(self):
        scores = [evaluate_immediate(candidate())]
        self.assertIsNone(select_executable(rerank_after_robot(scores, {"a": RobotVerification("FAIL")})))

    def test_pass_requires_full_cycle_time(self):
        with self.assertRaises(ValueError):
            RobotVerification("PASS")

    def test_time_cost_remains_strictly_monotonic_above_reference(self):
        t1 = evaluate_immediate(candidate(cycle_time_s=80)).features["T"]
        t2 = evaluate_immediate(candidate(cycle_time_s=160)).features["T"]
        self.assertLess(t1, t2)
        self.assertLess(t2, 1.0)

    def test_lower_used_height_can_distinguish_same_volume_candidates(self):
        low = evaluate_immediate(candidate("low", compactness=0.8))
        high = evaluate_immediate(candidate("high", compactness=0.4))
        self.assertGreater(low.total, high.total)

    def test_same_mean_but_bad_tail_has_higher_risk(self):
        def outcomes(values):
            return [RolloutOutcome(str(i), 100.0, 100 * value, False) for i, value in enumerate(values)]
        stable = summarize_future(outcomes([0.5] * 4), self.config)
        variable = summarize_future(outcomes([0.0, 0.5, 0.5, 1.0]), self.config)
        self.assertAlmostEqual(stable.mean, variable.mean)
        self.assertGreater(variable.risk, stable.risk)

    def test_empty_terminal_future_has_no_free_positive_reward(self):
        summary = summarize_future([RolloutOutcome("terminal", 0, 0, False)], self.config)
        self.assertEqual((summary.mean, summary.risk), (0, 0))

    def test_fractional_tail_uses_exact_empirical_mass(self):
        self.assertAlmostEqual(lower_tail_mean([0.1, 0.4, 0.9], 0.5), 0.2)

    def test_future_score_excludes_already_penalized_value(self):
        result = rank_final([evaluate_immediate(candidate())], {"a": [RolloutOutcome("s", 100, 50, True)]})[0]
        self.assertEqual(result.features["F"], 0.5)
        self.assertAlmostEqual(result.features["R"], 0.6)

    def test_shared_scenario_ids_are_required(self):
        scores = [evaluate_immediate(candidate("a")), evaluate_immediate(candidate("b"))]
        with self.assertRaises(ValueError):
            rank_final(scores, {"a": [RolloutOutcome("s0", 100, 90, False)], "b": [RolloutOutcome("s1", 100, 90, False)]})

    def test_shared_horizon_volumes_are_required(self):
        scores = [evaluate_immediate(candidate("a")), evaluate_immediate(candidate("b"))]
        with self.assertRaises(ValueError):
            rank_final(scores, {"a": [RolloutOutcome("s0", 100, 90, False)], "b": [RolloutOutcome("s0", 200, 180, False)]})

    def test_duplicate_scenarios_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_future([RolloutOutcome("same", 100, 90, False)] * 2, self.config)

    def test_invalid_ai_predictions_fall_back_to_fixed_ranking(self):
        scores = [evaluate_immediate(candidate("a")), evaluate_immediate(candidate("b", compactness=0.9))]
        shortlist = preselect(scores, 1, {"a": math.nan, "b": 0.8})
        self.assertEqual(shortlist.ranking_source, "fixed_score")
        self.assertEqual(shortlist.selected[0].candidate.candidate_id, "b")

    def test_ai_can_add_a_candidate_outside_heuristic_top_k(self):
        scores = [evaluate_immediate(candidate("a", compactness=0.9)),
                  evaluate_immediate(candidate("b", compactness=0.8)),
                  evaluate_immediate(candidate("c", compactness=0.7))]
        shortlist = preselect(scores, 2, {"a": 0.0, "b": 0.0, "c": 1.0}, heuristic_reserve=1)
        self.assertEqual({s.candidate.candidate_id for s in shortlist.selected}, {"a", "c"})
        self.assertEqual([s.candidate.candidate_id for s in shortlist.deferred], ["b"])

    def test_duplicate_candidate_ids_are_not_silently_overwritten(self):
        with self.assertRaises(ValueError):
            preselect([evaluate_immediate(candidate()), evaluate_immediate(candidate())], 1)

    def test_nan_invalid_ranges_and_invalid_weights_fail(self):
        for changes in ({"cycle_time_s": math.nan}, {"compactness": 1.01}, {"min_load_slack_ratio": -0.1}):
            with self.assertRaises(ValueError):
                candidate(**changes)
        with self.assertRaises(ValueError):
            ScoreWeights(U=0.8)

    def test_score_parts_sum_and_inputs_are_unchanged(self):
        source = candidate()
        before = repr(source)
        score = evaluate_immediate(source)
        self.assertAlmostEqual(score.total, sum(score.contributions.values()))
        self.assertTrue(all(0 <= value <= 1 for value in score.features.values()))
        self.assertEqual(repr(source), before)


if __name__ == "__main__":
    unittest.main()
