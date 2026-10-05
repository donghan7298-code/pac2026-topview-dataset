import unittest

from pattern_evaluation.tuning import choose_on_validation


def episode(fill, seconds, violations=0):
    return {"final_utilization": fill, "cycle_s_per_placed": seconds, "hard_violations": violations}


class TuningTests(unittest.TestCase):
    def test_unsafe_profile_is_excluded_even_with_best_fill(self):
        result = choose_on_validation({"unsafe": [episode(0.99, 1, 1)], "safe": [episode(0.7, 6)]})
        self.assertEqual(result["selected_profile"], "safe")

    def test_approximately_equal_fill_prefers_faster_profile(self):
        result = choose_on_validation({"slow": [episode(0.701, 10)], "fast": [episode(0.700, 5)]})
        self.assertEqual(result["selected_profile"], "fast")

    def test_no_safe_profile_is_not_silently_deployed(self):
        self.assertIsNone(choose_on_validation({"unsafe": [episode(0.99, 1, 1)]})["selected_profile"])


if __name__ == "__main__":
    unittest.main()
