import importlib.util
import os
import sys
import unittest
from pathlib import Path

if os.environ.get("CORE_ROOT"):
    sys.path.insert(0, str(Path(os.environ["CORE_ROOT"]).resolve()))

CORE_AVAILABLE = importlib.util.find_spec("palletizing_core") is not None


@unittest.skipUnless(CORE_AVAILABLE, "set CORE_ROOT to reference repo's src/palletizing_core")
class CoreAdapterTests(unittest.TestCase):
    def state(self):
        from palletizing_core.config import Config, PalletSpec
        from palletizing_core.model import BoxType, PalletState
        cfg = Config(pallet=PalletSpec(length=600, width=400, max_height=600, max_mass=300, grid=20))
        cfg.constraints.cog_radius_end = 0.4
        types = [BoxType("a", 200, 200, 100, 5, max_load=500), BoxType("b", 300, 200, 200, 10, max_load=700)]
        return PalletState.new(cfg, types, {"a": 3, "b": 3})

    def box(self, state, tid="a"):
        from palletizing_core.model import Box
        return Box.from_type("actual", state.types[tid])

    def test_reference_core_integration_does_not_mutate_live_state(self):
        import numpy as np
        from pattern_evaluation.core_adapter import CorePatternPlanner
        state = self.state()
        state.consume("a")
        before = (state.H.copy(), dict(state.inventory), state.mass, len(state.placed), state.moment.copy())
        plan = CorePatternPlanner(shortlist=2, scenario_count=2, depth=2).plan(state, self.box(state), ["b"])
        self.assertTrue(plan.ranked)
        self.assertIsNone(plan.executable)
        self.assertTrue(all(s.future.scenario_count == 2 for s in plan.ranked))
        np.testing.assert_array_equal(state.H, before[0])
        self.assertEqual((state.inventory, state.mass, len(state.placed)), before[1:4])
        np.testing.assert_array_equal(state.moment, before[4])

    def test_observed_prefix_and_inventory_conservation(self):
        from collections import Counter
        from pattern_evaluation.core_adapter import observed_scenarios
        state = self.state()
        state.consume("a")
        scenarios = observed_scenarios(state, ["b", "a"], seed=3, count=4, depth=20)
        for seq in scenarios:
            self.assertEqual(seq[:2], ["b", "a"])
            self.assertEqual(Counter(seq), Counter(state.inventory))
        self.assertEqual(scenarios, observed_scenarios(state, ["b", "a"], seed=3, count=4, depth=20))

    def test_terminal_state_and_damaged_box(self):
        from pattern_evaluation.core_adapter import CorePatternPlanner
        state = self.state()
        state.inventory = {"a": 0, "b": 0}
        box = self.box(state)
        planner = CorePatternPlanner(shortlist=2, scenario_count=2, depth=2)
        plan = planner.plan(state, box)
        self.assertTrue(plan.ranked)
        self.assertTrue(all(s.future is None and s.features["F"] == 0 for s in plan.ranked))
        box.damaged = True
        self.assertFalse(planner.plan(state, box).ranked)

    def test_ranker_failure_has_working_fixed_score_fallback(self):
        from pattern_evaluation.core_adapter import CorePatternPlanner
        def broken(_):
            raise RuntimeError("model unavailable")
        state = self.state()
        state.consume("a")
        plan = CorePatternPlanner(shortlist=2, scenario_count=2, depth=1, ranker=broken).plan(state, self.box(state))
        self.assertTrue(plan.ranked)
        self.assertEqual(plan.fallback_reason, "ranker_failed")
        self.assertEqual(plan.ranking_source, "fixed_score")


if __name__ == "__main__":
    unittest.main()
