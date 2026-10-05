import copy
import unittest

import numpy as np

from projects.pattern_evaluation.fit import require_disjoint
from projects.pattern_evaluation.model import Ranker, train
from projects.pattern_evaluation.scoring import FEATURE_NAMES
from projects.pattern_evaluation.rollout import OUTPUT_NAMES
from projects.pattern_evaluation.planner import load_config


class LearningTests(unittest.TestCase):
    def test_inventory_group_split_rejects_camera_leakage(self):
        with self.assertRaises(ValueError): require_disjoint([dict(base_group="same")], [dict(base_group="same")])

    def test_network_learns_and_serializes_future_outputs(self):
        rng = np.random.default_rng(3)
        groups = []
        for i in range(24):
            rows = []
            for j in range(8):
                x = rng.normal(0, .3, len(FEATURE_NAMES)); fit = float(np.clip(.5+.5*x[0], 0, 1))
                rows.append(dict(features=x.tolist(), static=dict(safety=1., space=.5, balance=.5, handling=.3),
                    future=dict(mean_fit=fit, mean_volume=fit*.1, cvar_fit=fit*.8, block_rate=1-fit, risk=.5*(1-fit)+.2*fit)))
            groups.append(dict(base_group=str(i), rows=rows))
        model, report = train(groups[:18], groups[18:], load_config()["weights"], seed=3, epochs=25, hidden=12)
        self.assertLess(min(r["validation_normalized_mse"] for r in report["history"]), report["history"][0]["validation_normalized_mse"]*.8)
        restored = Ranker(model.value)
        features = [row["features"] for row in groups[-1]["rows"]]
        self.assertEqual(model.predict(features), restored.predict(features))
        self.assertTrue(all(0 <= v[n] <= 1 for v in model.predict(features) for n in OUTPUT_NAMES))
        invalid = copy.deepcopy(model.value); invalid["feature_names"] = ["stale"]
        with self.assertRaises(ValueError): Ranker(invalid)


if __name__ == "__main__": unittest.main()
