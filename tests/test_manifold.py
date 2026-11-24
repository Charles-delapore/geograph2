import time
import unittest

import numpy as np

from geograph2.embedding import generate_synthetic_tessera_cube
from geograph2.manifold import (
    compare_continuous_vs_hard_classification,
    cosine_resistance_field,
    fit_few_shot_suitability,
    semantic_gradient_boundaries,
)


class TestContinuousManifold(unittest.TestCase):
    def test_cosine_resistance_and_semantic_boundaries(self):
        cube, true_suit, _ = generate_synthetic_tessera_cube(24, 24, dim=128, seed=42)
        # Pick highest-suitability pixel as reference habitat prototype
        best_idx = np.unravel_index(int(np.argmax(true_suit)), true_suit.shape)
        ref_emb = cube.embeddings[best_idx]

        res_field = cosine_resistance_field(cube, ref_emb, base_resistance=1.0, gamma=10.0)
        self.assertEqual(res_field.shape, (24, 24))
        self.assertAlmostEqual(float(res_field[best_idx]), 1.0, places=5)
        self.assertGreater(float(np.max(res_field)), float(res_field[best_idx]))

        grad = semantic_gradient_boundaries(cube)
        self.assertEqual(grad.shape, (24, 24))
        self.assertTrue(np.all(grad >= 0.0))
        self.assertGreater(float(np.max(grad)), 0.01)

    def test_cpu_few_shot_suitability_beats_hard_classification(self):
        cube, true_suit, hard_lc = generate_synthetic_tessera_cube(24, 24, dim=128, seed=42)
        rng = np.random.default_rng(123)
        # Only 12 labeled few-shot points across the 576-pixel landscape
        coords = [(int(rng.integers(0, 24)), int(rng.integers(0, 24))) for _ in range(12)]
        targets = [float(true_suit[r, c]) for r, c in coords]

        t0 = time.perf_counter()
        model = fit_few_shot_suitability(cube, coords, targets, l2_reg=0.2)
        pred = model.predict(cube)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertLess(elapsed_ms, 100.0)
        comp = compare_continuous_vs_hard_classification(pred, hard_lc, true_suit)
        # Continuous 128D manifold must achieve lower RMSE and higher correlation than 3-class hard LC
        self.assertLess(comp["rmse_continuous"], comp["rmse_hard_class"])
        self.assertGreater(comp["pearson_continuous"], comp["pearson_hard_class"])
        self.assertGreater(comp["spearman_continuous"], 0.90)


if __name__ == "__main__":
    unittest.main()
