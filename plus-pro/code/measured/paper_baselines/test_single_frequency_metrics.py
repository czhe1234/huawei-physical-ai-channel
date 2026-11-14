"""Numerical safeguards for the scientific single-frequency evaluator."""
import unittest

import numpy as np

from single_frequency_metrics import evaluate


class SingleFrequencyMetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261017)
        self.h = (rng.normal(size=(7, 2, 8, 4)) + 1j * rng.normal(size=(7, 2, 8, 4))).astype(np.complex64)

    def test_identity_and_no_delay_claim(self):
        result, _ = evaluate(self.h, self.h)
        self.assertEqual(result["raw_global_energy_ratio_nmse"], 0)
        self.assertAlmostEqual(result["array_pas_cosine"], 1)
        self.assertEqual(result["power_mae_db"], 0)
        self.assertIsNone(result["pdp"])
        self.assertIsNone(result["original_pas_pdp_nmse_composite_score"])

    def test_common_phase_is_not_removed_from_primary_metric(self):
        result, _ = evaluate(1j * self.h, self.h)
        self.assertAlmostEqual(result["raw_global_energy_ratio_nmse"], 2, places=12)
        self.assertAlmostEqual(result["diagnostic_panel_phase_aligned_nmse"], 0, places=12)
        self.assertAlmostEqual(result["array_pas_cosine"], 1)
        self.assertEqual(result["power_mae_db"], 0)

    def test_amplitude_and_global_energy_aggregation(self):
        result, _ = evaluate(2 * self.h, self.h)
        self.assertAlmostEqual(result["raw_global_energy_ratio_nmse"], 1)
        self.assertAlmostEqual(result["power_mae_db"], 20 * np.log10(2))
        pred = self.h.copy()
        pred[0] *= 0
        result, _ = evaluate(pred, self.h)
        expected = (np.abs(self.h[0].astype(np.complex128)) ** 2).sum() / (np.abs(self.h.astype(np.complex128)) ** 2).sum()
        self.assertAlmostEqual(result["raw_global_energy_ratio_nmse"], expected, places=12)

    def test_zero_predictor_and_single_antenna(self):
        result, _ = evaluate(np.zeros_like(self.h), self.h)
        self.assertEqual(result["raw_global_energy_ratio_nmse"], 1)
        self.assertEqual(result["array_pas_cosine"], 0)
        self.assertTrue(np.isfinite(result["power_mae_db"]))
        result, _ = evaluate(self.h[:, :, :1, :1], self.h[:, :, :1, :1])
        self.assertIsNone(result["array_pas_cosine"])

    def test_forbid_wideband_mislabel_and_nonfinite(self):
        with self.assertRaises(ValueError):
            evaluate(np.stack([self.h, self.h], axis=-1), np.stack([self.h, self.h], axis=-1))
        bad = self.h.copy()
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            evaluate(bad, self.h)


if __name__ == "__main__":
    unittest.main()
