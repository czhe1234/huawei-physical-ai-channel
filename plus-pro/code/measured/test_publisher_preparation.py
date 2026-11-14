"""Small CPU invariants for publisher preparation verification; no data scoring."""
from __future__ import annotations

import hashlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from verify_publisher_preparation import (
    compare_calibration, exact_equal, minimum_distances, reference_calibration,
    reference_selection, reference_coordinates, verify_split,
)


def small_protocol(position, block_m=4.0, guard_m=.30):
    # Build a fixture with brute-force distance, independent of production splitter.
    tile = np.floor(position[:, :2] / block_m).astype(np.int64)
    colors = (tile[:, 0] + 2 * tile[:, 1]) % 5
    splits = []
    for outer in range(5):
        test = np.flatnonzero(colors == outer)
        val = np.flatnonzero(colors == (outer + 1) % 5)
        possible = np.flatnonzero((colors != outer) & (colors != (outer + 1) % 5))
        held = np.r_[test, val]
        distance = np.linalg.norm(position[possible, None, :2]
                                  - position[None, held, :2], axis=-1).min(axis=1)
        fit = possible[distance >= guard_m]
        nearest = np.linalg.norm(position[test, None, :2]
                                 - position[None, fit, :2], axis=-1).min(axis=1)
        splits.append(dict(outer=outer, fit=fit.tolist(), val=val.tolist(), test=test.tolist(),
                           min_test_fit_distance=float(nearest.min()),
                           test_fit_distance_quantiles=np.quantile(
                               nearest, [0, .25, .5, .75, 1]).tolist()))
    return dict(block_m=block_m, guard_m=guard_m, fold=colors.tolist(), splits=splits,
                sha256=hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest())


class PublisherInvariants(unittest.TestCase):
    def test_fftshift_positive_sign_and_physical_mapping_all_ports(self):
        csi = np.zeros((64, 1024, 2), dtype=np.float32)
        csi[..., 0] = np.arange(64)[:, None] * 2048 + np.arange(1024)[None]
        assignments = np.arange(64, dtype=np.int64).reshape(2, 4, 8)[:, :, ::-1]
        offsets = dict(cpo=np.full(64, np.pi / 2), sto=np.zeros(64))
        freq = np.arange(0, 1024, 4, dtype=np.int64)
        expected = reference_calibration(csi, offsets, assignments, freq)
        self.assertEqual(expected.shape, (2, 8, 4, 256))
        for panel in range(2):
            for horizontal in range(8):
                for vertical in range(4):
                    port = assignments[panel, vertical, horizontal]
                    for output_index in (0, 32, 128, 255):
                        raw_index = (int(freq[output_index]) + 512) % 1024
                        self.assertAlmostEqual(expected[panel, horizontal, vertical,
                                                        output_index].imag,
                                               float(port * 2048 + raw_index), places=7)
                        self.assertLess(abs(expected[panel, horizontal, vertical,
                                                     output_index].real), 1e-9)

    def test_wrong_compensation_sign_or_port_swap_is_rejected(self):
        csi = np.zeros((64, 1024, 2), dtype=np.float32)
        csi[..., 0] = (np.arange(64) + 1)[:, None]
        assignments = np.arange(64, dtype=np.int64).reshape(2, 4, 8)
        freq = np.arange(0, 1024, 4, dtype=np.int64)
        offsets = dict(cpo=np.full(64, .4), sto=np.linspace(-.5, .5, 64))
        expected = reference_calibration(csi, offsets, assignments, freq)
        compare_calibration(expected.astype(np.complex64), expected)
        wrong = reference_calibration(
            csi, dict(cpo=-offsets["cpo"], sto=-offsets["sto"]), assignments, freq)
        with self.assertRaises(ValueError):
            compare_calibration(wrong, expected)
        wrong = expected.copy()
        wrong[[0, 1]] = wrong[[1, 0]]
        with self.assertRaises(ValueError):
            compare_calibration(wrong, expected)

    def test_near_zero_tolerance_and_nonfinite_are_distinct(self):
        reference = np.zeros((2, 8, 4, 256), dtype=np.complex128)
        compare_calibration(np.full_like(reference, 1e-13), reference)
        with self.assertRaises(ValueError):
            compare_calibration(np.full_like(reference, 1e-4), reference)
        bad = reference.copy()
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            compare_calibration(bad, reference)

    def test_horizontal_scene_frame_ignores_reference_height_difference(self):
        content = "Name,South,West,Height\nAU,0,0,7\nNWU,1,0,8\n"
        with patch("pathlib.Path.open", return_value=io.StringIO(content)):
            origin, rotation = reference_coordinates("synthetic.csv")
        np.testing.assert_array_equal(origin, [0, 0, 7])
        np.testing.assert_array_equal(rotation, np.eye(3))

    def test_representative_uses_only_position_center_and_source_id(self):
        positions = np.array([[.25, .5, 0], [.75, .5, 0], [.5, .5, 0],
                              [1.25, .5, 0], [1.75, .5, 0]], dtype=np.float64)
        ids, groups, count = reference_selection(positions, 1.0, 6000, 20261015)
        np.testing.assert_array_equal(ids, [2, 3])
        np.testing.assert_array_equal(groups, [0, 1])
        self.assertEqual(count, 2)
        # A source-order tie chooses the first frame, without CSI or timestamps.
        ids2, _, _ = reference_selection(positions[:2], 1.0, 6000, 20261015)
        np.testing.assert_array_equal(ids2, [0])

    def test_spatial_contract_and_duplicate_test_are_checked(self):
        positions = np.array([[4*x + dx, 4*y + dy, 1.5]
                              for x in range(6) for y in range(6)
                              for dx, dy in ((.1, .1), (3.9, .1), (.1, 3.9), (3.9, 3.9), (2., 2.))])
        protocol = small_protocol(positions)
        report = verify_split(positions, protocol, 4.0, .30)
        self.assertTrue(report["test_once"])
        bad = json.loads(json.dumps(protocol))
        bad["splits"][0]["test"].append(bad["splits"][0]["test"][0])
        bad["sha256"] = hashlib.sha256(json.dumps(
            bad["splits"], sort_keys=True).encode()).hexdigest()
        with self.assertRaises(ValueError):
            verify_split(positions, bad, 4.0, .30)

    def test_guard_removes_nearby_different_fold_and_exact_metadata_detects_dtype(self):
        distances = minimum_distances(np.array([[.1, 0, 0], [1, 0, 0]]),
                                     np.array([[0, 0, 0]]))
        np.testing.assert_allclose(distances, [.1, 1])
        self.assertFalse(bool(distances[0] >= .30))
        with self.assertRaises(ValueError):
            exact_equal(np.array([1], dtype=np.int32),
                        np.array([1], dtype=np.int64), "ID dtype")


if __name__ == "__main__":
    unittest.main(verbosity=2)
