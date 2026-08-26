import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anatomical_rule_classifier import classify_anatomical_rules


def open_hand():
    """Synthetic MediaPipe-layout open hand; kept local for portable sharing."""
    points = np.zeros((21, 2), dtype=float)
    points[0] = (0.0, 0.0)
    points[1:5] = [(-0.5, 0.2), (-0.9, 0.5), (-1.3, 0.9), (-1.8, 1.1)]
    for x, chain in zip(
        (-0.8, -0.25, 0.35, 0.95),
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)),
    ):
        for y, index in zip((0.8, 1.6, 2.4, 3.2), chain):
            points[index] = (x, y)
    return points


class AnatomicalRuleClassifierTest(unittest.TestCase):
    def test_open_hand(self):
        self.assertEqual(classify_anatomical_rules(open_hand())["state"], "OPEN")

    def test_thumb_may_overlap_index_lane(self):
        points = open_hand()
        points[3, 0] = points[5, 0]
        points[4, 0] = points[5, 0]
        self.assertEqual(classify_anatomical_rules(points)["state"], "OPEN")

    def test_thumb_cannot_reach_middle_lane(self):
        points = open_hand()
        points[4, 0] = points[9, 0]
        result = classify_anatomical_rules(points)
        self.assertEqual(result["state"], "CLOSED")
        self.assertTrue(result["lateral_crossings"])

    def test_middle_may_overlap_immediate_ring_lane(self):
        points = open_hand()
        points[11, 0] = points[13, 0]
        points[12, 0] = points[13, 0]
        self.assertEqual(classify_anatomical_rules(points)["state"], "OPEN")

    def test_middle_cannot_reach_pinky_lane(self):
        points = open_hand()
        points[12, 0] = points[17, 0]
        self.assertEqual(classify_anatomical_rules(points)["state"], "CLOSED")

    def test_tip_cannot_fold_back_to_second_joint(self):
        points = open_hand()
        points[8] = points[6]
        result = classify_anatomical_rules(points)
        self.assertEqual(result["state"], "CLOSED")
        self.assertTrue(any(item["finger"] == "index" for item in result["proximal_folds"]))

    def test_rotated_scaled_translated_open_hand(self):
        points = open_hand()
        angle = np.deg2rad(73.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        transformed = 91.0 * points @ rotation.T + np.array([510.0, 280.0])
        self.assertEqual(classify_anatomical_rules(transformed)["state"], "OPEN")


if __name__ == "__main__":
    unittest.main()
