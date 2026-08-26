import json
import sys
import unittest
from pathlib import Path

import numpy as np

COMPONENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_DIR))

from anatomical_rule_classifier import (  # noqa: E402
    AnatomicalRuleConfig,
    classify_anatomical_rules,
)


class RecordedFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_dir = COMPONENT_DIR / "test_data"
        cls.keypoints = json.loads((fixture_dir / "keypoints.json").read_text())
        cls.expected = json.loads(
            (fixture_dir / "expected_results.json").read_text()
        )

    def test_fixture_metadata_matches(self):
        self.assertEqual(len(self.keypoints["frames"]), 330)
        self.assertEqual(len(self.expected["frames"]), 330)
        self.assertEqual(self.keypoints["fps"], 15.0)
        self.assertEqual(self.keypoints["resolution"], [1280, 720])

    def test_classifier_matches_recorded_regression_baseline(self):
        config = AnatomicalRuleConfig(
            second_neighbor_margin_ratio=self.expected["config"][
                "second_neighbor_margin_ratio"
            ],
            minimum_tip_beyond_second_ratio=self.expected["config"][
                "minimum_tip_beyond_second_ratio"
            ],
        )
        actual_counts = {"OPEN": 0, "CLOSED": 0}

        self.assertEqual(
            [frame["frame_idx"] for frame in self.keypoints["frames"]],
            [frame["frame_idx"] for frame in self.expected["frames"]],
        )

        for keypoint_frame, expected_frame in zip(
            self.keypoints["frames"], self.expected["frames"]
        ):
            expected_hands = {
                hand["hand_index"]: hand for hand in expected_frame["hands"]
            }
            self.assertEqual(len(keypoint_frame["hands"]), len(expected_hands))

            for hand in keypoint_frame["hands"]:
                hand_index = hand["hand_index"]
                with self.subTest(
                    frame_idx=keypoint_frame["frame_idx"], hand_index=hand_index
                ):
                    result = classify_anatomical_rules(
                        np.asarray(hand["joints_2d"], dtype=np.float64), config
                    )
                    expected_state = expected_hands[hand_index]["state"]["state"]
                    self.assertEqual(result["state"], expected_state)
                    actual_counts[result["state"]] += 1

        self.assertEqual(actual_counts, self.expected["state_counts"])


if __name__ == "__main__":
    unittest.main()
