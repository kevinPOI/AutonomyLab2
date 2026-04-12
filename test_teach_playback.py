"""
Tests for teach_playback.py

Mocks FrankaArm so these can run anywhere without the real robot.
Run:  python3 test_teach_playback.py
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np

# We need to mock frankapy before importing our module
sys.modules["frankapy"] = MagicMock()

from teach_playback import (
    save_sequence,
    load_sequence,
    playback,
    record_pose,
)


class FakeFrankaArm:
    """Lightweight stand-in for FrankaArm."""

    def __init__(self):
        self._joints = np.array([0.0, -0.7, 0.0, -2.15, 0.0, 1.57, 0.7])
        self.calls = []  # track method calls for assertions

    def get_joints(self):
        return self._joints.copy()

    def get_pose(self):
        return MagicMock()

    def reset_joints(self):
        self.calls.append("reset_joints")

    def open_gripper(self):
        self.calls.append("open_gripper")

    def close_gripper(self):
        self.calls.append("close_gripper")

    def goto_joints(self, joints):
        self.calls.append(("goto_joints", list(joints)))

    def run_guide_mode(self, duration, block=False):
        self.calls.append("run_guide_mode")

    def stop_skill(self):
        self.calls.append("stop_skill")


class TestSaveLoad(unittest.TestCase):
    """Test serialisation round-trip."""

    def test_save_and_load(self):
        actions = [
            {"type": "waypoint", "joints": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
            {"type": "grasp", "joints": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
            {"type": "place", "joints": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            save_sequence(actions, path)
            loaded = load_sequence(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0]["type"], "waypoint")
            self.assertEqual(loaded[1]["type"], "grasp")
            self.assertEqual(loaded[2]["type"], "place")
            np.testing.assert_allclose(loaded[0]["joints"], actions[0]["joints"])
        finally:
            os.unlink(path)

    def test_empty_sequence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_sequence([], path)
            loaded = load_sequence(path)
            self.assertEqual(loaded, [])
        finally:
            os.unlink(path)


class TestRecordPose(unittest.TestCase):
    def test_records_current_joints(self):
        fa = FakeFrankaArm()
        fa._joints = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        joints = record_pose(fa)
        self.assertEqual(joints, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def test_returns_list_not_ndarray(self):
        fa = FakeFrankaArm()
        joints = record_pose(fa)
        self.assertIsInstance(joints, list)


class TestPlayback(unittest.TestCase):
    """Verify that playback issues the right commands in the right order."""

    def test_empty_playback(self):
        fa = FakeFrankaArm()
        playback(fa, [])
        goto_calls = [c for c in fa.calls if isinstance(c, tuple) and c[0] == "goto_joints"]
        self.assertEqual(len(goto_calls), 0)

    def test_waypoint_only(self):
        fa = FakeFrankaArm()
        actions = [
            {"type": "waypoint", "joints": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
        ]
        playback(fa, actions)
        self.assertIn("reset_joints", fa.calls)
        goto_calls = [c for c in fa.calls if isinstance(c, tuple) and c[0] == "goto_joints"]
        self.assertEqual(len(goto_calls), 1)
        self.assertEqual(fa.calls.count("close_gripper"), 0)

    def test_grasp_closes_gripper(self):
        fa = FakeFrankaArm()
        actions = [
            {"type": "grasp", "joints": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
        ]
        playback(fa, actions)
        self.assertIn("close_gripper", fa.calls)

    def test_place_opens_gripper(self):
        fa = FakeFrankaArm()
        actions = [
            {"type": "place", "joints": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
        ]
        playback(fa, actions)
        # open_gripper called at start AND during place
        self.assertGreaterEqual(fa.calls.count("open_gripper"), 2)

    def test_full_pick_and_place_sequence(self):
        """Simulate: waypoint -> grasp -> waypoint -> place -> waypoint"""
        fa = FakeFrankaArm()
        j = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        actions = [
            {"type": "waypoint", "joints": j},
            {"type": "grasp", "joints": j},
            {"type": "waypoint", "joints": j},
            {"type": "place", "joints": j},
            {"type": "waypoint", "joints": j},
        ]
        playback(fa, actions)
        goto_calls = [c for c in fa.calls if isinstance(c, tuple) and c[0] == "goto_joints"]
        self.assertEqual(len(goto_calls), 5)
        self.assertEqual(fa.calls.count("close_gripper"), 1)
        self.assertEqual(fa.calls.count("open_gripper"), 2)

    def test_multi_pick_place_cycle(self):
        """Simulate the user's workflow: r-g-p-r-g-p-r (two pick-and-place cycles)."""
        fa = FakeFrankaArm()
        j = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        actions = [
            {"type": "waypoint", "joints": j},
            {"type": "grasp", "joints": j},
            {"type": "place", "joints": j},
            {"type": "waypoint", "joints": j},
            {"type": "grasp", "joints": j},
            {"type": "place", "joints": j},
            {"type": "waypoint", "joints": j},
        ]
        playback(fa, actions)
        goto_calls = [c for c in fa.calls if isinstance(c, tuple) and c[0] == "goto_joints"]
        self.assertEqual(len(goto_calls), 7)
        self.assertEqual(fa.calls.count("close_gripper"), 2)
        self.assertEqual(fa.calls.count("open_gripper"), 3)

    def test_playback_order(self):
        """Verify the exact call order for a single pick-and-place."""
        fa = FakeFrankaArm()
        j = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        actions = [
            {"type": "grasp", "joints": j},
            {"type": "place", "joints": j},
        ]
        playback(fa, actions)
        expected = [
            "reset_joints",
            "open_gripper",
            ("goto_joints", j),
            "close_gripper",
            ("goto_joints", j),
            "open_gripper",
            "reset_joints",
        ]
        for exp, actual in zip(expected, fa.calls):
            self.assertEqual(exp, actual)


class TestJsonRoundTrip(unittest.TestCase):
    """Make sure a recorded sequence survives save/load and plays back identically."""

    def test_roundtrip_preserves_actions(self):
        original = [
            {"type": "waypoint", "joints": [0.1, -0.7, 0.0, -2.15, 0.0, 1.57, 0.7]},
            {"type": "grasp", "joints": [0.3, 0.38, 0.027, -2.62, -0.026, 3.05, 0.47]},
            {"type": "waypoint", "joints": [-0.26, -0.06, 0.12, -2.75, -0.026, 2.76, 0.67]},
            {"type": "place", "joints": [-0.15, 0.17, 0.20, -2.70, -0.026, 2.96, 0.87]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_sequence(original, path)
            loaded = load_sequence(path)
            for orig, load in zip(original, loaded):
                self.assertEqual(orig["type"], load["type"])
                np.testing.assert_allclose(orig["joints"], load["joints"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
