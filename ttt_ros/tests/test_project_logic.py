import xml.etree.ElementTree as ET

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ttt_ros"))

import mock_ros
mock_ros.install()

import project
import RobotUtil as rt
from game_logic import EMPTY, HUMAN, ROBOT
from ros_publisher import BlockPosePublisher, GridFrame


class FakeSimulator:
    def __init__(self):
        self.supply = project.BlockSupply()
        self.executed = []
        self.reset_calls = 0

    def execute_pose(self, pose_msg, grid):
        row, col = grid.pose_to_cell(pose_msg)
        block_index, block_name = self.supply.allocate_next()
        self.executed.append({
            "row": row,
            "col": col,
            "block_index": block_index,
            "block_name": block_name,
        })
        return self.executed[-1]

    def reset(self):
        self.reset_calls += 1
        self.executed.clear()
        self.supply.reset()

    def sync(self):
        return None


class TestProjectMapping(unittest.TestCase):
    def test_pose_to_dropoff_mapping_covers_all_cells(self):
        grid = GridFrame()
        pub = BlockPosePublisher(grid=grid, init_node=False, topic="/test/project_mapping")
        for row in range(3):
            for col in range(3):
                msg = pub.make_pose(row, col, ROBOT)
                resolved_row, resolved_col, xyz = project.pose_to_dropoff_position(msg, grid)
                self.assertEqual((resolved_row, resolved_col), (row, col))
                self.assertEqual(tuple(xyz), tuple(project.DROPOFF_POSITIONS[2 - col][row]))

    def test_grid_center_and_bottom_center_match_requested_positions(self):
        center = project.cell_to_dropoff_position(1, 1)
        bottom_center = project.cell_to_dropoff_position(2, 1)
        self.assertAlmostEqual(center[0], 0.520)
        self.assertAlmostEqual(center[1], -0.021)
        self.assertAlmostEqual(center[2], 0.021)
        self.assertAlmostEqual(bottom_center[0], 0.622)
        self.assertAlmostEqual(bottom_center[1], -0.027)
        self.assertAlmostEqual(bottom_center[2], 0.021)


class TestBlockSupply(unittest.TestCase):
    def test_sequential_supply_and_reset(self):
        supply = project.BlockSupply()
        allocated = [supply.allocate_next() for _ in range(4)]
        self.assertEqual(
            allocated,
            list(enumerate(("Block", "Block_2", "Block_3", "Block_4"))),
        )
        with self.assertRaises(RuntimeError):
            supply.allocate_next()
        supply.reset()
        self.assertEqual(supply.next_block_index, 0)


class TestProjectApp(unittest.TestCase):
    def setUp(self):
        mock_ros.clear_published()
        self.grid = GridFrame()
        self.publisher = BlockPosePublisher(
            grid=self.grid,
            init_node=False,
            topic="/test/project_app",
        )
        self.simulator = FakeSimulator()
        self.app = project.ProjectApp(
            grid=self.grid,
            publisher=self.publisher,
            simulator=self.simulator,
            robot_delay_ms=0,
        )

    def _published(self):
        return [msg for topic, msg in mock_ros.get_published() if topic == "/test/project_app"]

    def test_only_robot_turns_publish_and_execute(self):
        self.assertTrue(self.app.handle_human_move(0, 0, 0))
        self.assertEqual(len(self._published()), 0)

        result = self.app.step(0)
        self.assertIsNotNone(result)
        self.assertEqual(len(self._published()), 1)
        self.assertEqual(len(self.simulator.executed), 1)
        self.assertEqual((result.row, result.col), (self.simulator.executed[0]["row"], self.simulator.executed[0]["col"]))
        self.assertEqual(self.simulator.executed[0]["block_name"], "Block")

    def test_reset_restores_board_and_supply(self):
        self.app.handle_human_move(0, 0, 0)
        self.app.step(0)
        self.assertNotEqual(self.app.controller.game.board, [[EMPTY] * 3 for _ in range(3)])
        self.assertEqual(self.simulator.supply.next_block_index, 1)

        self.app.reset()

        self.assertEqual(self.app.controller.game.board, [[EMPTY] * 3 for _ in range(3)])
        self.assertIsNone(self.app.controller.robot_started_ms)
        self.assertEqual(self.simulator.reset_calls, 1)
        self.assertEqual(self.simulator.supply.next_block_index, 0)

    def test_no_robot_move_after_game_over(self):
        self.app.controller.game.board = [
            [HUMAN, HUMAN, HUMAN],
            [ROBOT, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ]
        self.app.controller.game.turn = ROBOT
        self.assertIsNone(self.app.step(0))
        self.assertEqual(len(self._published()), 0)
        self.assertEqual(len(self.simulator.executed), 0)


class TestRobotUtil(unittest.TestCase):
    def test_add_free_block_keeps_requested_name(self):
        root = ET.Element("mujoco")
        ET.SubElement(root, "worldbody")
        tree = ET.ElementTree(root)
        rt.add_free_block_to_model(tree, "Block", [0, 0, 0], 20, [0.02, 0.02, 0.02], [1, 0, 0, 1], True)
        body = tree.getroot().find("worldbody/body")
        self.assertEqual(body.attrib["name"], "Block")


if __name__ == "__main__":
    unittest.main()
