import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mock_ros
mock_ros.install()

from game_logic import EMPTY, HUMAN, ROBOT, TicTacToe
from ros_publisher import BlockPosePublisher, GridFrame, TOPIC


class TestGameLogic(unittest.TestCase):
    def test_place_and_turn(self):
        g = TicTacToe()
        self.assertTrue(g.place(0, 0, HUMAN))
        self.assertEqual(g.board[0][0], HUMAN)
        self.assertEqual(g.turn, ROBOT)
        self.assertFalse(g.place(0, 0, ROBOT))

    def test_row_win(self):
        g = TicTacToe()
        g.board[0] = [HUMAN, HUMAN, HUMAN]
        self.assertEqual(g.winner(), HUMAN)

    def test_col_win(self):
        g = TicTacToe()
        for r in range(3):
            g.board[r][2] = ROBOT
        self.assertEqual(g.winner(), ROBOT)

    def test_diag_win(self):
        g = TicTacToe()
        g.board[0][0] = g.board[1][1] = g.board[2][2] = ROBOT
        self.assertEqual(g.winner(), ROBOT)

    def test_draw(self):
        g = TicTacToe()
        g.board = [
            [HUMAN, ROBOT, HUMAN],
            [HUMAN, ROBOT, ROBOT],
            [ROBOT, HUMAN, HUMAN],
        ]
        self.assertIsNone(g.winner())
        self.assertTrue(g.is_over())

    def test_ai_blocks_immediate_threat(self):
        g = TicTacToe()
        g.board[0][0] = HUMAN
        g.board[0][1] = HUMAN
        g.board[1][0] = ROBOT
        g.turn = ROBOT
        self.assertEqual(g.best_move(ROBOT), (0, 2))

    def test_ai_takes_the_win(self):
        g = TicTacToe()
        g.board[0][0] = ROBOT
        g.board[0][1] = ROBOT
        g.board[1][1] = HUMAN
        g.turn = ROBOT
        self.assertEqual(g.best_move(ROBOT), (0, 2))

    def test_ai_never_loses_from_corner_opening(self):
        g = TicTacToe()
        g.place(0, 0, HUMAN)
        while not g.is_over():
            g.place(*g.best_move(g.turn), g.turn)
        self.assertNotEqual(g.winner(), HUMAN)


class TestPosePublisher(unittest.TestCase):
    def setUp(self):
        mock_ros.clear_published()
        self.grid = GridFrame(0.40, 0.0, 0.06, 0.05, "base_link")
        self.pub = BlockPosePublisher(grid=self.grid)

    def test_origin_cell(self):
        msg = self.pub.make_pose(0, 0, HUMAN)
        self.assertAlmostEqual(msg.pose.position.x, 0.40)
        self.assertAlmostEqual(msg.pose.position.y, 0.00)
        self.assertAlmostEqual(msg.pose.position.z, 0.05)
        self.assertEqual(msg.header.frame_id, "base_link")
        self.assertAlmostEqual(msg.pose.orientation.w, 1.0)

    def test_cell_offsets(self):
        m = self.pub.make_pose(2, 1, ROBOT)
        self.assertAlmostEqual(m.pose.position.x, 0.52)
        self.assertAlmostEqual(m.pose.position.y, 0.06)

    def test_publish_records_and_increments_seq(self):
        self.pub.publish_move(1, 2, HUMAN)
        self.pub.publish_move(0, 0, ROBOT)
        log = [m for t, m in mock_ros.get_published() if t == TOPIC]
        self.assertGreaterEqual(len(log), 2)
        self.assertEqual(log[-1].header.seq, log[-2].header.seq + 1)

    def test_every_cell_maps_correctly(self):
        for r in range(3):
            for c in range(3):
                m = self.pub.make_pose(r, c, HUMAN)
                self.assertAlmostEqual(m.pose.position.x, 0.40 + r * 0.06)
                self.assertAlmostEqual(m.pose.position.y, 0.00 + c * 0.06)

    def test_cell_round_trip(self):
        for r in range(3):
            for c in range(3):
                msg = self.pub.make_pose(r, c, ROBOT)
                self.assertEqual(self.grid.pose_to_cell(msg), (r, c))

    def test_xy_to_cell_rejects_outside_point(self):
        with self.assertRaises(ValueError):
            self.grid.xy_to_cell(10.0, 10.0)


class TestFullGamePublishes(unittest.TestCase):
    def test_one_pose_per_move(self):
        mock_ros.clear_published()
        pub = BlockPosePublisher(topic="/test/full_game")
        g = TicTacToe()
        moves = 0
        g.place(0, 0, HUMAN)
        pub.publish_move(0, 0, HUMAN)
        moves += 1
        while not g.is_over():
            mover = g.turn
            mv = g.best_move(mover)
            g.place(*mv, mover)
            pub.publish_move(*mv, mover)
            moves += 1
        log = [m for t, m in mock_ros.get_published() if t == "/test/full_game"]
        self.assertEqual(len(log), moves)


if __name__ == "__main__":
    unittest.main()
