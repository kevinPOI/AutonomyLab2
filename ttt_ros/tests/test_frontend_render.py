import importlib.util
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HAS_PYGAME = importlib.util.find_spec("pygame") is not None

import mock_ros
mock_ros.install()

import main
from ros_publisher import GridFrame, TOPIC


class TestFrontend(unittest.TestCase):
    @unittest.skipUnless(HAS_PYGAME, "pygame is required for frontend rendering tests")
    def test_scripted_game_publishes_and_renders(self):
        mock_ros.clear_published()
        grid = GridFrame()
        self.assertEqual(main.run(grid, headless_script=True,
                                  scripted_moves=[(1, 1), (0, 0), (2, 2), (0, 2), (2, 0)],
                                  max_frames=600), 0)

        log = [m for t, m in mock_ros.get_published() if t == TOPIC]
        self.assertGreater(len(log), 0)
        for msg in log:
            self.assertEqual(msg.header.frame_id, "base_link")
            self.assertAlmostEqual(msg.pose.orientation.w, 1.0)
            self.assertGreaterEqual(msg.pose.position.x, grid.origin_x - 1e-9)
            self.assertLessEqual(msg.pose.position.x, grid.origin_x + 2 * grid.cell_size + 1e-9)

        cx = grid.origin_x + grid.cell_size
        cy = grid.origin_y + grid.cell_size
        self.assertTrue(any(abs(m.pose.position.x - cx) < 1e-6 and
                            abs(m.pose.position.y - cy) < 1e-6 for m in log))

    def test_cell_pixel_round_trip(self):
        for r in range(3):
            for c in range(3):
                mx = main.MARGIN + c * main.CELL_PX + main.CELL_PX // 2
                my = main.STATUS_H + r * main.CELL_PX + main.CELL_PX // 2
                self.assertEqual(main.cell_from_pixel(mx, my), (r, c))
        self.assertIsNone(main.cell_from_pixel(1, 1))
        self.assertIsNone(main.cell_from_pixel(10000, 10000))


if __name__ == "__main__":
    unittest.main()
