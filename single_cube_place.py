from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project import MujocoTicTacToeSim


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Pick one cube from the supply area and place it at a target xyz position.",
    )
    parser.add_argument("x", type=float, help="Target x position in meters")
    parser.add_argument("y", type=float, help="Target y position in meters")
    parser.add_argument("z", type=float, help="Target z position in meters")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    simulator = MujocoTicTacToeSim()

    try:
        result = simulator.execute_xyz((args.x, args.y, args.z))
        simulator.sync()
        print(
            "[single_cube_place] completed "
            f"block={result['block_name']} "
            f"target=({result['target_xyz'][0]:.3f}, {result['target_xyz'][1]:.3f}, {result['target_xyz'][2]:.3f}) "
            f"orientation={result['orientation']}"
        )
        try:
            input("Placement complete. Press Enter to close the simulator...")
        except EOFError:
            pass
    finally:
        simulator.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
