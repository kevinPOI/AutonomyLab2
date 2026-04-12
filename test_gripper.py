"""
Gripper hardware test - run this on the real robot inside the docker container.

Tests all three gripper control methods to diagnose what works:
  1. close_gripper() / open_gripper()
  2. goto_gripper(width)
  3. Gripper width readback

Usage (inside docker):
  python3 test_gripper.py
"""

import time
from frankapy import FrankaArm

if __name__ == "__main__":
    print("Initializing FrankaArm...")
    fa = FrankaArm()
    fa.reset_joints()

    # --- Test 1: open_gripper / close_gripper ---
    print("\n=== Test 1: open_gripper() ===")
    fa.open_gripper()
    time.sleep(1.0)
    width = fa.get_gripper_width()
    print(f"  Gripper width after open_gripper(): {width:.4f} m")

    print("\n=== Test 2: close_gripper() ===")
    fa.close_gripper()
    time.sleep(1.0)
    width = fa.get_gripper_width()
    print(f"  Gripper width after close_gripper(): {width:.4f} m")

    print("\n=== Test 3: open_gripper() again ===")
    fa.open_gripper()
    time.sleep(1.0)
    width = fa.get_gripper_width()
    print(f"  Gripper width after open_gripper(): {width:.4f} m")

    # --- Test 2: goto_gripper with specific widths ---
    print("\n=== Test 4: goto_gripper(0.04) -- half open ===")
    fa.goto_gripper(0.04)
    time.sleep(1.0)
    width = fa.get_gripper_width()
    print(f"  Gripper width after goto_gripper(0.04): {width:.4f} m")

    print("\n=== Test 5: goto_gripper(0.01) -- nearly closed ===")
    fa.goto_gripper(0.01)
    time.sleep(1.0)
    width = fa.get_gripper_width()
    print(f"  Gripper width after goto_gripper(0.01): {width:.4f} m")

    print("\n=== Test 6: goto_gripper(0.08) -- fully open ===")
    fa.goto_gripper(0.08)
    time.sleep(1.0)
    width = fa.get_gripper_width()
    print(f"  Gripper width after goto_gripper(0.08): {width:.4f} m")

    # --- Summary ---
    print("\n=== Done ===")
    print("If none of the above moved the gripper, it's a hardware/server issue.")
    print("If only goto_gripper worked, use that instead of open/close.")
    print("If only open/close worked, use those.")
    fa.reset_joints()
