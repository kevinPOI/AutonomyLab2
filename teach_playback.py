"""
Teach-and-Playback for Franka Panda - Lab 3 Block Arrangement

Usage:
  1. Run: python3 teach_playback.py
  2. The robot enters guide mode - physically move it by hand.
  3. Press keys to record actions:
       r  - Record current pose as a waypoint (move-only)
       g  - Record current pose as a GRASP point (close gripper NOW + record)
       p  - Record current pose as a PLACE point (open gripper NOW + record)
       u  - Undo last recorded action
       d  - Display all recorded actions
       SPACE - Stop recording and play back the full sequence
       s  - Save the current sequence to a JSON file
       l  - Load a sequence from a JSON file and play it back
       q  - Quit without playing
"""

import numpy as np
import json
import time
import sys
import os
import termios
import tty

# ---------------------------------------------------------------------------
# Keyboard helpers (non-blocking single keypress without requiring Enter)
# ---------------------------------------------------------------------------

def get_key():
    """Read a single keypress from stdin (no Enter needed)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# ---------------------------------------------------------------------------
# Action recording / serialisation
# ---------------------------------------------------------------------------

def record_pose(fa):
    """Capture the robot's current joint angles."""
    return fa.get_joints().tolist()


def save_sequence(actions, filepath):
    """Save an action sequence to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(actions, f, indent=2)
    print(f"\nSaved {len(actions)} actions to {filepath}")


def load_sequence(filepath):
    """Load an action sequence from a JSON file."""
    with open(filepath, "r") as f:
        actions = json.load(f)
    print(f"\nLoaded {len(actions)} actions from {filepath}")
    return actions


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def playback(fa, actions):
    """Execute a recorded action sequence on the robot."""
    if not actions:
        print("\nNo actions to play back.")
        return

    print(f"\nPlaying back {len(actions)} actions...")
    fa.reset_joints()
    fa.open_gripper()
    time.sleep(0.5)

    for i, action in enumerate(actions):
        atype = action["type"]
        joints = action["joints"]
        print(f"  [{i+1}/{len(actions)}] {atype.upper()} -> joints {np.round(joints, 3).tolist()}")

        fa.goto_joints(joints)

        if atype == "grasp":
            time.sleep(0.3)
            fa.close_gripper()
            time.sleep(0.5)
        elif atype == "place":
            time.sleep(0.3)
            fa.open_gripper()
            time.sleep(0.5)

    print("\nPlayback complete. Resetting to home.")
    fa.reset_joints()


# ---------------------------------------------------------------------------
# Main teach loop
# ---------------------------------------------------------------------------

def teach_loop(fa):
    """Enter guide mode and let the user record poses interactively."""
    actions = []

    print("\n--- Guide mode active. Move the robot by hand. ---")
    print("Keys:  r=waypoint  g=grasp  p=place  u=undo  d=display  SPACE=play  s=save  l=load+play  q=quit\n")

    fa.run_guide_mode(10000, block=False)

    try:
        while True:
            key = get_key()

            if key == "r":
                joints = record_pose(fa)
                actions.append({"type": "waypoint", "joints": joints})
                print(f"  [WAYPOINT #{len(actions)}] recorded")

            elif key == "g":
                joints = record_pose(fa)
                # Stop guide mode, close gripper, re-enter guide mode
                fa.stop_skill()
                time.sleep(0.5)
                print(f"  Closing gripper...")
                fa.close_gripper()
                time.sleep(0.5)
                actions.append({"type": "grasp", "joints": joints})
                print(f"  [GRASP    #{len(actions)}] recorded -- gripper closed")
                fa.run_guide_mode(10000, block=False)

            elif key == "p":
                joints = record_pose(fa)
                # Stop guide mode, open gripper, re-enter guide mode
                fa.stop_skill()
                time.sleep(0.5)
                print(f"  Opening gripper...")
                fa.open_gripper()
                time.sleep(0.5)
                actions.append({"type": "place", "joints": joints})
                print(f"  [PLACE    #{len(actions)}] recorded -- gripper opened")
                fa.run_guide_mode(10000, block=False)

            elif key == "u":
                if actions:
                    removed = actions.pop()
                    print(f"  Undid {removed['type']} (now {len(actions)} actions)")
                else:
                    print("  Nothing to undo.")

            elif key == "d":
                if not actions:
                    print("  (no actions recorded yet)")
                else:
                    print(f"  --- {len(actions)} recorded actions ---")
                    for i, a in enumerate(actions):
                        print(f"    {i+1}. {a['type']}")

            elif key == " ":
                print("\nStopping guide mode for playback...")
                fa.stop_skill()
                time.sleep(1.0)
                playback(fa, actions)
                return actions

            elif key == "s":
                if not actions:
                    print("  Nothing to save.")
                    continue
                fa.stop_skill()
                # Switch back to cooked mode briefly for input()
                print()
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                filename = input("  Filename (without .json): ").strip()
                if not filename:
                    filename = "sequence"
                filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{filename}.json")
                save_sequence(actions, filepath)
                # Re-enter guide mode
                fa.run_guide_mode(10000, block=False)
                print("\n--- Guide mode re-activated. Keep recording or press SPACE to play. ---\n")

            elif key == "l":
                fa.stop_skill()
                print()
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                filename = input("  Filename to load (without .json): ").strip()
                if not filename:
                    filename = "sequence"
                filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{filename}.json")
                if not os.path.exists(filepath):
                    print(f"  File not found: {filepath}")
                    fa.run_guide_mode(10000, block=False)
                    continue
                loaded = load_sequence(filepath)
                time.sleep(1.0)
                playback(fa, loaded)
                return loaded

            elif key == "q":
                print("\nQuitting without playback.")
                fa.stop_skill()
                return actions

    except KeyboardInterrupt:
        print("\nInterrupted. Stopping guide mode.")
        fa.stop_skill()
        return actions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from frankapy import FrankaArm

    print("Initializing FrankaArm...")
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()

    teach_loop(fa)
    print("Done.")
