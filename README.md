# Lab 3 - Teach & Playback for Block Arrangement

Teach-and-playback system for the Franka Panda. Manually guide the robot to poses, record them with single keypresses, then play the whole sequence back autonomously.

## Quick Start

```bash
# Inside the docker container
python3 teach_playback.py
```

The robot enters **guide mode** -- physically move it by hand (no wrist buttons needed).

## Controls

| Key     | Action |
|---------|--------|
| `r`     | Record current pose as a **waypoint** (move-only, no gripper action) |
| `g`     | Record current pose as a **grasp** point (move there, then close gripper) |
| `p`     | Record current pose as a **place** point (move there, then open gripper) |
| `u`     | Undo last recorded action |
| `d`     | Display all recorded actions |
| `SPACE` | Stop guide mode and **play back** the full sequence |
| `s`     | Save the current sequence to a `.json` file |
| `l`     | Load a saved `.json` file and play it back |
| `q`     | Quit without playing |

## Typical Workflow

To pick up a block and place it somewhere:

1. Move the robot above the block, press `r` (safe waypoint)
2. Move the robot down to the block, press `g` (gripper will close here)
3. Move the robot to the destination, press `p` (gripper will open here)
4. Repeat for each block
5. Press `s` to save, then `SPACE` to play back

For a full 3x3 pattern with 8 blocks, the sequence looks like:

```
r - g - p - r - g - p - r - g - p - r - g - p - ... - s - SPACE
```

## Saving & Loading

- `s` prompts for a filename (e.g., `pickaxe`) and saves to `pickaxe.json`
- `l` prompts for a filename and loads + plays it immediately
- Saved files live in the same directory as the script

Recommended: save one file per pattern (`pickaxe.json`, `axe.json`, `box.json`, `bow.json`).

## Running Tests

```bash
python3 test_teach_playback.py -v
```

Tests mock FrankaArm so they run anywhere without the robot.
