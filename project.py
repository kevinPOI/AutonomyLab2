from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import RobotUtil as rt
from kinematics import FrankArm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ttt_ros.game_logic import HUMAN, ROBOT, TicTacToe
from ttt_ros.ros_publisher import BlockPosePublisher, GridFrame, USING_REAL_ROS

WINDOW = 920
MARGIN = 58
CELL_PX = 148
STATUS_H = 146
WINDOW_H = STATUS_H + 3 * CELL_PX + MARGIN
BOARD_LEFT = MARGIN
BOARD_TOP = STATUS_H
BOARD_SIZE = 3 * CELL_PX
BOARD_PANEL_PAD = 18
SIDEBAR_GAP = 30
SIDEBAR_LEFT = BOARD_LEFT + BOARD_SIZE + SIDEBAR_GAP
SIDEBAR_WIDTH = WINDOW - SIDEBAR_LEFT - MARGIN
HEADER_TOP = 26
HEADER_HEIGHT = 84
LINE_W = 8

BG_TOP = (245, 238, 229)
BG_BOTTOM = (226, 235, 241)
TITLE_COLOR = (34, 42, 58)
TEXT_PRIMARY = (62, 71, 84)
TEXT_MUTED = (108, 116, 128)
PANEL_BG = (251, 248, 243)
PANEL_BORDER = (214, 206, 194)
STATUS_PANEL_BG = (37, 50, 71)
STATUS_PANEL_BORDER = (70, 88, 118)
BOARD_BG = (248, 243, 236)
CELL_BG = (255, 252, 248)
GRID_COLOR = (64, 73, 82)
X_COLOR = (44, 103, 205)
O_COLOR = (204, 79, 64)
ACCENT_COLOR = (224, 167, 93)
KEYCAP_BG = (244, 239, 231)
KEYCAP_BORDER = (213, 202, 188)
KEYCAP_TEXT = (52, 60, 72)
SHADOW_RGBA = (16, 24, 40, 30)
ROBOT_DELAY_MS = 350

ROOT_MODEL_XML = ROOT_DIR / "franka_emika_panda/panda_torque_table.xml"
ROOT_MODEL_DIR = ROOT_MODEL_XML.parent
ASSET_DIR = ROOT_MODEL_DIR / "assets"

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0125
SEGMENT_DURATION_BASE = 0.3
SEGMENT_DURATION_PER_METER = 2.6
HOLD_DURATION_BASE = 0.05
HOLD_DURATION_PER_METER = 1.0
RENDER_SKIP = 5

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float) * 1.2
KD = np.array([18, 18, 12, 8, 6, 4, 3], dtype=float) * 2.5
Q_HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8], dtype=float)

END_OF_TABLE = 0.55 + 0.135 + 0.05
DOWN_GRASP_Y_RPY = (3.13714399, 0.00419659, -0.80013718 + np.pi / 2.0)
DOWN_GRASP_X_RPY = (3.13714399, 0.00419659, -0.80013718)
FREE_BLOCK_SIZE = [0.02, 0.02, 0.02]
BLOCK_DENSITY = 20
TABLE_RGBA = [1.0, 1.0, 1.0, 1.0]
GRID_RGBA = [0.1, 0.1, 0.1, 1.0]
GRID_LINE_RADIUS = 0.0035
HUMAN_BLOCK_SIZE = [0.02, 0.02, 0.02]
HUMAN_BLOCK_RGBA = [0.85, 0.1, 0.1, 1.0]

BLOCK_POSE_1 = [0.438, 0.288, 0.05]
BLOCK_POSE_2 = [0.438, 0.174, 0.05]
BLOCK_POSE_3 = [0.576, 0.174, 0.05]
BLOCK_POSE_4 = [0.576, 0.288, 0.05]

SUPPLY_BLOCKS = (
    {"name": "Block", "pos": BLOCK_POSE_1, "rgba": [0.0, 0.2, 0.9, 1.0]},
    {"name": "Block_2", "pos": BLOCK_POSE_2, "rgba": [0.0, 0.2, 0.9, 1.0]},
    {"name": "Block_3", "pos": BLOCK_POSE_3, "rgba": [0.0, 0.2, 0.9, 1.0]},
    {"name": "Block_4", "pos": BLOCK_POSE_4, "rgba": [0.0, 0.2, 0.9, 1.0]},
)

BLOCK_NAMES = tuple(block["name"] for block in SUPPLY_BLOCKS)
SUPPLY_POSITIONS = tuple(tuple(block["pos"]) for block in SUPPLY_BLOCKS)
HUMAN_BLOCK_NAMES = tuple(f"HumanBlock_{idx}" for idx in range(1, 6))

GRID_CENTER = np.array([0.520, -0.021, 0.05], dtype=float)
GRID_BOTTOM_CENTER = np.array([0.622, -0.027, 0.05], dtype=float)
GRID_LOGICAL_ROW_STEP = GRID_BOTTOM_CENTER - GRID_CENTER

GRID_LOGICAL_COL_STEP = np.array([0.0, 0.075, 0.0], dtype=float)


def _make_dropoff_positions():
    positions = []
    for sim_row in range(3):
        row_positions = []
        for sim_col in range(3):
            logical_row_offset = sim_col - 1
            logical_col_offset = 1 - sim_row
            xyz = (
                GRID_CENTER
                + logical_row_offset * GRID_LOGICAL_ROW_STEP
                + logical_col_offset * GRID_LOGICAL_COL_STEP
            )
            row_positions.append(tuple(float(v) for v in xyz))
        positions.append(tuple(row_positions))
    return tuple(positions)


DROPOFF_POSITIONS = _make_dropoff_positions()
HUMAN_BLOCK_PARK_POSITIONS = tuple((0.22 + 0.06 * idx, -0.42, 0.05) for idx in range(len(HUMAN_BLOCK_NAMES)))

STATIC_BLOCKS = [
    ["TablePlane", [END_OF_TABLE - 0.275, 0.0, -0.005], [0.275, 0.504, 0.0051]],
]


def _load_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            f"pygame is required to run project.py with interpreter: {sys.executable}"
        ) from exc
    return pygame


def gripper_cmd(g):
    return GRIPPER_OPEN if g <= 0.5 else GRIPPER_CLOSED


def waypoint_duration(distance, base_duration, distance_scale):
    return base_duration + distance_scale * distance


def _unwrap(q_new, q_ref):
    q = np.array(q_new, dtype=float)
    r = np.array(q_ref, dtype=float)
    q -= np.round((q - r) / (2 * np.pi)) * (2 * np.pi)
    return q


def _pace(wall_t0, sim_t):
    drift = sim_t - (time.time() - wall_t0)
    if drift > 0:
        time.sleep(drift)


def add_static_box_geom(tree, name, pos, size, rgba):
    worldbody = tree.getroot().find("worldbody")
    body = ET.SubElement(worldbody, "body", {
        "name": name,
        "pos": f"{pos[0]} {pos[1]} {pos[2]}",
    })
    ET.SubElement(body, "geom", {
        "type": "box",
        "size": f"{size[0]} {size[1]} {size[2]}",
        "rgba": f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
        "contype": "0",
        "conaffinity": "0",
    })


def add_static_capsule_line(tree, name, start_xyz, end_xyz, radius, rgba):
    worldbody = tree.getroot().find("worldbody")
    body = ET.SubElement(worldbody, "body", {"name": name})
    ET.SubElement(body, "geom", {
        "type": "capsule",
        "fromto": (
            f"{start_xyz[0]} {start_xyz[1]} {start_xyz[2]} "
            f"{end_xyz[0]} {end_xyz[1]} {end_xyz[2]}"
        ),
        "size": str(radius),
        "rgba": f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
        "contype": "0",
        "conaffinity": "0",
    })


def board_cell_position(row, col):
    return np.array(cell_to_dropoff_position(row, col), dtype=float)


def build_grid_line_segments():
    z = GRID_CENTER[2] - HUMAN_BLOCK_SIZE[2] + 0.002
    row_step = board_cell_position(2, 1) - board_cell_position(1, 1)
    col_step = board_cell_position(1, 2) - board_cell_position(1, 1)
    center = board_cell_position(1, 1)

    row_half_span = 1.5 * np.linalg.norm(col_step)
    col_half_span = 1.5 * np.linalg.norm(row_step)
    row_dir = col_step / np.linalg.norm(col_step)
    col_dir = row_step / np.linalg.norm(row_step)

    segments = []
    for offset in (-0.5, 0.5):
        line_center = center + offset * row_step
        start = line_center - row_half_span * row_dir
        end = line_center + row_half_span * row_dir
        start[2] = z
        end[2] = z
        segments.append((start, end))

    for offset in (-0.5, 0.5):
        line_center = center + offset * col_step
        start = line_center - col_half_span * col_dir
        end = line_center + col_half_span * col_dir
        start[2] = z
        end[2] = z
        segments.append((start, end))

    return segments


def cell_to_dropoff_position(row, col):
    # The MuJoCo board frame is rotated 90 deg clockwise relative to the UI.
    # Map logical cells into the simulation frame so both boards line up visually.
    sim_row = 2 - col
    sim_col = row
    return DROPOFF_POSITIONS[sim_row][sim_col]


def pose_to_dropoff_position(pose_msg, grid):
    row, col = grid.pose_to_cell(pose_msg)
    return row, col, cell_to_dropoff_position(row, col)


def build_pick_and_place_waypoints(source_xyz, target_xyz, current_xyz, grasp_orientation, place_orientation):
    current_xyz = np.array(current_xyz, dtype=float)
    sx, sy, sz = source_xyz
    tx, ty, tz = target_xyz
    mx, my, _ = 0.6 * np.array(source_xyz, dtype=float) + 0.4 * np.array(target_xyz, dtype=float)
    transfer_z = max(sz, tz) + 0.20
    return [
        ("rotate_to_grasp_orientation", [0.0, current_xyz[0], current_xyz[1], current_xyz[2], *grasp_orientation]),
        ("descend_to_grasp_pre", [0.0, sx, sy, sz + 0.13, *grasp_orientation]),
        ("descend_to_grasp", [0.0, sx, sy, sz + 0.07, *grasp_orientation]),
        ("close_gripper", [1.0, sx, sy, sz + 0.07, *grasp_orientation]),
        ("lift_from_supply", [1.0, sx, sy, sz + 0.15, *grasp_orientation]),
        ("move_halfway", [1.0, mx, my, transfer_z, *place_orientation]),
        ("move_above_target", [1.0, tx, ty, tz + 0.15, *place_orientation]),
        ("lower_to_target", [1.0, tx, ty, tz + 0.07, *place_orientation]),
        ("open_gripper", [0.0, tx, ty, tz + 0.07, *place_orientation]),
        ("retract_from_target", [0.0, tx, ty, tz + 0.15, *place_orientation]),
    ]


def cell_from_pixel(mx, my):
    if not (BOARD_LEFT <= mx < BOARD_LEFT + BOARD_SIZE):
        return None
    if not (BOARD_TOP <= my < BOARD_TOP + BOARD_SIZE):
        return None
    return (my - BOARD_TOP) // CELL_PX, (mx - BOARD_LEFT) // CELL_PX


def status_text(game):
    w = game.winner()
    if w == HUMAN:
        return "You win. Press A for an animated reset or R for a quick reset."
    if w == ROBOT:
        return "Robot wins. Press A to restore the scene or R to reset instantly."
    if game.is_full():
        return "Draw game. Press A to restore the scene or R to reset instantly."
    if game.turn == HUMAN:
        return f"Your turn ({HUMAN}). Click an empty cell to place a red block."
    return "Robot is planning the next blue block."


def _lerp_color(color_a, color_b, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(color_a, color_b))


def _wrap_text(text, font, max_width):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if font.size(trial)[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_gradient_background(screen, pygame_mod):
    for y in range(WINDOW_H):
        blend = y / max(1, WINDOW_H - 1)
        color = _lerp_color(BG_TOP, BG_BOTTOM, blend)
        pygame_mod.draw.line(screen, color, (0, y), (WINDOW, y))

    glow = pygame_mod.Surface((WINDOW, WINDOW_H), pygame_mod.SRCALPHA)
    pygame_mod.draw.circle(glow, (45, 103, 205, 34), (150, 110), 180)
    pygame_mod.draw.circle(glow, (204, 79, 64, 28), (WINDOW - 140, WINDOW_H - 120), 220)
    pygame_mod.draw.circle(glow, (224, 167, 93, 24), (WINDOW - 90, 120), 120)
    screen.blit(glow, (0, 0))


def _draw_panel(screen, rect, fill, border, pygame_mod):
    shadow = pygame_mod.Surface((rect.width, rect.height), pygame_mod.SRCALPHA)
    pygame_mod.draw.rect(shadow, SHADOW_RGBA, shadow.get_rect(), border_radius=24)
    screen.blit(shadow, (rect.x, rect.y + 8))
    pygame_mod.draw.rect(screen, fill, rect, border_radius=24)
    pygame_mod.draw.rect(screen, border, rect, width=2, border_radius=24)


def _draw_piece_mark(screen, piece, center, radius, color, stroke_width, pygame_mod):
    cx, cy = center
    if piece == ROBOT:
        pygame_mod.draw.line(screen, color, (cx - radius, cy - radius), (cx + radius, cy + radius), stroke_width)
        pygame_mod.draw.line(screen, color, (cx + radius, cy - radius), (cx - radius, cy + radius), stroke_width)
    elif piece == HUMAN:
        pygame_mod.draw.circle(screen, color, (cx, cy), radius, stroke_width)


def _draw_key_hint(screen, x, y, key_text, description, fonts, pygame_mod):
    key_width = max(58, fonts["key"].size(key_text)[0] + 24)
    key_rect = pygame_mod.Rect(x, y, key_width, 34)
    pygame_mod.draw.rect(screen, KEYCAP_BG, key_rect, border_radius=12)
    pygame_mod.draw.rect(screen, KEYCAP_BORDER, key_rect, width=2, border_radius=12)
    key_label = fonts["key"].render(key_text, True, KEYCAP_TEXT)
    screen.blit(key_label, key_label.get_rect(center=key_rect.center))
    desc_surface = fonts["body"].render(description, True, TEXT_PRIMARY)
    screen.blit(desc_surface, (key_rect.right + 14, y + 7))


def _status_badge(game):
    w = game.winner()
    if w == HUMAN:
        return "YOU WIN", O_COLOR
    if w == ROBOT:
        return "ROBOT WINS", X_COLOR
    if game.is_full():
        return "DRAW", ACCENT_COLOR
    if game.turn == HUMAN:
        return "YOUR TURN", O_COLOR
    return "ROBOT TURN", X_COLOR


def make_ui_fonts(pygame_mod):
    return {
        "title": pygame_mod.font.SysFont("DejaVu Serif", 34, bold=True),
        "subtitle": pygame_mod.font.SysFont("DejaVu Sans", 18),
        "card_title": pygame_mod.font.SysFont("DejaVu Sans", 21, bold=True),
        "body": pygame_mod.font.SysFont("DejaVu Sans", 20),
        "small": pygame_mod.font.SysFont("DejaVu Sans", 16),
        "key": pygame_mod.font.SysFont("DejaVu Sans Mono", 18, bold=True),
        "badge": pygame_mod.font.SysFont("DejaVu Sans", 16, bold=True),
    }


def draw_board(screen, game, status, fonts, pygame_mod):
    _draw_gradient_background(screen, pygame_mod)

    header_rect = pygame_mod.Rect(MARGIN, HEADER_TOP, WINDOW - 2 * MARGIN, HEADER_HEIGHT)
    board_panel_rect = pygame_mod.Rect(
        BOARD_LEFT - BOARD_PANEL_PAD,
        BOARD_TOP - BOARD_PANEL_PAD,
        BOARD_SIZE + 2 * BOARD_PANEL_PAD,
        BOARD_SIZE + 2 * BOARD_PANEL_PAD,
    )
    board_rect = pygame_mod.Rect(BOARD_LEFT, BOARD_TOP, BOARD_SIZE, BOARD_SIZE)
    status_rect = pygame_mod.Rect(SIDEBAR_LEFT, BOARD_TOP - 2, SIDEBAR_WIDTH, 140)
    controls_rect = pygame_mod.Rect(SIDEBAR_LEFT, status_rect.bottom + 18, SIDEBAR_WIDTH, 204)
    legend_rect = pygame_mod.Rect(SIDEBAR_LEFT, controls_rect.bottom + 18, SIDEBAR_WIDTH, 124)

    _draw_panel(screen, header_rect, PANEL_BG, PANEL_BORDER, pygame_mod)
    _draw_panel(screen, board_panel_rect, PANEL_BG, PANEL_BORDER, pygame_mod)
    _draw_panel(screen, status_rect, STATUS_PANEL_BG, STATUS_PANEL_BORDER, pygame_mod)
    _draw_panel(screen, controls_rect, PANEL_BG, PANEL_BORDER, pygame_mod)
    _draw_panel(screen, legend_rect, PANEL_BG, PANEL_BORDER, pygame_mod)

    title = fonts["title"].render("Robot Tic-Tac-Toe Arena", True, TITLE_COLOR)
    subtitle = fonts["subtitle"].render(
        "",
        True,
        TEXT_MUTED,
    )
    screen.blit(title, (header_rect.x + 26, header_rect.y + 14))
    screen.blit(subtitle, (header_rect.x + 28, header_rect.y + 54))

    board_label = fonts["small"].render("PHYSICAL BOARD", True, TEXT_MUTED)
    screen.blit(board_label, (board_panel_rect.x + 20, board_panel_rect.y + 16))
    pygame_mod.draw.rect(screen, BOARD_BG, board_rect, border_radius=28)

    cell_gap = 10
    for row in range(3):
        for col in range(3):
            cell_rect = pygame_mod.Rect(
                BOARD_LEFT + col * CELL_PX + cell_gap // 2,
                BOARD_TOP + row * CELL_PX + cell_gap // 2,
                CELL_PX - cell_gap,
                CELL_PX - cell_gap,
            )
            pygame_mod.draw.rect(screen, CELL_BG, cell_rect, border_radius=20)

    for i in range(4):
        x = BOARD_LEFT + i * CELL_PX
        pygame_mod.draw.line(screen, GRID_COLOR, (x, BOARD_TOP), (x, BOARD_TOP + BOARD_SIZE), LINE_W)
        y = BOARD_TOP + i * CELL_PX
        pygame_mod.draw.line(screen, GRID_COLOR, (BOARD_LEFT, y), (BOARD_LEFT + BOARD_SIZE, y), LINE_W)

    pad = CELL_PX // 5
    stroke = 10
    for row in range(3):
        for col in range(3):
            cell = game.board[row][col]
            cx = BOARD_LEFT + col * CELL_PX + CELL_PX // 2
            cy = BOARD_TOP + row * CELL_PX + CELL_PX // 2
            color = O_COLOR if cell == HUMAN else X_COLOR
            _draw_piece_mark(screen, cell, (cx, cy), CELL_PX // 2 - pad, color, stroke, pygame_mod)

    pygame_mod.draw.rect(screen, GRID_COLOR, board_rect, width=3, border_radius=28)

    badge_text, badge_color = _status_badge(game)
    badge_surface = fonts["badge"].render(badge_text, True, (255, 255, 255))
    badge_rect = pygame_mod.Rect(status_rect.x + 24, status_rect.y + 18, badge_surface.get_width() + 24, 32)
    pygame_mod.draw.rect(screen, badge_color, badge_rect, border_radius=16)
    screen.blit(badge_surface, badge_surface.get_rect(center=badge_rect.center))
    status_title = fonts["card_title"].render("Match status", True, (247, 250, 255))
    screen.blit(status_title, (status_rect.x + 24, status_rect.y + 60))

    status_lines = _wrap_text(status, fonts["body"], status_rect.width - 48)
    for idx, line in enumerate(status_lines[:3]):
        line_surface = fonts["body"].render(line, True, (233, 238, 245))
        screen.blit(line_surface, (status_rect.x + 24, status_rect.y + 92 + idx * 24))

    controls_title = fonts["card_title"].render("Controls", True, TITLE_COLOR)
    #controls_note = fonts["small"].render("Use A when you want the scene to restore with motion.", True, TEXT_MUTED)
    screen.blit(controls_title, (controls_rect.x + 24, controls_rect.y + 18))
    #screen.blit(controls_note, (controls_rect.x + 24, controls_rect.y + 50))
    _draw_key_hint(screen, controls_rect.x + 24, controls_rect.y + 80, "Click", "Place a red block", fonts, pygame_mod)
    _draw_key_hint(screen, controls_rect.x + 24, controls_rect.y + 122, "R", "Reset immediately", fonts, pygame_mod)
    _draw_key_hint(screen, controls_rect.x + 24, controls_rect.y + 164, "A", "Restore", fonts, pygame_mod)
    _draw_key_hint(screen, controls_rect.x + 184, controls_rect.y + 164, "Esc", "Quit", fonts, pygame_mod)

    legend_title = fonts["card_title"].render("Pieces", True, TITLE_COLOR)
    #legend_note = fonts["small"].render("Red is the human player, blue is the robot player.", True, TEXT_MUTED)
    screen.blit(legend_title, (legend_rect.x + 24, legend_rect.y + 18))
    #screen.blit(legend_note, (legend_rect.x + 24, legend_rect.y + 50))

    human_center = (legend_rect.x + 46, legend_rect.y + 88)
    robot_center = (legend_rect.x + 200, legend_rect.y + 88)
    _draw_piece_mark(screen, HUMAN, human_center, 18, O_COLOR, 6, pygame_mod)
    _draw_piece_mark(screen, ROBOT, robot_center, 18, X_COLOR, 6, pygame_mod)
    screen.blit(fonts["body"].render("You", True, TEXT_PRIMARY), (human_center[0] + 28, human_center[1] - 11))
    screen.blit(fonts["body"].render("Robot", True, TEXT_PRIMARY), (robot_center[0] + 28, robot_center[1] - 11))


@dataclass
class BlockSupply:
    block_names: tuple[str, ...] = BLOCK_NAMES
    next_block_index: int = 0

    def allocate_next(self):
        if self.next_block_index >= len(self.block_names):
            raise RuntimeError("no remaining simulation blocks to place")
        block_index = self.next_block_index
        block_name = self.block_names[block_index]
        self.next_block_index += 1
        return block_index, block_name

    def reset(self):
        self.next_block_index = 0


@dataclass
class RobotTurnResult:
    row: int
    col: int
    pose: object


class ProjectController:
    def __init__(self, grid=None, publisher=None, robot_delay_ms=ROBOT_DELAY_MS):
        self.grid = grid or GridFrame()
        self.publisher = publisher or BlockPosePublisher(
            grid=self.grid,
            node_name="tic_tac_toe_project_node",
        )
        self.robot_delay_ms = robot_delay_ms
        self.game = TicTacToe()
        self.robot_started_ms = None

    def reset(self):
        self.game.reset()
        self.robot_started_ms = None

    def handle_human_move(self, row, col, now_ms):
        if self.game.turn != HUMAN or self.game.is_over():
            return False
        if not self.game.place(row, col, HUMAN):
            return False
        self.robot_started_ms = now_ms
        return True

    def maybe_publish_robot_move(self, now_ms):
        if self.game.is_over() or self.game.turn != ROBOT:
            return None

        if self.robot_started_ms is None:
            self.robot_started_ms = now_ms
        if now_ms - self.robot_started_ms < self.robot_delay_ms:
            return None

        move = self.game.best_move(ROBOT)
        if move is None:
            self.robot_started_ms = None
            return None

        self.game.place(*move, ROBOT)
        pose = self.publisher.publish_move(move[0], move[1], ROBOT)
        self.robot_started_ms = None
        return RobotTurnResult(move[0], move[1], pose)


class ProjectApp:
    def __init__(self, grid=None, publisher=None, simulator=None, robot_delay_ms=ROBOT_DELAY_MS):
        self.grid = grid or GridFrame()
        self.controller = ProjectController(
            grid=self.grid,
            publisher=publisher,
            robot_delay_ms=robot_delay_ms,
        )
        self.simulator = simulator or MujocoTicTacToeSim()

    def reset(self):
        self.controller.reset()
        self.simulator.reset()

    def handle_human_move(self, row, col, now_ms):
        placed = self.controller.handle_human_move(row, col, now_ms)
        if placed:
            place_human = getattr(self.simulator, "place_human_move", None)
            if place_human is not None:
                place_human(row, col)
        return placed

    def step(self, now_ms):
        result = self.controller.maybe_publish_robot_move(now_ms)
        if result is not None:
            self.simulator.execute_pose(result.pose, self.grid)
        return result

    def restore_for_new_game(self):
        restore_scene = getattr(self.simulator, "restore_scene", None)
        if restore_scene is not None:
            restore_scene()
        self.controller.reset()

    def sync(self):
        sync = getattr(self.simulator, "sync", None)
        if sync is not None:
            sync()


class MujocoTicTacToeSim:
    def __init__(self):
        try:
            import mujoco as mj
            from mujoco import viewer
        except ImportError as exc:
            raise RuntimeError(
                f"mujoco is required to run the robot simulation with interpreter: {sys.executable}"
            ) from exc

        self.mj = mj
        self.viewer_module = viewer
        self.arm = FrankArm()
        T_home, _ = self.arm.ForwardKin(Q_HOME.tolist())
        self.home_ee_xyz = T_home[-1][0:3, 3].copy()
        self.supply = BlockSupply()
        self.human_supply = BlockSupply(HUMAN_BLOCK_NAMES)
        self.arm_idx = list(range(7))
        self.gripper_idx = 7
        self.temp_xml_path = None
        self.model = None
        self.data = None
        self.viewer = None
        self.current_q = Q_HOME.copy()
        self.free_body_ids = {}
        self.free_body_joint_addrs = {}

        self._build_scene()
        self._capture_initial_state()

    def _build_scene(self):
        model_tree = ET.parse(str(ROOT_MODEL_XML))
        compiler = model_tree.getroot().find("compiler")
        if compiler is not None:
            compiler.set("meshdir", str(ASSET_DIR.resolve()))
        for name, pos, size in STATIC_BLOCKS:
            rgba = TABLE_RGBA if name == "TablePlane" else [0.2, 0.2, 0.9, 1.0]
            rt.add_free_block_to_model(
                tree=model_tree,
                name=name,
                pos=pos,
                density=BLOCK_DENSITY,
                size=size,
                rgba=rgba,
                free=False,
            )

        for line_idx, (start_xyz, end_xyz) in enumerate(build_grid_line_segments(), start=1):
            add_static_capsule_line(
                tree=model_tree,
                name=f"GridLine_{line_idx}",
                start_xyz=start_xyz,
                end_xyz=end_xyz,
                radius=GRID_LINE_RADIUS,
                rgba=GRID_RGBA,
            )

        for block in SUPPLY_BLOCKS:
            rt.add_free_block_to_model(
                tree=model_tree,
                name=block["name"],
                pos=block["pos"],
                density=BLOCK_DENSITY,
                size=FREE_BLOCK_SIZE,
                rgba=block["rgba"],
                free=True,
            )

        for name, pos in zip(HUMAN_BLOCK_NAMES, HUMAN_BLOCK_PARK_POSITIONS):
            rt.add_free_block_to_model(
                tree=model_tree,
                name=name,
                pos=pos,
                density=BLOCK_DENSITY,
                size=HUMAN_BLOCK_SIZE,
                rgba=HUMAN_BLOCK_RGBA,
                free=True,
            )

        fd, temp_path = tempfile.mkstemp(prefix="project_scene_", suffix=".xml")
        os.close(fd)
        self.temp_xml_path = Path(temp_path)
        model_tree.write(self.temp_xml_path, encoding="utf-8", xml_declaration=True)

        self.model = self.mj.MjModel.from_xml_path(str(self.temp_xml_path))
        self.data = self.mj.MjData(self.model)
        self.data.qpos[self.arm_idx] = Q_HOME
        self.data.qvel[:] = 0.0
        if self.data.ctrl.size:
            self.data.ctrl[:] = 0.0
            self.data.ctrl[self.gripper_idx] = GRIPPER_OPEN
        self.mj.mj_forward(self.model, self.data)

        self.viewer = self.viewer_module.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth += 90
        for block in SUPPLY_BLOCKS:
            self._register_free_body(block["name"])
        for name in HUMAN_BLOCK_NAMES:
            self._register_free_body(name)

        self._park_all_human_blocks()

    def _capture_initial_state(self):
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.initial_act = self.data.act.copy()
        self.initial_ctrl = self.data.ctrl.copy()
        self.initial_time = float(self.data.time)

    def _register_free_body(self, body_name):
        body_id = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, body_name)
        self.free_body_ids[body_name] = body_id
        joint_id = self.model.body_jntadr[body_id]
        self.free_body_joint_addrs[body_name] = (
            self.model.jnt_qposadr[joint_id],
            self.model.jnt_dofadr[joint_id],
        )

    def _set_free_body_pose(self, body_name, xyz, quat=(1.0, 0.0, 0.0, 0.0)):
        qpos_adr, qvel_adr = self.free_body_joint_addrs[body_name]
        self.data.qpos[qpos_adr:qpos_adr + 3] = np.array(xyz, dtype=float)
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.array(quat, dtype=float)
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    def _park_all_human_blocks(self):
        for name, park_xyz in zip(HUMAN_BLOCK_NAMES, HUMAN_BLOCK_PARK_POSITIONS):
            self._set_free_body_pose(name, park_xyz)
        self.mj.mj_forward(self.model, self.data)

    def _block_xyz(self, block_name):
        body_id = self.free_body_ids[block_name]
        return self.data.xpos[body_id].copy()

    def _solve_with_seed(self, seed, T_goal, q_prev):
        q_raw, _ = self.arm.IterInvKin(seed, T_goal)
        q_new = _unwrap(q_raw[:7], q_prev)
        jump = np.max(np.abs(q_new - q_prev))
        return q_new, jump

    def _current_ee_xyz(self):
        T_curr, _ = self.arm.ForwardKin(self.current_q.tolist())
        return T_curr[-1][0:3, 3].copy()

    def _solve_ik_chain(self, cart_waypoints):
        jump_thresh = 1.0
        q_seed = np.array(self.current_q, dtype=float)
        joint_waypoints = [q_seed.tolist() + [GRIPPER_OPEN]]
        max_jump = 0.0
        total_jump = 0.0

        for _, waypoint in cart_waypoints:
            g, x, y, z, r, p, yaw = waypoint
            T_goal = rt.rpyxyz2H([r, p, yaw], [x, y, z])
            q_new, jump = self._solve_with_seed(q_seed.tolist(), T_goal, q_seed)
            if jump > jump_thresh:
                q_home, jump_home = self._solve_with_seed(Q_HOME.tolist(), T_goal, q_seed)
                if jump_home < jump:
                    q_new = q_home
                    jump = jump_home
            max_jump = max(max_jump, float(jump))
            total_jump += float(np.linalg.norm(q_new - q_seed))
            q_seed = q_new.copy()
            joint_waypoints.append(q_new.tolist() + [gripper_cmd(g)])

        q_home = _unwrap(Q_HOME, q_seed)
        max_jump = max(max_jump, float(np.max(np.abs(q_home - q_seed))))
        total_jump += float(np.linalg.norm(q_home - q_seed))
        joint_waypoints.append(q_home.tolist() + [GRIPPER_OPEN])
        metrics = {"max_jump": max_jump, "total_jump": total_jump}
        return np.array(joint_waypoints, dtype=float), metrics

    def _select_best_motion_plan(self, source_xyz, target_xyz, grasp_candidates, place_candidates, label_fn):
        current_xyz = self._current_ee_xyz()
        candidates = []
        for grasp_label, grasp_orientation in grasp_candidates:
            for place_label, place_orientation in place_candidates:
                cart_waypoints = build_pick_and_place_waypoints(
                    source_xyz=source_xyz,
                    target_xyz=target_xyz,
                    current_xyz=current_xyz,
                    grasp_orientation=grasp_orientation,
                    place_orientation=place_orientation,
                )
                joint_waypoints, metrics = self._solve_ik_chain(cart_waypoints)
                candidates.append(
                    (
                        metrics["max_jump"],
                        metrics["total_jump"],
                        label_fn(grasp_label, place_label),
                        cart_waypoints,
                        joint_waypoints,
                    )
                )

        _, _, label, cart_waypoints, joint_waypoints = min(candidates, key=lambda item: (item[0], item[1]))
        return label, cart_waypoints, joint_waypoints

    def _select_motion_plan(self, source_xyz, target_xyz):
        return self._select_best_motion_plan(
            source_xyz=source_xyz,
            target_xyz=target_xyz,
            grasp_candidates=(("down_y", DOWN_GRASP_Y_RPY),),
            place_candidates=(
                ("down_y", DOWN_GRASP_Y_RPY),
                ("down_x", DOWN_GRASP_X_RPY),
            ),
            label_fn=lambda _grasp_label, place_label: place_label,
        )

    def _select_restore_motion_plan(self, source_xyz, target_xyz):
        return self._select_best_motion_plan(
            source_xyz=source_xyz,
            target_xyz=target_xyz,
            grasp_candidates=(
                ("down_y", DOWN_GRASP_Y_RPY),
                ("down_x", DOWN_GRASP_X_RPY),
            ),
            place_candidates=(("down_y", DOWN_GRASP_Y_RPY),),
            label_fn=lambda grasp_label, _place_label: grasp_label,
        )

    def _trajectory_durations(self, cart_waypoints):
        ee_positions = [self._current_ee_xyz()]
        ee_positions.extend(np.array(waypoint[1:4], dtype=float) for _, waypoint in cart_waypoints)
        ee_positions.append(np.array(self.home_ee_xyz, dtype=float))

        segment_durations = []
        hold_durations = []
        for i in range(len(ee_positions) - 1):
            cart_distance = np.linalg.norm(ee_positions[i + 1] - ee_positions[i])
            segment_durations.append(
                waypoint_duration(cart_distance, SEGMENT_DURATION_BASE, SEGMENT_DURATION_PER_METER)
            )
            hold_durations.append(
                waypoint_duration(cart_distance, HOLD_DURATION_BASE, HOLD_DURATION_PER_METER)
            )
        return segment_durations, hold_durations

    def _run_joint_trajectory(self, joint_waypoints, cart_waypoints):
        dt = self.model.opt.timestep
        segment_durations, hold_durations = self._trajectory_durations(cart_waypoints)

        for i in range(len(joint_waypoints) - 1):
            waypoint_name = cart_waypoints[i][0] if i < len(cart_waypoints) else "return_home"
            q_start = joint_waypoints[i][self.arm_idx].copy()
            q_goal = joint_waypoints[i + 1][self.arm_idx].copy()
            segment_duration = segment_durations[i]
            hold_duration = hold_durations[i]
            segment_steps = max(1, int(segment_duration / dt))
            hold_steps = int(hold_duration / dt)
            wall_t0 = time.time()
            sim_t = 0.0
            t = 0.0

            for step in range(segment_steps + hold_steps):
                q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, segment_duration)
                q = self.data.qpos[self.arm_idx].copy()
                qd = self.data.qvel[self.arm_idx].copy()
                tau = KP * (q_des - q) + KD * (qd_des - qd)
                self.data.ctrl[self.arm_idx] = tau + self.data.qfrc_bias[:7]
                self.data.ctrl[self.gripper_idx] = joint_waypoints[i][-1]
                self.mj.mj_step(self.model, self.data)
                t += dt
                sim_t += dt

                if step % RENDER_SKIP == 0:
                    self.viewer.sync()
                    _pace(wall_t0, sim_t)

            print(f"[project] completed waypoint: {waypoint_name}")

        self.current_q = self.data.qpos[self.arm_idx].copy()

    def _execute_target_xyz(
        self,
        block_name,
        source_xyz,
        target_xyz,
        target_label,
        motion_plan_selector=None,
        action_label="placing",
    ):
        plan_t0 = time.time()
        print(
            f"[project] planning {block_name} for {target_label} "
            f"target=({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})"
        )
        if motion_plan_selector is None:
            motion_plan_selector = self._select_motion_plan
        orientation_label, cart_waypoints, joint_waypoints = motion_plan_selector(source_xyz, target_xyz)
        plan_dt = time.time() - plan_t0
        print(
            f"[project] {action_label} {block_name} at {target_label} "
            f"target=({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}) "
            f"orientation={orientation_label} "
            f"planning_time={plan_dt:.2f}s"
        )
        self._run_joint_trajectory(joint_waypoints, cart_waypoints)
        return {
            "block_name": block_name,
            "target_xyz": tuple(target_xyz),
            "orientation": orientation_label,
        }

    def execute_pose(self, pose_msg, grid):
        row, col, target_xyz = pose_to_dropoff_position(pose_msg, grid)
        block_index, block_name = self.supply.allocate_next()
        source_xyz = SUPPLY_POSITIONS[block_index]
        result = self._execute_target_xyz(
            block_name=block_name,
            source_xyz=source_xyz,
            target_xyz=target_xyz,
            target_label=f"cell=({row},{col})",
        )
        result.update({"row": row, "col": col})
        return result

    def execute_xyz(self, target_xyz):
        block_index, block_name = self.supply.allocate_next()
        source_xyz = SUPPLY_POSITIONS[block_index]
        return self._execute_target_xyz(
            block_name=block_name,
            source_xyz=source_xyz,
            target_xyz=tuple(float(v) for v in target_xyz),
            target_label="xyz",
        )

    def place_human_move(self, row, col):
        block_index, block_name = self.human_supply.allocate_next()
        target_xyz = np.array(cell_to_dropoff_position(row, col), dtype=float)
        self._set_free_body_pose(block_name, target_xyz)
        self.mj.mj_forward(self.model, self.data)
        self.sync()
        print(
            f"[project] spawned {block_name} for human at cell=({row},{col}) "
            f"target=({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})"
        )
        return {"block_name": block_name, "block_index": block_index, "row": row, "col": col}

    def restore_scene(self):
        print("[project] restoring scene for a new game")
        self._park_all_human_blocks()
        used_count = self.supply.next_block_index
        for block_index in range(used_count - 1, -1, -1):
            block_name = BLOCK_NAMES[block_index]
            source_xyz = self._block_xyz(block_name)
            target_xyz = SUPPLY_POSITIONS[block_index]
            self._execute_target_xyz(
                block_name=block_name,
                source_xyz=source_xyz,
                target_xyz=target_xyz,
                target_label=f"restore->{block_name}",
                motion_plan_selector=self._select_restore_motion_plan,
                action_label="restoring",
            )

        self.supply.reset()
        self.human_supply.reset()
        self.sync()

    def reset(self):
        self.supply.reset()
        self.human_supply.reset()
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.act[:] = self.initial_act
        self.data.ctrl[:] = self.initial_ctrl
        self.data.time = self.initial_time
        self.mj.mj_forward(self.model, self.data)
        self._park_all_human_blocks()
        self.current_q = Q_HOME.copy()
        self.sync()

    def sync(self):
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.temp_xml_path is not None and self.temp_xml_path.exists():
            self.temp_xml_path.unlink(missing_ok=True)


def run(grid):
    pygame = _load_pygame()
    pygame.init()
    pygame.display.set_caption("Robot Tic-Tac-Toe Arena")
    screen = pygame.display.set_mode((WINDOW, WINDOW_H))
    fonts = make_ui_fonts(pygame)

    app = None

    try:
        app = ProjectApp(grid=grid)
        print(f"[project] ROS backend: {'real rospy' if USING_REAL_ROS else 'mock rospy'}")
        print(
            f"[project] Grid origin=({grid.origin_x}, {grid.origin_y}) "
            f"cell={grid.cell_size} frame={grid.frame_id}"
        )

        clock = pygame.time.Clock()
        running = True
        while running:
            now_ms = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        app.reset()
                    elif event.key == pygame.K_a:
                        app.restore_for_new_game()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    cell = cell_from_pixel(*event.pos)
                    if cell is not None:
                        app.handle_human_move(cell[0], cell[1], now_ms)

            app.step(now_ms)
            draw_board(screen, app.controller.game, status_text(app.controller.game), fonts, pygame)
            pygame.display.flip()
            app.sync()
            clock.tick(60)
    finally:
        if app is not None:
            app.simulator.close()
        pygame.quit()

    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-x", type=float, default=0.40)
    parser.add_argument("--origin-y", type=float, default=0.00)
    parser.add_argument("--cell", type=float, default=0.06)
    parser.add_argument("--place-z", type=float, default=0.05)
    parser.add_argument("--frame-id", default="base_link")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    grid = GridFrame(args.origin_x, args.origin_y, args.cell, args.place_z, args.frame_id)
    return run(grid)


if __name__ == "__main__":
    sys.exit(main())
