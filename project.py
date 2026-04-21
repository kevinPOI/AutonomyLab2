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

WINDOW = 600
MARGIN = 40
CELL_PX = (WINDOW - 2 * MARGIN) // 3
STATUS_H = 80
LINE_W = 6

BG = (245, 245, 240)
GRID_COLOR = (40, 40, 40)
X_COLOR = (30, 90, 200)
O_COLOR = (200, 60, 60)
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

GRID_CENTER = np.array([0.520, -0.021, 0.05], dtype=float)
GRID_BOTTOM_CENTER = np.array([0.622, -0.027, 0.05], dtype=float)
GRID_LOGICAL_ROW_STEP = GRID_BOTTOM_CENTER - GRID_CENTER
# The user only specified the row direction; keep the existing 7.5 cm column spacing.
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
    if not (MARGIN <= mx < MARGIN + 3 * CELL_PX):
        return None
    if not (STATUS_H <= my < STATUS_H + 3 * CELL_PX):
        return None
    return (my - STATUS_H) // CELL_PX, (mx - MARGIN) // CELL_PX


def status_text(game):
    w = game.winner()
    if w == HUMAN:
        return "You win! Press R to restart."
    if w == ROBOT:
        return "Robot wins. Press R to restart."
    if game.is_full():
        return "Draw. Press R to restart."
    if game.turn == HUMAN:
        return f"Your turn ({HUMAN}). Click a cell."
    return "Robot thinking..."


def draw_board(screen, game, status, font, pygame_mod):
    screen.fill(BG)
    screen.blit(font.render(status, True, (20, 20, 20)), (MARGIN, 10))

    top = STATUS_H
    for i in range(4):
        x = MARGIN + i * CELL_PX
        pygame_mod.draw.line(screen, GRID_COLOR, (x, top), (x, top + 3 * CELL_PX), LINE_W)
        y = top + i * CELL_PX
        pygame_mod.draw.line(screen, GRID_COLOR, (MARGIN, y), (MARGIN + 3 * CELL_PX, y), LINE_W)

    pad = CELL_PX // 5
    for row in range(3):
        for col in range(3):
            cell = game.board[row][col]
            cx = MARGIN + col * CELL_PX + CELL_PX // 2
            cy = top + row * CELL_PX + CELL_PX // 2
            color = O_COLOR if cell == HUMAN else X_COLOR
            if cell == ROBOT:
                a = CELL_PX // 2 - pad
                pygame_mod.draw.line(screen, color, (cx - a, cy - a), (cx + a, cy + a), 10)
                pygame_mod.draw.line(screen, color, (cx + a, cy - a), (cx - a, cy + a), 10)
            elif cell == HUMAN:
                pygame_mod.draw.circle(screen, color, (cx, cy), CELL_PX // 2 - pad, 10)


@dataclass
class BlockSupply:
    block_names: tuple[str, ...] = BLOCK_NAMES
    next_block_index: int = 0

    def allocate_next(self):
        if self.next_block_index >= len(self.block_names):
            raise RuntimeError("robot has no remaining simulation blocks to place")
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
        return self.controller.handle_human_move(row, col, now_ms)

    def step(self, now_ms):
        result = self.controller.maybe_publish_robot_move(now_ms)
        if result is not None:
            self.simulator.execute_pose(result.pose, self.grid)
        return result

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
        self.arm_idx = list(range(7))
        self.gripper_idx = 7
        self.temp_xml_path = None
        self.model = None
        self.data = None
        self.viewer = None
        self.current_q = Q_HOME.copy()

        self._build_scene()
        self._capture_initial_state()

    def _build_scene(self):
        model_tree = ET.parse(str(ROOT_MODEL_XML))
        compiler = model_tree.getroot().find("compiler")
        if compiler is not None:
            compiler.set("meshdir", str(ASSET_DIR.resolve()))
        for name, pos, size in STATIC_BLOCKS:
            rt.add_free_block_to_model(
                tree=model_tree,
                name=name,
                pos=pos,
                density=BLOCK_DENSITY,
                size=size,
                rgba=[0.2, 0.2, 0.9, 1.0],
                free=False,
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
        self.block_joint_addrs = {}
        for block in SUPPLY_BLOCKS:
            body_id = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, block["name"])
            joint_id = self.model.body_jntadr[body_id]
            self.block_joint_addrs[block["name"]] = (
                self.model.jnt_qposadr[joint_id],
                self.model.jnt_dofadr[joint_id],
            )

    def _capture_initial_state(self):
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.initial_act = self.data.act.copy()
        self.initial_ctrl = self.data.ctrl.copy()
        self.initial_time = float(self.data.time)

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

    def _select_motion_plan(self, source_xyz, target_xyz):
        current_xyz = self._current_ee_xyz()
        candidates = []
        for label, place_orientation in (
            ("down_y", DOWN_GRASP_Y_RPY),
            ("down_x", DOWN_GRASP_X_RPY),
        ):
            cart_waypoints = build_pick_and_place_waypoints(
                source_xyz=source_xyz,
                target_xyz=target_xyz,
                current_xyz=current_xyz,
                grasp_orientation=DOWN_GRASP_Y_RPY,
                place_orientation=place_orientation,
            )
            joint_waypoints, metrics = self._solve_ik_chain(cart_waypoints)
            candidates.append((metrics["max_jump"], metrics["total_jump"], label, cart_waypoints, joint_waypoints))

        _, _, label, cart_waypoints, joint_waypoints = min(candidates, key=lambda item: (item[0], item[1]))
        return label, cart_waypoints, joint_waypoints

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

    def _execute_target_xyz(self, block_name, source_xyz, target_xyz, target_label):
        plan_t0 = time.time()
        print(
            f"[project] planning {block_name} for {target_label} "
            f"target=({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})"
        )
        orientation_label, cart_waypoints, joint_waypoints = self._select_motion_plan(source_xyz, target_xyz)
        plan_dt = time.time() - plan_t0
        print(
            f"[project] placing {block_name} at {target_label} "
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

    def reset(self):
        self.supply.reset()
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.act[:] = self.initial_act
        self.data.ctrl[:] = self.initial_ctrl
        self.data.time = self.initial_time
        self.mj.mj_forward(self.model, self.data)
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
    pygame.display.set_caption("Tic-Tac-Toe / MuJoCo robot player")
    screen = pygame.display.set_mode((WINDOW, STATUS_H + 3 * CELL_PX + MARGIN))
    font = pygame.font.SysFont("sans", 22)

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
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    cell = cell_from_pixel(*event.pos)
                    if cell is not None:
                        app.handle_human_move(cell[0], cell[1], now_ms)

            app.step(now_ms)
            draw_board(screen, app.controller.game, status_text(app.controller.game), font, pygame)
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
