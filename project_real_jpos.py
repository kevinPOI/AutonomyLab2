from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import RobotUtil as rt
from kinematics import FrankArm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ttt_ros.game_logic import HUMAN, ROBOT, TicTacToe
from ttt_ros.ros_publisher import BlockPosePublisher, GridFrame, USING_REAL_ROS, _ros_is_initialized


def _ensure_ros_node(node_name="tic_tac_toe_project_node"):
    if _ros_is_initialized():
        return
    try:
        import rospy
    except ImportError:
        return
    try:
        rospy.init_node(node_name, anonymous=True, disable_signals=True)
    except Exception as exc:
        print(f"[project_real_jpos] rospy.init_node failed: {exc}")

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

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0125
SEGMENT_DURATION_BASE = 0.3
SEGMENT_DURATION_PER_METER = 2.6
HOLD_DURATION_BASE = 0.05
HOLD_DURATION_PER_METER = 1.0
MIN_JERK_PEAK_SPEED_SCALE = 1.875
DEFAULT_MOTION_DURATION_SCALE = 2.0
ROTATE_TO_GRASP_STEPS = 3
ROTATE_IN_PLACE_DISTANCE_EPS = 0.01
ROTATE_IN_PLACE_MIN_DURATION = 3.0
DEFAULT_MAX_EEF_LINEAR_SPEED = 0.15
DEFAULT_WORKSPACE_MARGIN = 0.01
DEFAULT_SAFETY_POLL_PERIOD = 0.05
DEFAULT_HOME_DURATION = 8.0
RUNTIME_LIMIT_TOLERANCE = 1.25
# Absolute tolerance (m/s) added to the EEF linear speed limit to absorb
# measurement jitter and compliance motion during stationary waypoints.
DEFAULT_EEF_SPEED_NOISE_FLOOR = 0.05
DEFAULT_MAX_JOINT_VELOCITY_SCALE = 0.3
MIN_SEGMENT_DURATION = 1.0
# Official Franka Panda max joint velocities (rad/s). Scaled by
# max_joint_velocity_scale at runtime for a safety margin.
FALLBACK_JOINT_VELOCITY_LIMITS = [2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]

Q_HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8], dtype=float)

DOWN_GRASP_Y_RPY = (3.13714399, 0.00419659, -0.80013718 + np.pi / 2.0)
DOWN_GRASP_X_RPY = (3.13714399, 0.00419659, -0.80013718)

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

DEFAULT_CARTESIAN_IMPEDANCES = [3000, 3000, 100, 300, 300, 300]
WORLD_FRAME = "world"
TOOL_FRAME = "franka_tool"
FALLBACK_FORCE_THRESHOLDS = [120.0, 120.0, 120.0, 125.0, 125.0, 125.0]
FALLBACK_TORQUE_THRESHOLDS = [120.0, 120.0, 118.0, 118.0, 116.0, 114.0, 112.0]
FALLBACK_WORKSPACE_WALLS = np.array(
    [
        [0.15, 0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1],
        [0.15, -0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1],
        [-0.41, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
        [0.75, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
        [0.2, 0, 1, 0, 0, 0, 1.2, 1, 0.01],
        [0.2, 0, -0.05, 0, 0, 0, 1.2, 1, 0.01],
    ],
    dtype=float,
)


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


def _load_pygame():
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            f"pygame is required to run project_real.py with interpreter: {sys.executable}"
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


def cell_to_dropoff_position(row, col):
    # Keep the same logical-to-physical mapping as the original project file.
    sim_row = 2 - col
    sim_col = row
    return DROPOFF_POSITIONS[sim_row][sim_col]


def pose_to_dropoff_position(pose_msg, grid):
    row, col = grid.pose_to_cell(pose_msg)
    return row, col, cell_to_dropoff_position(row, col)


def _build_rotate_to_grasp_waypoints(current_xyz, current_rotation, grasp_orientation, steps):
    current_xyz = np.array(current_xyz, dtype=float)
    if current_rotation is None or steps <= 1:
        return [
            (
                "rotate_to_grasp_orientation",
                [0.0, current_xyz[0], current_xyz[1], current_xyz[2], *grasp_orientation],
            )
        ]

    target_rotation = np.array(
        rt.rpyxyz2H(grasp_orientation, [0.0, 0.0, 0.0])[:3, :3],
        dtype=float,
    )
    current_rotation = np.array(current_rotation, dtype=float)
    delta_rotation = target_rotation @ current_rotation.T
    axis, angle = rt.R2axisang(delta_rotation)
    axis = np.array(axis, dtype=float)

    if np.linalg.norm(axis) < 1e-9 or abs(angle) < 1e-9:
        return [
            (
                "rotate_to_grasp_orientation",
                [0.0, current_xyz[0], current_xyz[1], current_xyz[2], *grasp_orientation],
            )
        ]

    waypoints = []
    for step_idx in range(1, steps + 1):
        frac = step_idx / steps
        interp_rotation = rt.MatrixExp(axis, angle * frac)[:3, :3] @ current_rotation
        interp_rpy = rt.R2rpy(interp_rotation)
        waypoint_name = (
            "rotate_to_grasp_orientation"
            if step_idx == steps
            else f"rotate_to_grasp_orientation_{step_idx}"
        )
        waypoints.append(
            (
                waypoint_name,
                [
                    0.0,
                    current_xyz[0],
                    current_xyz[1],
                    current_xyz[2],
                    float(interp_rpy[0]),
                    float(interp_rpy[1]),
                    float(interp_rpy[2]),
                ],
            )
        )
    return waypoints


def build_pick_and_place_waypoints(
    source_xyz,
    target_xyz,
    current_xyz,
    grasp_orientation,
    place_orientation,
    current_rotation=None,
    rotate_to_grasp_steps=1,
):
    current_xyz = np.array(current_xyz, dtype=float)
    sx, sy, sz = source_xyz
    tx, ty, tz = target_xyz
    mx, my, _ = 0.6 * np.array(source_xyz, dtype=float) + 0.4 * np.array(target_xyz, dtype=float)
    transfer_z = max(sz, tz) + 0.20
    return _build_rotate_to_grasp_waypoints(
        current_xyz=current_xyz,
        current_rotation=current_rotation,
        grasp_orientation=grasp_orientation,
        steps=rotate_to_grasp_steps,
    ) + [
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
            raise RuntimeError("robot has no remaining supply blocks to place")
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
            init_node=False,
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
        self.simulator = simulator or RealTicTacToeRobot()

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


class RealTicTacToeRobot:
    def __init__(
        self,
        cartesian_impedances=None,
        use_impedance=False,
        max_eef_linear_speed=DEFAULT_MAX_EEF_LINEAR_SPEED,
        eef_speed_noise_floor=DEFAULT_EEF_SPEED_NOISE_FLOOR,
        max_joint_velocity_scale=DEFAULT_MAX_JOINT_VELOCITY_SCALE,
        joint_velocity_limits=None,
        motion_duration_scale=DEFAULT_MOTION_DURATION_SCALE,
        workspace_margin=DEFAULT_WORKSPACE_MARGIN,
        safety_poll_period=DEFAULT_SAFETY_POLL_PERIOD,
        force_thresholds=None,
        torque_thresholds=None,
        home_duration=DEFAULT_HOME_DURATION,
        enable_workspace_check=True,
        enable_joint_reachability_check=True,
        enable_collision_check=True,
        enable_eef_speed_check=False,
        enable_joint_velocity_check=True,
        enable_force_check=True,
        enable_torque_check=True,
    ):
        try:
            from autolab_core import RigidTransform
            from frankapy import FrankaArm as RealFrankaArm
            from frankapy.franka_constants import FrankaConstants
        except ImportError as exc:
            raise RuntimeError(
                "project_real.py requires frankapy and autolab_core. "
                "Run it on the Franka lab machine/container after starting the Control PC server."
            ) from exc

        self.RigidTransform = RigidTransform
        self.FrankaConstants = FrankaConstants
        self.fa = RealFrankaArm(init_node=False)
        self.arm = FrankArm()
        T_home, _ = self.arm.ForwardKin(Q_HOME.tolist())
        self.home_H = np.array(T_home[-1], dtype=float)
        self.home_ee_xyz = self.home_H[0:3, 3].copy()
        self.supply = BlockSupply()
        self.cartesian_impedances = list(
            DEFAULT_CARTESIAN_IMPEDANCES if cartesian_impedances is None else cartesian_impedances
        )
        self.use_impedance = use_impedance
        self.max_eef_linear_speed = float(max_eef_linear_speed)
        self.eef_speed_noise_floor = float(eef_speed_noise_floor)
        self.max_joint_velocity_scale = float(max_joint_velocity_scale)
        self.motion_duration_scale = float(motion_duration_scale)
        self.workspace_margin = float(workspace_margin)
        self.safety_poll_period = float(safety_poll_period)
        self.home_duration = float(home_duration)
        self.enable_workspace_check = bool(enable_workspace_check)
        self.enable_joint_reachability_check = bool(enable_joint_reachability_check)
        self.enable_collision_check = bool(enable_collision_check)
        self.enable_eef_speed_check = bool(enable_eef_speed_check)
        self.enable_joint_velocity_check = bool(enable_joint_velocity_check)
        self.enable_force_check = bool(enable_force_check)
        self.enable_torque_check = bool(enable_torque_check)
        self.joint_velocity_limits = self._joint_velocity_limit_array(joint_velocity_limits)
        self.force_thresholds = self._threshold_array(
            force_thresholds
            if force_thresholds is not None
            else getattr(self.FrankaConstants, "DEFAULT_UPPER_FORCE_THRESHOLDS_NOMINAL", FALLBACK_FORCE_THRESHOLDS),
            6,
            "force_thresholds",
        )
        self.torque_thresholds = self._threshold_array(
            torque_thresholds
            if torque_thresholds is not None
            else getattr(self.FrankaConstants, "DEFAULT_UPPER_TORQUE_THRESHOLDS_NOMINAL", FALLBACK_TORQUE_THRESHOLDS),
            7,
            "torque_thresholds",
        )
        self.workspace_min, self.workspace_max = self._workspace_bounds()
        self.current_q = Q_HOME.copy()
        self.current_gripper = GRIPPER_OPEN
        self._monitor_pose = None
        self._monitor_time = None
        self._monitor_joints = None

        if self.max_eef_linear_speed <= 0.0:
            raise ValueError("max_eef_linear_speed must be positive")
        if self.eef_speed_noise_floor < 0.0:
            raise ValueError("eef_speed_noise_floor must be non-negative")
        if self.max_joint_velocity_scale <= 0.0 or self.max_joint_velocity_scale > 1.0:
            raise ValueError("max_joint_velocity_scale must be in (0, 1]")
        if self.motion_duration_scale <= 0.0:
            raise ValueError("motion_duration_scale must be positive")
        if self.workspace_margin < 0.0:
            raise ValueError("workspace_margin must be non-negative")
        if self.safety_poll_period <= 0.0:
            raise ValueError("safety_poll_period must be positive")
        if self.home_duration <= 0.0:
            raise ValueError("home_duration must be positive")
        self.reset()

    def _threshold_array(self, values, expected_len, name):
        arr = np.array(values, dtype=float)
        if arr.shape != (expected_len,):
            raise ValueError(f"{name} must contain exactly {expected_len} values")
        if np.any(arr <= 0.0):
            raise ValueError(f"{name} must be strictly positive")
        return arr

    def _joint_velocity_limit_array(self, values):
        if values is None:
            values = getattr(
                self.FrankaConstants,
                "MAX_JOINT_VELOCITIES",
                FALLBACK_JOINT_VELOCITY_LIMITS,
            )
        arr = np.array(values, dtype=float)
        if arr.shape != (7,):
            raise ValueError("joint_velocity_limits must contain exactly 7 values")
        if np.any(arr <= 0.0):
            raise ValueError("joint_velocity_limits must be strictly positive")
        return arr

    def _workspace_bounds(self):
        workspace_walls = np.array(
            getattr(self.FrankaConstants, "WORKSPACE_WALLS", FALLBACK_WORKSPACE_WALLS),
            dtype=float,
        )
        mins = workspace_walls[:, :3].min(axis=0) + self.workspace_margin
        maxs = workspace_walls[:, :3].max(axis=0) - self.workspace_margin
        if np.any(mins >= maxs):
            raise ValueError("workspace_margin is too large for the configured Franka workspace")
        return mins, maxs

    def _refresh_joints(self):
        self.current_q = np.array(self.fa.get_joints(), dtype=float)
        return self.current_q

    def _solve_with_seed(self, seed, T_goal, q_prev):
        q_raw, _ = self.arm.IterInvKin(seed, T_goal)
        q_new = _unwrap(q_raw[:7], q_prev)
        jump = np.max(np.abs(q_new - q_prev))
        return q_new, jump

    def _current_ee_xyz(self):
        return np.array(self.fa.get_pose().translation, dtype=float)

    def _solve_ik_chain(self, cart_waypoints):
        jump_thresh = 1.0
        q_seed = np.array(self._refresh_joints(), dtype=float)
        joint_waypoints = [q_seed.tolist() + [self.current_gripper]]
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
        current_pose = self.fa.get_pose()
        current_xyz = np.array(current_pose.translation, dtype=float)
        current_rotation = np.array(current_pose.rotation, dtype=float)
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
                current_rotation=current_rotation,
                rotate_to_grasp_steps=ROTATE_TO_GRASP_STEPS,
            )
            joint_waypoints, metrics = self._solve_ik_chain(cart_waypoints)
            candidates.append((metrics["max_jump"], metrics["total_jump"], label, cart_waypoints, joint_waypoints))

        _, _, label, cart_waypoints, joint_waypoints = min(candidates, key=lambda item: (item[0], item[1]))
        return label, cart_waypoints, joint_waypoints

    def _trajectory_durations(self, cart_waypoints, joint_waypoints):
        ee_positions = [self._current_ee_xyz()]
        ee_positions.extend(np.array(waypoint[1:4], dtype=float) for _, waypoint in cart_waypoints)
        ee_positions.append(np.array(self.home_ee_xyz, dtype=float))

        joint_positions = np.array(joint_waypoints, dtype=float)[:, :7]
        scaled_joint_velocity_limits = self.joint_velocity_limits * self.max_joint_velocity_scale

        segment_durations = []
        hold_durations = []
        for i in range(len(ee_positions) - 1):
            waypoint_name = cart_waypoints[i][0] if i < len(cart_waypoints) else "return_home"
            cart_distance = np.linalg.norm(ee_positions[i + 1] - ee_positions[i])
            joint_delta = np.abs(joint_positions[i + 1] - joint_positions[i])

            nominal_duration = waypoint_duration(cart_distance, SEGMENT_DURATION_BASE, SEGMENT_DURATION_PER_METER)
            eef_speed_limited = MIN_JERK_PEAK_SPEED_SCALE * cart_distance / self.max_eef_linear_speed
            joint_speed_limited = MIN_JERK_PEAK_SPEED_SCALE * float(
                np.max(joint_delta / scaled_joint_velocity_limits)
            )

            segment_duration = max(nominal_duration, eef_speed_limited, joint_speed_limited)
            if (
                waypoint_name.startswith("rotate_to_grasp_orientation")
                and cart_distance <= ROTATE_IN_PLACE_DISTANCE_EPS
            ):
                segment_duration = max(
                    segment_duration,
                    ROTATE_IN_PLACE_MIN_DURATION / ROTATE_TO_GRASP_STEPS,
                )
            segment_duration = max(segment_duration * self.motion_duration_scale, MIN_SEGMENT_DURATION)
            segment_durations.append(segment_duration)
            hold_durations.append(
                waypoint_duration(cart_distance, HOLD_DURATION_BASE, HOLD_DURATION_PER_METER)
            )
        return segment_durations, hold_durations

    def _make_pose(self, waypoint):
        _, x, y, z, r, p, yaw = waypoint
        T_goal = rt.rpyxyz2H([r, p, yaw], [x, y, z])
        return self.RigidTransform(
            rotation=np.array(T_goal[:3, :3], dtype=float),
            translation=np.array(T_goal[:3, 3], dtype=float),
            from_frame=TOOL_FRAME,
            to_frame=WORLD_FRAME,
        )

    def _validate_workspace_translation(self, xyz, context):
        if not self.enable_workspace_check:
            return
        xyz = np.array(xyz, dtype=float)
        if np.any(xyz < self.workspace_min) or np.any(xyz > self.workspace_max):
            raise RuntimeError(
                f"{context} is outside workspace bounds: "
                f"xyz={tuple(float(v) for v in xyz)} "
                f"allowed_min={tuple(float(v) for v in self.workspace_min)} "
                f"allowed_max={tuple(float(v) for v in self.workspace_max)}"
            )

    def _validate_pose_in_workspace(self, pose, context):
        self._validate_workspace_translation(pose.translation, context)

    def _validate_joint_waypoint(self, joints, context):
        q = np.array(joints, dtype=float)
        is_reachable = getattr(self.fa, "is_joints_reachable", None)
        if self.enable_joint_reachability_check and is_reachable is not None and not is_reachable(q.tolist()):
            raise RuntimeError(f"{context} is outside Franka joint limits: joints={tuple(float(v) for v in q)}")

        in_collision = getattr(self.fa, "is_joints_in_collision_with_boxes", None)
        if self.enable_collision_check and self.enable_workspace_check and in_collision is not None and in_collision(joints=q.tolist()):
            raise RuntimeError(f"{context} is in collision with Franka collision boxes or workspace walls")

    def _validate_motion_plan(self, cart_waypoints, joint_waypoints):
        for i, (waypoint_name, waypoint) in enumerate(cart_waypoints):
            self._validate_pose_in_workspace(self._make_pose(waypoint), f"{waypoint_name} target pose")
            self._validate_joint_waypoint(joint_waypoints[i + 1][:7], f"{waypoint_name} joint waypoint")
        self._validate_joint_waypoint(Q_HOME, "return_home joint waypoint")

    def _motion_kwargs(self, skill_desc):
        return {
            "force_thresholds": self.force_thresholds.tolist() if self.enable_force_check else None,
            "torque_thresholds": self.torque_thresholds.tolist() if self.enable_torque_check else None,
            "block": False,
            "ignore_errors": False,
            "ignore_virtual_walls": not self.enable_workspace_check,
            "skill_desc": skill_desc,
        }

    def _start_motion_monitor(self, context):
        pose = self.fa.get_pose()
        self._validate_pose_in_workspace(pose, f"{context} current pose")
        self._monitor_pose = np.array(pose.translation, dtype=float)
        self._monitor_time = time.monotonic()
        get_joints = getattr(self.fa, "get_joints", None)
        if get_joints is not None:
            try:
                self._monitor_joints = np.array(get_joints(), dtype=float)
            except Exception:
                self._monitor_joints = None
        else:
            self._monitor_joints = None
        self._check_runtime_safety(context)

    def _abort_motion(self, reason):
        try:
            self.fa.stop_skill()
        except Exception:
            pass
        raise RuntimeError(reason)

    def _check_runtime_safety(self, context):
        pose = self.fa.get_pose()
        pose_xyz = np.array(pose.translation, dtype=float)
        self._validate_workspace_translation(pose_xyz, f"{context} current pose")

        now = time.monotonic()
        if self.enable_eef_speed_check and self._monitor_pose is not None and self._monitor_time is not None:
            dt = now - self._monitor_time
            if dt >= max(1e-3, 0.5 * self.safety_poll_period):
                linear_speed = np.linalg.norm(pose_xyz - self._monitor_pose) / dt
                effective_limit = (
                    self.max_eef_linear_speed * RUNTIME_LIMIT_TOLERANCE
                    + self.eef_speed_noise_floor
                )
                if linear_speed > effective_limit:
                    self._abort_motion(
                        f"{context} exceeded EEF linear speed limit: "
                        f"measured={linear_speed:.3f} m/s "
                        f"limit={self.max_eef_linear_speed:.3f} m/s "
                        f"effective={effective_limit:.3f} m/s"
                    )

        if self.enable_joint_velocity_check:
            get_joint_velocities = getattr(self.fa, "get_joint_velocities", None)
            joint_velocities = None
            if get_joint_velocities is not None:
                try:
                    joint_velocities = np.array(get_joint_velocities(), dtype=float)
                except Exception:
                    joint_velocities = None
            if joint_velocities is None:
                get_joints = getattr(self.fa, "get_joints", None)
                if get_joints is not None and self._monitor_joints is not None and self._monitor_time is not None:
                    dt_j = now - self._monitor_time
                    if dt_j >= max(1e-3, 0.5 * self.safety_poll_period):
                        try:
                            q_now = np.array(get_joints(), dtype=float)
                            joint_velocities = (q_now - self._monitor_joints) / dt_j
                        except Exception:
                            joint_velocities = None
            if joint_velocities is not None and joint_velocities.shape == (7,):
                joint_velocity_limits = self.joint_velocity_limits * RUNTIME_LIMIT_TOLERANCE
                if np.any(np.abs(joint_velocities) > joint_velocity_limits):
                    self._abort_motion(
                        f"{context} exceeded joint velocity limit: "
                        f"velocities={tuple(float(v) for v in joint_velocities)} "
                        f"limits={tuple(float(v) for v in self.joint_velocity_limits)}"
                    )

        if self.enable_torque_check:
            joint_torques = np.array(self.fa.get_joint_torques(), dtype=float)
            if np.any(np.abs(joint_torques) > self.torque_thresholds * RUNTIME_LIMIT_TOLERANCE):
                self._abort_motion(
                    f"{context} exceeded joint torque limit: "
                    f"torques={tuple(float(v) for v in joint_torques)} "
                    f"limits={tuple(float(v) for v in self.torque_thresholds)}"
                )

        if self.enable_force_check:
            ee_force_torque = np.array(self.fa.get_ee_force_torque(), dtype=float)
            if np.any(np.abs(ee_force_torque) > self.force_thresholds * RUNTIME_LIMIT_TOLERANCE):
                self._abort_motion(
                    f"{context} exceeded end-effector wrench limit: "
                    f"force_torque={tuple(float(v) for v in ee_force_torque)} "
                    f"limits={tuple(float(v) for v in self.force_thresholds)}"
                )

        self._monitor_pose = pose_xyz
        self._monitor_time = now
        get_joints = getattr(self.fa, "get_joints", None)
        if get_joints is not None:
            try:
                self._monitor_joints = np.array(get_joints(), dtype=float)
            except Exception:
                self._monitor_joints = None

    def _monitor_skill(self, context):
        self._start_motion_monitor(context)
        while True:
            if self.fa.is_skill_done(ignore_errors=False):
                break
            self._check_runtime_safety(context)
            time.sleep(self.safety_poll_period)
        self._check_runtime_safety(f"{context} final")

    def _hold_with_safety(self, duration, context):
        end_time = time.monotonic() + max(0.0, float(duration))
        while time.monotonic() < end_time:
            self._check_runtime_safety(context)
            time.sleep(min(self.safety_poll_period, max(0.0, end_time - time.monotonic())))

    def _goto_joints_waypoint(self, q_target, duration, waypoint_name):
        q_target = np.array(q_target, dtype=float)
        self._validate_joint_waypoint(q_target, f"{waypoint_name} joint waypoint")
        self.fa.goto_joints(
            q_target.tolist(),
            duration=float(duration),
            **self._motion_kwargs(f"project_real_jpos:{waypoint_name}"),
        )
        self._monitor_skill(waypoint_name)
        self.current_q = q_target.copy()

    def _apply_gripper_command(self, waypoint_name):
        if waypoint_name == "close_gripper":
            self.fa.goto_gripper(GRIPPER_CLOSED)
            self.current_gripper = GRIPPER_CLOSED
        elif waypoint_name == "open_gripper":
            self.fa.goto_gripper(GRIPPER_OPEN)
            self.current_gripper = GRIPPER_OPEN

    def _goto_home(self):
        self._validate_joint_waypoint(Q_HOME, "return_home joint waypoint")
        self.fa.goto_joints(
            Q_HOME.tolist(),
            duration=self.home_duration,
            **self._motion_kwargs("project_real_jpos:return_home"),
        )
        self._monitor_skill("return_home")
        self.current_q = Q_HOME.copy()

    def _run_joint_trajectory(self, joint_waypoints, cart_waypoints):
        segment_durations, hold_durations = self._trajectory_durations(cart_waypoints, joint_waypoints)

        for i, (waypoint_name, _waypoint) in enumerate(cart_waypoints):
            q_target = joint_waypoints[i + 1][:7]
            self._goto_joints_waypoint(q_target, segment_durations[i], waypoint_name)
            self._apply_gripper_command(waypoint_name)
            self._hold_with_safety(hold_durations[i], f"{waypoint_name} hold")
            print(f"[project_real_jpos] completed waypoint: {waypoint_name}")

        home_duration = max(segment_durations[-1], self.home_duration)
        self._validate_joint_waypoint(Q_HOME, "return_home joint waypoint")
        self.fa.goto_joints(
            Q_HOME.tolist(),
            duration=float(home_duration),
            **self._motion_kwargs("project_real_jpos:return_home"),
        )
        self._monitor_skill("return_home")
        self.fa.goto_gripper(GRIPPER_OPEN)
        self.current_gripper = GRIPPER_OPEN
        self._hold_with_safety(hold_durations[-1], "return_home hold")
        print("[project_real_jpos] completed waypoint: return_home")
        self.current_q = Q_HOME.copy()

    def _execute_target_xyz(self, block_name, source_xyz, target_xyz, target_label):
        plan_t0 = time.time()
        print(
            f"[project_real_jpos] planning {block_name} for {target_label} "
            f"target=({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})"
        )
        orientation_label, cart_waypoints, joint_waypoints = self._select_motion_plan(source_xyz, target_xyz)
        plan_dt = time.time() - plan_t0
        print(
            f"[project_real_jpos] placing {block_name} at {target_label} "
            f"target=({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}) "
            f"orientation={orientation_label} "
            f"planning_time={plan_dt:.2f}s"
        )
        self._validate_motion_plan(cart_waypoints, joint_waypoints)
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
        self._goto_home()
        self.fa.goto_gripper(GRIPPER_OPEN)
        self.current_q = Q_HOME.copy()
        self.current_gripper = GRIPPER_OPEN

    def sync(self):
        return None

    def close(self):
        return None


def run(grid, robot_kwargs=None):
    pygame = _load_pygame()
    pygame.init()
    pygame.display.set_caption("Tic-Tac-Toe / Real robot player")
    screen = pygame.display.set_mode((WINDOW, STATUS_H + 3 * CELL_PX + MARGIN))
    font = pygame.font.SysFont("sans", 22)

    app = None

    try:
        _ensure_ros_node()
        app = ProjectApp(
            grid=grid,
            simulator=RealTicTacToeRobot(**(robot_kwargs or {})),
        )
        print(f"[project_real] ROS backend: {'real rospy' if USING_REAL_ROS else 'mock rospy'}")
        print(
            f"[project_real] Grid origin=({grid.origin_x}, {grid.origin_y}) "
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
    parser.add_argument("--max-eef-speed", type=float, default=DEFAULT_MAX_EEF_LINEAR_SPEED)
    parser.add_argument(
        "--eef-speed-noise-floor",
        type=float,
        default=DEFAULT_EEF_SPEED_NOISE_FLOOR,
        help="Absolute m/s tolerance added to the EEF speed limit to absorb measurement jitter.",
    )
    parser.add_argument(
        "--max-joint-velocity-scale",
        type=float,
        default=DEFAULT_MAX_JOINT_VELOCITY_SCALE,
        help="Fraction (0, 1] of Franka's per-joint velocity limits used for planning/runtime checks.",
    )
    parser.add_argument(
        "--joint-velocity-limits",
        type=float,
        nargs=7,
        default=None,
        help="Override the 7 per-joint velocity limits (rad/s).",
    )
    parser.add_argument("--motion-duration-scale", type=float, default=DEFAULT_MOTION_DURATION_SCALE)
    parser.add_argument("--workspace-margin", type=float, default=DEFAULT_WORKSPACE_MARGIN)
    parser.add_argument("--safety-poll-period", type=float, default=DEFAULT_SAFETY_POLL_PERIOD)
    parser.add_argument("--home-duration", type=float, default=DEFAULT_HOME_DURATION)
    parser.add_argument("--force-thresholds", type=float, nargs=6, default=None)
    parser.add_argument("--torque-thresholds", type=float, nargs=7, default=None)
    parser.add_argument("--disable-workspace-check", action="store_true")
    parser.add_argument("--disable-joint-reachability-check", action="store_true")
    parser.add_argument("--disable-collision-check", action="store_true")
    parser.add_argument(
        "--enable-eef-speed-check",
        action="store_true",
        help="Enable runtime end-effector linear speed monitoring (off by default for joint-position control).",
    )
    parser.add_argument("--disable-joint-velocity-check", action="store_true")
    parser.add_argument("--disable-force-check", action="store_true")
    parser.add_argument("--disable-torque-check", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    grid = GridFrame(args.origin_x, args.origin_y, args.cell, args.place_z, args.frame_id)
    robot_kwargs = {
        "max_eef_linear_speed": args.max_eef_speed,
        "eef_speed_noise_floor": args.eef_speed_noise_floor,
        "max_joint_velocity_scale": args.max_joint_velocity_scale,
        "joint_velocity_limits": args.joint_velocity_limits,
        "motion_duration_scale": args.motion_duration_scale,
        "workspace_margin": args.workspace_margin,
        "safety_poll_period": args.safety_poll_period,
        "force_thresholds": args.force_thresholds,
        "torque_thresholds": args.torque_thresholds,
        "home_duration": args.home_duration,
        "enable_workspace_check": not args.disable_workspace_check,
        "enable_joint_reachability_check": not args.disable_joint_reachability_check,
        "enable_collision_check": not args.disable_collision_check,
        "enable_eef_speed_check": bool(args.enable_eef_speed_check),
        "enable_joint_velocity_check": not args.disable_joint_velocity_check,
        "enable_force_check": not args.disable_force_check,
        "enable_torque_check": not args.disable_torque_check,
    }
    return run(grid, robot_kwargs=robot_kwargs)


if __name__ == "__main__":
    sys.exit(main())
