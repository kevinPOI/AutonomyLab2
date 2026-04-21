import argparse
import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET
import time

import RobotUtil as rt
from kinematics import FrankArm

ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
MODEL_XML = "franka_emika_panda/panda_torque_table_shelves.xml"

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.008

SEGMENT_DURATION = 2.0
HOLD_DURATION = 0.5
RENDER_SKIP = 5

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float)

Q_HOME = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]

R_DOWN       = (3.14, 0.00419659, -0.80013718 + np.pi / 2.0)
R_SIDE_LEFT  = (1.53, 0.77, 3.12)
R_SIDE_RIGHT = (1.53, 0.77, 0.0)

EndofTable = 0.55 + 0.135 + 0.05

BLOCKS = [
    ["TablePlane",        [EndofTable-0.275, 0., -0.005],                                        [0.275, 0.504, 0.0051]],
    ["LShelfDistal",      [EndofTable-0.09-0.0225, 0.504-0.045-0.0225, 0.315],                  [0.0225, 0.0225, 0.315]],
    ["LShelfProximal",    [EndofTable-0.55-0.0225, 0.504-0.045-0.0225, 0.3825-0.135],           [0.0225, 0.0225, 0.3825]],
    ["LShelfBack",        [EndofTable-0.55-0.0225-0.09, 0.504-0.045-0.0225, 0.3825-0.135],      [0.0225, 0.0225, 0.3825]],
    ["LShelfMid",         [EndofTable-0.32, 0.504-0.045-0.0225, 0.315],                         [0.0225, 0.0225, 0.315]],
    ["LShelfArch",        [EndofTable-0.275-0.135+0.0225, 0.504-0.045-0.0225, 0.63+0.0225],     [0.315, 0.0225, 0.0225]],
    ["LShelfBottom",      [EndofTable-0.275-0.135+0.0225, 0.504-0.09-0.135/2., 0.1375+0.005],   [0.2525, 0.135/2., 0.005]],
    ["LShelfBottomSupp1", [EndofTable-0.55-0.0225-0.09+0.045, 0.504-0.225/2., 0.1375-0.0225],   [0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSupp2", [EndofTable-0.32-0.045, 0.504-0.225/2., 0.1375-0.0225],               [0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSupp3", [EndofTable-0.09-0.0225-0.045, 0.504-0.225/2., 0.1375-0.0225],        [0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSuppB", [EndofTable-0.275-0.135+0.0225, 0.504-0.0225, 0.1375+0.0225],         [0.315, 0.0225, 0.0225]],
    ["RShelfDistal",      [EndofTable-0.09-0.0225, -0.504+0.045+0.0225, 0.315],                 [0.0225, 0.0225, 0.315]],
    ["RShelfProximal",    [EndofTable-0.55-0.0225, -0.504+0.045+0.0225, 0.3825-0.135],          [0.0225, 0.0225, 0.3825]],
    ["RShelfBack",        [EndofTable-0.55-0.0225-0.09, -0.504+0.045+0.0225, 0.3825-0.135],     [0.0225, 0.0225, 0.3825]],
    ["RShelfMid",         [EndofTable-0.32, -0.504+0.045+0.0225, 0.315],                        [0.0225, 0.0225, 0.315]],
    ["RShelfArch",        [EndofTable-0.275-0.135+0.0225, -0.504+0.045+0.0225, 0.63+0.0225],    [0.315, 0.0225, 0.0225]],
    ["RShelfBottom",      [EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005],  [0.2525, 0.135/2., 0.005]],
    ["RShelfBottomSupp1", [EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225],  [0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSupp2", [EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225],              [0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSupp3", [EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225],       [0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSuppB", [EndofTable-0.275-0.135+0.0225, -0.504+0.0225, 0.1375+0.0225],        [0.315, 0.0225, 0.0225]],
    ["RShelfMiddle",      [EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.2],  [0.2525, 0.135/2., 0.005]],
    ["RShelfMiddleSupp1", [EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.2],  [0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSupp2", [EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],              [0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSupp3", [EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],       [0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSuppB", [EndofTable-0.275-0.135+0.0225, -0.504+0.0225, 0.1375+0.0225+.2],        [0.315, 0.0225, 0.0225]],
    ["RShelfTop",         [EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.4],  [0.2525, 0.135/2., 0.005]],
    ["RShelfTopSupp1",    [EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.4],  [0.0225, 0.1125, 0.0225]],
    ["RShelfTopSupp2",    [EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],              [0.0225, 0.1125, 0.0225]],
    ["RShelfTopSupp3",    [EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],       [0.0225, 0.1125, 0.0225]],
    ["RShelfTopSuppB",    [EndofTable-0.275-0.135+0.0225, -0.504+0.0225, 0.1375+0.0225+.4],        [0.315, 0.0225, 0.0225]],
]

TRIAL_CONFIGS = [
    {"label": "Trial 1 -> Right Middle Shelf",
     "right_shelf": "RShelfMiddle",  "left_dx":  0.00, "right_dx":  0.00},
    {"label": "Trial 2 -> Right Bottom Shelf",
     "right_shelf": "RShelfBottom",  "left_dx": -0.01, "right_dx":  0.02},
    {"label": "Trial 3 -> Right Top Shelf",
     "right_shelf": "RShelfTop",     "left_dx":  0.01, "right_dx": -0.02},
]

### random range for point
# LEFT_DX_RANGE = (-0.035, 0.035)
# RIGHT_DX_RANGE = (-0.035, 0.035)
LEFT_DX_RANGE = (-0.1, 0.1)
RIGHT_DX_RANGE = (-0.1, 0.1)
# Which right shelf each trial targets when using --sample (cycled if n_trials > len)
SAMPLE_RIGHT_SHELVES = ["RShelfMiddle", "RShelfBottom", "RShelfTop"]

# Mocap bodies (added to MJCF) for showing planned release poses in the viewer
VIZ_BODY_LEFT = "viz_place_left"
VIZ_BODY_RIGHT = "viz_place_right"
FREE_BLOCK_SIZE = [0.02, 0.02, 0.02]
FREE_BLOCK_RGBA = [0.0, 0.9, 0.2, 1]
BLOCK_X_OFFSET = -0.1
#row1
BLOCK_POSE_1 = [EndofTable - 0.235 + BLOCK_X_OFFSET, 0.22, 0.05]
BLOCK_POSE_2 = [EndofTable - 0.175 + BLOCK_X_OFFSET, 0.22, 0.05]
BLOCK_POSE_3 = [EndofTable - 0.115 + BLOCK_X_OFFSET, 0.22, 0.05]
BLOCK_POSE_4 = [EndofTable - 0.055 + BLOCK_X_OFFSET, 0.22, 0.05]

#row2
BLOCK_POSE_5 = [EndofTable - 0.235 + BLOCK_X_OFFSET, -0.22, 0.05]
BLOCK_POSE_6 = [EndofTable - 0.175 + BLOCK_X_OFFSET, -0.22, 0.05]
BLOCK_POSE_7 = [EndofTable - 0.115 + BLOCK_X_OFFSET, -0.22, 0.05]
BLOCK_POSE_8 = [EndofTable - 0.055 + BLOCK_X_OFFSET, -0.22, 0.05]

SPAWN_BLOCKS = [
    {"name": "Block", "pos": BLOCK_POSE_1},
    {"name": "Block_2", "pos": BLOCK_POSE_2},
    {"name": "Block_3", "pos": BLOCK_POSE_3},
    {"name": "Block_4", "pos": BLOCK_POSE_4},
    {"name": "Block_5", "pos": BLOCK_POSE_5},
    {"name": "Block_6", "pos": BLOCK_POSE_6},
    {"name": "Block_7", "pos": BLOCK_POSE_7},
    {"name": "Block_8", "pos": BLOCK_POSE_8},
]


def add_mocap_marker_sphere(tree, name, radius=0.012, rgba="1 0 0 0.4"):
    # mocap marker for viz only
    worldbody = tree.getroot().find("worldbody")
    b = ET.SubElement(worldbody, "body", {"name": name, "mocap": "true"})
    ET.SubElement(b, "geom", {
        "type": "sphere",
        "size": str(radius),
        "rgba": rgba,
        "contype": "0",
        "conaffinity": "0",
    })


def release_xyz_from_refs(left_ref, right_ref):
    # where the gripper opens in phase1 / phase2
    lx, ly, lz = left_ref
    rx, ry, rz = right_ref
    p_left = np.array([lx, ly - 0.05, lz + 0.08], dtype=float)
    p_right = np.array([rx, ry + 0.10, rz + 0.03], dtype=float)
    return p_left, p_right


def update_place_markers(model, data, left_ref, right_ref):
    p_l, p_r = release_xyz_from_refs(left_ref, right_ref)
    bid_l = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, VIZ_BODY_LEFT)
    bid_r = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, VIZ_BODY_RIGHT)
    mid_l = model.body_mocapid[bid_l]
    mid_r = model.body_mocapid[bid_r]
    data.mocap_pos[mid_l] = p_l
    data.mocap_pos[mid_r] = p_r
    mj.mj_forward(model, data)


def sample_trial_configs(rng, n_trials, shelf_names):
    cfgs = []
    for i in range(n_trials):
        name = shelf_names[i % len(shelf_names)]
        ldx = float(rng.uniform(*LEFT_DX_RANGE))
        rdx = float(rng.uniform(*RIGHT_DX_RANGE))
        cfgs.append({
            "label": f"Trial {i + 1} -> {name}  (left_dx={ldx:+.4f}, right_dx={rdx:+.4f})",
            "right_shelf": name,
            "left_dx": ldx,
            "right_dx": rdx,
        })
    return cfgs

# --------------------------------------------------------------------- #
#                          Helper functions                              #
# --------------------------------------------------------------------- #

def get_block_pos(blocks, name):
    for b in blocks:
        if b[0] == name:
            return list(b[1])
    raise ValueError(f"Block {name} not found")


def gripper_cmd(g):
    return GRIPPER_OPEN if g <= 0.5 else GRIPPER_CLOSED


def build_full_wps(block_xyz, left_ref, right_ref):
    # ref IK only
    return build_phase1_wps(block_xyz, left_ref) + \
           build_phase2_wps([left_ref[0], left_ref[1]-0.05, left_ref[2]+0.02],
                            left_ref, right_ref, block_xyz)


def build_phase1_wps(block_xyz, left_ref):
    rd, pd, yd = R_DOWN
    lx, ly, lz = left_ref
    return [
        ("pre-descend_to_grasp",    [0, block_xyz[0], block_xyz[1], block_xyz[2]+0.27,  rd, pd, yd]),
        ("descend_to_grasp",    [0, block_xyz[0], block_xyz[1], block_xyz[2]+0.07,  rd, pd, yd]),
        ("close_gripper",       [1, block_xyz[0], block_xyz[1], block_xyz[2]+0.07,  rd, pd, yd]),
        ("lift_block",          [1, block_xyz[0], block_xyz[1], block_xyz[2]+0.15,  rd, pd, yd]),
        ("above_left_shelf",    [1, lx, ly-0.06, lz+0.22,  rd, pd, yd]),
        ("lower_to_left",       [1, lx, ly-0.05, lz+0.08,  rd, pd, yd]),
        ("release_on_left",     [0, lx, ly-0.05, lz+0.08,  rd, pd, yd]),
        ("retract_up",          [0, lx, ly-0.05, lz+0.22,  rd, pd, yd]),
        ("retract_clear",       [0, lx, ly-0.24, lz+0.24,  rd, pd, yd]),
    ]


def build_phase2_wps(block_pos, left_ref, right_ref, block_xyz):
    rd, pd, yd = R_DOWN
    rsl, psl, ysl = R_SIDE_LEFT
    rsr, psr, ysr = R_SIDE_RIGHT
    lx, ly, lz = left_ref
    rx, ry, rz = right_ref
    bx, by, bz = block_pos

    mid_z = max(0.25, rz - 0.04)

    return [
        ("side_reorient",       [0, bx, ly-0.24, bz+0.08,      rsl, psl, ysl]),
        ("side_prepose",        [0, bx, ly-0.20, bz,            rsl, psl, ysl]),
        ("side_approach",       [0, bx, by-0.1, bz,            rsl, psl, ysl]),
        ("side_close",          [1, bx, by-0.1, bz,            rsl, psl, ysl]),
        ("side_lift",           [1, bx+0.03, ly-0.15, bz+0.10,  rsl, psl, ysl]),
        ("orient_down",         [1, bx+0.03, ly-0.15, bz+0.10,  rd, pd, yd]),
        ("turn_to_right",       [1, bx+0.03, ly-0.20, bz+0.10,  rsr, psr, ysr]),
        ("transfer_mid",        [1, (bx+rx)/2, (ly+ry)/2, mid_z, rsr, psr, ysr]),
        ("above_right_shelf",   [1, rx, ry+0.10, rz+0.08,       rsr, psr, ysr]),
        ("lower_to_right",      [1, rx, ry+0.10, rz+0.03,       rsr, psr, ysr]),
        ("release_on_right",    [0, rx, ry+0.10, rz+0.03,       rsr, psr, ysr]),
        ("retract_right",       [0, rx, ry+0.18, rz+0.08,       rsr, psr, ysr]),
        ("return_hover",        [0, block_xyz[0], block_xyz[1], block_xyz[2]+0.20,  rd, pd, yd]),
    ]


def _unwrap(q_new, q_ref):
    # keep angles near previous soln so IK doesnt jump weirdly
    q = np.array(q_new, dtype=float)
    r = np.array(q_ref, dtype=float)
    q -= np.round((q - r) / (2 * np.pi)) * (2 * np.pi)
    return q


def _solve_with_seed(arm, seed, T_goal, q_prev):
    q_raw, _ = arm.IterInvKin(seed, T_goal)
    q = _unwrap(q_raw[:7], q_prev)
    jump = np.max(np.abs(q - q_prev))
    return q, jump


def solve_ik_chain(cart_wps, arm, q_seed, ref_seeds=None, return_home=True):
    JUMP_THRESH = 1.0

    q_s = np.array(q_seed, dtype=float)
    jnt_wps = [list(q_seed) + [GRIPPER_OPEN]]

    for idx, (_, wp) in enumerate(cart_wps):
        g, x, y, z, r, p, yaw = wp
        T_goal = rt.rpyxyz2H([r, p, yaw], [x, y, z])

        q_new, jump = _solve_with_seed(arm, q_s.tolist(), T_goal, q_s)

        if jump > JUMP_THRESH and ref_seeds is not None and idx < len(ref_seeds):
            q_ref, jump_ref = _solve_with_seed(
                arm, ref_seeds[idx].tolist(), T_goal, q_s)
            if jump_ref < jump:
                q_new, jump = q_ref, jump_ref

        if jump > JUMP_THRESH:
            q_home, jump_home = _solve_with_seed(arm, Q_HOME, T_goal, q_s)
            if jump_home < jump:
                q_new, jump = q_home, jump_home

        q_s = q_new.copy()
        jnt_wps.append(q_new.tolist() + [gripper_cmd(g)])

    if return_home:
        q_home = _unwrap(Q_HOME, q_s)
        jnt_wps.append(q_home.tolist() + [GRIPPER_OPEN])

    return np.array(jnt_wps, dtype=float)


def _pace(wall_t0, sim_t):
    drift = sim_t - (time.time() - wall_t0)
    if drift > 0:
        time.sleep(drift)


def run_trajectory(model, data, v, jnt_wps, cart_wps):
    arm_idx = list(range(7))
    dt = model.opt.timestep
    seg_steps = max(1, int(SEGMENT_DURATION / dt))
    hold_steps = int(HOLD_DURATION / dt)

    for i in range(len(jnt_wps) - 1):
        name = cart_wps[i][0] if i < len(cart_wps) else "done"
        q_start = jnt_wps[i][arm_idx].copy()
        q_goal  = jnt_wps[i + 1][arm_idx].copy()

        t = 0.0
        wall_t0 = time.time()
        for step in range(seg_steps + hold_steps):
            q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, SEGMENT_DURATION)
            q  = data.qpos[arm_idx].copy()
            qd = data.qvel[arm_idx].copy()
            tau = KP * (q_des - q) + KD * (qd_des - qd)
            data.ctrl[arm_idx] = tau + data.qfrc_bias[:7]
            data.ctrl[7] = jnt_wps[i][-1]
            mj.mj_step(model, data)
            t += dt
            if step % RENDER_SKIP == 0:
                v.sync()
                _pace(wall_t0, t)


def reset_sim(model, data, v):
    arm_idx = list(range(7))
    dt = model.opt.timestep

    mj.mj_resetData(model, data)
    data.qpos[arm_idx] = Q_HOME
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)

    q_home = np.array(Q_HOME)
    wall_t0 = time.time()
    sim_t = 0.0
    for step in range(int(1.5 / dt)):
        tau = KP * (q_home - data.qpos[arm_idx]) + KD * (0.0 - data.qvel[arm_idx])
        data.ctrl[arm_idx] = tau + data.qfrc_bias[:7]
        data.ctrl[7] = GRIPPER_OPEN
        mj.mj_step(model, data)
        sim_t += dt
        if step % RENDER_SKIP == 0:
            v.sync()
            _pace(wall_t0, sim_t)


# --------------------------------------------------------------------- #
#                              Main                                     #
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    np.random.seed(args.seed)

    if args.sample:
        rng = np.random.default_rng(args.seed)
        trial_configs = sample_trial_configs(rng, args.trials, SAMPLE_RIGHT_SHELVES)
    else:
        if args.trials <= len(TRIAL_CONFIGS):
            trial_configs = TRIAL_CONFIGS[: args.trials]
        else:
            trial_configs = [TRIAL_CONFIGS[i % len(TRIAL_CONFIGS)] for i in range(args.trials)]

    modelTree = ET.parse(ROOT_MODEL_XML)

    block_xyz = list(BLOCK_POSE_1)
    left_shelf_base = get_block_pos(BLOCKS, "LShelfBottom")

    for b in BLOCKS:
        rt.add_free_block_to_model(
            tree=modelTree, name=b[0], pos=b[1],
            density=20, size=b[2], rgba=[0.2, 0.2, 0.9, 1], free=False)
    for block in SPAWN_BLOCKS:
        rt.add_free_block_to_model(
            tree=modelTree, name=block["name"], pos=block["pos"],
            density=20, size=FREE_BLOCK_SIZE, rgba=FREE_BLOCK_RGBA, free=True)
    add_mocap_marker_sphere(modelTree, VIZ_BODY_LEFT)
    add_mocap_marker_sphere(modelTree, VIZ_BODY_RIGHT)
    modelTree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)

    model = mj.MjModel.from_xml_path(MODEL_XML)
    data = mj.MjData(model)
    arm_idx = list(range(7))

    data.qpos[arm_idx] = Q_HOME
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)

    v = viewer.launch_passive(model, data)
    v.cam.distance = 3.0
    v.cam.azimuth += 90

    arm = FrankArm()

    # Pre-compute reference IK chain (middle shelf, no offsets) for seed fallback
    ref_left = list(left_shelf_base)
    ref_right = get_block_pos(BLOCKS, "RShelfMiddle")
    ref_cart = build_full_wps(block_xyz, ref_left, ref_right)
    N_P1 = len(build_phase1_wps(block_xyz, ref_left))
    ref_jnt = solve_ik_chain(ref_cart, arm, Q_HOME)
    ref_seeds_all = ref_jnt[1:-1, :7]
    ref_seeds_p1 = ref_seeds_all[:N_P1]
    ref_seeds_p2 = ref_seeds_all[N_P1:]

    block_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "Block")

    try:
        for t_idx, cfg in enumerate(trial_configs):
            left_ref = [
                left_shelf_base[0] + cfg["left_dx"],
                left_shelf_base[1],
                left_shelf_base[2],
            ]
            right_base = get_block_pos(BLOCKS, cfg["right_shelf"])
            right_ref = [
                right_base[0] + cfg["right_dx"],
                right_base[1],
                right_base[2],
            ]

            update_place_markers(model, data, left_ref, right_ref)
            v.sync()

            # --- Phase 1: pick from table, place on left shelf, retract ---
            p1_wps = build_phase1_wps(block_xyz, left_ref)
            p1_jnt = solve_ik_chain(p1_wps, arm, Q_HOME,
                                    ref_seeds=ref_seeds_p1, return_home=False)
            run_trajectory(model, data, v, p1_jnt, p1_wps)

            # Read actual block position from simulation
            actual_pos = data.xpos[block_body_id].copy()

            # --- Phase 2: side grasp (actual pos) -> transfer -> right shelf ---
            q_seed_p2 = p1_jnt[-1, :7].tolist()
            p2_wps = build_phase2_wps(actual_pos, left_ref, right_ref, block_xyz)
            p2_jnt = solve_ik_chain(p2_wps, arm, q_seed_p2,
                                    ref_seeds=ref_seeds_p2, return_home=True)
            run_trajectory(model, data, v, p2_jnt, p2_wps)

            if t_idx < len(trial_configs) - 1:
                reset_sim(model, data, v)
        dt = model.opt.timestep
        q_home = np.array(Q_HOME)
        wall_t0 = time.time()
        sim_t = 0.0
        for step in range(int(3.0 / dt)):
            tau = KP * (q_home - data.qpos[arm_idx]) + KD * (0.0 - data.qvel[arm_idx])
            data.ctrl[arm_idx] = tau + data.qfrc_bias[:7]
            data.ctrl[7] = GRIPPER_OPEN
            mj.mj_step(model, data)
            sim_t += dt
            if step % RENDER_SKIP == 0:
                v.sync()
                _pace(wall_t0, sim_t)

    finally:
        if v is not None:
            v.close()
