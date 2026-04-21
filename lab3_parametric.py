import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET

import RobotUtil as rt
from kinematics import FrankArm

ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml" 
MODEL_XML = "franka_emika_panda/panda_torque_table_shelves.xml" 

GRIPPER_OPEN = 0.03
GRIPPER_CLOSED = 0.015

SEGMENT_DURATION_BASE = 0.3
SEGMENT_DURATION_PER_METER = 2.6
HOLD_DURATION_BASE = 0.05
HOLD_DURATION_PER_METER = 1.0

# Controller gains (per-joint)
KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float) * 1.2
KD = np.array([  18,   18,   12,  8,  6,  4,  3], dtype=float) * 2.5

# ---------------------------------------------------------

  
if __name__ == "__main__":
    #Load a model, add objects, and save a new model

    #Set random generator seed for reproducing results (useful for debugging)
    np.random.seed(13)

    #Generate scene and obstacle boxes for collison checking (points and axes)
    EndofTable=0.55+0.135+0.05
    modelTree = ET.parse(ROOT_MODEL_XML)

    BLOCKS=[
        ["TablePlane",[EndofTable-0.275,0.,-0.005],[0.275, 0.504, 0.0051]],
        ["LShelfDistal",[EndofTable-0.09-0.0225, 0.504-0.045-0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["LShelfProximal",[EndofTable-0.55-0.0225, 0.504-0.045-0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["LShelfBack",[EndofTable-0.55-0.0225-0.09, 0.504-0.045-0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["LShelfMid",[EndofTable-0.32, 0.504-0.045-0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["LShelfArch",[EndofTable-0.275-0.135+0.0225, 0.504-0.045-0.0225, 0.63+0.0225],[0.315, 0.0225, 0.0225]],
        ["LShelfBottom",[EndofTable-0.275-0.135+0.0225, 0.504-0.09-0.135/2., 0.1375+0.005],[0.2525, 0.135/2., 0.005]],
        ["LShelfBottomSupp1",[EndofTable-0.55-0.0225-0.09+0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSupp2",[EndofTable-0.32-0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSupp3",[EndofTable-0.09-0.0225-0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["LShelfBottomSuppB",[EndofTable-0.275-0.135+0.0225, 0.504-0.0225,0.1375+0.0225],[0.315, 0.0225, 0.0225]],
        ["RShelfDistal",[EndofTable-0.09-0.0225, -0.504+0.045+0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["RShelfProximal",[EndofTable-0.55-0.0225, -0.504+0.045+0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["RShelfBack",[EndofTable-0.55-0.0225-0.09, -0.504+0.045+0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
        ["RShelfMid",[EndofTable-0.32, -0.504+0.045+0.0225, 0.315],[0.0225, 0.0225, 0.315]],
        ["RShelfArch",[EndofTable-0.275-0.135+0.0225, -0.504+0.045+0.0225, 0.63+0.0225],[0.315, 0.0225, 0.0225]],
        ["RShelfBottom",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005],[0.2525, 0.135/2., 0.005]],
        ["RShelfBottomSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
        ["RShelfBottomSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225],[0.315, 0.0225, 0.0225]],
        ["RShelfMiddle",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.2],[0.2525, 0.135/2., 0.005]],
        ["RShelfMiddleSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
        ["RShelfMiddleSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225+.2],[0.315, 0.0225, 0.0225]],
        ["RShelfTop",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.4],[0.2525, 0.135/2., 0.005]],
        ["RShelfTopSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
        ["RShelfTopSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
        ["RShelfTopSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
        ["RShelfTopSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225+.4],[0.315, 0.0225, 0.0225]],
    ]

    def get_block_pos(blocks, name):
        for b in blocks:
            if b[0] == name:
                return b[1]
        raise ValueError(f"Block {name} not found in BLOCKS")

    def gripper_cmd(g):
        return GRIPPER_OPEN if g <= 0.5 else GRIPPER_CLOSED

    def waypoint_duration(distance, base_duration, distance_scale):
        return base_duration + distance_scale * distance

    def add_in_place_grasp_rotations(cart_waypoints, initial_xyz):
        augmented_waypoints = []
        current_xyz = np.array(initial_xyz, dtype=float)
        for name, wp in cart_waypoints:
            g, x, y, z, r, p, yaw = wp
            if name == "descend_to_grasp_pre":
                augmented_waypoints.append(
                    ("rotate_to_grasp_orientation", [g, current_xyz[0], current_xyz[1], current_xyz[2], r, p, yaw])
                )
            augmented_waypoints.append((name, wp))
            current_xyz = np.array([x, y, z], dtype=float)
        return augmented_waypoints

    BLOCK_X_OFFSET = -0.13
    BLOCK_ROW_START_X = EndofTable - 0.255 + BLOCK_X_OFFSET
    BLOCK_ROW_SPACING_X = 0.075

    #row1
    BLOCK_POSE_1 = [BLOCK_ROW_START_X + 0 * BLOCK_ROW_SPACING_X, 0.20, 0.05]
    BLOCK_POSE_2 = [BLOCK_ROW_START_X + 1 * BLOCK_ROW_SPACING_X, 0.20, 0.05]
    BLOCK_POSE_3 = [BLOCK_ROW_START_X + 2 * BLOCK_ROW_SPACING_X, 0.20, 0.05]
    BLOCK_POSE_4 = [BLOCK_ROW_START_X + 3 * BLOCK_ROW_SPACING_X, 0.20, 0.05]

    #row2
    BLOCK_POSE_5 = [BLOCK_ROW_START_X + 0 * BLOCK_ROW_SPACING_X, -0.20, 0.05]
    BLOCK_POSE_6 = [BLOCK_ROW_START_X + 1 * BLOCK_ROW_SPACING_X, -0.20, 0.05]
    BLOCK_POSE_7 = [BLOCK_ROW_START_X + 2 * BLOCK_ROW_SPACING_X, -0.20, 0.05]
    BLOCK_POSE_8 = [BLOCK_ROW_START_X + 3 * BLOCK_ROW_SPACING_X, -0.20, 0.05]

    DROPOFF_CENTER = [EndofTable - 0.145 + BLOCK_X_OFFSET, 0.0, 0.05]
    DROPOFF_DX = 0.065
    DROPOFF_DY = 0.065
    DROPOFF_POSITIONS = [
        [
            [DROPOFF_CENTER[0] - DROPOFF_DX, DROPOFF_CENTER[1] + DROPOFF_DY, DROPOFF_CENTER[2]],
            [DROPOFF_CENTER[0],             DROPOFF_CENTER[1] + DROPOFF_DY, DROPOFF_CENTER[2]],
            [DROPOFF_CENTER[0] + DROPOFF_DX, DROPOFF_CENTER[1] + DROPOFF_DY, DROPOFF_CENTER[2]],
        ],
        [
            [DROPOFF_CENTER[0] - DROPOFF_DX, DROPOFF_CENTER[1],             DROPOFF_CENTER[2]],
            [DROPOFF_CENTER[0],             DROPOFF_CENTER[1],             DROPOFF_CENTER[2]],
            [DROPOFF_CENTER[0] + DROPOFF_DX, DROPOFF_CENTER[1],             DROPOFF_CENTER[2]],
        ],
        [
            [DROPOFF_CENTER[0] - DROPOFF_DX, DROPOFF_CENTER[1] - DROPOFF_DY, DROPOFF_CENTER[2]],
            [DROPOFF_CENTER[0],             DROPOFF_CENTER[1] - DROPOFF_DY, DROPOFF_CENTER[2]],
            [DROPOFF_CENTER[0] + DROPOFF_DX, DROPOFF_CENTER[1] - DROPOFF_DY, DROPOFF_CENTER[2]],
        ],
    ]

    SPAWN_BLOCKS = [
        {"name": "Block", "pos": BLOCK_POSE_1, "rgba": [0.0, 0.2, 0.9, 1.0]},
        {"name": "Block_2", "pos": BLOCK_POSE_2, "rgba": [0.0, 0.2, 0.9, 1.0]},
        {"name": "Block_3", "pos": BLOCK_POSE_3, "rgba": [0.0, 0.2, 0.9, 1.0]},
        {"name": "Block_4", "pos": BLOCK_POSE_4, "rgba": [0.0, 0.2, 0.9, 1.0]},
        {"name": "Block_5", "pos": BLOCK_POSE_5, "rgba": [0.9, 0.1, 0.1, 1.0]},
        {"name": "Block_6", "pos": BLOCK_POSE_6, "rgba": [0.9, 0.1, 0.1, 1.0]},
        {"name": "Block_7", "pos": BLOCK_POSE_7, "rgba": [0.9, 0.1, 0.1, 1.0]},
        {"name": "Block_8", "pos": BLOCK_POSE_8, "rgba": [0.9, 0.1, 0.1, 1.0]},
    ]

    block_xyzs = [
        BLOCK_POSE_1,
        BLOCK_POSE_2,
        BLOCK_POSE_3,
        BLOCK_POSE_4,
        BLOCK_POSE_5,
        BLOCK_POSE_6,
        BLOCK_POSE_7,
        BLOCK_POSE_8,
    ]
    block_xyzs_grid = [
        [BLOCK_POSE_1, BLOCK_POSE_2, BLOCK_POSE_3, BLOCK_POSE_4],
        [BLOCK_POSE_5, BLOCK_POSE_6, BLOCK_POSE_7, BLOCK_POSE_8],
    ]
    left_shelf_xyz = get_block_pos(BLOCKS, "LShelfBottom")

    pi = np.pi
    down_grasp_y = (3.13714399, 0.00419659, -0.80013718 + pi/2)
    side_grasp = (1.53, 0.77, 3.12)
    down_grasp_x = (3.13714399, 0.00419659, -0.80013718)

    def resolve_pick_xyz(pick_pos):
        if len(pick_pos) == 3 and pick_pos[0] == "dropoff":
            _, row, col = pick_pos
            return DROPOFF_POSITIONS[row][col]
        row, col = pick_pos
        return block_xyzs_grid[row][col]

    def resolve_place_xyz(place_pos):
        if len(place_pos) == 3 and place_pos[0] == "pickup":
            _, row, col = place_pos
            return block_xyzs_grid[row][col]
        row, col = place_pos
        return DROPOFF_POSITIONS[row][col]

    def build_cart_waypoints(grasp_block, target_pos, grasp_orientation, place_orientation):
        grasp_xyz = resolve_pick_xyz(grasp_block)
        dropoff_xyz = resolve_place_xyz(target_pos)
        halfway_xyz = (0.6 * np.array(grasp_xyz, dtype=float) + 0.4 * np.array(dropoff_xyz, dtype=float))
        return [
            ("descend_to_grasp_pre", [0.4, grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + 0.13, *grasp_orientation]),
            ("descend_to_grasp", [0.4, grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + 0.07, *grasp_orientation]),
            ("close_gripper", [0.7, grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + 0.07, *grasp_orientation]),
            ("lift_block", [0.7, grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + 0.15, *grasp_orientation]),
            ("move_halfway_to_dropoff", [1, halfway_xyz[0], halfway_xyz[1], halfway_xyz[2] + 0.15, *place_orientation]),

            ("move_above_dropoff", [0.7, dropoff_xyz[0], dropoff_xyz[1], dropoff_xyz[2] + 0.15, *place_orientation]),
            ("lower_to_dropoff", [0.7, dropoff_xyz[0], dropoff_xyz[1], dropoff_xyz[2] + 0.07, *place_orientation]),
            ("open_at_dropoff", [0.4, dropoff_xyz[0], dropoff_xyz[1], dropoff_xyz[2] + 0.07, *place_orientation]),
            ("retract_from_dropoff", [0.4, dropoff_xyz[0], dropoff_xyz[1], dropoff_xyz[2] + 0.15, *place_orientation]),
        ]

    grasp_block = (0, 0)
    target_pos = (0, 0)
    raw_cart_waypoints = (
        # #1
        build_cart_waypoints((0,0), (0,0), down_grasp_y, down_grasp_y) + 
        build_cart_waypoints((0,1), (1,0), down_grasp_y, down_grasp_y) +
        build_cart_waypoints((0,2), (2,0), down_grasp_y, down_grasp_y) +
        build_cart_waypoints((1,0), (1,1), down_grasp_y, down_grasp_x) +
        build_cart_waypoints((1,1), (1,2), down_grasp_y, down_grasp_x)
        +
        #3
        build_cart_waypoints(("dropoff", 1,1), (0,1), down_grasp_x, down_grasp_x) + 
        build_cart_waypoints(("dropoff", 1,2), (2,1), down_grasp_x, down_grasp_x) +
        build_cart_waypoints(("dropoff", 1,0), (0,2), down_grasp_y, down_grasp_x) +
        build_cart_waypoints((0,3), (2,2), down_grasp_y, down_grasp_x) +
        build_cart_waypoints((1,2), (1,0), down_grasp_y, down_grasp_y) + 
        build_cart_waypoints((1,3), (1,2), down_grasp_y, down_grasp_y)
        +
        #4
        build_cart_waypoints(("dropoff", 0,1), ("pickup", 1,1), down_grasp_x, down_grasp_y) + 
        build_cart_waypoints(("dropoff", 2,1), ("pickup", 1,2), down_grasp_x, down_grasp_y) + 
        build_cart_waypoints(("dropoff", 0,2), (2,1), down_grasp_y, down_grasp_x) + 
        build_cart_waypoints(("dropoff", 0,0), ("pickup", 1,3), down_grasp_y, down_grasp_y) + 
        build_cart_waypoints((1,1), (0,1), down_grasp_y, down_grasp_x)
    )
    arm = FrankArm()
    init_pose_jointspace = [0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04]
    T_init_fk, _ = arm.ForwardKin(init_pose_jointspace[:7])
    init_ee_xyz = T_init_fk[-1][0:3, 3].copy()
    cart_waypoints = add_in_place_grasp_rotations(raw_cart_waypoints, init_ee_xyz)
    q_seed = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]
    WAYPOINTS = []
    for idx, (name, wp) in enumerate(cart_waypoints):
        g, x, y, z, r, p, yaw = wp
        T_goal = rt.rpyxyz2H([r, p, yaw], [x, y, z])
        q_sol, _ = arm.IterInvKin(q_seed, T_goal)
        T_fk, _ = arm.ForwardKin(q_sol)
        T_fk = T_fk[-1]
        p_fk = T_fk[0:3, 3]
        R_fk = T_fk[0:3, 0:3]
        p_t = T_goal[0:3, 3]
        R_t = T_goal[0:3, 0:3]
        rpy_fk = rt.R2rpy(R_fk)
        print(f"[WP {idx}] {name} target xyz: {p_t}, target R:\n{R_t}")
        print(f"[WP {idx}] fk xyz:     {p_fk}, fk R:\n{R_fk}")
        print(f"[WP {idx}] fk rpy:     {rpy_fk}")
        q_seed = q_sol
        WAYPOINTS.append(q_sol + [gripper_cmd(g)])
    WAYPOINTS.insert(0, init_pose_jointspace)
    WAYPOINTS = np.array(WAYPOINTS, dtype=float)
    input("Finished computing IK for all waypoints. Press Enter to proceed...")

    ee_positions = [T_init_fk[-1][0:3, 3].copy()] + [np.array(wp[1:4], dtype=float) for _, wp in cart_waypoints]
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

    for i in range(len(BLOCKS)):
        rt.add_free_block_to_model(tree=modelTree, name=BLOCKS[i][0], pos=BLOCKS[i][1], density= 20 , size=BLOCKS[i][2] , rgba=[0.5, 0.5, 0.5, 1],free=False)

    for block in SPAWN_BLOCKS:
        rt.add_free_block_to_model(
            tree=modelTree, name=block["name"], pos=block["pos"],
            density=20, size=[0.02, 0.02, 0.02], rgba=block["rgba"], free=True)

    modelTree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)
        
    ###### EXECUTE PLAN ######

    #Load the model
    model = mj.MjModel.from_xml_path(MODEL_XML)
    data = mj.MjData(model)
    arm_idx = [0,1,2,3,4,5,6]
    gripper_idx = 7
    link7_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link7")

    # Initialize arm at first waypoint
    data.qpos[arm_idx] = WAYPOINTS[0][arm_idx]
    data.qvel[arm_idx] = 0.0
    mj.mj_forward(model, data)

    dt = model.opt.timestep

    #Launch Viewer
    v = viewer.launch_passive(model, data)
    v.cam.distance=3.0 #Shift camera 
    v.cam.azimuth += 90
    

    try:
        #Step through each waypoint
        for i in range(len(WAYPOINTS) - 1):
            wp_name, wp = cart_waypoints[i]
            g, x, y, z, r, p, yaw = wp
            q_target = WAYPOINTS[i + 1][arm_idx].copy()
            segment_duration = segment_durations[i]
            hold_duration = hold_durations[i]
            segment_steps = max(1, int(segment_duration / dt))
            hold_steps = max(0, int(hold_duration / dt))
            print(f"Moving to: {i+1}, {wp_name}, xyz=({x:.4f}, {y:.4f}, {z:.4f}), "
                  f"rpy=({r:.4f}, {p:.4f}, {yaw:.4f}), q={q_target}, "
                  f"segment_duration={segment_duration:.3f}, hold_duration={hold_duration:.3f}")
            q_start = WAYPOINTS[i][arm_idx].copy()
            q_goal  = WAYPOINTS[i + 1][arm_idx].copy()
            t = 0.0
            #Compute and control along a minjerk trajectory for each waypoint 
            for k in range(segment_steps+hold_steps):
                # Desired joint state 
                q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, segment_duration)

                # Current joint state
                q = data.qpos[arm_idx].copy()
                qd = data.qvel[arm_idx].copy()

                # Compute PD + gravity compensation torque
                data.ctrl[arm_idx] = 0.0
                tau = KP * (q_des - q) + KD * (qd_des - qd)
                data.ctrl[arm_idx] = tau+data.qfrc_bias[:7]
                data.ctrl[gripper_idx] = WAYPOINTS[i][-1]

                # Step sim
                mj.mj_step(model, data)
                # if k % 50 == 0:
                #     gt_pos = data.xpos[link7_id].copy()
                #     gt_R = data.xmat[link7_id].reshape(3, 3).copy()
                #     gt_rpy = rt.R2rpy(gt_R)
                #     print(f"[WP {i+1} step {k}] GT link7 xyz: {gt_pos}, rpy: {gt_rpy}")
                v.sync()
                t += dt


    finally:
        if v is not None:
            v.close() #cleanup viewer at the end
