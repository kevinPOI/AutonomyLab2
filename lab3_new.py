import numpy as np
import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET

import RobotUtil as rt
from kinematics import FrankArm

ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml" 
MODEL_XML = "franka_emika_panda/panda_torque_table_shelves.xml" 

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0125

SEGMENT_DURATION = 1.0
HOLD_DURATION = 0.3

# Controller gains (per-joint)
KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float)

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

    DROPOFF_CENTER = [EndofTable - 0.145 + BLOCK_X_OFFSET, 0.0, 0.05]
    DROPOFF_DX = 0.125
    DROPOFF_DY = 0.075
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
        {"name": "Block", "pos": BLOCK_POSE_1},
        {"name": "Block_2", "pos": BLOCK_POSE_2},
        {"name": "Block_3", "pos": BLOCK_POSE_3},
        {"name": "Block_4", "pos": BLOCK_POSE_4},
        {"name": "Block_5", "pos": BLOCK_POSE_5},
        {"name": "Block_6", "pos": BLOCK_POSE_6},
        {"name": "Block_7", "pos": BLOCK_POSE_7},
        {"name": "Block_8", "pos": BLOCK_POSE_8},
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
    left_shelf_xyz = get_block_pos(BLOCKS, "LShelfBottom")
    right_shelf_xyz = DROPOFF_POSITIONS[0][0]

    pi = np.pi
    r_down, p_down, y_down = 3.13714399, 0.00419659, -0.80013718 + pi/2
    r_side, p_side, y_side = 1.53, 0.77, 3.12
    cart_waypoints = [
        ("descend_to_grasp_pre", [0, block_xyzs[0][0], block_xyzs[0][1], block_xyzs[0][2] + 0.22, r_down, p_down, y_down]),
        ("descend_to_grasp", [0, block_xyzs[0][0], block_xyzs[0][1], block_xyzs[0][2] + 0.07, r_down, p_down, y_down]),
        ("close_gripper", [1, block_xyzs[0][0], block_xyzs[0][1], block_xyzs[0][2] + 0.07, r_down, p_down, y_down]),
        ("lift_block", [1, block_xyzs[0][0], block_xyzs[0][1], block_xyzs[0][2] + 0.15, r_down, p_down, y_down]),

        ("move_above_dropoff_0_0", [1, right_shelf_xyz[0], right_shelf_xyz[1], right_shelf_xyz[2] + 0.15, r_down, p_down, y_down]),
        ("lower_to_dropoff_0_0", [1, right_shelf_xyz[0], right_shelf_xyz[1], right_shelf_xyz[2] + 0.07, r_down, p_down, y_down]),
        ("open_at_dropoff_0_0", [0, right_shelf_xyz[0], right_shelf_xyz[1], right_shelf_xyz[2] + 0.07, r_down, p_down, y_down]),
        ("retract_from_dropoff_0_0", [0, right_shelf_xyz[0], right_shelf_xyz[1], right_shelf_xyz[2] + 0.15, r_down, p_down, y_down]),

        # ("move_above_left_shelf", [1, left_shelf_xyz[0], left_shelf_xyz[1] - 0.06, left_shelf_xyz[2] + 0.22, r_down, p_down, y_down]),
        # ("lower_to_shelf", [1, left_shelf_xyz[0], left_shelf_xyz[1] - 0.05, left_shelf_xyz[2] + 0.08, r_down, p_down, y_down]),
        # ("open_gripper", [0, left_shelf_xyz[0], left_shelf_xyz[1] - 0.05, left_shelf_xyz[2] + 0.08, r_down, p_down, y_down]),
        # ("retract_up", [0, left_shelf_xyz[0], left_shelf_xyz[1] - 0.05 , left_shelf_xyz[2] + 0.22, r_down, p_down, y_down]),
        # ("retract_up_right", [0, left_shelf_xyz[0], left_shelf_xyz[1] - 0.24 , left_shelf_xyz[2] + 0.24, r_down, p_down, y_down]),

        # ("side_prepose", [0, left_shelf_xyz[0] + 0.0, left_shelf_xyz[1] - 0.20, left_shelf_xyz[2] + 0.03, r_side, p_side, y_side]),
        # ("side_approach", [0, left_shelf_xyz[0] + 0.0, left_shelf_xyz[1] - 0.10, left_shelf_xyz[2] + 0.03, r_side, p_side, y_side]),
        # ("side_close_gripper", [1, left_shelf_xyz[0] + 0.0, left_shelf_xyz[1] - 0.10, left_shelf_xyz[2] + 0.03, r_side, p_side, y_side]),
        # ("side_lift", [1, left_shelf_xyz[0] + 0.03, left_shelf_xyz[1] - 0.15, left_shelf_xyz[2] + 0.12, r_side, p_side, y_side]),

        # ("orient_down", [1, left_shelf_xyz[0] + 0.03, left_shelf_xyz[1] - 0.15, left_shelf_xyz[2] + 0.12, r_down, p_down, y_down]),
        # ("turn_mid_air", [1, left_shelf_xyz[0] + 0.03, left_shelf_xyz[1] -0.20, left_shelf_xyz[2] + 0.12, r_side, p_side, 0]),
        # ("move_to_midpoint", [1, (right_shelf_xyz[0] + left_shelf_xyz[0]) / 2, (right_shelf_xyz[1] + left_shelf_xyz[1])/2, right_shelf_xyz[2] -0.04, r_side, p_side, 0]),
        # ("move_above_right_shelf", [1, right_shelf_xyz[0], right_shelf_xyz[1] + 0.1, right_shelf_xyz[2] + 0.08, r_side, p_side, 0]),
        
        # ("lower_to_place", [1, right_shelf_xyz[0], right_shelf_xyz[1] + 0.1, right_shelf_xyz[2] + 0.03, r_side, p_side, 0]),
        # ("open_to_release", [0, right_shelf_xyz[0], right_shelf_xyz[1] + 0.1, right_shelf_xyz[2] + 0.03, r_side, p_side, 0]),
        # ("retract_up_right", [0, right_shelf_xyz[0], right_shelf_xyz[1] + 0.18, right_shelf_xyz[2] + 0.08, r_side, p_side, 0]),
        # ("return_hover_start", [0, block_xyzs[0][0], block_xyzs[0][1], block_xyzs[0][2] + 0.20, r_down, p_down, y_down]),
    ]

    arm = FrankArm()
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
    init_pose_jointspace = [0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8, 0.04]
    WAYPOINTS.insert(0, init_pose_jointspace)
    WAYPOINTS = np.array(WAYPOINTS, dtype=float)

    for i in range(len(BLOCKS)):
        rt.add_free_block_to_model(tree=modelTree, name=BLOCKS[i][0], pos=BLOCKS[i][1], density= 20 , size=BLOCKS[i][2] , rgba=[0.2, 0.2, 0.9, 1],free=False)

    for block in SPAWN_BLOCKS:
        rt.add_free_block_to_model(
            tree=modelTree, name=block["name"], pos=block["pos"],
            density=20, size=[0.02, 0.02, 0.02], rgba=[0.0, 0.9, 0.2, 1], free=True)

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

    # Map step duration to steps
    dt = model.opt.timestep
    segment_steps = max(1, int(SEGMENT_DURATION / dt))
    hold_steps =int(HOLD_DURATION/dt)

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
            print(f"Moving to: {i+1}, {wp_name}, xyz=({x:.4f}, {y:.4f}, {z:.4f}), "
                  f"rpy=({r:.4f}, {p:.4f}, {yaw:.4f}), q={q_target}")
            q_start = WAYPOINTS[i][arm_idx].copy()
            q_goal  = WAYPOINTS[i + 1][arm_idx].copy()
            t = 0.0
            #Compute and control along a minjerk trajectory for each waypoint 
            for k in range(segment_steps+hold_steps):
                # Desired joint state 
                q_des, qd_des = rt.interp_min_jerk(q_start, q_goal, t, SEGMENT_DURATION)

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
