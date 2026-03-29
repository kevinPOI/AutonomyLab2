import numpy as np
import RobotUtil as rt
import math


class FrankArm:
    def __init__(self):
        # Robot descriptor taken from URDF file (rpy xyz for each rigid link transform)
        self.Rdesc = [
            [0, 0, 0, 0., 0, 0.333],  # From robot base to joint1
            [-np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0, -0.316, 0],
            [np.pi/2, 0, 0, 0.0825, 0, 0],
            [-np.pi/2, 0, 0, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0.088, 0, 0],
            [0, 0, 0, 0, 0, 0.107]  # From joint5 to end-effector center
        ]

        # Define the axis of rotation for each joint
        self.axis = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]

        # Set base coordinate frame as identity
        self.Tbase = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

        # Initialize matrices
        self.Tlink = []  # Transforms for each link (const)
        self.Tjoint = []  # Transforms for each joint (init eye)
        self.Tcurr = []  # Coordinate frame of current (init eye)

        for i in range(len(self.Rdesc)):
            self.Tlink.append(rt.rpyxyz2H(
                self.Rdesc[i][0:3], self.Rdesc[i][3:6]))
            self.Tcurr.append([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, 0.], [0, 0, 0, 1]])
            self.Tjoint.append([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0.], [0, 0, 0, 1]])

        self.Tlinkzero = rt.rpyxyz2H(self.Rdesc[0][0:3], self.Rdesc[0][3:6])

        self.Tlink[0] = np.matmul(self.Tbase, self.Tlink[0])

        # initialize Jacobian matrix
        self.J = np.zeros((6, 7))

        self.q = [0., 0., 0., 0., 0., 0., 0.]
        self.ForwardKin([0., 0., 0., 0., 0., 0., 0.])

    def NumericalJacobian(self, ang, eps=1e-6):
        '''
        inputs: joint angles, finite-difference step
        outputs: numerical Jacobian (6x7), using position + axis-angle orientation error
        '''
        ang = np.array(ang, dtype=float)
        Tcurr, _ = self.ForwardKin(ang.tolist())
        T0 = Tcurr[-1]
        p0 = T0[0:3, 3]
        R0 = T0[0:3, 0:3]

        Jn = np.zeros((6, 7))
        for i in range(7):
            ang_pert = ang.copy()
            ang_pert[i] += eps
            Tpert, _ = self.ForwardKin(ang_pert.tolist())
            T1 = Tpert[-1]

            p1 = T1[0:3, 3]
            R1 = T1[0:3, 0:3]

            dp = (p1 - p0) / eps
            R_err = np.matmul(R1, R0.T)
            axis, ang_err = rt.R2axisang(R_err)
            dr = (np.array(axis) * ang_err) / eps

            Jn[0:3, i] = dp
            Jn[3:6, i] = dr

        return Jn

    def ForwardKin(self, ang):
        '''
        inputs: joint angles
        outputs: joint transforms for each joint, Jacobian matrix
        '''

        self.q[0:7] = ang

        # Compute current joint and end effector coordinate frames (self.Tjoint).
        # Joint axes are defined in the joint frame; convert to base frame when needed.
        self.J = np.zeros((6, 7))
        T = np.eye(4)
        for i in range(len(self.Rdesc)):
            T = np.matmul(T, self.Tlink[i])
            if i < 7:
                axis_local = np.asarray(self.axis[i], dtype=float)
            else:
                axis_local = np.zeros(3)

            if np.linalg.norm(axis_local) > 0:
                self.Tjoint[i] = T
                # Rotate about the joint's local axis, then update current frame
                T = np.matmul(T, rt.MatrixExp(axis_local, ang[i]))
                self.Tcurr[i] = T
            else:
                # Fixed transform (no joint)
                self.Tcurr[i] = T

        # Jacobian
        O_n = self.Tcurr[-1][0:3, 3]
        for i in range(7):
            O_i = self.Tjoint[i][0:3, 3]
            axis_local = np.asarray(self.axis[i], dtype=float)
            z_i = np.matmul(self.Tjoint[i][0:3, 0:3], axis_local)
            self.J[0:3, i] = np.cross(z_i, (O_n - O_i))
            self.J[3:6, i] = z_i

        return self.Tcurr, self.J

    def IterInvKin(self, ang, TGoal, x_eps=1e-3, r_eps=1e-3):
        '''
        inputs: starting joint angles (ang), target end effector pose (TGoal)

        outputs: computed joint angles to achieve desired end effector pose,
        Error in your IK solution compared to the desired target
        '''

        W = np.diag([1, 1, 100, 100, 1, 1, 100])
        W[6, 0] = 1
        # W[6, 6] = 1
        C = np.diag([1000000, 1000000, 1000000, 1000, 1000, 1000])

        q = np.array(ang, dtype=float)
        max_iters = 1000

        x_goal = TGoal[0:3, 3]
        r_goal = TGoal[0:3, 0:3]

        ang_err = np.inf

        Err = np.zeros(6)

        for _ in range(max_iters):
            Tcurr, J = self.ForwardKin(q.tolist())
            T = Tcurr[-1]

            p_curr = T[0:3, 3]
            R_curr = T[0:3, 0:3]

            # Rotational error
            R_err = np.matmul(r_goal, R_curr.T)
            axis, ang_err = rt.R2axisang(R_err)
            ang_err_v = ang_err * np.array(axis)

            # Translation
            x_err_v = x_goal - p_curr
            Err = np.concatenate([x_err_v, ang_err_v])

            if np.linalg.norm(x_err_v) < x_eps and abs(ang_err) < r_eps:
                break

            JJt = np.matmul(np.matmul(J, W), J.T)
            dls = np.matmul(J.T, np.linalg.inv(JJt + np.linalg.inv(C)))
            dq = np.matmul(dls, Err)
            q = q + dq

        # Wrap +- pi
        q = (q + np.pi) % (2 * np.pi) - np.pi
        self.q[0:7] = q.tolist()

        return self.q, Err
