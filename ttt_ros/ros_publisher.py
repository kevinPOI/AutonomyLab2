import math
from dataclasses import dataclass

try:
    import rospy
    from geometry_msgs.msg import PoseStamped
except ImportError:
    try:
        from . import mock_ros
    except ImportError:
        import mock_ros
    mock_ros.install()
    import rospy
    from geometry_msgs.msg import PoseStamped

USING_REAL_ROS = not getattr(rospy, "_is_mock", False)

TOPIC = "/tic_tac_toe/block_placement"


def _ros_is_initialized():
    core = getattr(rospy, "core", None)
    is_initialized = getattr(core, "is_initialized", None)
    if callable(is_initialized):
        try:
            return bool(is_initialized())
        except Exception:
            return False
    return False


@dataclass
class GridFrame:
    origin_x: float = 0.40
    origin_y: float = 0.00
    cell_size: float = 0.06
    place_z: float = 0.05
    frame_id: str = "base_link"

    def cell_to_xy(self, row, col):
        # Row steps in +x (away from the robot), column in +y (to its right).
        return self.origin_x + row * self.cell_size, self.origin_y + col * self.cell_size

    def xy_to_cell(self, x, y, tol=None):
        tol = self.cell_size * 0.45 if tol is None else tol
        row_f = (x - self.origin_x) / self.cell_size
        col_f = (y - self.origin_y) / self.cell_size
        row = int(math.floor(row_f + 0.5))
        col = int(math.floor(col_f + 0.5))

        if not (0 <= row < 3 and 0 <= col < 3):
            raise ValueError(
                f"point ({x:.4f}, {y:.4f}) is outside the 3x3 grid "
                f"origin=({self.origin_x:.4f}, {self.origin_y:.4f}) cell={self.cell_size:.4f}"
            )

        cx, cy = self.cell_to_xy(row, col)
        if abs(x - cx) > tol or abs(y - cy) > tol:
            raise ValueError(
                f"point ({x:.4f}, {y:.4f}) is not within {tol:.4f} m of cell ({row}, {col})"
            )

        return row, col

    def pose_to_cell(self, pose_stamped, tol=None):
        return self.xy_to_cell(
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            tol=tol,
        )


class BlockPosePublisher:
    def __init__(self, grid=None, topic=TOPIC, node_name="tic_tac_toe_node", init_node=True):
        self.grid = grid or GridFrame()
        self.topic = topic
        if init_node and not _ros_is_initialized():
            rospy.init_node(node_name, anonymous=True)
        self.publisher = rospy.Publisher(topic, PoseStamped, queue_size=10)
        self._seq = 0

    def make_pose(self, row, col, player):
        x, y = self.grid.cell_to_xy(row, col)
        msg = PoseStamped()
        msg.header.seq = self._seq
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.grid.frame_id
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = self.grid.place_z
        msg.pose.orientation.w = 1.0
        self._seq += 1
        return msg

    def publish_move(self, row, col, player):
        msg = self.make_pose(row, col, player)
        self.publisher.publish(msg)
        rospy.loginfo(
            f"move player={player} cell=({row},{col}) "
            f"-> pose=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, "
            f"{msg.pose.position.z:.3f}) frame={msg.header.frame_id}"
        )
        return msg
