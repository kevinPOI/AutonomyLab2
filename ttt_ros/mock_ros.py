import sys
import time
import types
from dataclasses import dataclass, field


@dataclass
class Time:
    secs: int = 0
    nsecs: int = 0

    @classmethod
    def now(cls):
        t = time.time()
        return cls(int(t), int((t - int(t)) * 1e9))

    def to_sec(self):
        return self.secs + self.nsecs * 1e-9


@dataclass
class Header:
    seq: int = 0
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose:
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseStamped:
    header: Header = field(default_factory=Header)
    pose: Pose = field(default_factory=Pose)


class Publisher:
    _log = []

    def __init__(self, topic, msg_type, queue_size=10):
        self.topic = topic
        self.msg_type = msg_type
        self.queue_size = queue_size
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)
        Publisher._log.append((self.topic, msg))


class Rate:
    def __init__(self, hz):
        self.period = 1.0 / hz

    def sleep(self):
        time.sleep(self.period)


def _make_rospy():
    mod = types.ModuleType("rospy")
    state = {"shutdown": False}

    def init_node(name, anonymous=False, **_):
        print(f"[mock_rospy] init_node('{name}')")

    def loginfo(m): print(f"[mock_rospy][INFO] {m}")
    def logwarn(m): print(f"[mock_rospy][WARN] {m}")
    def logerr(m):  print(f"[mock_rospy][ERR ] {m}")
    def is_shutdown(): return state["shutdown"]
    def signal_shutdown(_reason=""): state["shutdown"] = True

    mod.init_node = init_node
    mod.Publisher = Publisher
    mod.Rate = Rate
    mod.Time = Time
    mod.loginfo = loginfo
    mod.logwarn = logwarn
    mod.logerr = logerr
    mod.is_shutdown = is_shutdown
    mod.signal_shutdown = signal_shutdown
    mod._is_mock = True
    return mod


def install():
    if "rospy" in sys.modules and not getattr(sys.modules["rospy"], "_is_mock", False):
        return

    rospy = _make_rospy()

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PoseStamped = PoseStamped
    geometry_msgs_msg.Pose = Pose
    geometry_msgs_msg.Point = Point
    geometry_msgs_msg.Quaternion = Quaternion
    geometry_msgs.msg = geometry_msgs_msg

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Header = Header
    std_msgs.msg = std_msgs_msg

    sys.modules["rospy"] = rospy
    sys.modules["geometry_msgs"] = geometry_msgs
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs_msg


def get_published():
    return list(Publisher._log)


def clear_published():
    Publisher._log.clear()
