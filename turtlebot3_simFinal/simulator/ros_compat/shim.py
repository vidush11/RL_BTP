"""ROS-free stand-in for the small part of ROS 2 that your agent scripts use.

Why this exists
---------------
``gvf_horde_agent.py`` / ``gvf_horde_agent_evaluate.py`` (and, presumably, your
DQN agent) are ``rclpy`` nodes that talk to the environment through three
services, read ``/odom`` and publish a couple of topics.  Instead of rewriting
those scripts, this module provides *fake* ``rclpy`` / message modules whose
services are answered directly by :class:`simulator.TurtleBot3LidarEnv`.  The
agent code then runs byte-for-byte unmodified, with no ROS installed.

Only the API surface those scripts actually use is implemented (inspected, not
guessed): ``rclpy.init/shutdown/ok/spin/spin_once/spin_until_future_complete``,
``Node`` (``declare_parameter``, ``get_parameter``, ``get_logger``,
``create_client/publisher/subscription``, ``destroy_node``), QoS helpers,
callback groups, and the message/service types ``Dqn``, ``Goal``, ``Empty``,
``Float32MultiArray``, ``Odometry``.  Anything else (e.g. ``create_timer``,
``create_service``) raises ``NotImplementedError`` with an explanation instead
of silently doing nothing.

Service contract (identical to the Gazebo environment node)
-----------------------------------------------------------
``make_environment`` (Empty)      no-op; the world already exists.
``reset_environment`` (Dqn)       -> ``state``: list of 24 LiDAR ranges in METRES
                                  (float64, ``max_range`` for "no hit"), i.e. the
                                  same un-normalised readings ``dqn_environment.py``
                                  returns.  The Horde agents normalise by 3.5 themselves.
``rl_agent_interface`` (Dqn)      ``action`` -> ``state`` (metres), ``reward``,
                                  ``done`` (= terminated OR truncated).

There is one backend (one environment) per process; this is the only global
state, and it lives in this compat layer, not in the simulator.
"""

from __future__ import annotations

import logging
import math
import sys
import types
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import yaml

_log = logging.getLogger("ros_compat")


# =============================================================== messages ===
class Empty:
    class Request:
        pass

    class Response:
        pass


class Dqn:
    class Request:
        def __init__(self):
            self.action = 0

    class Response:
        def __init__(self):
            self.state: List[float] = []
            self.reward = 0.0
            self.done = False


class Goal:
    class Request:
        pass

    class Response:
        def __init__(self):
            self.pose_x = 0.0
            self.pose_y = 0.0
            self.success = False


class Float32MultiArray:
    def __init__(self):
        self.data: List[float] = []


class Odometry:
    def __init__(self):
        self.pose = SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))
        self.twist = SimpleNamespace(twist=SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.0)))


class _Plain:  # placeholders so ``from ... import X`` works for unused types
    pass


class QoSProfile:
    def __init__(self, depth: int = 10, **kwargs):
        self.depth = depth


class MutuallyExclusiveCallbackGroup:
    pass


class ReentrantCallbackGroup:
    pass


# ================================================================ backend ===
class SimBackend:
    """Answers the three environment services from a ``TurtleBot3LidarEnv``."""

    def __init__(self, env, seed: Optional[int] = None):
        self.env = env
        self._seed = seed
        self._odom_callbacks: List[Callable] = []
        self.last_info: Dict[str, Any] = {}
        self.episodes = 0
        self.total_steps = 0
        self.terminations = 0     # episodes ended by collision / goal
        self.truncations = 0      # episodes ended by the step limit

    def add_odom_callback(self, cb: Callable) -> None:
        self._odom_callbacks.append(cb)

    def _publish_odom(self) -> None:
        if not self._odom_callbacks:
            return
        r = self.env.unwrapped.robot
        msg = Odometry()
        msg.pose.pose.position.x, msg.pose.pose.position.y = r.x, r.y
        msg.pose.pose.orientation.z = math.sin(r.theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(r.theta / 2.0)
        msg.twist.twist.linear.x = r.v
        msg.twist.twist.angular.z = r.omega
        for cb in self._odom_callbacks:
            cb(msg)

    def call(self, service: str, request):
        name = service.lstrip("/").split("/")[-1]
        if name == "make_environment":
            return Empty.Response()
        if name == "reset_environment":
            _obs, info = self.env.reset(seed=self._seed)
            self._seed = None            # seed once; later resets continue the stream
            self.episodes += 1
            self.last_info = info
            self._publish_odom()
            resp = Dqn.Response()
            resp.state = [float(x) for x in info["ranges_m"]]
            return resp
        if name == "rl_agent_interface":
            _obs, reward, terminated, truncated, info = self.env.step(int(request.action))
            self.total_steps += 1
            info = dict(info, terminated=terminated, truncated=truncated)
            self.terminations += int(terminated)
            self.truncations += int(truncated)
            self.last_info = info
            self._publish_odom()
            resp = Dqn.Response()
            resp.state = [float(x) for x in info["ranges_m"]]
            resp.reward = float(reward)
            resp.done = bool(terminated or truncated)
            return resp
        raise NotImplementedError(
            f"Service '{service}' is not provided by the simulator. Only make_environment, "
            f"reset_environment and rl_agent_interface exist (the Gazebo-side task_succeed/"
            f"task_failed/initialize_env services are handled inside the simulator).")


class _State:
    backend: Optional[SimBackend] = None
    overrides: Dict[str, Any] = {}
    ok = True
    warned_topics: set = set()


class _Future:
    def __init__(self, result=None):
        self._result = result

    def done(self) -> bool:
        return True

    def result(self):
        return self._result

    def exception(self):
        return None


class _Client:
    def __init__(self, srv_type, name):
        self.srv_type, self.name = srv_type, name

    def wait_for_service(self, timeout_sec=None) -> bool:
        return _State.backend is not None

    def call_async(self, request) -> _Future:
        return _Future(_State.backend.call(self.name, request))


class _Publisher:
    def __init__(self, topic):
        self.topic = topic
        self.last = None

    def publish(self, msg) -> None:
        self.last = msg


class _Logger:
    def __init__(self, name: str):
        self._l = logging.getLogger(name)

    def info(self, msg, *a, **k): self._l.info(msg)
    def warn(self, msg, *a, **k): self._l.warning(msg)
    warning = warn
    def error(self, msg, *a, **k): self._l.error(msg)
    def debug(self, msg, *a, **k): self._l.debug(msg)
    fatal = error


class _Parameter:
    def __init__(self, name, value):
        self.name, self.value = name, value


def _coerce(raw, default):
    """Mimic ROS's ``-p name:=value`` typing: parse as YAML, then match the
    declared default's type."""
    if not isinstance(raw, str) or isinstance(default, str):
        return raw
    parsed = yaml.safe_load(raw)
    if isinstance(default, bool):
        return bool(parsed)
    if isinstance(default, int):
        return int(parsed) if float(parsed) == int(float(parsed)) else parsed
    if isinstance(default, float):
        return float(parsed)
    return parsed


class Node:
    def __init__(self, node_name: str, **kwargs):
        self._node_name = node_name
        self._params: Dict[str, Any] = {}
        self._logger = _Logger(node_name)

    def get_logger(self) -> _Logger:
        return self._logger

    def declare_parameter(self, name, value=None, *args, **kwargs):
        if name in _State.overrides:
            value = _coerce(_State.overrides[name], value)
        self._params[name] = value
        return _Parameter(name, value)

    def get_parameter(self, name) -> _Parameter:
        if name not in self._params:
            raise KeyError(f"Parameter '{name}' was not declared")
        return _Parameter(name, self._params[name])

    def create_client(self, srv_type, name, callback_group=None, **kw):
        return _Client(srv_type, name)

    def create_publisher(self, msg_type, topic, qos=10, **kw):
        return _Publisher(topic)

    def create_subscription(self, msg_type, topic, callback, qos=10, **kw):
        if msg_type is Odometry:
            _State.backend.add_odom_callback(callback)
        elif topic not in _State.warned_topics:
            _State.warned_topics.add(topic)
            _log.warning("Subscription to '%s' is accepted but never fed by the simulator "
                         "(only odometry is provided).", topic)
        return SimpleNamespace(topic=topic)

    def create_service(self, *a, **k):
        raise NotImplementedError(
            "create_service: the environment services are provided by the simulator, "
            "not by agent code.")

    def create_timer(self, *a, **k):
        raise NotImplementedError(
            "create_timer: wall-clock timers have no meaning in a stepped simulator. "
            "If an agent needs one, tell me which one so it can be mapped to sim time.")

    def destroy_node(self):
        pass


# ===================================================================== rclpy ===
def _init(args=None, **kw):
    _State.ok = True
    args = list(args) if args is not None else []
    if "--ros-args" in args:
        i = 0
        while i < len(args):
            if args[i] in ("-p", "--param") and i + 1 < len(args) and ":=" in args[i + 1]:
                k, v = args[i + 1].split(":=", 1)
                _State.overrides.setdefault(k, v)   # explicit install() overrides win
                i += 2
            else:
                i += 1


def _shutdown(**kw):
    _State.ok = False


def _ok(**kw):
    return _State.ok


def _spin(node=None, **kw):
    # Your scripts do all their work in __init__ (process()); a real spin would
    # block until Ctrl-C.  Returning is the correct headless behaviour.
    return None


def _spin_once(node=None, timeout_sec=None, **kw):
    return None


def _spin_until_future_complete(node, future, **kw):
    return None   # service calls are synchronous: the future is already complete


# ================================================================== install ===
_MODULE_NAMES = (
    "rclpy", "rclpy.node", "rclpy.qos", "rclpy.callback_groups",
    "std_srvs", "std_srvs.srv", "std_msgs", "std_msgs.msg", "nav_msgs", "nav_msgs.msg",
    "geometry_msgs", "geometry_msgs.msg", "sensor_msgs", "sensor_msgs.msg",
    "turtlebot3_msgs", "turtlebot3_msgs.srv",
)


def _mod(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, m)
    return m


def install(env, seed: Optional[int] = None, param_overrides: Optional[dict] = None
            ) -> SimBackend:
    """Put the fake ROS modules into ``sys.modules`` and bind ``env`` to them.

    Call this *before* importing your agent script.  It deliberately replaces
    a real ``rclpy`` if one is installed, so the same command behaves the same
    on a laptop with ROS sourced and on the cluster.
    """
    _State.backend = SimBackend(env, seed=seed)
    _State.overrides = dict(param_overrides or {})
    _State.ok = True
    _State.warned_topics = set()

    _mod("rclpy", init=_init, shutdown=_shutdown, ok=_ok, spin=_spin, spin_once=_spin_once,
         spin_until_future_complete=_spin_until_future_complete, _IS_SIM_SHIM=True)
    _mod("rclpy.node", Node=Node)
    _mod("rclpy.qos", QoSProfile=QoSProfile, qos_profile_sensor_data=QoSProfile(depth=5))
    _mod("rclpy.callback_groups", MutuallyExclusiveCallbackGroup=MutuallyExclusiveCallbackGroup,
         ReentrantCallbackGroup=ReentrantCallbackGroup)
    _mod("std_srvs")
    _mod("std_srvs.srv", Empty=Empty)
    _mod("std_msgs")
    _mod("std_msgs.msg", Float32MultiArray=Float32MultiArray)
    _mod("nav_msgs")
    _mod("nav_msgs.msg", Odometry=Odometry)
    _mod("geometry_msgs")
    _mod("geometry_msgs.msg", Twist=_Plain, TwistStamped=_Plain, Pose=_Plain)
    _mod("sensor_msgs")
    _mod("sensor_msgs.msg", LaserScan=_Plain)
    _mod("turtlebot3_msgs")
    _mod("turtlebot3_msgs.srv", Dqn=Dqn, Goal=Goal)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                            format="[%(levelname)s] [%(name)s]: %(message)s")
    return _State.backend


def uninstall() -> None:
    for name in _MODULE_NAMES:
        sys.modules.pop(name, None)
    _State.backend = None
    _State.overrides = {}
