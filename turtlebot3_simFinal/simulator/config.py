"""Configuration for the TurtleBot3 Burger LiDAR simulator.

All defaults live here, so the environment works with no YAML file at all.
YAML files (see ``configs/``) only override what they mention.  Unknown keys
raise ``ConfigError`` so a typo can never be silently ignored.

Where a default comes from
--------------------------
* ``robot``, ``world`` size, ``dt``, ``actions`` ... : the values given in the
  project brief.
* ``lidar.max_range = 3.5``, 24 beams, beam 0 straight ahead, counter-clockwise,
  360 deg : taken from the existing code (``dqn_environment.py`` replaces Inf by
  3.5, the Horde agents normalise by 3.5 and use ``lidar_size = 24``).
* ``termination.collision_lidar_threshold = 0.15`` and the ``obstacle_avoidance``
  reward (constant +1 per step, no collision penalty) : the *live* code path of
  ``dqn_environment.py`` (``calculate_state`` / ``calculate_reward``).
* ``goal_navigation`` reward terms : the commented-out code in the same file.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import yaml


class ConfigError(ValueError):
    """Raised for an invalid or unknown configuration value."""


# Action tables --------------------------------------------------------------
# Brief's table: action 0 = most negative omega (clockwise), action 4 = most
# positive omega (counter-clockwise).  v = 0.15 m/s.
SPEC_ACTION_TABLE: Tuple[Tuple[float, float], ...] = (
    (0.15, -1.50), (0.15, -0.75), (0.15, 0.00), (0.15, 0.75), (0.15, 1.50),
)
# What the Gazebo environment actually sends (dqn_environment.py:
# linear.x = 0.2, angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]).  Positive
# angular.z is counter-clockwise (left) in ROS (REP-103).
GAZEBO_ACTION_TABLE: Tuple[Tuple[float, float], ...] = (
    (0.20, 1.50), (0.20, 0.75), (0.20, 0.00), (0.20, -0.75), (0.20, -1.50),
)


@dataclass
class WorldConfig:
    width: float = 4.0                      # metres (x extent)
    height: float = 4.0                     # metres (y extent)
    center: Tuple[float, float] = (0.0, 0.0)  # world spans center +- size/2
    # Static rectangular obstacles, each [xmin, ymin, xmax, ymax].
    obstacles: List[List[float]] = field(default_factory=list)
    # Random layouts are OFF unless explicitly enabled.
    randomize_obstacles: bool = False
    random_obstacle_count: int = 3
    random_obstacle_size_range: Tuple[float, float] = (0.2, 0.5)  # side length


@dataclass
class RobotConfig:
    collision_radius: float = 0.033   # circular footprint used for hard collisions
    wheel_radius: float = 0.033       # only used for wheel-speed diagnostics
    wheel_separation: float = 0.160   # only used for wheel-speed diagnostics


@dataclass
class MotionConfig:
    dt: float = 0.1                   # seconds per simulation sub-step
    action_repeat: int = 1            # simulation sub-steps per agent step
    # list of [v, omega]; index == action id.
    actions: List[List[float]] = field(
        default_factory=lambda: [list(a) for a in SPEC_ACTION_TABLE])
    # Swept-collision polyline resolution: chords are at most this fraction of
    # the collision radius long.
    max_chord_fraction: float = 0.5
    # --- opt-in dynamics (defaults keep the original kinematic table model) ---
    # "kinematic": (v, omega) taken from the action table every step (original).
    # "inertial" : actions are accelerations; v, omega integrate, decay by
    #              friction every sub-step and are clipped (RL_BTP env.py model).
    dynamics: str = "kinematic"
    # "velocity_table": `actions` above.  "accel": FORWARD/BACKWARD/LEFT/RIGHT/STOP
    # (STOP = coast under friction); requires dynamics: inertial.
    action_mode: str = "velocity_table"
    enabled_actions: Optional[List[str]] = None   # accel mode: subset, in this order
    lin_accel: float = 0.8            # m/s^2   (RL_BTP)
    ang_accel: float = 3.0            # rad/s^2 (RL_BTP)
    friction: float = 0.10            # fraction of v, omega lost per sub-step
    v_min: float = -0.6               # inertial clip limits (RL_BTP defaults)
    v_max: float = 0.6
    w_max: float = 3.0


@dataclass
class SpawnConfig:
    mode: str = "fixed"               # "fixed" (Gazebo behaviour) | "random"
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    clearance: float = 0.35           # min distance from walls/obstacles at spawn


@dataclass
class LidarConfig:
    num_beams: int = 24
    fov_deg: float = 360.0
    angle_min_deg: float = 0.0        # robot-frame angle of beam 0 (0 = straight ahead)
    max_range: float = 3.5            # metres
    noise_std: float = 0.0            # metres, Gaussian; 0 = noiseless


@dataclass
class TaskConfig:
    mode: str = "obstacle_avoidance"  # "obstacle_avoidance" | "goal_navigation"


@dataclass
class GoalConfig:
    # "random_grid" mimics dqn_gazebo.py: x, y = randrange(-21, 21) / 10
    # (rejection-sampled to free space).  "fixed": use ``position``.
    mode: str = "random_grid"
    position: Tuple[float, float] = (0.5, 0.0)
    grid_min: float = -2.1
    grid_max: float = 2.0
    grid_step: float = 0.1
    min_distance_from_robot: float = 0.5
    clearance: float = 0.3


@dataclass
class TerminationConfig:
    # Existing rule: episode ends when min(lidar) < 0.15 m.  null disables it
    # (only the geometric collision then ends the episode).
    collision_lidar_threshold: Optional[float] = 0.15
    terminate_on_collision: bool = True
    # The existing file defines max_step = 800 (its check is commented out).
    # null = no limit.
    max_episode_steps: Optional[int] = 800
    goal_tolerance: float = 0.20      # from the commented-out goal check
    terminate_on_goal: bool = True


@dataclass
class RewardConfig:
    # Defaults reproduce the live code: reward = 1.0 every step, including the
    # collision step.
    per_step: float = 1.0
    collision_penalty: float = 0.0
    goal_reward: float = 0.0
    # "add": terminal value is added to the step reward.
    # "replace": terminal value replaces it (what the commented-out code did).
    terminal_mode: str = "add"
    yaw_coef: float = 0.0             # weight of 1 - 2|goal_angle|/pi
    obstacle_coef: float = 0.0        # weight of the weighted obstacle term
    progress_coef: float = 0.0        # weight of (previous - current) goal distance
    # Opt-in task reward for the safety task: speed_coef * clip(v, 0, v_target)/v_target
    speed_coef: float = 0.0
    v_target: float = 0.2


ACCEL_ACTIONS = ("FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP")   # == RL_BTP env.Action order
OBS_CHANNELS = ("lidar", "velocity", "ang_velocity", "bump", "compass", "light")


@dataclass
class ObservationConfig:
    # Default ["lidar"], frame_stack 1 == the original Box(0, 1, (num_beams,)).
    channels: List[str] = field(default_factory=lambda: ["lidar"])
    compass: str = "cos"              # "cos" | "cos_sin"
    frame_stack: int = 1              # concatenate the last k frames (oldest first)


@dataclass
class LightConfig:
    enabled: bool = False
    positions: List[List[float]] = field(default_factory=lambda: [[1.6, 1.6]])
    occlusion: bool = False           # obstacles between robot and light block it


@dataclass
class SafetyConfig:
    """Paper Sec. III-A front-safety cumulant from a separate fine scan."""
    enabled: bool = False
    num_beams: int = 360
    max_range: float = 3.5
    margin_w: float = 0.10            # zone width = 2*collision_radius + margin_w
    tau: float = 1.5                  # headway (s)
    d_min: float = 0.4                # stand-still distance (m)
    beta_f: int = 0                   # unsafe if more than beta_f points in the zone
    rear: bool = False                # optional rear zone (implemented, unused)


@dataclass
class SimConfig:
    seed: Optional[int] = None
    world: WorldConfig = field(default_factory=WorldConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    light: LightConfig = field(default_factory=LightConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "SimConfig":
        data = {} if data is None else dict(data)
        sections = {
            "world": WorldConfig, "robot": RobotConfig, "motion": MotionConfig,
            "spawn": SpawnConfig, "lidar": LidarConfig, "task": TaskConfig,
            "goal": GoalConfig, "termination": TerminationConfig,
            "reward": RewardConfig, "observation": ObservationConfig,
            "light": LightConfig, "safety": SafetyConfig,
        }
        unknown = set(data) - set(sections) - {"seed"}
        if unknown:
            raise ConfigError(
                f"Unknown top-level key(s) {sorted(unknown)}; valid keys are "
                f"{sorted(set(sections) | {'seed'})}")
        kwargs: dict = {"seed": data.get("seed")}
        for name, section_cls in sections.items():
            kwargs[name] = _build_section(section_cls, data.get(name), name)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SimConfig":
        with open(path, "r") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict:
        return copy.deepcopy(dataclasses.asdict(self))

    # ---------------------------------------------------------- validation
    def validate(self) -> None:
        w, r, m, s, l, t, g, te, rw = (
            self.world, self.robot, self.motion, self.spawn, self.lidar,
            self.task, self.goal, self.termination, self.reward)
        _require(w.width > 0 and w.height > 0, "world.width/height must be > 0")
        _require(r.collision_radius > 0, "robot.collision_radius must be > 0")
        _require(r.wheel_radius > 0 and r.wheel_separation > 0,
                 "robot.wheel_radius/wheel_separation must be > 0")
        _require(m.dt > 0, "motion.dt must be > 0")
        _require(int(m.action_repeat) >= 1, "motion.action_repeat must be >= 1")
        _require(0 < m.max_chord_fraction <= 1, "motion.max_chord_fraction in (0, 1]")
        _require(len(m.actions) >= 1, "motion.actions must not be empty")
        for i, a in enumerate(m.actions):
            _require(len(a) == 2, f"motion.actions[{i}] must be [v, omega]")
        _require(s.mode in ("fixed", "random"), "spawn.mode must be 'fixed' or 'random'")
        _require(s.clearance >= 0, "spawn.clearance must be >= 0")
        _require(int(l.num_beams) >= 1, "lidar.num_beams must be >= 1")
        _require(0 < l.fov_deg <= 360, "lidar.fov_deg must be in (0, 360]")
        _require(l.max_range > 0, "lidar.max_range must be > 0")
        _require(l.noise_std >= 0, "lidar.noise_std must be >= 0")
        _require(t.mode in ("obstacle_avoidance", "goal_navigation"),
                 "task.mode must be 'obstacle_avoidance' or 'goal_navigation'")
        _require(g.mode in ("random_grid", "fixed"), "goal.mode must be 'random_grid' or 'fixed'")
        _require(g.grid_step > 0 and g.grid_max > g.grid_min, "goal grid invalid")
        _require(te.max_episode_steps is None or int(te.max_episode_steps) >= 1,
                 "termination.max_episode_steps must be null or >= 1")
        _require(te.collision_lidar_threshold is None or te.collision_lidar_threshold >= 0,
                 "termination.collision_lidar_threshold must be null or >= 0")
        _require(te.goal_tolerance > 0, "termination.goal_tolerance must be > 0")
        _require(rw.terminal_mode in ("add", "replace"),
                 "reward.terminal_mode must be 'add' or 'replace'")
        for k, o in enumerate(w.obstacles):
            _require(len(o) == 4 and o[2] > o[0] and o[3] > o[1],
                     f"world.obstacles[{k}] must be [xmin, ymin, xmax, ymax] with max > min")
        lo, hi = w.random_obstacle_size_range
        _require(0 < lo <= hi, "world.random_obstacle_size_range must satisfy 0 < lo <= hi")
        # --- opt-in extensions ---
        _require(m.dynamics in ("kinematic", "inertial"), "motion.dynamics must be kinematic|inertial")
        _require(m.action_mode in ("velocity_table", "accel"),
                 "motion.action_mode must be velocity_table|accel")
        _require((m.dynamics, m.action_mode) in (("kinematic", "velocity_table"), ("inertial", "accel")),
                 "supported combinations: kinematic+velocity_table, inertial+accel")
        if m.enabled_actions is not None:
            _require(m.action_mode == "accel", "motion.enabled_actions requires action_mode: accel")
            _require(len(m.enabled_actions) >= 1 and len(set(m.enabled_actions)) == len(m.enabled_actions)
                     and all(a in ACCEL_ACTIONS for a in m.enabled_actions),
                     f"motion.enabled_actions must be unique names from {ACCEL_ACTIONS}")
        _require(0.0 <= m.friction < 1.0, "motion.friction must be in [0, 1)")
        _require(m.v_min <= m.v_max and m.w_max > 0, "need motion.v_min <= v_max and w_max > 0")
        _require(m.lin_accel >= 0 and m.ang_accel >= 0, "accelerations must be >= 0")
        o = self.observation
        _require(len(o.channels) >= 1 and len(set(o.channels)) == len(o.channels)
                 and all(c in OBS_CHANNELS for c in o.channels),
                 f"observation.channels must be unique names from {OBS_CHANNELS}")
        _require(o.compass in ("cos", "cos_sin"), "observation.compass must be cos|cos_sin")
        _require(int(o.frame_stack) >= 1, "observation.frame_stack must be >= 1")
        if "light" in o.channels:
            _require(self.light.enabled, "observation channel 'light' requires light.enabled")
        if self.light.enabled:
            _require(len(self.light.positions) >= 1 and all(len(p) == 2 for p in self.light.positions),
                     "light.positions must be a list of [x, y]")
        sf = self.safety
        _require(int(sf.num_beams) >= 1 and sf.max_range > 0 and sf.margin_w >= 0 and sf.tau >= 0
                 and sf.d_min >= 0 and int(sf.beta_f) >= 0, "invalid safety parameters")
        _require(rw.v_target > 0, "reward.v_target must be > 0")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _build_section(section_cls, data, name):
    if data is None:
        return section_cls()
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config section '{name}' must be a mapping, got {type(data).__name__}")
    valid = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {sorted(unknown)} in section '{name}'; valid keys: {sorted(valid)}")
    obj = section_cls()
    for key, value in data.items():
        default = getattr(obj, key)
        if isinstance(default, tuple) and isinstance(value, Sequence) and not isinstance(value, str):
            value = tuple(value)
        elif isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        setattr(obj, key, value)
    return obj


def load_config(source: Union[None, str, Path, Mapping[str, Any], SimConfig] = None) -> SimConfig:
    """Accept ``None`` (defaults), a YAML path, a mapping, or a ``SimConfig``."""
    if source is None:
        cfg = SimConfig()
        cfg.validate()
        return cfg
    if isinstance(source, SimConfig):
        source.validate()
        return copy.deepcopy(source)
    if isinstance(source, Mapping):
        return SimConfig.from_dict(source)
    return SimConfig.from_yaml(source)
