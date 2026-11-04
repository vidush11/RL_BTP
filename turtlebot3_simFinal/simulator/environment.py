"""Gymnasium environment: TurtleBot3 Burger, LiDAR-only observation by default.

Observation   default ``Box(0, 1, (num_beams,), float32)`` -- normalised LiDAR only.
              Opt-in (``observation`` config): channels from
              [lidar, velocity, ang_velocity, bump, compass, light] and frame
              stacking. Pose/goal are never in the observation unless a channel says so.
Actions       default ``Discrete(len(motion.actions))`` via the velocity table.
              Opt-in ``motion.action_mode: accel`` (with ``dynamics: inertial``):
              FORWARD/BACKWARD/LEFT/RIGHT/STOP accelerations with friction,
              optionally restricted by ``motion.enabled_actions``.
Termination   ``terminated``: collision (geometric swept-disc contact, or the
              legacy LiDAR proximity rule ``min(range) < collision_lidar_threshold``)
              or goal reached (goal_navigation only). Collision ends the episode on
              the step where it happens; step() before reset() raises.
              ``truncated``: ``termination.max_episode_steps`` reached without a
              terminal event (bootstrap through it in learning targets).
Opt-in info   ``light`` (two light sensors), ``safety`` (paper Sec. III-A front
              safety cumulant from a separate fine scan), ``friction`` (accel mode).
Randomness    One ``numpy.random.Generator`` per instance (``self.np_random``).
With every opt-in feature at its default, behaviour is bit-identical to the
original package (tests/test_golden_regression.py).
"""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import ACCEL_ACTIONS, LidarConfig, SimConfig, load_config
from .lidar import Lidar
from .rewards import RewardModel, goal_angle as _goal_angle
from .robot import ActionTable, InvalidActionError, Robot
from .world import World, segments_to_rects_distance


class TurtleBot3LidarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Union[None, str, Mapping[str, Any], SimConfig] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        if render_mode is not None:
            raise ValueError("This simulator is headless; render_mode must be None")
        self.cfg: SimConfig = load_config(config)
        cfg = self.cfg
        m = cfg.motion

        self.world = World(cfg.world)
        self.robot = Robot(cfg.robot, cfg.motion)
        self.lidar = Lidar(cfg.lidar)
        self.accel_mode = m.action_mode == "accel"
        if self.accel_mode:
            self.action_names = list(m.enabled_actions or ACCEL_ACTIONS)
            self.actions = None
            self.action_space = spaces.Discrete(len(self.action_names))
        else:
            self.action_names = None
            self.actions = ActionTable(m.actions)
            self.action_space = spaces.Discrete(len(self.actions))
        self.reward_model = RewardModel(cfg.reward, self.lidar.ros_angles)

        # ---- observation -------------------------------------------------
        oc = cfg.observation
        self._default_obs = list(oc.channels) == ["lidar"] and int(oc.frame_stack) == 1
        if m.dynamics == "inertial":
            self._v_lim = max(abs(m.v_min), abs(m.v_max)) or 1.0
            self._w_lim = m.w_max
        else:
            tab = np.asarray(m.actions, float)
            self._v_lim = float(np.abs(tab[:, 0]).max()) or 1.0
            self._w_lim = float(np.abs(tab[:, 1]).max()) or 1.0
        self._lights = np.asarray(cfg.light.positions, float) if cfg.light.enabled else None
        if self._default_obs:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.lidar.num_beams,), dtype=np.float32)
        else:
            lo, hi = self._frame_bounds()
            k = int(oc.frame_stack)
            self.observation_space = spaces.Box(low=np.tile(lo, k), high=np.tile(hi, k), dtype=np.float32)
        self._frames: deque = deque(maxlen=int(oc.frame_stack))

        # ---- safety cumulant scanner (separate fine scan, never observed) --
        self._safety_lidar = None
        if cfg.safety.enabled:
            self._safety_lidar = Lidar(LidarConfig(num_beams=int(cfg.safety.num_beams),
                                                   max_range=cfg.safety.max_range, noise_std=0.0))

        self.goal_enabled = cfg.task.mode == "goal_navigation"
        self._pending_seed = cfg.seed
        self._elapsed = 0
        self._needs_reset = True
        self._goal: Optional[Tuple[float, float]] = None
        self._prev_goal_dist = 0.0
        self._last_collision = False
        self._last_action_name: Optional[str] = None
        self.last_ranges = np.zeros(self.lidar.num_beams)

    # ------------------------------------------------------------- helpers
    def _frame_bounds(self):
        lo, hi = [], []
        for ch in self.cfg.observation.channels:
            if ch == "lidar":
                lo += [0.0] * self.lidar.num_beams; hi += [1.0] * self.lidar.num_beams
            elif ch in ("velocity", "ang_velocity"):
                lo.append(-1.0); hi.append(1.0)
            elif ch == "bump":
                lo.append(0.0); hi.append(1.0)
            elif ch == "compass":
                n = 1 if self.cfg.observation.compass == "cos" else 2
                lo += [-1.0] * n; hi += [1.0] * n
            elif ch == "light":
                n = float(len(self._lights))
                lo += [0.0, 0.0]; hi += [n, n]
        return np.asarray(lo, np.float32), np.asarray(hi, np.float32)

    def _frame(self, ranges: np.ndarray, bump: bool) -> np.ndarray:
        r = self.robot
        parts = []
        for ch in self.cfg.observation.channels:
            if ch == "lidar":
                parts.append(self.lidar.normalize(ranges))
            elif ch == "velocity":
                parts.append([np.clip(r.v / self._v_lim, -1.0, 1.0)])
            elif ch == "ang_velocity":
                parts.append([np.clip(r.omega / self._w_lim, -1.0, 1.0)])
            elif ch == "bump":
                parts.append([1.0 if bump else 0.0])
            elif ch == "compass":
                parts.append([math.cos(r.theta)] if self.cfg.observation.compass == "cos"
                             else [math.cos(r.theta), math.sin(r.theta)])
            elif ch == "light":
                parts.append(list(self.light_sensors()))
        return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts]).astype(np.float32)

    def _observe(self, ranges: np.ndarray, bump: bool, reset: bool) -> np.ndarray:
        if self._default_obs:
            return self.lidar.normalize(ranges)
        f = self._frame(ranges, bump)
        if reset:
            self._frames.clear()
            for _ in range(self._frames.maxlen):
                self._frames.append(f)
        else:
            self._frames.append(f)
        return np.concatenate(list(self._frames)).astype(np.float32)

    def light_sensors(self) -> Tuple[float, float]:
        """RL_BTP env.py light model, summed over light sources:
        max(0, cos(th+-1)*dx + sin(th+-1)*dy) / (dist + 1e-9) / (1 + 0.5*dist^2).
        With light.occlusion, a light whose straight segment to the robot crosses
        an obstacle contributes 0."""
        if self._lights is None:
            return 0.0, 0.0
        x, y, th = self.robot.x, self.robot.y, self.robot.theta
        left = right = 0.0
        for lx, ly in self._lights:
            dx, dy = lx - x, ly - y
            dist = math.hypot(dx, dy)
            if self.cfg.light.occlusion and self.world.obstacles.shape[0]:
                d = segments_to_rects_distance(np.array([[x, y]]), np.array([[lx, ly]]), self.world.obstacles)
                if (d <= 0.0).any():
                    continue
            att = 1.0 / (1.0 + 0.5 * dist * dist)
            left += max(0.0, math.cos(th + 1.0) * dx + math.sin(th + 1.0) * dy) / (dist + 1e-9) * att
            right += max(0.0, math.cos(th - 1.0) * dx + math.sin(th - 1.0) * dy) / (dist + 1e-9) * att
        return float(left), float(right)

    def safety_zone(self) -> Dict[str, Any]:
        """Paper Sec. III-A front safety: count fine-scan points inside the box of
        width 2*collision_radius + margin_w and length h_f = d_min + v*tau ahead of
        the robot centre; c_front = 0 if n_front > beta_f else 1."""
        sc, r = self.cfg.safety, self.robot
        d = self._safety_lidar.cast(self.world, r.x, r.y, r.theta)
        hit = d < sc.max_range - 1e-9
        ang = self._safety_lidar.angles
        u, w = d * np.cos(ang), d * np.sin(ang)
        half = self.cfg.robot.collision_radius + 0.5 * sc.margin_w
        v = max(r.v, 0.0)
        hf = sc.d_min + v * sc.tau
        n_f = int(np.count_nonzero(hit & (u >= 0.0) & (u <= hf) & (np.abs(w) <= half)))
        out = {"c_front": 0.0 if n_f > sc.beta_f else 1.0, "n_front": n_f, "h_front": float(hf),
               "zone_length": float(hf), "zone_width": float(2 * half), "speed": float(r.v)}
        if sc.rear:
            hr = sc.d_min + abs(r.v) * sc.tau
            n_r = int(np.count_nonzero(hit & (u <= 0.0) & (u >= -hr) & (np.abs(w) <= half)))
            out["rear"] = {"c_rear": 0.0 if n_r > sc.beta_f else 1.0, "n_rear": n_r, "h_rear": float(hr)}
        return out

    def set_friction(self, friction: float) -> None:
        """Switch the 'floor' (inertial dynamics) -- takes effect on the next step."""
        if not 0.0 <= friction < 1.0:
            raise ValueError("friction must be in [0, 1)")
        self.robot.friction = float(friction)

    def _validate(self, action) -> int:
        if not self.accel_mode:
            return self.actions.validate(action)
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise InvalidActionError(f"Action must be an integer, got {action!r}")
        a = int(action)
        if not 0 <= a < len(self.action_names):
            raise InvalidActionError(f"Action {a} out of range 0..{len(self.action_names) - 1}")
        return a

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None and self._pending_seed is not None:
            seed = self._pending_seed
        self._pending_seed = None
        super().reset(seed=seed)
        options = options or {}
        cfg, rng, world = self.cfg, self.np_random, self.world
        required = max(cfg.robot.collision_radius,
                       cfg.termination.collision_lidar_threshold or 0.0)

        # 1. obstacles ("never random unless explicitly enabled")
        if "obstacles" in options:
            world.set_obstacles(options["obstacles"])
        elif cfg.world.randomize_obstacles:
            world.reset_obstacles()
            keep_out = []
            if cfg.spawn.mode == "fixed":
                keep_out.append((cfg.spawn.x, cfg.spawn.y, cfg.spawn.clearance + 0.5))
            extra = world.sample_obstacles(
                rng, cfg.world.random_obstacle_count,
                tuple(cfg.world.random_obstacle_size_range), keep_out)
            world.set_obstacles(np.vstack([world._static, extra]))
        else:
            world.reset_obstacles()

        # 2. spawn pose (always collision-free)
        if "spawn" in options:
            x, y, theta = options["spawn"]
        elif cfg.spawn.mode == "random":
            x, y = world.sample_free(rng, cfg.spawn.clearance)
            theta = float(rng.uniform(-np.pi, np.pi))
        else:
            x, y, theta = cfg.spawn.x, cfg.spawn.y, cfg.spawn.theta
        if world.clearance(x, y) < required:
            raise ValueError(
                f"Spawn pose ({x:.3f}, {y:.3f}) is not collision-free: clearance "
                f"{world.clearance(x, y):.3f} m < required {required:.3f} m")
        self.robot.reset(x, y, theta)
        if "friction" in options:
            self.set_friction(float(options["friction"]))
        if "velocity" in options:                      # e.g. approach-a-wall tests
            self.robot.v, self.robot.omega = (float(options["velocity"][0]), float(options["velocity"][1]))

        # 3. goal (hidden from the observation)
        self._goal = None
        if self.goal_enabled:
            self._goal = self._pick_goal(options.get("goal"))
            self._prev_goal_dist = self._goal_distance()

        self._elapsed = 0
        self._needs_reset = False
        self._last_collision = False
        self._last_action_name = None
        self.last_ranges = self.lidar.cast(world, x, y, self.robot.theta, rng)
        obs = self._observe(self.last_ranges, False, reset=True)
        return obs, self._extras(self._info(False, "none", False))

    def _pick_goal(self, forced) -> Tuple[float, float]:
        cfg, world, rng = self.cfg, self.world, self.np_random
        gc = cfg.goal
        rx, ry = self.robot.x, self.robot.y

        def ok(gx, gy):
            return (world.clearance(gx, gy) >= gc.clearance and
                    np.hypot(gx - rx, gy - ry) >= gc.min_distance_from_robot)

        if forced is not None:
            gx, gy = float(forced[0]), float(forced[1])
            if world.clearance(gx, gy) < cfg.robot.collision_radius:
                raise ValueError(f"Goal ({gx:.3f}, {gy:.3f}) is inside an obstacle or wall")
            return gx, gy
        if gc.mode == "fixed":
            gx, gy = gc.position
            if not ok(gx, gy):
                raise ValueError(f"Fixed goal {gc.position} violates clearance/min-distance rules")
            return float(gx), float(gy)
        # "random_grid": same lattice as dqn_gazebo.generate_goal_pose
        n = int(round((gc.grid_max - gc.grid_min) / gc.grid_step)) + 1
        for _ in range(5000):
            gx = gc.grid_min + gc.grid_step * int(rng.integers(0, n))
            gy = gc.grid_min + gc.grid_step * int(rng.integers(0, n))
            if ok(gx, gy):
                return float(gx), float(gy)
        raise RuntimeError("Could not sample a valid goal position; check the layout")

    def _goal_distance(self) -> float:
        return float(np.hypot(self._goal[0] - self.robot.x, self._goal[1] - self.robot.y))

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Episode has ended; call reset() before step()")
        a = self._validate(action)
        cfg, robot, world = self.cfg, self.robot, self.world
        if self.accel_mode:
            name = self.action_names[a]
            self._last_action_name = name
        else:
            v, omega = self.actions.table[a]
        radius = cfg.robot.collision_radius

        # --- motion with swept collision ---------------------------------
        robot.collided = False
        geometric = False
        for _ in range(int(cfg.motion.action_repeat)):
            if self.accel_mode:
                v, omega = robot.inertial_twist(name, cfg.motion.dt)
            end, pts, sag = robot.plan(v, omega, cfg.motion.dt)
            if world.path_hits(pts, radius + sag):
                robot.block()
                geometric = True
                break
            robot.commit(end, v, omega)

        # --- sensing -------------------------------------------------------
        ranges = self.lidar.cast(world, robot.x, robot.y, robot.theta, self.np_random)
        self.last_ranges = ranges
        thr = cfg.termination.collision_lidar_threshold
        lidar_hit = thr is not None and float(ranges.min()) < thr
        collision = geometric or lidar_hit
        ctype = "geometric" if geometric else ("lidar" if lidar_hit else "none")
        self._last_collision = collision

        # --- goal ------------------------------------------------------------
        goal_reached = False
        g_angle = progress = None
        if self.goal_enabled:
            dist = self._goal_distance()
            goal_reached = (not collision) and dist < cfg.termination.goal_tolerance
            g_angle = _goal_angle(robot.x, robot.y, robot.theta, *self._goal)
            progress = self._prev_goal_dist - dist
            self._prev_goal_dist = dist

        # --- termination (Gymnasium semantics) ------------------------------
        terminated = bool((collision and cfg.termination.terminate_on_collision) or
                          (goal_reached and cfg.termination.terminate_on_goal))
        self._elapsed += 1
        limit = cfg.termination.max_episode_steps
        truncated = bool((not terminated) and limit is not None and self._elapsed >= limit)
        self._needs_reset = terminated or truncated

        reward = self.reward_model.compute(ranges, collision, goal_reached, g_angle, progress,
                                           speed=robot.v)
        obs = self._observe(ranges, collision, reset=False)
        return (obs, reward, terminated, truncated,
                self._extras(self._info(collision, ctype, goal_reached)))

    # ------------------------------------------------------------------- info
    def _info(self, collision: bool, ctype: str, goal_reached: bool) -> Dict[str, Any]:
        r = self.robot
        info: Dict[str, Any] = {
            "step": self._elapsed,
            "collision": collision,
            "collision_type": ctype,          # "none" | "geometric" | "lidar"
            "goal_reached": goal_reached,
            "min_range": float(self.last_ranges.min()),
            "ranges_m": self.last_ranges,     # un-normalised, metres (float64); diagnostics/legacy bridge
            "linear_velocity": r.v,           # odometry-like twist for the velocity demons
            "angular_velocity": r.omega,
            "pose": (r.x, r.y, r.theta),      # NOT part of the observation
        }
        if self._goal is not None:
            info["goal"] = self._goal
            info["goal_distance"] = self._goal_distance()
            info["goal_angle"] = _goal_angle(r.x, r.y, r.theta, *self._goal)
        return info

    def _extras(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Opt-in info keys; nothing is added for the original configs."""
        if self.accel_mode:
            info["friction"] = self.robot.friction
            info["action_name"] = self._last_action_name
        if self._lights is not None:
            info["light"] = self.light_sensors()
        if self._safety_lidar is not None:
            info["safety"] = self.safety_zone()
        return info

    # --------------------------------------------------------------- state io
    def get_state(self, include_rng: bool = False) -> Dict[str, Any]:
        """Everything needed to resume this episode deterministically."""
        r = self.robot
        st = {"pose": (r.x, r.y, r.theta), "v": r.v, "omega": r.omega, "collided": r.collided,
              "elapsed": self._elapsed, "friction": r.friction, "goal": self._goal,
              "prev_goal_dist": self._prev_goal_dist, "needs_reset": self._needs_reset,
              "obstacles": self.world.obstacles.copy(), "last_ranges": self.last_ranges.copy(),
              "frames": [f.copy() for f in self._frames], "last_collision": self._last_collision,
              "last_action_name": self._last_action_name}
        if include_rng:
            st["rng"] = copy.deepcopy(self.np_random.bit_generator.state)
        return st

    def set_state(self, st: Dict[str, Any]) -> None:
        r = self.robot
        r.x, r.y, r.theta = (float(v) for v in st["pose"])
        r.v, r.omega, r.collided = float(st["v"]), float(st["omega"]), bool(st["collided"])
        r.friction = float(st["friction"])
        self._elapsed = int(st["elapsed"])
        self._goal = None if st["goal"] is None else tuple(st["goal"])
        self._prev_goal_dist = float(st["prev_goal_dist"])
        self._needs_reset = bool(st["needs_reset"])
        self.world.set_obstacles(st["obstacles"])
        self.last_ranges = np.asarray(st["last_ranges"], float).copy()
        self._frames.clear()
        for f in st["frames"]:
            self._frames.append(np.asarray(f, np.float32).copy())
        self._last_collision = bool(st["last_collision"])
        self._last_action_name = st.get("last_action_name")
        if "rng" in st:
            self.np_random.bit_generator.state = copy.deepcopy(st["rng"])

    def close(self) -> None:  # nothing to release; kept for API completeness
        pass
