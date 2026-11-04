#!/usr/bin/env python3
#################################################################################
# Copyright 2026
# Multi-GVF Horde Agent for TurtleBot3 Burger ROS 2
# Based on the Horde Architecture (Sutton et al., 2011) using GQ(lambda) +
# fixed-horizon off-policy LMS prediction.
#
# Five demons, ONE shared behavior trajectory, running in parallel every
# step, each with its own independent weights and its own target policy:
#
#   1. TTC_forward   - GQ(lambda), target=deterministic-FORWARD,
#                      gamma = 0 once the FULL 24-ray scan sees an
#                      obstacle <= 0.25m (unchanged from gvf_agent.py)
#   2. TTC_left      - GQ(lambda), target=deterministic-LEFT,
#                      gamma = 0 on the same FULL 24-ray condition
#   3. TTC_right     - GQ(lambda), target=deterministic-RIGHT,
#                      gamma = 0 on the same FULL 24-ray condition
#   (All three TTC demons share the same cumulant, gamma and termination;
#    they differ ONLY in their target policy.)
#   4. Future linear velocity   - fixed-horizon (H-step) LMS, target =
#                      epsilon-greedy(FORWARD, eps=0.05)
#   5. Future angular velocity  - same, reading /odom angular.z instead
#
# Behavior policy mu(a|s): SHARED across all 5 demons, uniform random
# (equivalent to epsilon-greedy with eps=1.0) - this is deliberate, not a
# simplification: it is exactly what TTC_forward/left/right already
# require (their original scripts use plain random.randint), and it still
# gives genuine off-policy separation for the velocity demons since their
# target policy (eps=0.05) is very different from eps=1.0.
#
# Each demon keeps its OWN target policy pi - this is what makes the
# off-policy learning meaningful: five different questions, one shared
# stream of experience, each demon's rho_t = pi_d(A_t|S_t) / mu(A_t|S_t)
# computed with ITS OWN pi_d.
#################################################################################

import collections
import datetime
import math
import numpy as np
import os
import random
import sys
import time
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry

from turtlebot3_msgs.srv import Dqn


# =============================================================================
# Action-index convention (shared by every demon):
#   0 = hard-left, 1 = left, 2 = straight/forward, 3 = right, 4 = hard-right
# =============================================================================
HARD_LEFT_ACTION = 0
LEFT_ACTION = 1
FORWARD_ACTION = 2
RIGHT_ACTION = 3
HARD_RIGHT_ACTION = 4

# LiDAR range used by every TTC demon's gamma function: the FULL 24-ray scan
# (slice(0, 24)), exactly as in the original gvf_agent.py. TTC_forward,
# TTC_left and TTC_right all use it, so the three differ only in their
# target policy.
FULL_SCAN_IDX = slice(0, 24)


def compute_regression_metrics(predictions, targets):
    """MAE, RMSE, and R^2 of `predictions` against `targets`. R^2 is NaN if
    the targets have (numerically) zero variance, since it is undefined
    there rather than merely small."""
    predictions_arr = np.asarray(predictions, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)
    errors = predictions_arr - targets_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((targets_arr - np.mean(targets_arr)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# =============================================================================
# Demon 1/2/3 - Directional TTC GVF (GQ(lambda))
# Exactly the equations from gvf_agent.py's update_gq_lambda(), parameterised
# per-demon by (target action, gamma LiDAR range).
# =============================================================================
class DirectionalTTCGVF:

    def __init__(self, feature_dim, action_size, target_action, sector_idx,
                 alpha_theta, alpha_w, trace_lambda, obstacle_threshold, name):
        self.name = name
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.target_action = target_action
        self.sector_idx = sector_idx
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.trace_lambda = trace_lambda
        self.obstacle_threshold = obstacle_threshold

        self.theta = np.zeros(feature_dim, dtype=np.float64)
        self.w = np.zeros(feature_dim, dtype=np.float64)
        self.e = np.zeros(feature_dim, dtype=np.float64)

    def target_policy_prob(self, action):
        return 1.0 if action == self.target_action else 0.0

    def compute_gamma(self, lidar_scan):
        sector = lidar_scan[self.sector_idx]
        min_dist = float(np.min(sector)) if sector.size > 0 else 1.0
        return 0.0 if min_dist <= self.obstacle_threshold else 1.0

    def predict(self, phi):
        return float(np.dot(self.theta, phi))

    def reset_trace(self):
        self.e.fill(0.0)

    def update(self, phi_t, phi_target_next, lidar_t, lidar_tp1, action,
               behaviour_prob, done):
        """phi_target_next = get_feature_vector(next_state, self.target_action)
        - since this demon's target policy is fully deterministic, phi_bar_tp1
        collapses to exactly that one feature vector (no need to loop/sum
        over all 5 actions)."""
        gamma_t = self.compute_gamma(lidar_t)
        gamma_tp1 = 0.0 if done else self.compute_gamma(lidar_tp1)
        r_tp1 = 1.0
        z_tp1 = 0.0

        q_current = float(np.dot(self.theta, phi_t))
        q_next_target = float(np.dot(self.theta, phi_target_next))
        delta_t = r_tp1 + (1.0 - gamma_tp1) * z_tp1 + gamma_tp1 * q_next_target - q_current

        pi_a = self.target_policy_prob(action)
        b_a = max(float(behaviour_prob), 1e-8)
        rho_t = pi_a / b_a

        new_e = phi_t + gamma_t * self.trace_lambda * rho_t * self.e

        w_dot_e = float(np.dot(self.w, new_e))
        w_dot_phi = float(np.dot(self.w, phi_t))

        theta_update = self.alpha_theta * (
            delta_t * new_e - gamma_tp1 * (1.0 - self.trace_lambda) * w_dot_e * phi_target_next)
        w_update = self.alpha_w * (delta_t * new_e - w_dot_phi * phi_t)

        if (not np.isfinite(delta_t) or not np.all(np.isfinite(theta_update))
                or not np.all(np.isfinite(w_update))):
            self.e.fill(0.0)
            return False

        self.e = new_e
        self.theta += theta_update
        self.w += w_update
        return True


# =============================================================================
# Demon 4/5 - Fixed-Horizon GVF (future linear / angular velocity)
# Exactly the class from gvf_future_linear_velocity_agent.py /
# gvf_future_angular_velocity_agent.py, unmodified.
# =============================================================================
class FixedHorizonGVF:

    def __init__(self, feature_dim, horizon, alpha_w, rho_product_clip):
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.alpha_w = alpha_w
        self.rho_product_clip = rho_product_clip

        self.w = np.zeros(feature_dim, dtype=np.float64)
        self._pending = collections.deque()

    def predict(self, phi):
        return float(np.dot(self.w, phi))

    def reset_window(self):
        self._pending.clear()

    def push_and_maybe_update(self, phi_t, rho_t, current_measurement):
        self._pending.append((phi_t, rho_t))
        if len(self._pending) <= self.horizon:
            return None

        old_phi, _old_rho_unused = self._pending.popleft()
        rho_product = 1.0
        for _, rho in list(self._pending)[:self.horizon - 1]:
            rho_product *= rho
        rho_product = float(np.clip(rho_product, 0.0, self.rho_product_clip))

        prediction = self.predict(old_phi)
        delta = float(current_measurement) - prediction
        w_update = self.alpha_w * rho_product * delta * old_phi

        if not np.isfinite(delta) or not np.all(np.isfinite(w_update)):
            return None

        self.w += w_update
        return prediction, float(current_measurement), abs(delta)


class GVFAgent(Node):

    def __init__(self):
        super().__init__('gvf_horde_agent')

        # ---------------------------------------------------------------
        # Node parameters
        # ---------------------------------------------------------------
        self.declare_parameter('max_training_episodes', 10000)
        self.max_training_episodes = self.get_parameter('max_training_episodes').value

        self.declare_parameter('horizon_steps', 20)
        self.horizon = self.get_parameter('horizon_steps').value

        # --- Resume support ---
        # Preferred: resume everything from one combined checkpoint saved
        # by THIS script:
        #   -p resume_checkpoint:=/path/to/gvf_weights_ep_XXXXXX.npz
        # Bootstrap option: load each demon's weights from its OWN
        # separately-trained checkpoint (the files your individual
        # gvf_ttc_left_agent.py / gvf_ttc_right_agent.py / gvf_agent.py /
        # gvf_future_*_velocity_agent.py runs already produced), so you
        # don't lose that progress when switching to this merged script:
        #   -p resume_ttc_forward_checkpoint:=...
        #   -p resume_ttc_left_checkpoint:=...
        #   -p resume_ttc_right_checkpoint:=...
        #   -p resume_future_linear_velocity_checkpoint:=...
        #   -p resume_future_angular_velocity_checkpoint:=...
        # If any of these per-demon paths are given, resume_checkpoint is
        # ignored.
        self.declare_parameter('resume_checkpoint', '')
        self.declare_parameter('resume_ttc_forward_checkpoint', '')
        self.declare_parameter('resume_ttc_left_checkpoint', '')
        self.declare_parameter('resume_ttc_right_checkpoint', '')
        self.declare_parameter('resume_future_linear_velocity_checkpoint', '')
        self.declare_parameter('resume_future_angular_velocity_checkpoint', '')
        resume_checkpoint = self.get_parameter('resume_checkpoint').value
        resume_fwd = self.get_parameter('resume_ttc_forward_checkpoint').value
        resume_left = self.get_parameter('resume_ttc_left_checkpoint').value
        resume_right = self.get_parameter('resume_ttc_right_checkpoint').value
        resume_linvel = self.get_parameter('resume_future_linear_velocity_checkpoint').value
        resume_angvel = self.get_parameter('resume_future_angular_velocity_checkpoint').value

        # Environment & State configuration
        self.lidar_size = 24
        self.action_size = 5

        # -------------------------------------------------------------
        # SHARED behavior policy mu(a|s): uniform random over all 5
        # actions. See the module docstring for why this single policy
        # is correct for all 5 demons simultaneously.
        # -------------------------------------------------------------
        def behavior_policy_prob(_action):
            return 1.0 / float(self.action_size)
        self.behavior_policy_prob = behavior_policy_prob

        def get_behavior_action():
            return random.randint(0, self.action_size - 1)
        self.get_behavior_action = get_behavior_action

        # -------------------------------------------------------------
        # TTC hyperparameters. Defaults are UNCHANGED from the original
        # (gvf_agent.py / gvf_ttc_left_agent.py / gvf_ttc_right_agent.py);
        # they are now ROS parameters so a controlled experiment can override
        # them, e.g. -p alpha_theta:=0.2, without editing this file. With no
        # override the behaviour is byte-for-byte the original. (The original
        # 0.005 is ~40x too small for the forward-TTC target magnitude: see
        # DIAGNOSIS_GVF.md.)
        # -------------------------------------------------------------
        self.declare_parameter('alpha_theta', 0.005)
        self.declare_parameter('alpha_w', 0.0001)
        self.declare_parameter('trace_lambda', 0.4)
        self.declare_parameter('obstacle_threshold', 0.25)
        self.alpha_theta = self.get_parameter('alpha_theta').value
        self.alpha_w = self.get_parameter('alpha_w').value
        self.trace_lambda = self.get_parameter('trace_lambda').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value

        # Fixed-horizon hyperparameters (defaults unchanged). Set
        # -p velocity_importance_sampling:=false to drop the off-policy rho
        # product for the two velocity demons. That is justified when the
        # predicted quantity is policy-INDEPENDENT (TurtleBot3's commanded
        # linear speed is a constant 0.15 m/s for every action), where the
        # H-step IS product is both unnecessary and so near-zero that it keeps
        # the weights pinned at 0. Default true = original behaviour.
        self.declare_parameter('epsilon_target_velocity', 0.05)
        self.declare_parameter('alpha_w_horizon', 0.001)
        self.declare_parameter('rho_product_clip', 50.0)
        self.declare_parameter('velocity_importance_sampling', True)
        self.epsilon_target_velocity = self.get_parameter('epsilon_target_velocity').value
        self.alpha_w_horizon = self.get_parameter('alpha_w_horizon').value
        self.rho_product_clip = self.get_parameter('rho_product_clip').value
        self.velocity_importance_sampling = self.get_parameter('velocity_importance_sampling').value

        self.feature_dim = self.lidar_size * self.action_size

        # ---------------------------------------------------------------
        # The five demons
        # ---------------------------------------------------------------
        self.ttc_forward = DirectionalTTCGVF(
            self.feature_dim, self.action_size, FORWARD_ACTION, FULL_SCAN_IDX,
            self.alpha_theta, self.alpha_w, self.trace_lambda,
            self.obstacle_threshold, name='ttc_forward')
        self.ttc_left = DirectionalTTCGVF(
            self.feature_dim, self.action_size, LEFT_ACTION, FULL_SCAN_IDX,
            self.alpha_theta, self.alpha_w, self.trace_lambda,
            self.obstacle_threshold, name='ttc_left')
        self.ttc_right = DirectionalTTCGVF(
            self.feature_dim, self.action_size, RIGHT_ACTION, FULL_SCAN_IDX,
            self.alpha_theta, self.alpha_w, self.trace_lambda,
            self.obstacle_threshold, name='ttc_right')
        self.ttc_demons = [self.ttc_forward, self.ttc_left, self.ttc_right]

        self.future_linear_vel = FixedHorizonGVF(
            self.feature_dim, self.horizon, self.alpha_w_horizon, self.rho_product_clip)
        self.future_angular_vel = FixedHorizonGVF(
            self.feature_dim, self.horizon, self.alpha_w_horizon, self.rho_product_clip)
        self.horizon_demons = [
            ('future_linear_velocity', self.future_linear_vel),
            ('future_angular_velocity', self.future_angular_vel),
        ]

        # --- Resume ---
        self.start_episode = 1
        per_demon_resumes = {
            'ttc_forward': resume_fwd, 'ttc_left': resume_left,
            'ttc_right': resume_right, 'future_linear_velocity': resume_linvel,
            'future_angular_velocity': resume_angvel,
        }
        if any(per_demon_resumes.values()):
            self._load_per_demon_checkpoints(per_demon_resumes)
        elif resume_checkpoint:
            self._load_combined_checkpoint(resume_checkpoint)

        # ---------------------------------------------------------------
        # Odometry - needed only by the two velocity demons.
        # ---------------------------------------------------------------
        self._latest_linear_vel = 0.0
        self._latest_angular_vel = 0.0
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10)

        # ROS 2 Service Clients & Publishers
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        # -------------------------------------------------------------
        # Checkpointing - ONE combined file per interval, all 5 demons.
        # -------------------------------------------------------------
        self.checkpoint_dir = 'gvf_checkpoints_horde'
        self.checkpoint_every = 50
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # -------------------------------------------------------------
        # ONE shared matplotlib window, one panel per demon.
        # -------------------------------------------------------------
        self._max_plot_points = 2000
        self.episodes_list = []
        self.mae_lists = {d.name: [] for d in self.ttc_demons}
        self.rmse_lists = {name: [] for name, _d in self.horizon_demons}

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.canvas.manager.set_window_title('GVF Horde Agent Training Metrics')
        self.axes = self.axes.flatten()
        self.axes[5].axis('off')

        self.process()

    # -----------------------------------------------------------------
    def _odom_callback(self, msg):
        self._latest_linear_vel = msg.twist.twist.linear.x
        self._latest_angular_vel = msg.twist.twist.angular.z

    # -----------------------------------------------------------------
    # Feature machinery - identical to every original script.
    # -----------------------------------------------------------------
    def _extract_lidar(self, state):
        raw_state = np.copy(state).squeeze()
        if raw_state.size >= 26:
            return raw_state[2:26]
        elif raw_state.size == 24:
            return raw_state
        else:
            return raw_state[-self.lidar_size:]

    def get_feature_vector(self, state, action):
        phi = np.zeros((self.action_size, self.lidar_size), dtype=np.float64)
        lidar_scan = self._extract_lidar(state)
        norm_lidar = np.clip(lidar_scan / 3.5, 0.0, 1.0)
        phi[action, :] = norm_lidar
        return phi.flatten()

    def future_vel_target_prob(self, action):
        eps = self.epsilon_target_velocity
        if action == FORWARD_ACTION:
            return (1.0 - eps) + eps / self.action_size
        return eps / self.action_size

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------
    def save_checkpoint(self, episode):
        filepath = os.path.join(self.checkpoint_dir, f'gvf_weights_ep_{episode:06d}.npz')
        np.savez(
            filepath,
            episode=episode,
            theta_forward=self.ttc_forward.theta, w_forward=self.ttc_forward.w,
            theta_left=self.ttc_left.theta, w_left=self.ttc_left.w,
            theta_right=self.ttc_right.theta, w_right=self.ttc_right.w,
            w_future_linear_velocity=self.future_linear_vel.w,
            w_future_angular_velocity=self.future_angular_vel.w,
        )
        self.get_logger().info(f'Saved combined Horde checkpoint: {filepath}')

    def _load_combined_checkpoint(self, path):
        if not os.path.exists(path):
            self.get_logger().error(f'resume_checkpoint not found: {path}')
            raise FileNotFoundError(path)
        data = np.load(path)
        self.ttc_forward.theta = data['theta_forward']; self.ttc_forward.w = data['w_forward']
        self.ttc_left.theta = data['theta_left']; self.ttc_left.w = data['w_left']
        self.ttc_right.theta = data['theta_right']; self.ttc_right.w = data['w_right']
        self.future_linear_vel.w = data['w_future_linear_velocity']
        self.future_angular_vel.w = data['w_future_angular_velocity']
        self.start_episode = int(data['episode']) + 1
        self.get_logger().info(
            f'Resumed all 5 demons from combined checkpoint {path}, '
            f'continuing from episode {self.start_episode}')
        self._validate_start_episode()

    def _load_per_demon_checkpoints(self, paths: dict):
        """Bootstraps each demon independently from its own pre-existing,
        separately-trained checkpoint file (old single-demon .npz format:
        'theta'+'w' for the TTC demons, 'w' only for the velocity demons).
        Episode numbering then continues from the HIGHEST episode number
        found among the provided checkpoints - every demon trains together
        in lockstep from that point on, even one(s) that were further
        behind get carried forward without re-zeroing their weights."""
        max_episode = 0
        demon_lookup = {
            'ttc_forward': self.ttc_forward, 'ttc_left': self.ttc_left,
            'ttc_right': self.ttc_right,
        }
        for name, path in paths.items():
            if not path:
                continue
            if not os.path.exists(path):
                self.get_logger().error(f'{name} checkpoint not found: {path}')
                raise FileNotFoundError(path)
            data = np.load(path)
            if name in demon_lookup:
                demon_lookup[name].theta = data['theta']
                demon_lookup[name].w = data['w']
            elif name == 'future_linear_velocity':
                self.future_linear_vel.w = data['w']
            elif name == 'future_angular_velocity':
                self.future_angular_vel.w = data['w']
            ep = int(data['episode'])
            max_episode = max(max_episode, ep)
            self.get_logger().info(f'Bootstrapped {name} from {path} (episode {ep})')
        self.start_episode = max_episode + 1
        self.get_logger().info(
            f'All provided demons bootstrapped; training continues from '
            f'episode {self.start_episode} (max of the provided checkpoints). '
            f'Any demon NOT given a checkpoint starts fresh from zero.')
        self._validate_start_episode()

    def _validate_start_episode(self):
        if self.start_episode > self.max_training_episodes:
            self.get_logger().error(
                'Resumed episode >= max_training_episodes - increase '
                'max_training_episodes before resuming.')
            raise ValueError('max_training_episodes must exceed the resumed episode.')

    # -----------------------------------------------------------------
    # Live plot
    # -----------------------------------------------------------------
    def _trim_plot_lists(self):
        if len(self.episodes_list) > self._max_plot_points:
            del self.episodes_list[0]
            for lst in list(self.mae_lists.values()) + list(self.rmse_lists.values()):
                if lst:
                    del lst[0]

    def update_live_plot(self):
        panels = [
            ('TTC Forward - MAE', self.mae_lists['ttc_forward']),
            ('TTC Left - MAE', self.mae_lists['ttc_left']),
            ('TTC Right - MAE', self.mae_lists['ttc_right']),
            ('Future Linear Vel - RMSE', self.rmse_lists['future_linear_velocity']),
            ('Future Angular Vel - RMSE', self.rmse_lists['future_angular_velocity']),
        ]
        for ax, (title, series) in zip(self.axes[:5], panels):
            ax.clear()
            n = len(series)
            xs = self.episodes_list[-n:] if n > 0 else []
            ax.plot(xs, series, marker='o', markersize=3)
            ax.set_title(title)
            ax.set_xlabel('Episode')
            ax.grid(True)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    # -----------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------
    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode = self.start_episode
        while episode <= self.max_training_episodes:
            state = self.reset_environment()
            for d in self.ttc_demons:
                d.reset_trace()
            for _name, d in self.horizon_demons:
                d.reset_window()

            local_step = 0
            score = 0.0
            ttc_predictions = {d.name: [] for d in self.ttc_demons}
            horizon_pairs = {'future_linear_velocity': [], 'future_angular_velocity': []}

            time.sleep(0.5)

            while True:
                local_step += 1

                action = self.get_behavior_action()
                behaviour_prob = self.behavior_policy_prob(action)

                lidar_t = self._extract_lidar(state)
                phi_t = self.get_feature_vector(state, action)

                predicted_ttc_forward = self.ttc_forward.predict(phi_t)
                if math.isnan(predicted_ttc_forward) or math.isinf(predicted_ttc_forward):
                    predicted_ttc_forward = 0.0
                    self.get_logger().warn('Warning: TTC forward prediction diverged (NaN/Inf).')
                safe_ttc_forward = max(min(predicted_ttc_forward, 1e30), -1e30)

                next_state, reward, done = self.step(action)
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(safe_ttc_forward)]
                self.action_pub.publish(msg)

                # --- FIX FOR GAZEBO SENSOR SYNC BUG (unchanged) ---
                if done and local_step <= 2:
                    self.get_logger().warn(
                        'Ghost collision detected (Gazebo sync delay). Retrying episode...')
                    break

                lidar_tp1 = self._extract_lidar(next_state)

                # TTC demons: each one's target policy is deterministic, so
                # phi_bar_tp1 collapses to exactly phi(next_state, target_action).
                for d in self.ttc_demons:
                    phi_target_next = self.get_feature_vector(next_state, d.target_action)
                    pred = d.predict(phi_t)
                    ttc_predictions[d.name].append(
                        max(min(pred, 1e30), -1e30) if math.isfinite(pred) else 0.0)
                    d.update(phi_t, phi_target_next, lidar_t, lidar_tp1, action,
                             behaviour_prob, done)

                # Velocity demons: shared target policy (epsilon-greedy
                # toward FORWARD, eps=0.05).
                pi_vel = self.future_vel_target_prob(action)
                rho_vel = (pi_vel / max(behaviour_prob, 1e-8)
                           if self.velocity_importance_sampling else 1.0)

                lin_result = self.future_linear_vel.push_and_maybe_update(
                    phi_t, rho_vel, self._latest_linear_vel)
                if lin_result is not None:
                    prediction, target, _err = lin_result
                    horizon_pairs['future_linear_velocity'].append((prediction, target))

                ang_result = self.future_angular_vel.push_and_maybe_update(
                    phi_t, rho_vel, self._latest_angular_vel)
                if ang_result is not None:
                    prediction, target, _err = ang_result
                    horizon_pairs['future_angular_velocity'].append((prediction, target))

                state = next_state

                if done:
                    self._finish_episode(episode, local_step, score,
                                          ttc_predictions, horizon_pairs)
                    episode += 1
                    break

                time.sleep(0.01)

        if (episode - 1) % self.checkpoint_every != 0:
            self.save_checkpoint(episode - 1)

    def _finish_episode(self, episode, local_step, score, ttc_predictions, horizon_pairs):
        targets = np.arange(local_step, 0, -1, dtype=np.float64)
        ttc_metrics = {}
        for d in self.ttc_demons:
            preds = ttc_predictions[d.name]
            if len(preds) == len(targets):
                ttc_metrics[d.name] = compute_regression_metrics(preds, targets)
            else:
                ttc_metrics[d.name] = {'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan')}
            self.mae_lists[d.name].append(ttc_metrics[d.name]['mae'])

        horizon_metrics = {}
        for name, _d in self.horizon_demons:
            pairs = horizon_pairs[name]
            if pairs:
                preds, targs = zip(*pairs)
                horizon_metrics[name] = compute_regression_metrics(preds, targs)
            else:
                horizon_metrics[name] = {'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan')}
            self.rmse_lists[name].append(horizon_metrics[name]['rmse'])

        msg = Float32MultiArray()
        msg.data = [float(score), float(ttc_metrics['ttc_forward']['mae'])]
        self.result_pub.publish(msg)

        self.get_logger().info(
            f"Episode: {episode} | Steps: {local_step} | Reward: {score:.2f} | "
            f"TTC_fwd MAE: {ttc_metrics['ttc_forward']['mae']:.2f} | "
            f"TTC_L MAE: {ttc_metrics['ttc_left']['mae']:.2f} | "
            f"TTC_R MAE: {ttc_metrics['ttc_right']['mae']:.2f} | "
            f"FutVel RMSE: {horizon_metrics['future_linear_velocity']['rmse']:.3f} | "
            f"FutAngVel RMSE: {horizon_metrics['future_angular_velocity']['rmse']:.3f}"
        )

        self.episodes_list.append(episode)
        self._trim_plot_lists()
        self.update_live_plot()

        if episode % self.checkpoint_every == 0:
            self.save_checkpoint(episode)

    # -----------------------------------------------------------------
    # ROS service plumbing - unchanged.
    # -----------------------------------------------------------------
    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for make_environment service...')
        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for reset_environment service...')
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state_array = np.asarray(future.result().state)
            state = np.reshape(state_array, [1, -1])
        else:
            self.get_logger().error(f'Failed to reset environment: {future.exception()}')
            state = np.zeros((1, self.lidar_size))
        return state

    def step(self, action):
        req = Dqn.Request()
        req.action = action
        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for rl_agent_interface service...')
        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state_array = np.asarray(future.result().state)
            next_state = np.reshape(state_array, [1, -1])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(f'Step service call failed: {future.exception()}')
            next_state = np.zeros((1, self.lidar_size))
            reward = 0.0
            done = True
        return next_state, reward, done


def main(args=None):
    rclpy.init(args=args)
    gvf_agent = GVFAgent()
    try:
        rclpy.spin(gvf_agent)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()
        gvf_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
