#!/usr/bin/env python3
#################################################################################
# Copyright 2026
# Multi-GVF Horde Agent for TurtleBot3 Burger ROS 2
# EVALUATION SCRIPT (one evaluate agent, all 5 demons)
#
# ---------------------------------------------------------------------------
# IMPORTANT - why this runs 4 separate rollout PHASES per checkpoint, not one
# ---------------------------------------------------------------------------
# During TRAINING, all 5 demons learn off-policy from ONE shared random
# trajectory - that is exactly what off-policy learning is for: answering
# several different "what if I followed policy X" questions from a single
# stream of experience.
#
# EVALUATION is different on purpose: its whole point is to run EACH demon's
# OWN target policy ON-POLICY and see how it actually performs when
# followed. But TTC_forward's target policy is "always go straight forward",
# TTC_left's is "always turn left", and TTC_right's is "always turn right" -
# these are mutually exclusive, deterministic behaviors. A single robot
# cannot simultaneously drive forward, left, AND right to produce one
# trajectory that is on-policy for all three at once.
#
# So this script runs each checkpoint through 4 distinct rollout phases:
#   Phase A (deterministic FORWARD)        -> evaluates TTC_forward
#   Phase B (deterministic LEFT)           -> evaluates TTC_left
#   Phase C (deterministic RIGHT)          -> evaluates TTC_right
#   Phase D (epsilon-greedy FORWARD, 0.05) -> evaluates BOTH velocity demons
#            together (they share the same target policy, so one rollout
#            phase serves both - no conflict between them).
#
# Still "one evaluate agent": one script, one process, one combined
# checkpoint format, one combined graph - it just does its job in 4 rollout
# phases per checkpoint instead of pretending one trajectory can answer five
# different policy questions at once.
#################################################################################

import collections
import glob
import math
import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry

from turtlebot3_msgs.srv import Dqn


FORWARD_ACTION = 2
LEFT_ACTION = 1
RIGHT_ACTION = 3


def compute_regression_metrics(predictions, targets):
    predictions_arr = np.asarray(predictions, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)
    errors = predictions_arr - targets_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((targets_arr - np.mean(targets_arr)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


class GVFEvaluationAgent(Node):

    def __init__(self):
        super().__init__('gvf_horde_agent_evaluation_agent')

        self.declare_parameter('checkpoint_dir', 'gvf_checkpoints_horde')
        self.checkpoint_dir = self.get_parameter('checkpoint_dir').value

        self.declare_parameter('num_eval_episodes', 20)
        self.num_eval_episodes = self.get_parameter('num_eval_episodes').value

        self.declare_parameter('horizon_steps', 20)
        self.horizon = self.get_parameter('horizon_steps').value

        self.lidar_size = 24
        self.action_size = 5
        self.epsilon_target_velocity = 0.05
        self.feature_dim = self.lidar_size * self.action_size

        # Weights for all 5 demons - overwritten per checkpoint.
        self.theta_forward = np.zeros(self.feature_dim, dtype=np.float64)
        self.theta_left = np.zeros(self.feature_dim, dtype=np.float64)
        self.theta_right = np.zeros(self.feature_dim, dtype=np.float64)
        self.w_linear_vel = np.zeros(self.feature_dim, dtype=np.float64)
        self.w_angular_vel = np.zeros(self.feature_dim, dtype=np.float64)

        self._latest_linear_vel = 0.0
        self._latest_angular_vel = 0.0
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10)

        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')

        # -------------------------------------------------------------
        # ONE shared matplotlib window, one panel per demon - same layout
        # as the training agent's.
        # -------------------------------------------------------------
        self.checkpoints_list = []
        self.metric_lists = {
            'ttc_forward': [], 'ttc_left': [], 'ttc_right': [],
            'future_linear_velocity': [], 'future_angular_velocity': [],
        }

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.canvas.manager.set_window_title('GVF Horde Agent Checkpoint Evaluation')
        self.axes = self.axes.flatten()
        self.axes[5].axis('off')

        self.process()

    def _odom_callback(self, msg):
        self._latest_linear_vel = msg.twist.twist.linear.x
        self._latest_angular_vel = msg.twist.twist.angular.z

    # ---------------------------------------------------------------
    # Feature machinery - identical to every training agent.
    # ---------------------------------------------------------------
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

    def predict_ttc(self, theta, state, action):
        phi = self.get_feature_vector(state, action)
        return float(np.dot(theta, phi))

    def predict_vel(self, w, phi):
        return float(np.dot(w, phi))

    def compute_episode_mae(self, episode_predictions, episode_length):
        predictions_arr = np.array(episode_predictions, dtype=np.float64)
        targets = np.arange(episode_length, 0, -1, dtype=np.float64)
        errors = np.abs(predictions_arr - targets)
        return float(np.mean(errors))

    # ---------------------------------------------------------------
    # Checkpoint discovery
    # ---------------------------------------------------------------
    def find_checkpoints(self):
        pattern = os.path.join(self.checkpoint_dir, 'gvf_weights_ep_*.npz')
        candidate_files = glob.glob(pattern)
        checkpoints = []
        for filepath in candidate_files:
            try:
                data = np.load(filepath)
                checkpoints.append((int(data['episode']), filepath))
            except Exception as exc:
                self.get_logger().warn(f'Skipping unreadable checkpoint {filepath}: {exc}')
        checkpoints.sort(key=lambda pair: pair[0])
        return checkpoints

    def update_live_plot(self):
        panels = [
            ('ttc_forward', 'TTC Forward - Avg MAE'),
            ('ttc_left', 'TTC Left - Avg MAE'),
            ('ttc_right', 'TTC Right - Avg MAE'),
            ('future_linear_velocity', 'Future Linear Vel - Avg RMSE'),
            ('future_angular_velocity', 'Future Angular Vel - Avg RMSE'),
        ]
        for ax, (key, title) in zip(self.axes[:5], panels):
            ax.clear()
            series = self.metric_lists[key]
            n = len(series)
            xs = self.checkpoints_list[-n:] if n > 0 else []
            ax.plot(xs, series, marker='o', markersize=4, color='green')
            ax.set_title(title)
            ax.set_xlabel('Training Episode at Checkpoint')
            ax.grid(True)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    # ---------------------------------------------------------------
    # Main evaluation loop
    # ---------------------------------------------------------------
    def process(self):
        self.env_make()
        time.sleep(1.0)

        checkpoints = self.find_checkpoints()
        num_checkpoints = len(checkpoints)
        if num_checkpoints == 0:
            self.get_logger().warn(f'No checkpoints found in "{self.checkpoint_dir}".')
            return
        self.get_logger().info(f'Found {num_checkpoints} checkpoint(s) in "{self.checkpoint_dir}".')

        for idx, (training_episode, filepath) in enumerate(checkpoints, start=1):
            data = np.load(filepath)
            self.theta_forward = data['theta_forward']
            self.theta_left = data['theta_left']
            self.theta_right = data['theta_right']
            self.w_linear_vel = data['w_future_linear_velocity']
            self.w_angular_vel = data['w_future_angular_velocity']

            self.get_logger().info(
                f'[{idx}/{num_checkpoints}] Evaluating checkpoint from training episode '
                f'{training_episode} ({os.path.basename(filepath)})'
            )

            # --- Phase A: deterministic FORWARD -> TTC_forward ---
            fwd_maes = [self._run_ttc_phase(FORWARD_ACTION, self.theta_forward)
                        for _ in range(self.num_eval_episodes)]
            mae_forward = float(np.mean(fwd_maes))

            # --- Phase B: deterministic LEFT -> TTC_left ---
            left_maes = [self._run_ttc_phase(LEFT_ACTION, self.theta_left)
                         for _ in range(self.num_eval_episodes)]
            mae_left = float(np.mean(left_maes))

            # --- Phase C: deterministic RIGHT -> TTC_right ---
            right_maes = [self._run_ttc_phase(RIGHT_ACTION, self.theta_right)
                          for _ in range(self.num_eval_episodes)]
            mae_right = float(np.mean(right_maes))

            # --- Phase D: epsilon-greedy FORWARD (0.05) -> both velocity demons ---
            all_lin_pairs, all_ang_pairs = [], []
            for _ in range(self.num_eval_episodes):
                lin_pairs, ang_pairs = self._run_velocity_phase()
                all_lin_pairs.extend(lin_pairs)
                all_ang_pairs.extend(ang_pairs)
            lin_metrics = (compute_regression_metrics(*zip(*all_lin_pairs)) if all_lin_pairs
                           else {'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan')})
            ang_metrics = (compute_regression_metrics(*zip(*all_ang_pairs)) if all_ang_pairs
                           else {'mae': float('nan'), 'rmse': float('nan'), 'r2': float('nan')})

            self.checkpoints_list.append(training_episode)
            self.metric_lists['ttc_forward'].append(mae_forward)
            self.metric_lists['ttc_left'].append(mae_left)
            self.metric_lists['ttc_right'].append(mae_right)
            self.metric_lists['future_linear_velocity'].append(lin_metrics['rmse'])
            self.metric_lists['future_angular_velocity'].append(ang_metrics['rmse'])
            self.update_live_plot()

            self.get_logger().info(
                f"[{idx}/{num_checkpoints}] Episode {training_episode} | "
                f"TTC_fwd MAE: {mae_forward:.2f} | TTC_L MAE: {mae_left:.2f} | "
                f"TTC_R MAE: {mae_right:.2f} | FutLinVel RMSE: {lin_metrics['rmse']:.3f} | "
                f"FutAngVel RMSE: {ang_metrics['rmse']:.3f}"
            )

        self.get_logger().info('Finished evaluating all stored checkpoints.')

    # ---------------------------------------------------------------
    # Rollout phases
    # ---------------------------------------------------------------
    def _run_ttc_phase(self, deterministic_action, theta):
        """One on-policy episode, always taking `deterministic_action`,
        scored against `theta`. Retries on the Gazebo ghost-collision
        artifact, same as every training/eval script in this project."""
        while True:
            state = self.reset_environment()
            time.sleep(0.5)
            local_step = 0
            episode_predictions = []

            while True:
                local_step += 1
                predicted = self.predict_ttc(theta, state, deterministic_action)
                if math.isnan(predicted) or math.isinf(predicted):
                    predicted = 0.0
                safe_pred = max(min(predicted, 1e30), -1e30)
                episode_predictions.append(safe_pred)

                next_state, _reward, done = self.step(deterministic_action)
                state = next_state

                if done:
                    if local_step <= 2:
                        self.get_logger().warn('Ghost collision detected. Retrying episode...')
                        break
                    return self.compute_episode_mae(episode_predictions, local_step)

                time.sleep(0.01)

    def _run_velocity_phase(self):
        """One on-policy episode under the velocity demons' shared target
        policy (epsilon-greedy toward FORWARD, eps=0.05). Returns
        (linear_pairs, angular_pairs) - the matured (prediction, target)
        tuples collected during this episode for each demon."""
        eps = self.epsilon_target_velocity
        while True:
            state = self.reset_environment()
            lin_pending = collections.deque()
            ang_pending = collections.deque()
            time.sleep(0.5)
            local_step = 0
            lin_pairs, ang_pairs = [], []

            while True:
                local_step += 1
                action = (random.randint(0, self.action_size - 1)
                          if random.random() < eps else FORWARD_ACTION)
                phi_t = self.get_feature_vector(state, action)

                next_state, _reward, done = self.step(action)
                state = next_state

                if done and local_step <= 2:
                    self.get_logger().warn('Ghost collision detected. Retrying episode...')
                    break

                lin_pending.append(phi_t)
                if len(lin_pending) > self.horizon:
                    old_phi = lin_pending.popleft()
                    lin_pairs.append((self.predict_vel(self.w_linear_vel, old_phi),
                                       self._latest_linear_vel))

                ang_pending.append(phi_t)
                if len(ang_pending) > self.horizon:
                    old_phi = ang_pending.popleft()
                    ang_pairs.append((self.predict_vel(self.w_angular_vel, old_phi),
                                       self._latest_angular_vel))

                if done:
                    return lin_pairs, ang_pairs

                time.sleep(0.01)

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
    eval_agent = GVFEvaluationAgent()
    try:
        plt.ioff()
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        eval_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
