#!/usr/bin/env python3
#################################################################################
# Copyright 2026
# GVF Horde Evaluation Agent for TurtleBot3 Burger ROS 2
# EVALUATION SCRIPT
#
# Loads every checkpoint saved by the training agent (gvf_checkpoints/gvf_weights
# _ep_*.npz), and for each one runs num_eval_episodes (default 20) on-policy
# episodes: the behavior policy here IS the target policy (always FORWARD_ACTION,
# deterministic, no exploration, no learning). This measures how the frozen
# theta at that point in training performs when actually followed, rather than
# under the random behavior policy used for off-policy training.
#
# For each checkpoint: num_eval_episodes episodes are rolled out, each demon's
# per-step predictions are compared against its true discounted return to get
# a per-episode MAE per demon, and the average of those MAEs becomes a single
# point per demon on a Checkpoint vs Avg MAE graph.
#################################################################################

import glob
import math
import os
import numpy as np
import time
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn


class GVFEvaluationAgent(Node):

    def __init__(self):
        super().__init__('gvf_evaluation_agent')

        self.declare_parameter('checkpoint_dir', 'gvf_checkpoints')
        self.checkpoint_dir = self.get_parameter('checkpoint_dir').value

        # How many on-policy episodes to roll out per checkpoint - the MAE
        # reported for a checkpoint is the average across all of them.
        self.declare_parameter('num_eval_episodes', 20)
        self.num_eval_episodes = self.get_parameter('num_eval_episodes').value

        # Must match the training agent's configuration exactly - the feature
        # encoding has to line up with how theta was actually learned.
        self.lidar_size = 24
        self.action_size = 5
        self.FORWARD_ACTION = 2
        self.feature_dim = self.lidar_size * self.action_size
        self.obstacle_threshold = 0.25

        self.num_demons = 7
        self.demon_names = ['ttc', 'goal', 'ir0', 'ir1', 'ir2', 'ir3', 'imu']

        # theta gets overwritten with each checkpoint's weights before that
        # checkpoint's evaluation episode runs.
        self.theta = np.zeros((self.num_demons, self.feature_dim), dtype=np.float64)

        # ROS 2 Service Clients
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')

        # -------------------------------------------------------------
        # Matplotlib Graph Setup (Evaluation MAE per Checkpoint)
        # -------------------------------------------------------------
        self.checkpoints_list = []   # training episode number each checkpoint was saved at
        self.mae_logs = [[] for _ in range(self.num_demons)]

        plt.ion()
        self.fig, self.axes = plt.subplots(7, 1, figsize=(8, 12), sharex=True)
        self.fig.canvas.manager.set_window_title('GVF Checkpoint Evaluation')

        self.process()

    # ---------------------------------------------------------------
    # Feature machinery - identical to the training agent, since theta
    # only means anything under the exact same feature encoding.
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

    def predict_demon(self, state, action, d=0):
        phi = self.get_feature_vector(state, action)
        return float(np.dot(self.theta[d], phi))

    def get_action(self):
        # Behavior policy == target policy for evaluation: always drive
        # straight forward, deterministically. No exploration.
        return self.FORWARD_ACTION

    # ---------------------------------------------------------------
    # Checkpoint discovery
    # ---------------------------------------------------------------
    def find_checkpoints(self):
        """Returns a list of (training_episode, filepath) tuples, sorted by
        training episode ascending. The episode number is read from inside
        each .npz file (not parsed from the filename), so this stays robust
        even if the naming convention changes."""
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
        for d in range(self.num_demons):
            self.axes[d].clear()
            self.axes[d].plot(self.checkpoints_list, self.mae_logs[d], marker='o', markersize=3)
            self.axes[d].set_title(f'{self.demon_names[d]} avg MAE ({self.num_eval_episodes} eps)')
            self.axes[d].grid(True)
        self.axes[-1].set_xlabel('Training Episode at Checkpoint')
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
            theta = data['theta']
            if np.shape(theta) != (self.num_demons, self.feature_dim):
                self.get_logger().warn(f'Skipping {os.path.basename(filepath)}: expected theta shape {(self.num_demons, self.feature_dim)}, got {np.shape(theta)}')
                continue
            self.theta = theta

            self.get_logger().info(
                f'[{idx}/{num_checkpoints}] Evaluating checkpoint from training episode '
                f'{training_episode} ({os.path.basename(filepath)}) over {self.num_eval_episodes} episodes'
            )

            episode_maes = [[] for _ in range(self.num_demons)]
            for eval_ep in range(1, self.num_eval_episodes + 1):
                demon_maes = self.run_evaluation_episode()
                for d in range(self.num_demons):
                    episode_maes[d].append(demon_maes[d])
                self.get_logger().info(
                    f'    eval episode {eval_ep}/{self.num_eval_episodes} -> ' + ' '.join(f'{n}:{m:.2f}' for n, m in zip(self.demon_names, demon_maes))
                )

            avg_maes = [float(np.mean(episode_maes[d])) for d in range(self.num_demons)]

            self.checkpoints_list.append(training_episode)
            for d in range(self.num_demons):
                self.mae_logs[d].append(avg_maes[d])
            self.update_live_plot()

            self.get_logger().info(
                f'[{idx}/{num_checkpoints}] Training Episode: {training_episode} | '
                f'Avg MAE over {self.num_eval_episodes} episodes: ' + ' '.join(f'{n}:{m:.2f}' for n, m in zip(self.demon_names, avg_maes))
            )

        self.get_logger().info('Finished evaluating all stored checkpoints.')

    def run_evaluation_episode(self):
        """Runs exactly one on-policy (always-forward) episode with the
        currently loaded theta, frozen - no learning happens here. Retries
        on the same Gazebo ghost-collision artifact the training agent
        guards against, and returns the MAE per demon for the first real episode."""
        while True:
            state = self.reset_environment()
            time.sleep(0.5)

            local_step = 0
            ep_preds = [[] for _ in range(self.num_demons)]
            ep_C = [[] for _ in range(self.num_demons)]
            ep_Gnext = [[] for _ in range(self.num_demons)]

            while True:
                local_step += 1

                action = self.get_action()
                all_pred = [self.predict_demon(state, action, d) for d in range(self.num_demons)]
                safe_all = np.clip(np.nan_to_num(np.array(all_pred, dtype=np.float64), nan=0.0, posinf=1e30, neginf=-1e30), -1e30, 1e30)
                for d in range(self.num_demons):
                    ep_preds[d].append(float(safe_all[d]))

                next_state, reward, done = self.step(action)

                ns = np.copy(next_state).squeeze()
                lidar_next = ns[2:26]
                min_front_next = float(np.min(lidar_next))
                cumulants = [1.0, math.cos(float(ns[1])), float(np.min(lidar_next[0:6])) / 3.5, float(np.min(lidar_next[6:12])) / 3.5, float(np.min(lidar_next[12:18])) / 3.5, float(np.min(lidar_next[18:24])) / 3.5, math.sin(float(ns[-1]))]
                gamma_next_ttc = 0.0 if done or min_front_next <= self.obstacle_threshold else 1.0
                gamma_next_rest = 0.0 if done else 0.9
                gamma_nexts = [gamma_next_ttc, gamma_next_rest, gamma_next_rest, gamma_next_rest, gamma_next_rest, gamma_next_rest, gamma_next_rest]
                for d in range(self.num_demons):
                    ep_C[d].append(float(cumulants[d]))
                    ep_Gnext[d].append(float(gamma_nexts[d]))
                state = next_state

                if done:
                    if local_step <= 2:
                        self.get_logger().warn('Ghost collision detected during evaluation. Retrying episode...')
                        break  # retries the reset loop above, same checkpoint
                    demon_maes = []
                    for d in range(self.num_demons):
                        running_return = 0.0
                        errors = []
                        for j in range(local_step - 1, -1, -1):
                            running_return = ep_C[d][j] + ep_Gnext[d][j] * running_return
                            errors.append(abs(ep_preds[d][j] - running_return))
                        demon_maes.append(float(np.mean(errors)) if errors else 0.0)
                    return demon_maes

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
            state = np.zeros((1, self.lidar_size + 3))
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
            next_state = np.zeros((1, self.lidar_size + 3))
            reward = 0.0
            done = True

        return next_state, reward, done


def main(args=None):
    rclpy.init(args=args)
    eval_agent = GVFEvaluationAgent()
    try:
        plt.ioff()
        plt.show()  # Keep the graph window open once evaluation finishes
    except KeyboardInterrupt:
        pass
    finally:
        eval_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()