#!/usr/bin/env python3
#################################################################################
# Copyright 2026
# GVF Horde Prediction Agent for TurtleBot3 Burger ROS 2
# Based on the Horde Architecture (Sutton et al., 2011) using GQ(lambda)
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

from turtlebot3_msgs.srv import Dqn


class GVFAgent(Node):

    def __init__(self):
        super().__init__('gvf_agent')

        # Node parameters
        self.declare_parameter('max_training_episodes', 10000)
        self.max_training_episodes = self.get_parameter('max_training_episodes').value

        # --- Resume support ---
        # Pass -p resume_checkpoint:=/path/to/gvf_weights_ep_XXXXXX.npz to
        # continue training from a previously saved checkpoint instead of
        # starting theta/w from zero at episode 1.
        self.declare_parameter('resume_checkpoint', '')
        resume_checkpoint = self.get_parameter('resume_checkpoint').value

        # Environment & State configurations
        self.lidar_size = 24  # Using strictly the 24 Lidar points
        self.action_size = 5
        self.FORWARD_ACTION = 2  # Action index corresponding to straight forward

        # -------------------------------------------------------------
        # Horde GVF Question & Answer Hyperparameters (Section 5.1)
        # -------------------------------------------------------------
        self.alpha_theta = 0.005      # Step size for main weights theta
        self.alpha_w = 0.0001     # Step size for secondary weights w
        self.trace_lambda = 0.4    # Eligibility trace decay lambda
        self.obstacle_threshold = 0.25  # Front laser collision threshold in meters

        # Feature vector dimension (only lidar features per action)
        self.feature_dim = self.lidar_size * self.action_size

        # 7 demons: 0:TTC, 1:goal_angle, 2-5:IR sectors, 6:imu yaw
        self.num_demons = 7
        self.demon_names = ['ttc', 'goal', 'ir0', 'ir1', 'ir2', 'ir3', 'imu']
        self.theta = np.zeros((self.num_demons, self.feature_dim), dtype=np.float64)
        self.w = np.zeros((self.num_demons, self.feature_dim), dtype=np.float64)
        self.e = np.zeros((self.num_demons, self.feature_dim), dtype=np.float64)

        # --- Resume: load theta/w from checkpoint if one was given ---
        self.start_episode = 1
        if resume_checkpoint:
            if os.path.exists(resume_checkpoint):
                data = np.load(resume_checkpoint)
                self.theta = data['theta']
                self.w = data['w']
                self.start_episode = int(data['episode']) + 1
                self.get_logger().info(
                    f'Resumed from {resume_checkpoint}: theta/w loaded, '
                    f'continuing from episode {self.start_episode}')
                if self.start_episode > self.max_training_episodes:
                    self.get_logger().error(
                        'Loaded checkpoint episode >= max_training_episodes - '
                        'increase max_training_episodes before resuming.')
                    raise ValueError(
                        'max_training_episodes must exceed the resumed episode.')
            else:
                self.get_logger().error(f'resume_checkpoint not found: {resume_checkpoint}')
                raise FileNotFoundError(resume_checkpoint)

        # ROS 2 Service Clients & Publishers
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        # -------------------------------------------------------------
        # Checkpointing: save theta/w every N episodes for later evaluation
        # -------------------------------------------------------------
        self.checkpoint_dir = 'gvf_checkpoints'
        self.checkpoint_every = 50
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # -------------------------------------------------------------
        # Matplotlib Graph Setup (MAE of Demon 0 per Episode)
        # -------------------------------------------------------------
        self.episodes_list = []
        self.mae_logs = [[] for _ in range(self.num_demons)]

        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(7, 1, figsize=(8, 12), sharex=True)
        self.fig.canvas.manager.set_window_title('GVF Agent Training Metrics')

        self.process()

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

    def target_policy_prob(self, action):
        return 1.0 if action == self.FORWARD_ACTION else 0.0

    def behavior_policy_prob(self, action):
        return 1.0 / float(self.action_size)

    def get_behavior_action(self):
        return random.randint(0, self.action_size - 1)

    def compute_gamma(self, state):
        lidar_scan = self._extract_lidar(state)
        min_dist = np.min(lidar_scan) if lidar_scan.size > 0 else 1.0
        if min_dist <= self.obstacle_threshold:
            return 0.0 
        return 1.0

    def compute_reward(self, state):
        return 1.0

    def compute_terminal_reward(self, state):
        return 0.0

    def predict_demon(self, state, action, d=0):
        phi = self.get_feature_vector(state, action)
        return float(np.dot(self.theta[d], phi))

    def update_gq_lambda(self, state, action, next_state, done):
        phi_t = self.get_feature_vector(state, action)

        phi_bar_tp1 = np.zeros(self.feature_dim, dtype=np.float64)
        for act in range(self.action_size):
            pi_prob = self.target_policy_prob(act)
            if pi_prob > 0.0:
                phi_bar_tp1 += pi_prob * self.get_feature_vector(next_state, act)

        ns = np.copy(next_state).squeeze()
        lidar_next = ns[2:26]
        goal_angle_next = float(ns[1])
        theta_next = float(ns[-1])
        min_front_next = float(np.min(lidar_next))
        cumulant_ttc = 1.0
        cumulant_goal = math.cos(goal_angle_next)
        cumulant_ir0 = float(np.min(lidar_next[0:6])) / 3.5
        cumulant_ir1 = float(np.min(lidar_next[6:12])) / 3.5
        cumulant_ir2 = float(np.min(lidar_next[12:18])) / 3.5
        cumulant_ir3 = float(np.min(lidar_next[18:24])) / 3.5
        cumulant_imu = math.sin(theta_next)
        cumulants = [cumulant_ttc, cumulant_goal, cumulant_ir0, cumulant_ir1, cumulant_ir2, cumulant_ir3, cumulant_imu]

        s = np.copy(state).squeeze()
        min_front_now = float(np.min(s[2:26]))

        gamma_next_ttc = 0.0 if done or min_front_next <= self.obstacle_threshold else 1.0
        gamma_now_ttc = 0.0 if min_front_now <= self.obstacle_threshold else 1.0
        gamma_next_rest = 0.0 if done else 0.9
        gamma_nexts = [gamma_next_ttc, gamma_next_rest, gamma_next_rest, gamma_next_rest, gamma_next_rest, gamma_next_rest, gamma_next_rest]
        gamma_nows = [gamma_now_ttc, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

        pi_a = self.target_policy_prob(action)
        b_a = self.behavior_policy_prob(action)
        importance_ratio = pi_a / b_a if b_a > 0.0 else 0.0

        for d in range(self.num_demons):
            gamma_tp1 = gamma_nexts[d]
            gamma_t = gamma_nows[d]
            delta_t = cumulants[d] + gamma_tp1 * np.dot(self.theta[d], phi_bar_tp1) - np.dot(self.theta[d], phi_t)
            new_e = phi_t + gamma_t * self.trace_lambda * importance_ratio * self.e[d]
            self.e[d] = new_e
            self.theta[d] += self.alpha_theta * (delta_t * new_e - gamma_tp1 * (1.0 - self.trace_lambda) * np.dot(self.w[d], new_e) * phi_bar_tp1)
            self.w[d] += self.alpha_w * (delta_t * new_e - np.dot(self.w[d], phi_t) * phi_t)

        return cumulants, gamma_nexts

    def save_checkpoint(self, episode):
        """Saves theta/w (and the episode number they were learned at) to
        gvf_checkpoints/gvf_weights_ep_{episode}.npz - the evaluation script
        looks for files matching this exact naming pattern."""
        filepath = os.path.join(self.checkpoint_dir, f'gvf_weights_ep_{episode:06d}.npz')
        np.savez(filepath, theta=self.theta, w=self.w, episode=episode)
        self.get_logger().info(f'Saved GVF checkpoint: {filepath}')

    def update_live_plot(self):
        """Updates the Matplotlib window at the end of each episode."""
        for d in range(self.num_demons):
            self.axes[d].clear()
            self.axes[d].plot(self.episodes_list, self.mae_logs[d], marker='o', markersize=3)
            self.axes[d].set_title(f'{self.demon_names[d]} MAE')
            self.axes[d].grid(True)
        self.axes[-1].set_xlabel('Episode Number')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def process(self):
        self.env_make()
        time.sleep(1.0)
        
        episode = self.start_episode
        while episode <= self.max_training_episodes:
            state = self.reset_environment()
            self.e.fill(0.0)  
            
            local_step = 0
            score = 0.0
            ep_preds = [[] for _ in range(self.num_demons)]
            ep_C = [[] for _ in range(self.num_demons)]
            ep_Gnext = [[] for _ in range(self.num_demons)]
            demon_sums = np.zeros(self.num_demons)

            time.sleep(0.5)

            while True:
                local_step += 1

                action = self.get_behavior_action()
                all_pred = [self.predict_demon(state, action, d) for d in range(self.num_demons)]
                pred = all_pred[0]
                if math.isnan(pred) or math.isinf(pred):
                    pred = 0.0
                    self.get_logger().warn('Warning: demon prediction diverged (NaN/Inf detected).')
                
                safe_pred = max(min(pred, 1e30), -1e30)
                safe_all = np.clip(np.nan_to_num(np.array(all_pred, dtype=np.float64), nan=0.0, posinf=1e30, neginf=-1e30), -1e30, 1e30)
                for d in range(self.num_demons):
                    ep_preds[d].append(float(safe_all[d]))
                demon_sums += safe_all

                next_state, reward, done = self.step(action)
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(safe_pred)]
                self.action_pub.publish(msg)

                # --- FIX FOR GAZEBO SENSOR SYNC BUG ---
                # Check for the ghost collision BEFORE learning from this
                # transition. A "done" flag this early is a Gazebo timing
                # artifact, not a real terminal transition - feeding it into
                # update_gq_lambda would fold a spurious gamma=0 update into
                # theta/w even though the whole episode gets discarded.
                if done and local_step <= 2:
                    self.get_logger().warn('Ghost collision detected (Gazebo sync delay). Retrying episode...')
                    break

                step_C, step_G = self.update_gq_lambda(state, action, next_state, done)
                for d in range(self.num_demons):
                    ep_C[d].append(float(step_C[d]))
                    ep_Gnext[d].append(float(step_G[d]))
                state = next_state

                if done:
                    demon_maes = []
                    demon_means = demon_sums / max(local_step, 1)
                    for d in range(self.num_demons):
                        running_return = 0.0
                        errors = []
                        for j in range(local_step - 1, -1, -1):
                            running_return = ep_C[d][j] + ep_Gnext[d][j] * running_return
                            errors.append(abs(ep_preds[d][j] - running_return))
                        demon_maes.append(float(np.mean(errors)) if errors else 0.0)

                    msg = Float32MultiArray()
                    msg.data = [float(score)] + [float(v) for v in demon_maes] + [float(v) for v in demon_means]
                    self.result_pub.publish(msg)

                    self.get_logger().info(
                        f'Episode: {episode} | Steps: {local_step} | Score: {score:.2f} | ' + ' '.join(f'{n}-MAE:{m:.2f}' for n, m in zip(self.demon_names, demon_maes))
                    )

                    # --- EPISODE GRAPH UPDATE ---
                    self.episodes_list.append(episode)
                    for d in range(self.num_demons):
                        self.mae_logs[d].append(demon_maes[d])
                    self.update_live_plot()

                    # --- PERIODIC CHECKPOINT SAVE ---
                    if episode % self.checkpoint_every == 0:
                        self.save_checkpoint(episode)

                    episode += 1
                    break

                time.sleep(0.01)

        # Final safety-save once training completes, in case the last
        # episode wasn't itself a multiple of checkpoint_every.
        if (episode - 1) % self.checkpoint_every != 0:
            self.save_checkpoint(episode - 1)

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
    gvf_agent = GVFAgent()
    try:
        rclpy.spin(gvf_agent)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show() # Keep the graph window open when you stop the node
        gvf_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
