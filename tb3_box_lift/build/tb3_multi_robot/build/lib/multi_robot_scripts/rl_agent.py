#!/usr/bin/env python3
"""
RL Agent for Multi-Robot Box Lifting Task
==========================================
Each robot independently learns to position itself around a box so
that when all 3 are in their correct positions, the box lifts.

Algorithm : Independent Q-Learning (simple, hackathon-friendly)
           Can be upgraded to MAPPO for bonus RL points.

Observation per robot (6 values):
  [my_x, my_y, box_x, box_y, dist_to_box, angle_to_box]

Action space (5 discrete):
  0=forward, 1=backward, 2=left, 3=right, 4=stop

Reward:
  +100  → all 3 robots in correct lift positions (box lifts)
  +1    → individual robot closer to its target slot
  -0.1  → time penalty per step
  -5    → collision
"""

import math
import random
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Bool

# ── Hyperparameters ──────────────────────────────────────────────
ALPHA        = 0.1      # learning rate
GAMMA        = 0.95     # discount factor
EPSILON      = 1.0      # initial exploration
EPSILON_MIN  = 0.05
EPSILON_DECAY= 0.995
GRID_SIZE    = 10       # discretize continuous space into 10×10 grid

# Target positions relative to box centre (triangle formation)
# Robot slots at 120° apart, ~0.35 m from box
LIFT_RADIUS  = 0.35
TARGET_SLOTS = [
    ( -LIFT_RADIUS,          0.0         ),   # slot 0: left
    (  LIFT_RADIUS * 0.5,    LIFT_RADIUS * 0.866),  # slot 1: right-front
    (  LIFT_RADIUS * 0.5,   -LIFT_RADIUS * 0.866),  # slot 2: right-back
]
POSITION_TOLERANCE = 0.08   # metres — how close = "in position"

LINEAR_VEL   = 0.15
ANGULAR_VEL  = 0.0


class RLBoxLiftAgent(Node):
    """One RL agent node per robot. Robots share a common /lift_status topic."""

    def __init__(self, robot_name: str, slot_index: int):
        super().__init__(f'rl_agent_{robot_name}')
        self.robot_name = robot_name
        self.slot_index = slot_index
        self.target = TARGET_SLOTS[slot_index]

        # State
        self.my_x = 0.0
        self.my_y = 0.0
        self.box_x = 0.0   # updated via /box_pose topic
        self.box_y = 0.0
        self.in_position = False
        self.episode_step = 0

        # Q-table  key: (obs_disc, action)  value: Q
        self.q_table = {}
        self.epsilon = EPSILON

        # ROS interfaces
        ns = f'/{robot_name}'
        self.cmd_pub = self.create_publisher(TwistStamped, f'{ns}/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, f'{ns}/odom', self.odom_cb, 10)
        self.box_sub = self.create_subscription(
            Float32MultiArray, '/box_pose', self.box_cb, 10)

        # Publish whether this robot is in position
        self.status_pub = self.create_publisher(Bool, f'{ns}/in_position', 10)

        # Control loop at 10 Hz
        self.create_timer(0.1, self.step)
        self.get_logger().info(
            f'[{robot_name}] RL agent started — slot {slot_index} '
            f'target offset {self.target}')

    # ── Callbacks ────────────────────────────────────────────────
    def odom_cb(self, msg: Odometry):
        self.my_x = msg.pose.pose.position.x
        self.my_y = msg.pose.pose.position.y

    def box_cb(self, msg: Float32MultiArray):
        self.box_x = msg.data[0]
        self.box_y = msg.data[1]

    # ── Helpers ──────────────────────────────────────────────────
    def _abs_target(self):
        """Absolute target position = box centre + slot offset."""
        return (self.box_x + self.target[0],
                self.box_y + self.target[1])

    def _dist_to_target(self):
        tx, ty = self._abs_target()
        return math.hypot(self.my_x - tx, self.my_y - ty)

    def _get_obs(self):
        """Discretized observation tuple for Q-table lookup."""
        tx, ty = self._abs_target()
        dx = np.clip(int((self.my_x - tx) / 0.1) + GRID_SIZE // 2,
                     0, GRID_SIZE - 1)
        dy = np.clip(int((self.my_y - ty) / 0.1) + GRID_SIZE // 2,
                     0, GRID_SIZE - 1)
        return (dx, dy)

    def _get_q(self, obs, action):
        return self.q_table.get((obs, action), 0.0)

    def _best_action(self, obs):
        return max(range(5), key=lambda a: self._get_q(obs, a))

    def _choose_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        return self._best_action(obs)

    def _apply_action(self, action):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        tx, ty = self._abs_target()
        angle_to_target = math.atan2(ty - self.my_y, tx - self.my_x)

        if action == 0:    # move toward target
            msg.twist.linear.x  = LINEAR_VEL
        elif action == 1:  # move away
            msg.twist.linear.x  = -LINEAR_VEL
        elif action == 2:  # rotate left
            msg.twist.angular.z =  ANGULAR_VEL + 0.4
        elif action == 3:  # rotate right
            msg.twist.angular.z = -ANGULAR_VEL - 0.4
        else:              # stop
            pass
        self.cmd_pub.publish(msg)

    def _compute_reward(self, prev_dist, curr_dist):
        if curr_dist < POSITION_TOLERANCE:
            return 10.0          # reached slot
        return (prev_dist - curr_dist) * 10 - 0.1   # shaping + time penalty

    # ── Main RL step ─────────────────────────────────────────────
    def step(self):
        prev_dist = self._dist_to_target()
        obs       = self._get_obs()
        action    = self._choose_action(obs)

        self._apply_action(action)

        curr_dist = self._dist_to_target()
        reward    = self._compute_reward(prev_dist, curr_dist)
        next_obs  = self._get_obs()

        # Q-update
        best_next = max(self._get_q(next_obs, a) for a in range(5))
        old_q     = self._get_q(obs, action)
        new_q     = old_q + ALPHA * (reward + GAMMA * best_next - old_q)
        self.q_table[(obs, action)] = new_q

        # Epsilon decay
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        # Publish in-position status
        self.in_position = curr_dist < POSITION_TOLERANCE
        status_msg = Bool()
        status_msg.data = self.in_position
        self.status_pub.publish(status_msg)

        self.episode_step += 1
        if self.episode_step % 500 == 0:
            self.get_logger().info(
                f'[{self.robot_name}] step={self.episode_step} '
                f'ε={self.epsilon:.3f}  dist={curr_dist:.3f}  '
                f'in_pos={self.in_position}')


def main(args=None):
    rclpy.init(args=args)
    import sys
    robot_name = sys.argv[1] if len(sys.argv) > 1 else 'tb1'
    slot_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    node = RLBoxLiftAgent(robot_name, slot_index)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
