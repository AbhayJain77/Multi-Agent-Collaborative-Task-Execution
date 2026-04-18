#!/usr/bin/env python3
import os
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from stable_baselines3 import PPO

# Box is at world origin (0,0)
# Targets = tight triangle around box
_R = 0.25
TARGETS = {
    "tb1": np.array([3.0,  3.0 ]) + np.array([ 0.0,  _R   ]),
    "tb2": np.array([-3.0, 3.0 ]) + np.array([-_R,  -_R/2 ]),
    "tb3": np.array([0.0, -4.0 ]) + np.array([ _R,  -_R/2 ]),
}

# RL action → heading angle
ACTION_ANGLES = {
    0: None,
    1: 0.0,
    2: math.pi,
    3: math.pi / 2,
    4: -math.pi / 2,
}

LIFT_THRESHOLD = 0.4
ROBOT_NAMES    = ["tb1", "tb2", "tb3"]
LINEAR_VEL     = 0.3
ANGULAR_VEL    = 0.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "box_lift_final")


class BoxLiftController(Node):
    def __init__(self):
        super().__init__("box_lift_controller")

        self.model = PPO.load(MODEL_PATH)
        self.get_logger().info("RL model loaded successfully")

        self._pos    = {n: None for n in ROBOT_NAMES}
        self._yaw    = {n: 0.0  for n in ROBOT_NAMES}
        self._lifted = False
        self._pubs   = {}

        for name in ROBOT_NAMES:
            self._pubs[name] = self.create_publisher(
                TwistStamped, f"/{name}/cmd_vel", 10)
            self.create_subscription(
                Odometry, f"/{name}/odom",
                lambda msg, n=name: self._odom_cb(msg, n), 10)

        self.create_timer(0.1, self._loop)
        self.get_logger().info("Controller started — waiting for odom...")
        for name in ROBOT_NAMES:
            self.get_logger().info(f"  {name} target: {TARGETS[name]}")

    def _odom_cb(self, msg, name):
        p = msg.pose.pose.position
        new_pos = np.array([p.x, p.y])
        if self._pos[name] is not None:
            if np.linalg.norm(new_pos - self._pos[name]) > 5.0:
                return
        self._pos[name] = new_pos
        _, _, yaw = euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w])
        self._yaw[name] = yaw

    def _dist(self, name):
        if self._pos[name] is None:
            return 999.0
        return float(np.linalg.norm(self._pos[name] - TARGETS[name]))

    def _loop(self):
        if any(self._pos[n] is None for n in ROBOT_NAMES):
            self.get_logger().info("Waiting for odom...", throttle_duration_sec=2.0)
            return

        if self._lifted:
            self._stop_all()
            return

        # Build obs — relative vector to target per robot
        obs = []
        for name in ROBOT_NAMES:
            rel = TARGETS[name] - self._pos[name]
            obs += [float(rel[0]), float(rel[1])]
        obs = np.array(obs, dtype=np.float32)

        # RL decides action
        joint_action, _ = self.model.predict(obs, deterministic=True)

        dists = {n: self._dist(n) for n in ROBOT_NAMES}
        self.get_logger().info(
            f"tb1:{dists['tb1']:.2f}  tb2:{dists['tb2']:.2f}  "
            f"tb3:{dists['tb3']:.2f}  rl:{joint_action}",
            throttle_duration_sec=1.0)

        if all(d < LIFT_THRESHOLD for d in dists.values()):
            self._lifted = True
            self.get_logger().info("BOX LIFTED! All robots in position.")
            self._stop_all()
            return

        for i, name in enumerate(ROBOT_NAMES):
            self._move(name, int(joint_action[i]))

    def _move(self, name, rl_action):
        dist = self._dist(name)
        msg  = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        if dist < LIFT_THRESHOLD or rl_action == 0:
            msg.twist.linear.x  = 0.0
            msg.twist.angular.z = 0.0
        else:
            desired_yaw = ACTION_ANGLES[rl_action]
            yaw_err     = desired_yaw - self._yaw[name]
            yaw_err     = (yaw_err + math.pi) % (2 * math.pi) - math.pi
            msg.twist.linear.x  = LINEAR_VEL * (1.0 - min(abs(yaw_err), 1.0))
            msg.twist.angular.z = ANGULAR_VEL * yaw_err

        self._pubs[name].publish(msg)

    def _stop_all(self):
        for name in ROBOT_NAMES:
            msg = TwistStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            self._pubs[name].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(BoxLiftController())
    rclpy.shutdown()


if __name__ == "__main__":
    main()