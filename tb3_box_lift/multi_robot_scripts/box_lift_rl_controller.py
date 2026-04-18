#!/usr/bin/env python3
import os
import math
import subprocess
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from stable_baselines3 import PPO

_R             = 0.6
LIFT_THRESHOLD = 0.7
ROBOT_NAMES    = ["tb1", "tb2", "tb3"]
LINEAR_VEL     = 0.5
ANGULAR_VEL    = 0.5

# Triangle offsets around box
OFFSETS = {
    "tb1": np.array([ 0.0,   _R   ]),   # north
    "tb2": np.array([-_R,    0.0  ]),   # west  
    "tb3": np.array([ _R,    0.0  ]),   # east
}

ACTION_ANGLES = {
    0: None,
    1: 0.0,
    2: math.pi,
    3: math.pi / 2,
    4: -math.pi / 2,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "box_lift_final")

# Gazebo model names for each robot
GAZEBO_NAMES = {
    "tb1": "tb1_burger",
    "tb2": "tb2_burger",
    "tb3": "tb3_burger",
}


def gz_get_pose(model_name):
    """Read world position of any Gazebo model."""
    try:
        result = subprocess.run(
            ["gz", "model", "-m", model_name, "-p"],
            capture_output=True, text=True, timeout=3
        )
        lines = result.stdout.split("\n")
        for i, line in enumerate(lines):
            if "Pose [ XYZ" in line:
                coord_line = lines[i + 1].strip().strip("[]")
                parts = coord_line.split()
                x = float(parts[0])
                y = float(parts[1])
                return np.array([x, y])
    except Exception as e:
        pass
    return None


class BoxLiftController(Node):
    def __init__(self):
        super().__init__("box_lift_controller")

        self.model = PPO.load(MODEL_PATH)
        self.get_logger().info("RL model loaded successfully")

        self._odom_pos    = {n: None for n in ROBOT_NAMES}  # raw odom
        self._world_pos   = {n: None for n in ROBOT_NAMES}  # world frame
        self._spawn_pos   = {n: None for n in ROBOT_NAMES}  # read from Gazebo
        self._yaw         = {n: 0.0  for n in ROBOT_NAMES}
        self._box_pos     = None
        self._lifted      = False
        self._approaching = False
        self._initialized = False
        self._pubs        = {}

        for name in ROBOT_NAMES:
            self._pubs[name] = self.create_publisher(
                TwistStamped, f"/{name}/cmd_vel", 10)
            self.create_subscription(
                Odometry, f"/{name}/odom",
                lambda msg, n=name: self._odom_cb(msg, n), 10)

        # Try to initialize from Gazebo every second until success
        self.create_timer(1.0, self._try_init)
        self.create_timer(0.1, self._loop)
        self.get_logger().info("Waiting for Gazebo initialization...")

    def _try_init(self):
        """Read all positions from Gazebo at startup."""
        if self._initialized:
            return

        # Read box position
        box = gz_get_pose("lift_box")
        if box is None:
            self.get_logger().info("Waiting for box...", throttle_duration_sec=2.0)
            return
        self._box_pos = box

        # Read each robot's spawn position
        all_found = True
        for name in ROBOT_NAMES:
            gz_name = GAZEBO_NAMES[name]
            pos = gz_get_pose(gz_name)
            if pos is None:
                self.get_logger().info(f"Waiting for {gz_name}...",
                                       throttle_duration_sec=2.0)
                all_found = False
            else:
                self._spawn_pos[name] = pos

        if not all_found:
            return

        self._initialized = True
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Box position: {self._box_pos}")
        for name in ROBOT_NAMES:
            t = self._get_target(name)
            self.get_logger().info(
                f"  {name} spawn:{self._spawn_pos[name]}  target:{t}")
        self.get_logger().info("=" * 50)

    def _odom_cb(self, msg, name):
        p = msg.pose.pose.position
        self._odom_pos[name] = np.array([p.x, p.y])

        # Convert to world frame using Gazebo spawn position
        if self._spawn_pos[name] is not None:
            world = self._spawn_pos[name] + np.array([p.x, p.y])
            if self._world_pos[name] is not None:
                if np.linalg.norm(world - self._world_pos[name]) > 5.0:
                    return
            self._world_pos[name] = world

        _, _, yaw = euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w])
        self._yaw[name] = yaw

    def _get_target(self, name):
        if self._box_pos is None:
            return None
        return self._box_pos + OFFSETS[name]

    def _dist(self, name):
        if self._world_pos[name] is None:
            return 999.0
        target = self._get_target(name)
        if target is None:
            return 999.0
        return float(np.linalg.norm(self._world_pos[name] - target))

    def _loop(self):
        if not self._initialized:
            return

        if any(self._world_pos[n] is None for n in ROBOT_NAMES):
            self.get_logger().info("Waiting for odom...", throttle_duration_sec=2.0)
            return

        if self._approaching:
            self._approach_box()
            return

        if self._lifted:
            self._stop_all()
            return

        # Build obs — relative vector to target in world frame
        obs = []
        for name in ROBOT_NAMES:
            target = self._get_target(name)
            rel    = target - self._world_pos[name]
            obs   += [float(rel[0]), float(rel[1])]
        obs = np.array(obs, dtype=np.float32)

        joint_action, _ = self.model.predict(obs, deterministic=True)

        dists = {n: self._dist(n) for n in ROBOT_NAMES}
        self.get_logger().info(
            f"box:{self._box_pos.round(2)}  "
            f"tb1:{dists['tb1']:.2f}  tb2:{dists['tb2']:.2f}  "
            f"tb3:{dists['tb3']:.2f}  rl:{joint_action}",
            throttle_duration_sec=1.0)

        if all(d < LIFT_THRESHOLD for d in dists.values()):
            self._lifted      = True
            self._approaching = True
            self.get_logger().info("BOX LIFTED! Approaching box...")
            return

        for i, name in enumerate(ROBOT_NAMES):
            self._move(name, int(joint_action[i]))

    def _approach_box(self):
        all_close = True
        for name in ROBOT_NAMES:
            if self._world_pos[name] is None:
                continue
            to_box = self._box_pos - self._world_pos[name]
            dist   = np.linalg.norm(to_box)
            msg    = TwistStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            if dist > 0.17:
                all_close           = False
                desired_yaw         = math.atan2(to_box[1], to_box[0])
                yaw_err             = desired_yaw - self._yaw[name]
                yaw_err             = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                msg.twist.linear.x  = 0.3 * (1.0 - min(abs(yaw_err), 1.0))
                msg.twist.angular.z = 0.5 * yaw_err
            self._pubs[name].publish(msg)

        if all_close:
            self._approaching = False
            self.get_logger().info("All robots touching box!")
            self._stop_all()

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

        # Collision avoidance
        for other_name in ROBOT_NAMES:
            if other_name == name or self._world_pos[other_name] is None:
                continue
            to_other   = self._world_pos[name] - self._world_pos[other_name]
            dist_other = np.linalg.norm(to_other)
            if dist_other < 0.5 and dist_other > 0:
                push = (to_other / dist_other) * (0.5 - dist_other)
                msg.twist.linear.x  -= push[0] * 0.3
                msg.twist.angular.z += push[1] * 0.3

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