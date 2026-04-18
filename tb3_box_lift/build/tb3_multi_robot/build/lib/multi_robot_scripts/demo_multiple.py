#!/usr/bin/env python3
"""
Run 3 demo episodes — robots start from different positions each time.
Shows RL generalizes across different starting configurations.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from stable_baselines3 import PPO
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose

TARGETS = {
    "tb1": np.array([5.0,  5.5]),
    "tb2": np.array([4.57, 4.75]),
    "tb3": np.array([5.43, 4.75]),
}

# Different start positions for each episode
EPISODE_STARTS = [
    # Episode 1 — robots far from box
    {"tb1": (0.0, -3.0), "tb2": (-3.0, 3.0), "tb3": (3.0, 3.0)},
    # Episode 2 — robots on opposite sides
    {"tb1": (8.0, 8.0),  "tb2": (2.0, 8.0),  "tb3": (8.0, 2.0)},
    # Episode 3 — robots close but wrong side
    {"tb1": (3.0, 3.0),  "tb2": (7.0, 3.0),  "tb3": (5.0, 8.0)},
]

LIFT_THRESHOLD = 0.5
ROBOT_NAMES    = ["tb1", "tb2", "tb3"]
LINEAR_VEL     = 0.3
ANGULAR_VEL    = 0.3
MODEL_PATH     = "/home/abhay/ros2_ws/src/tb3_box_lift/multi_robot_scripts/box_lift_final"


class MultiEpisodeDemo(Node):
    def __init__(self):
        super().__init__("multi_episode_demo")
        self.model = PPO.load(MODEL_PATH)
        self.get_logger().info("RL model loaded")

        self._pos    = {n: None for n in ROBOT_NAMES}
        self._yaw    = {n: 0.0  for n in ROBOT_NAMES}
        self._lifted = False
        self._episode = 0
        self._pubs   = {}

        for name in ROBOT_NAMES:
            self._pubs[name] = self.create_publisher(
                TwistStamped, f"/{name}/cmd_vel", 10)
            self.create_subscription(
                Odometry, f"/{name}/odom",
                lambda msg, n=name: self._odom_cb(msg, n), 10)

        # Gazebo set entity state client for teleporting robots
        self._set_state = self.create_client(
            SetEntityState, "/gazebo/set_entity_state")

        self.create_timer(0.1, self._loop)
        self.get_logger().info(f"Starting Episode 1/{len(EPISODE_STARTS)}")

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

    def _teleport_robots(self, episode):
        starts = EPISODE_STARTS[episode]
        for name, (x, y) in starts.items():
            req = SetEntityState.Request()
            req.state = EntityState()
            req.state.name = name
            req.state.pose = Pose()
            req.state.pose.position.x = x
            req.state.pose.position.y = y
            req.state.pose.position.z = 0.01
            self._set_state.call_async(req)
        self.get_logger().info(f"Teleported robots to Episode {episode+1} positions")

    def _loop(self):
        if any(self._pos[n] is None for n in ROBOT_NAMES):
            return

        if self._lifted:
            self._stop_all()
            # Start next episode after 3 seconds
            if self._episode < len(EPISODE_STARTS) - 1:
                self._episode += 1
                self._lifted = False
                self._pos = {n: None for n in ROBOT_NAMES}
                self._teleport_robots(self._episode)
                self.get_logger().info(
                    f"Starting Episode {self._episode+1}/{len(EPISODE_STARTS)}")
            else:
                self.get_logger().info("All episodes complete!")
            return

        obs = []
        for name in ROBOT_NAMES:
            rel = TARGETS[name] - self._pos[name]
            obs += [rel[0], rel[1]]
        obs = np.array(obs, dtype=np.float32)
        joint_action, _ = self.model.predict(obs, deterministic=True)

        dists = {n: self._dist(n) for n in ROBOT_NAMES}
        self.get_logger().info(
            f"Ep{self._episode+1} — tb1:{dists['tb1']:.2f}  "
            f"tb2:{dists['tb2']:.2f}  tb3:{dists['tb3']:.2f}",
            throttle_duration_sec=1.0)

        if all(d < LIFT_THRESHOLD for d in dists.values()):
            self._lifted = True
            self.get_logger().info(
                f"Episode {self._episode+1} — BOX LIFTED!")
            return

        for i, name in enumerate(ROBOT_NAMES):
            self._move(name, joint_action[i])

    def _move(self, name, action):
        to_target = TARGETS[name] - self._pos[name]
        dist      = np.linalg.norm(to_target)
        msg       = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        if dist < LIFT_THRESHOLD:
            msg.twist.linear.x  = 0.0
            msg.twist.angular.z = 0.0
        else:
            desired_yaw = math.atan2(to_target[1], to_target[0])
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
    rclpy.spin(MultiEpisodeDemo())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
