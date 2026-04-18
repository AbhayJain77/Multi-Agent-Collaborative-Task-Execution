#!/usr/bin/env python3
import subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class BoxPosePublisher(Node):
    def __init__(self):
        super().__init__("box_pose_publisher")
        self._pub = self.create_publisher(PoseStamped, "/box_pose", 10)
        self.create_timer(0.5, self._publish)
        self.get_logger().info("Box pose publisher started")

    def _publish(self):
        try:
            result = subprocess.run(
                ["gz", "model", "-m", "lift_box", "-p"],
                capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.strip().split("\n"):
                if "Pose" in line or "position" in line.lower():
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x = float(parts[-6])
                            y = float(parts[-5])
                            msg = PoseStamped()
                            msg.header.stamp    = self.get_clock().now().to_msg()
                            msg.header.frame_id = "world"
                            msg.pose.position.x = x
                            msg.pose.position.y = y
                            self._pub.publish(msg)
                            return
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            self.get_logger().warn(f"Box pose error: {e}",
                                   throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(BoxPosePublisher())
    rclpy.shutdown()


if __name__ == "__main__":
    main()