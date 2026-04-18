#!/usr/bin/env python3
"""
Lift Coordinator Node
======================
Monitors all 3 robots' /in_position topics.
When ALL three are in position → publishes /box_lifted = True
and logs the successful lift event.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

ROBOT_NAMES = ['tb1', 'tb2', 'tb3']


class LiftCoordinator(Node):
    def __init__(self):
        super().__init__('lift_coordinator')
        self.statuses = {name: False for name in ROBOT_NAMES}
        self.lift_count = 0

        for name in ROBOT_NAMES:
            self.create_subscription(
                Bool, f'/{name}/in_position',
                lambda msg, n=name: self._status_cb(msg, n), 10)

        self.lifted_pub = self.create_publisher(Bool, '/box_lifted', 10)
        self.create_timer(0.2, self._check)
        self.get_logger().info('Lift coordinator started — watching 3 robots')

    def _status_cb(self, msg: Bool, name: str):
        self.statuses[name] = msg.data

    def _check(self):
        all_in = all(self.statuses.values())
        msg = Bool()
        msg.data = all_in
        self.lifted_pub.publish(msg)
        if all_in:
            self.lift_count += 1
            if self.lift_count % 10 == 1:   # log every 10th trigger
                self.get_logger().info(
                    f'BOX LIFTED! All robots in position. '
                    f'(event #{self.lift_count})')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LiftCoordinator())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
