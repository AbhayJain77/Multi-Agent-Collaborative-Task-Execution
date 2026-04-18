#!/usr/bin/env python3
"""
RL Box-Lift Launch File
========================
Launches:
  1. box_pose_publisher  — publishes /box_pose
  2. lift_coordinator    — monitors all robots, publishes /box_lifted
  3. rl_agent (×3)       — one per robot with its slot index
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # Box pose publisher
    ld.add_action(Node(
        package='tb3_multi_robot',
        executable='box_pose_publisher',
        output='screen',
    ))

    # Lift coordinator
    ld.add_action(Node(
        package='tb3_multi_robot',
        executable='lift_coordinator',
        output='screen',
    ))

    # RL agents — one per robot, each assigned a unique slot (0, 1, 2)
    for i, robot in enumerate(['tb1', 'tb2', 'tb3']):
        ld.add_action(Node(
            package='tb3_multi_robot',
            executable='rl_agent',
            name=f'rl_agent_{robot}',
            arguments=[robot, str(i)],
            output='screen',
        ))

    return ld
