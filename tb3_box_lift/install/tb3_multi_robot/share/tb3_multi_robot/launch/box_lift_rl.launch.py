#!/usr/bin/env python3
"""
Launch file: starts Gazebo world + 3 TurtleBots + RL controller node.
Usage:
    ros2 launch tb3_multi_robot box_lift_rl.launch.py
"""

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    pkg = get_package_share_directory('tb3_multi_robot')

    # ── Gazebo world (reuse existing world launch) ─────────────────────────
    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'tb3_world.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # ── RL Controller node ─────────────────────────────────────────────────
    rl_controller_node = Node(
        package='tb3_multi_robot',
        executable='box_lift_rl_controller',
        name='box_lift_rl_controller',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation (Gazebo) clock'
        ),
        world_launch,
        rl_controller_node,
    ])
