"""Uruchamia aruco_detector, cube_search_server (akcja /search_cubes) i explore_lite.

Wymaga dzialajacej symulacji: ros2 launch arm05_sim world.launch.py
Start wyszukiwania:          ros2 run cube_search search_client
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('cube_search'), 'config',
                                  'params.yaml')
    params = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    common = [params, {'use_sim_time': use_sim_time}]

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('explore', default_value='true',
                              description='uruchom explore_lite'),
        Node(package='cube_search', executable='aruco_detector', name='aruco_detector',
             output='screen', parameters=common),
        Node(package='cube_search', executable='search_server', name='cube_search_server',
             output='screen', parameters=common),
        Node(package='explore_lite', executable='explore', name='explore_node',
             output='screen', parameters=common,
             condition=IfCondition(LaunchConfiguration('explore'))),
    ])
