"""Ewaluacja: pose_error_logger (ground truth z Gazebo przez gz-transport, /odom, TF AMCL).

Uruchamiac obok symulacji i cube_search.launch.py, z tym samym GZ_IP co symulacja;
wyniki w ~/.ros/cube_search/eval/ (pose_errors.csv, detections.csv),
wykresy: tools/plot_eval.py.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    topic = PathJoinSubstitution(['/world', LaunchConfiguration('world'), 'dynamic_pose', 'info'])
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='default'),
        Node(package='cube_search', executable='pose_error_logger', name='pose_error_logger',
             parameters=[{'use_sim_time': True, 'gt_topic': topic}], output='screen'),
    ])
