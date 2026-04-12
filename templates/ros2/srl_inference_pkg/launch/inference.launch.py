from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path"),
            DeclareLaunchArgument("checkpoint_path"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument("hz", default_value="50.0"),
            Node(
                package="srl_inference_pkg",
                executable="srl_inference_node",
                name="srl_inference_node",
                output="screen",
                parameters=[
                    {
                        "config_path": LaunchConfiguration("config_path"),
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "device": LaunchConfiguration("device"),
                        "hz": LaunchConfiguration("hz"),
                    }
                ],
            ),
        ]
    )