from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("policy_path"),
            DeclareLaunchArgument("checkpoint", default_value="auto"),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument("hz", default_value="50.0"),
            Node(
                package="srl_inference_pkg",
                executable="srl_inference_node",
                name="srl_inference_node",
                output="screen",
                parameters=[
                    {
                        "policy_path": LaunchConfiguration("policy_path"),
                        "checkpoint": LaunchConfiguration("checkpoint"),
                        "config_path": LaunchConfiguration("config_path"),
                        "device": LaunchConfiguration("device"),
                        "hz": LaunchConfiguration("hz"),
                    }
                ],
            ),
        ]
    )
