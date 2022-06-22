import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    package_name = "protac_perception"

    upper_arm_force_sensing = Node(
        package="protac_perception",
        namespace="cam4",
        executable="force_sensing_node",
        name='force_sensing_node',
        remappings=[
        ('/video_frames', '/cam4/video_frames'),
        ]
    )

    return LaunchDescription([
        upper_arm_force_sensing,
    ])