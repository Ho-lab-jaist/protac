import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    package_name = "protac_perception"
    config = os.path.join(
    get_package_share_directory(package_name),
    'config',
    'depth_sensing.config.yaml'
    )


    img1_depth_sensing = Node(
        package="protac_perception",
        namespace="cam1",
        executable="depth_sensing_node",
        name='depth_sensing_node',
        parameters = [config],
        remappings=[
        ('/video_frames', '/cam1/video_frames'),
        ]
    )

    img2_depth_sensing = Node(
        package="protac_perception",
        namespace="cam2",
        executable="depth_sensing_node",
        name='depth_sensing_node',
        parameters = [config],
        remappings=[
        ('/video_frames', '/cam2/video_frames'),
        ]
    )

    img3_depth_sensing = Node(
        package="protac_perception",
        namespace="cam3",
        executable="depth_sensing_node",
        name='depth_sensing_node',
        parameters = [config],
        remappings=[
        ('/video_frames', '/cam3/video_frames'),
        ]
    )


    img4_depth_sensing = Node(
        package="protac_perception",
        namespace="cam4",
        executable="depth_sensing_node",
        name='depth_sensing_node',
        parameters = [config],
        remappings=[
        ('/video_frames', '/cam4/video_frames'),
        ]
    )

    return LaunchDescription([
        img1_depth_sensing,
        img2_depth_sensing,
        img3_depth_sensing,
        img4_depth_sensing,
    ])