import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    package_name = "protac_perception"
    config = os.path.join(
    get_package_share_directory(package_name),
    'config',
    'human_detector.config.yaml'
    )


    img1_distance_sensing = Node(
        package="protac_perception",
        namespace="cam1",
        executable="distance_sensing_node",
        name='distance_sensing_node',
        parameters = [config],
        remappings=[
        ('/video_frames', '/cam1/video_frames'),
        ('/protac_perception/object_area', '/cam1/protac_perception/object_area'),
        ]
    )

    img2_distance_sensing = Node(
        package="protac_perception",
        namespace="cam2",
        executable="distance_sensing_node",
        name='distance_sensing_node',
        parameters = [config],
        remappings=[
        ('/video_frames', '/cam2/video_frames'),
        ('/protac_perception/object_area', '/cam2/protac_perception/object_area'),
        ]
    )

    img3_distance_sensing = Node(
        package="protac_perception",
        namespace="cam3",
        executable="distance_sensing_node",
        name='distance_sensing_node',
        parameters = [config],
        remappings=[
        ('/video_frames', '/cam3/video_frames'),
        ('/protac_perception/object_area', '/cam3/protac_perception/object_area'),
        ]
    )

    img4_distance_sensing = Node(
        package="protac_perception",
        namespace="cam4",
        executable="distance_sensing_node",
        name='distance_sensing_node',
        parameters = [config],
        remappings=[
        ('/video_frames', '/cam4/video_frames'),
        ('/protac_perception/object_area', '/cam4/protac_perception/object_area'),
        ]
    )

    return LaunchDescription([
        img1_distance_sensing,
        img2_distance_sensing,
        img3_distance_sensing,
        img4_distance_sensing,
    ])